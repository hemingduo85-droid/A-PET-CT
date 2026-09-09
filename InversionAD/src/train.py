
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path

import copy
import argparse
import yaml
from pprint import pprint

import types

from src.datasets import build_dataset
from src.utils import get_optimizer, get_lr_scheduler
from src.denoiser import get_denoiser, Denoiser
from src.backbones import get_backbone, get_backbone_feature_shape
from src.evaluate import evaluate_inv
from src.evaluate import format_protocol_metrics

from einops import rearrange
from sklearn.metrics import roc_curve, roc_auc_score

import wandb
from dotenv import load_dotenv

# Load environment variables from .env file
try: 
    load_dotenv()
    use_wandb = (os.getenv("WANDB_API_KEY") is not None)
    if use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
except ImportError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="InvAD Training")
    
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')
    args = parser.parse_args()
    return args

def postprocess(x):
    x = x / 2 + 0.5
    return x.clamp(0, 1)

def convert2image(x):
    if x.dim() == 3:
        return x.permute(1, 2, 0).cpu().numpy()
    elif x.dim() == 4:
        return x.permute(0, 2, 3, 1).cpu().numpy()
    else:
        return x.cpu().numpy()
    
def main(config):
    pprint(config)
    
    if use_wandb:
        # create wandb project
        project = os.environ.get("WANDB_PROJECT")
        if project is None:
            raise ValueError("Please set the WANDB_PROJECT environment variable.")
        entity = os.environ.get("WANDB_ENTITY")
        if entity is None:
            raise ValueError("Please set the WANDB_ENTITY environment variable.")
        wandb.init(project=project, entity=entity, config=config)
    
    # set seed
    seed = config['meta']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    dataset_config = copy.deepcopy(config['data'])
    device = config['meta']['device']
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        dev_idx = torch.device(device).index
        if dev_idx is None:
            dev_idx = torch.cuda.current_device()
        torch.cuda.set_device(dev_idx)
        free_mem, total_mem = torch.cuda.mem_get_info(dev_idx)
        print(
            f"Training device: cuda:{dev_idx} ({torch.cuda.get_device_name(dev_idx)}), "
            f"free={free_mem / 1024**3:.2f}GiB / total={total_mem / 1024**3:.2f}GiB"
        )
    else:
        print(f"Training device: {device}")
    batch_size = config['data']['batch_size']
    train_dataset = build_dataset(**dataset_config)
    eval_dataset_config = copy.deepcopy(dataset_config)
    eval_dataset_config['train'] = False
    eval_dataset_config['anom_only'] = True
    anom_dataset = build_dataset(**eval_dataset_config)
    eval_dataset_config['anom_only'] = False
    eval_dataset_config['normal_only'] = True
    normal_dataset = build_dataset(**eval_dataset_config)
    anom_loader = [DataLoader(anom_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)]
    normal_loader = [DataLoader(normal_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)]

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, \
        pin_memory=config['data']['pin_memory'], num_workers=config['data']['num_workers'], drop_last=True)

    diff_in_sh = get_backbone_feature_shape(model_type=config['backbone']['model_type'])
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    ema_decay = config['diffusion']['ema_decay']
    model_ema = copy.deepcopy(model)
    model.to(device)
    model_ema.to(device)

    backbone_kwargs = config['backbone']
    print(f"Using feature space reconstruction with {backbone_kwargs['model_type']} backbone")
    
    feature_extractor = get_backbone(**backbone_kwargs)
    feature_extractor.to(device).eval()

    optimizer = get_optimizer([model], **config['optimizer'])
    if config['optimizer']['scheduler_type'] == 'none':
        pass
    else:
        scheduler = get_lr_scheduler(optimizer, **config['optimizer'], iter_per_epoch=len(train_loader))
    
    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # save config
    save_path = save_dir / "config.yaml"
    with open(save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config is saved at {save_path}")
    
    # Number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")
    
    model.train()
    print(f"Steps per epoch: {len(train_loader)}")
    
    num_epochs = config['optimizer']['num_epochs']
    final_epoch = num_epochs
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            img, labels = data["samples"], data["clslabels"]    # (B, C, H, W), (B,)
            img = img.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                x, _ = feature_extractor(img)  # (B, c, h, w)
            loss = model(x, labels)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
                    
            if config['optimizer']['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['grad_clip'])
            optimizer.step()
            
            scheduler.step()
            
            # update ema
            for ema_param, model_param in zip(model_ema.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1.0 - ema_decay)
            
            if i % config["logging"]["log_interval"] == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss {loss.item()}")      
                if use_wandb:
                    wandb.log({"Loss": loss.item(), "LR": scheduler.get_last_lr()})
                
        if (epoch + 1) == final_epoch:
            model_path = save_dir / f"model_epoch_{final_epoch}.pth"
            torch.save(model.state_dict(), model_path)
            latest_path = save_dir / "model_latest.pth"
            torch.save(model.state_dict(), latest_path)
            ema_latest_path = save_dir / "model_ema_latest.pth"
            torch.save(model_ema.state_dict(), ema_latest_path)
            print(f"Final epoch model is saved at {model_path}")

            save_heatmaps = config.get("evaluation", {}).get("save_heatmaps", 0)
            dummy_args = types.SimpleNamespace(
                visualize_samples=save_heatmaps != 0,
                save_heatmaps=save_heatmaps,
                save_all_heatmaps=save_heatmaps == -1,
                save_dir=str(save_dir),
            )
            metrics_dict = evaluate_inv(
                model,
                feature_extractor,
                anom_loader,
                normal_loader,
                config, 
                diff_in_sh,
                epoch + 1,
                config["evaluation"]["eval_step"],
                device,
                dummy_args,
            )
            cat_metrics = metrics_dict[config["data"]["category"]]

            if use_wandb:
                wandb.log({
                    "img_auroc": cat_metrics["img_auroc"],
                    "img_ap": cat_metrics["img_ap"],
                    "img_f1": cat_metrics["img_f1"],
                    "px_auroc_abn": cat_metrics["px_auroc_abn"],
                    "px_aupr_abn": cat_metrics["px_aupr_abn"],
                    "pat_auroc": cat_metrics["pat_auroc"],
                    "pat_ap": cat_metrics["pat_ap"],
                    "pat_f1": cat_metrics["pat_f1"],
                })
            print(f"Epoch {epoch + 1}\n{format_protocol_metrics(cat_metrics)}")
            
    print("Training is done!")

if __name__ == "__main__":
    args = parse_args()
    main(args)

    
    
    
        
      
            
    
        
            
            
            
            
            
            
