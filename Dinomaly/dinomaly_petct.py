"""
Dinomaly for PET-CT anomaly detection.
Supports three input modes: ct, pet, dual (CT+PET pseudo-RGB).
Trains for 30 epochs by default, evaluates once after training, saves the final epoch.

单模态
nohup python dinomaly_petct.py --input_mode pet --gpu 2 --save_name pet_psma > pet_psma.log 2>&1 &

双模态
cd /data/cyf/codes/A-PET-CT/Dinomaly
python dinomaly_petct.py --input_mode dual --gpu 2 --save_name petct_fdg > petct_fdg.log 2>&1 &

调参
 python dinomaly_petct.py \
    --input_mode pet \
    --gpu 0 \
    --num_epochs 30 \
    --batch_size 16 \
    --seed 42 \
    --save_name pet_psma

导出npz
nohup python -u dinomaly_petct.py \
  --eval_only \
  --export_only \
  --ckpt_path ./saved_results/pet_psma/epoch_30_model.pth \
  --input_mode dual \
  --gpu 4 \
  --batch_size 16 \
  --seed 42 \
  --data_path /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA \
  --save_name pet_psma_export \
  --vis_num 0 \
  --cache_path ./saved_results/pet_psma_export/PSMA_petct_eval_cache.npz \
  > psma_export.log 2>&1 &

计算
nohup python -u compute_petct_cache_metrics.py \
  --cache ./saved_results/pet_psma_export/PSMA_petct_eval_cache.npz \
  --output ./saved_results/pet_psma_export/PSMA_petct_metrics_ci.txt \
  --bootstrap_iters 500 \
  > psma_ci_npz.log 2>&1 &
"""

import torch
import torch.nn as nn
from dataset import get_data_transforms, PetCTDataset
import numpy as np
import random
import os
from torch.utils.data import DataLoader

from models.uad import ViTill
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
import argparse
from utils import evaluation_petct, format_petct_metrics, global_cosine_hm_percent, WarmCosineScheduler
from functools import partial
from optimizers import StableAdamW
from petct_run_config import resolve_anomaly_dir, resolve_checkpoint_path
import warnings
import logging

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_petct_loaders(args, train=True):
    image_size = 256
    crop_size = 252  # must be divisible by patch_size 14: 252/14=18

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    test_data = PetCTDataset(
        data_root=args.data_path,
        transform=data_transform,
        gt_transform=gt_transform,
        phase='test',
        input_mode=args.input_mode,
    )

    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    if not train:
        return None, test_dataloader, None, test_data, image_size, crop_size

    train_data = PetCTDataset(
        data_root=args.data_path,
        transform=data_transform,
        gt_transform=gt_transform,
        phase='train',
        input_mode=args.input_mode,
    )
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    return train_dataloader, test_dataloader, train_data, test_data, image_size, crop_size


def build_model(device):
    encoder_name = 'dinov2reg_vit_base_14'

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    embed_dim, num_heads = 768, 12

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])

    decoder = nn.ModuleList()
    for _ in range(8):
        blk = VitBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn_drop=0., attn=LinearAttention2,
        )
        decoder.append(blk)

    model = ViTill(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder,
        target_layers=target_layers, mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    )
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])
    return model, trainable


def init_trainable(trainable):
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def run_final_evaluation(model, test_dataloader, args, print_fn, device):
    print_fn('--- Export eval cache ---' if args.export_only else '--- Final evaluation ---')
    anomaly_dir = resolve_anomaly_dir(args)
    slice_metrics, patient_metrics, saved_vis = evaluation_petct(
        model,
        test_dataloader,
        device,
        max_ratio=args.max_ratio,
        vis_dir=anomaly_dir,
        vis_num=args.vis_num,
        ci_bootstrap=args.ci_bootstrap,
        ci_seed=args.ci_seed,
        ci_pixel_max_samples=args.ci_pixel_max_samples,
        exact_pixel_points=not args.hist_pixel_points,
        exact_pixel_auroc=args.exact_pixel_auroc,
        exact_pixel_aupr=args.exact_pixel_aupr,
        progress_fn=print_fn,
        progress_interval=args.progress_interval,
        cache_path=args.cache_path,
        export_only=args.export_only,
    )
    if args.cache_path:
        print_fn(f'Saved eval cache to {args.cache_path}')
    if args.export_only:
        if saved_vis:
            print_fn(f'Saved anomaly images to {anomaly_dir} ({saved_vis} cases)')
        print_fn('Export only: skipped metric computation.')
        return None, None
    print_fn(format_petct_metrics(slice_metrics, patient_metrics))
    if saved_vis:
        print_fn(f'Saved anomaly images to {anomaly_dir} ({saved_vis} cases)')
    return slice_metrics, patient_metrics


def load_checkpoint(model, ckpt_path, device):
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def train(args, print_fn, device):
    setup_seed(args.seed)
    train_dataloader, test_dataloader, train_data, test_data, image_size, crop_size = build_petct_loaders(args, train=True)
    print_fn(f'Input mode: {args.input_mode}')
    print_fn(f'Image size: {image_size}, Crop size: {crop_size}')
    print_fn(f'Train samples: {len(train_data)}, Test samples: {len(test_data)}')

    model, trainable = build_model(device)
    init_trainable(trainable)

    total_iters = args.num_epochs * len(train_dataloader)
    warmup_iters = min(100, total_iters // 10)

    optimizer = StableAdamW(
        [{'params': trainable.parameters()}],
        lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-8,
    )
    lr_scheduler = WarmCosineScheduler(
        optimizer, base_value=2e-3, final_value=2e-4,
        total_iters=total_iters, warmup_iters=warmup_iters,
    )

    print_fn(f'Trainable parameters: {count_parameters(trainable)}')
    print_fn(f'Total iterations: {total_iters}, Warmup: {warmup_iters}')

    it = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        loss_list = []

        for img, label in train_dataloader:
            img = img.to(device)

            en, de = model(img)

            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()
            it += 1

        avg_loss = np.mean(loss_list)
        print_fn(f'Epoch [{epoch}/{args.num_epochs}], Loss: {avg_loss:.4f}')

    save_path = os.path.join(args.save_dir, args.save_name, f'epoch_{args.num_epochs:02d}_model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'input_mode': args.input_mode,
    }, save_path)

    print_fn('=' * 80)
    print_fn(f'Saved final epoch model to {save_path}')
    slice_metrics, patient_metrics = run_final_evaluation(model, test_dataloader, args, print_fn, device)

    return slice_metrics, patient_metrics


def evaluate_only(args, print_fn, device):
    setup_seed(args.seed)
    _, test_dataloader, _, test_data, image_size, crop_size = build_petct_loaders(args, train=False)
    print_fn(f'Input mode: {args.input_mode}')
    print_fn(f'Image size: {image_size}, Crop size: {crop_size}')
    print_fn(f'Test samples: {len(test_data)}')

    model, _trainable = build_model(device)
    ckpt_path = resolve_checkpoint_path(args)
    print_fn(f'Loading checkpoint from {ckpt_path}')
    checkpoint = load_checkpoint(model, ckpt_path, device)
    if checkpoint.get('input_mode') and checkpoint.get('input_mode') != args.input_mode:
        print_fn(f"Warning: checkpoint input_mode={checkpoint.get('input_mode')} but current input_mode={args.input_mode}")
    return run_final_evaluation(model, test_dataloader, args, print_fn, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dinomaly PET-CT Anomaly Detection')
    parser.add_argument('--data_path', type=str,
                        default='/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG')
    parser.add_argument('--input_mode', type=str, default='dual',
                        choices=['ct', 'pet', 'dual'],
                        help='ct: CT only, pet: PET only, dual: [CT,PET,PET] pseudo-RGB')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default=None,
                        help='Experiment name. Auto-generated if not provided.')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only load a checkpoint and run PET-CT evaluation.')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Checkpoint path for --eval_only. Defaults to save_dir/save_name/epoch_N_model.pth.')
    parser.add_argument('--max_ratio', type=float, default=0.01,
                        help='Top pixel ratio used for image-level anomaly score.')
    parser.add_argument('--vis_num', type=int, default=20,
                        help='Number of top-scored anomaly maps to save.')
    parser.add_argument('--anomaly_dir', type=str, default=None,
                        help='Directory for saved anomaly heatmaps. Defaults to save_dir/save_name/anomaly_maps.')
    parser.add_argument('--ci_bootstrap', type=int, default=500,
                        help='Bootstrap iterations for AUPR/F1 95 CI.')
    parser.add_argument('--ci_seed', type=int, default=42,
                        help='Random seed for bootstrap confidence intervals.')
    parser.add_argument('--ci_pixel_max_samples', type=int, default=200000,
                        help='Compatibility option only; pixel CI now bootstraps abnormal slices and uses all pixels from sampled slices.')
    parser.add_argument('--hist_pixel_points', action='store_true',
                        help='Use histogram pixel point estimates for speed. Default uses exact sklearn full-pixel point estimates.')
    parser.add_argument('--exact_pixel_auroc', type=float, default=None,
                        help='Optional exact full-pixel AUROC point estimate, e.g. 0.9518.')
    parser.add_argument('--exact_pixel_aupr', type=float, default=None,
                        help='Optional exact full-pixel AUPR point estimate, e.g. 0.2436.')
    parser.add_argument('--cache_path', type=str, default=None,
                        help='Optional .npz path to save labels, masks, anomaly maps, image scores, and paths.')
    parser.add_argument('--export_only', action='store_true',
                        help='Only export eval cache and skip metric computation. Requires --cache_path for useful output.')
    parser.add_argument('--progress_interval', type=int, default=20,
                        help='Log evaluation/export progress every N batches. Use 0 to disable.')
    args = parser.parse_args()

    if args.save_name is None:
        args.save_name = f'dinomaly_petct_{args.input_mode}_ep{args.num_epochs}_s{args.seed}'
    if args.export_only and not args.cache_path:
        parser.error('--export_only requires --cache_path')

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    print_fn(f'Device: {device}')
    print_fn(f'Args: {args}')

    if args.eval_only:
        evaluate_only(args, print_fn, device)
    else:
        train(args, print_fn, device)
