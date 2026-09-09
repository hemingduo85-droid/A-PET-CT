# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import sys
import time

"""
nohup python3 -u test_modified.py \
  --dataset psma \
  --modalities pet,ct \
  --gpu 7 \
  > rd4ad_psma_ci.log 2>&1 &

nohup python3 -u test_modified.py \
  --dataset fdg \
  --modalities pet,ct \
  --gpu 7 \
  > fdg_npz.log 2>&1 &

nohup python3 -u test_modified.py \
  --dataset psma \
  --modalities pet,ct \
  --gpu 5 \
  > psma_npz.log 2>&1 &
"""


def _configure_visible_gpu_from_argv():
    if "--gpu" not in sys.argv:
        return None
    idx = sys.argv.index("--gpu")
    if idx + 1 >= len(sys.argv):
        return None
    gpu = sys.argv[idx + 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return gpu


REQUESTED_PHYSICAL_GPU = _configure_visible_gpu_from_argv()

import torch
from dataset import get_data_transforms
import numpy as np
import random
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from dataset import MVTecDataset
import argparse
from test_modified import evaluate, print_metrics, save_checkpoint_metadata
from torch.nn import functional as F
from config import (
    checkpoint_path,
    get_dataset_config,
    input_channels,
    parse_modalities,
    resolve_input_mode,
)
from model_utils import adapt_first_conv, select_device

"""
cpu高
# PSMA 单模态 PET（默认）
cd /data/cyf/codes/A-PET-CT/RD4AD
python main.py --dataset psma --modality pet --gpu 4

# FDG 双模态伪 RGB [CT, PET, PET]
python main.py --dataset fdg --modalities pet,ct

# 后台训练示例
nohup python -u main.py --dataset psma --modality pet > rd4ad_psma_pet.log 2>&1 &
nohup python -u main.py --dataset psma --modalities pet,ct > rd4ad_psma_ct_pet_pet.log 2>&1 &

# 仅测试（需先训练得到 checkpoint）
python test_modified.py --dataset psma --modality pet
python test_modified.py --dataset psma --modalities pet,ct --save-heatmaps all
"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def train(dataset_name, modalities, data_root=None, input_mode='auto',
          save_heatmaps='0', heatmap_dir='./heatmaps', gpu=None,
          epochs=30, batch_size=16, num_workers=4, progress_every=50,
          deterministic=False):
    dataset_config = get_dataset_config(dataset_name)
    data_root = data_root or dataset_config['data_root']
    input_mode = resolve_input_mode(modalities, input_mode)
    print(f"dataset: {dataset_config['name']}")
    print(f"data root: {data_root}")
    print(f"modalities: {modalities}")
    print(f"input mode: {input_mode}")
    learning_rate = 0.005
    image_size = 256
    
    # 确保checkpoints目录存在
    os.makedirs('./checkpoints', exist_ok=True)
        
    device = select_device(physical_gpu=REQUESTED_PHYSICAL_GPU)
    setup_seed(111, deterministic=deterministic)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_root = os.path.join(data_root, 'train')
    test_root = os.path.join(data_root, 'test')
    ckp_path = checkpoint_path(dataset_config, modalities, input_mode)
    
    # 多模态数据集
    train_data = MVTecDataset(root=train_root, transform=data_transform, gt_transform=None,
                              phase="train", modalities=modalities,
                              input_mode=input_mode, image_size=image_size)
    test_data = MVTecDataset(root=test_root, transform=data_transform, gt_transform=gt_transform,
                             phase="test", modalities=modalities,
                             input_mode=input_mode, image_size=image_size)
    
    print(f"train samples: {len(train_data)}")
    print(f"test samples: {len(test_data)}")
    print(f"batch size: {batch_size}; train steps/epoch: {len(train_data) // batch_size + int(len(train_data) % batch_size != 0)}")
    print(f"num workers: {num_workers}")
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, **loader_kwargs
    )

    encoder, bn = wide_resnet50_2(pretrained=True)
    in_ch = input_channels(modalities, input_mode)
    if in_ch != 3:
        encoder = adapt_first_conv(encoder, in_ch)
    print(f"encoder input channels: {in_ch}")
    encoder = encoder.to(device)
    encoder.requires_grad_(False)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    print("models moved to device; training starts now", flush=True)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        epoch_start = time.time()
        batch_start = time.time()
        last_print_step = 0
        print(f"epoch [{epoch + 1}/{epochs}] start")
        for step, (img, _, label, _) in enumerate(train_dataloader, start=1):
            # img shape follows --input-mode: rgb/ct_pet_pet -> 3 channels, concat -> 3*N.
            img = img.to(device, non_blocking=True)
            with torch.no_grad():
                inputs = encoder(img)
            outputs = decoder(bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            if step == 1 or step % progress_every == 0 or step == len(train_dataloader):
                elapsed = time.time() - batch_start
                steps_since_print = step - last_print_step
                print(
                    "epoch [{}/{}] step [{}/{}], loss:{:.4f}, {:.2f}s/{} step(s)".format(
                        epoch + 1, epochs, step, len(train_dataloader),
                        np.mean(loss_list[-min(len(loss_list), progress_every):]),
                        elapsed, steps_since_print,
                    ),
                    flush=True,
                )
                batch_start = time.time()
                last_print_step = step
        print('epoch [{}/{}] done, loss:{:.4f}, time:{:.1f}s'.format(
            epoch + 1, epochs, np.mean(loss_list), time.time() - epoch_start
        ), flush=True)
    torch.save({
        'bn': bn.state_dict(),
        'decoder': decoder.state_dict(),
        'dataset': dataset_config['name'],
        'modalities': modalities,
        'input_mode': input_mode,
        'epoch': epochs,
    }, ckp_path)
    save_checkpoint_metadata(ckp_path, dataset_config, modalities, input_mode, data_root)
    print(f"saved epoch-{epochs} checkpoint: {ckp_path}")

    slice_metrics, patient_metrics = evaluate(
        encoder, bn, decoder, test_dataloader, device,
        save_heatmaps=save_heatmaps,
        heatmap_dir=heatmap_dir,
        run_name=f"{dataset_config['name']}_{input_mode}",
    )
    print_metrics(slice_metrics, patient_metrics)
    return slice_metrics, patient_metrics




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RD4AD AutoPET anomaly detection')
    parser.add_argument('--dataset', type=str, default='psma', choices=['psma', 'fdg'],
                        help='数据集配置名，自动决定 data-root 与 checkpoint 名')
    parser.add_argument('--data-root', type=str,
                        default='',
                        help='覆盖配置中的数据集根目录，内含 train/ 与 test/')
    parser.add_argument('--modality', type=str, default='pet',
                        choices=['pet', 'ct'],
                        help='单模态：pet 或 ct（--modalities 留空时生效）')
    parser.add_argument('--modalities', type=str, default='',
                        help='多模态列表，逗号分隔，如 pet,ct；留空则仅用 --modality 单模态')
    parser.add_argument('--input-mode', type=str, default='auto',
                        choices=['auto', 'rgb', 'ct_pet_pet', 'concat'],
                        help='auto: pet,ct 默认 [CT,PET,PET]；concat 为旧 6 通道拼接')
    parser.add_argument('--save-heatmaps', type=str, default='0',
                        help='训练后测试保存热力图数量：0、整数或 all，默认 0')
    parser.add_argument('--heatmap-dir', type=str, default='./heatmaps',
                        help='热力图输出目录')
    parser.add_argument('--gpu', type=int, default=None,
                        help='指定 nvidia-smi 中的物理 GPU id，如 --gpu 4')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数，默认 30')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='训练 batch size，默认 16')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader worker 数，默认 4；读图慢可调大')
    parser.add_argument('--progress-every', type=int, default=50,
                        help='每多少个 batch 打印一次训练进度，默认 50')
    parser.add_argument('--deterministic', action='store_true',
                        help='启用确定性 cuDNN；默认关闭以优先训练速度')
    args = parser.parse_args()
    modalities = parse_modalities(args.modality, args.modalities)
    train(args.dataset, modalities, data_root=args.data_root or None,
          input_mode=args.input_mode, save_heatmaps=args.save_heatmaps,
          heatmap_dir=args.heatmap_dir, gpu=args.gpu,
          epochs=args.epochs, batch_size=args.batch_size,
          num_workers=args.num_workers, progress_every=args.progress_every,
          deterministic=args.deterministic)
