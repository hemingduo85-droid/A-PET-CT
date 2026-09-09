"""
Dinomaly2 for PET-CT anomaly detection.
支持五种输入模式:
  ct            — 仅CT伪RGB [CT,CT,CT]
  pet           — 仅PET伪RGB [PET,PET,PET]
  dual          — [CT,PET,PET] 伪RGB（像素级早期融合）
  dual_rgbd     — 6通道输入 [CT,CT,CT,PET,PET,PET]，DinomalyRGBD 特征级融合
  dual_mulsen   — CT和PET分开编码，推理时异常图融合（MulSen式分数级融合）

运行示例:
cd /data/cyf/codes/A-PET-CT/Dinomaly2-main

# 单模态
python dinomaly2_petct.py --input_mode ct --gpu 0
python dinomaly2_petct.py --input_mode pet --gpu 0

# 双模态 — 伪RGB（最简单）
nohup python dinomaly2_petct.py --input_mode dual --data_path /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50_ganzkoerper/FDG \
     --gpu 0 --save_dir ./saved_results/dual_rgb_fdg > dual_rgb_fdg.log 2>&1 &

# 双模态 — 6通道特征级融合（DinomalyRGBD）
nohup python dinomaly2_petct.py --input_mode dual_rgbd --gpu 7 --data_path /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/FDG \
     --save_dir ./saved_results/dual_rgbd_fdg_old > dual_rgbd_fdg_old.log 2>&1 &

nohup python dinomaly2_petct.py --input_mode dual_rgbd --gpu 3 --data_path /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50_ganzkoerper/FDG \
     --save_dir ./saved_results/dual_rgbd_fdg > dual_rgbd_fdg.log 2>&1 &

# 双模态 — MulSen式分数级融合
nohup python dinomaly2_petct.py --input_mode dual_mulsen --gpu 2 --data_path /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50_ganzkoerper/FDG \
     --save_dir ./saved_results/dual_mulsen_fdg > dual_mulsen.log 2>&1 &

# 测试
python dinomaly2_petct.py \
  --eval_only \
  --checkpoint /data/cyf/codes/A-PET-CT/Dinomaly2-main/saved_results/dual_rgbd_fdg/dinomaly2_petct_dual_rgbd_ep30_s42/model.pth \
  --data_path /data/cyf/shared_data/PET-CT/AutoPET/2d_equal/FDG \
  --input_mode dual_rgbd \
  --gpu 1 \
  --batch_size 16 \
  --num_workers 4
"""

import torch
import torch.nn as nn
from dataset import get_data_transforms, PetCTDataset
import numpy as np
import random
import os
from torch.utils.data import DataLoader

from models.uad import Dinomaly, DinomalyRGBD
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, LinearAttention2
from utils import (
    evaluation_petct, evaluation_petct_mulsen,
    format_petct_metrics,
    global_cosine_hm_percent, WarmupCosineScheduler,
)
from functools import partial
from optimizers import StableAdamW
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


def build_model(encoder, embed_dim, num_heads, target_layers,
                fuse_layer_encoder, fuse_layer_decoder,
                input_mode, dropout=0.4):
    """Build bottleneck, decoder, and the appropriate model for the given input_mode."""

    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=dropout)),
        nn.Sequential(nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(p=dropout),
                       nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(p=dropout)),
    ])

    decoder = nn.ModuleList()
    for _ in range(8):
        blk = VitBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=partial(LinearAttention2, eps=1e-8),
        )
        decoder.append(blk)

    if input_mode == 'dual_rgbd':
        model = DinomalyRGBD(
            encoder=encoder, bottleneck=bottleneck, decoder=decoder,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            fuse_layer_bottleneck=list(range(len(target_layers))),
            remove_class_token=False,
            norm_encoder_token=True,
            rgb_ratio=0.5,
        )
    else:
        model = Dinomaly(
            encoder=encoder, bottleneck=bottleneck, decoder=decoder,
            target_layers=target_layers,
            remove_class_token=False,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            context_aware_recenter=True,
        )

    return model, bottleneck, decoder


def train(args, print_fn, device):
    setup_seed(args.seed)

    image_size = 256
    crop_size = 252  # 必须是 patch_size 14 的倍数: 252/14=18

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data = PetCTDataset(
        data_root=args.data_path,
        transform=data_transform,
        gt_transform=gt_transform,
        phase='train',
        input_mode=args.input_mode,
    )
    test_data = PetCTDataset(
        data_root=args.data_path,
        transform=data_transform,
        gt_transform=gt_transform,
        phase='test',
        input_mode=args.input_mode,
    )

    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    mode_desc = {
        'ct':          '仅CT [CT,CT,CT] 伪RGB',
        'pet':         '仅PET [PET,PET,PET] 伪RGB',
        'dual':        '[CT,PET,PET] 伪RGB（像素级早期融合）',
        'dual_rgbd':   '6通道 [CT³,PET³] DinomalyRGBD（特征级融合）',
        'dual_mulsen': 'CT+PET 分开编码（MulSen式分数级融合）',
    }
    print_fn(f'输入模式: {args.input_mode} — {mode_desc[args.input_mode]}')
    print_fn(f'图像尺寸: {image_size}, 裁剪尺寸: {crop_size}')
    print_fn(f'训练样本: {len(train_data)}, 测试样本: {len(test_data)}')

    # ── 编码器 ────────────────────────────────────────────────────────────
    encoder_name = 'dinov2reg_vit_base_14'
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    encoder = vit_encoder.load(encoder_name)
    embed_dim, num_heads = 768, 12

    # ── 构建模型 ──────────────────────────────────────────────────────────
    model, bottleneck, decoder = build_model(
        encoder, embed_dim, num_heads, target_layers,
        fuse_layer_encoder, fuse_layer_decoder,
        input_mode=args.input_mode,
    )
    model = model.to(device)
    model.init_weights()

    trainable = nn.ModuleList([bottleneck, decoder])

    total_iters = args.num_epochs * len(train_dataloader)
    warmup_iters = min(100, total_iters // 10)

    optimizer = StableAdamW([
        {'params': bottleneck[0].parameters(), 'lr': 2e-4},
        {'params': bottleneck[1].parameters()},
        {'params': decoder.parameters()},
    ], lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=False, eps=1e-10)

    lr_scheduler = WarmupCosineScheduler(
        optimizer, final_ratio=1.0, total_epochs=total_iters, warmup_epochs=warmup_iters,
    )

    print_fn(f'模型类型: {"DinomalyRGBD" if args.input_mode == "dual_rgbd" else "Dinomaly"}')
    print_fn(f'可训练参数: {count_parameters(trainable)}')
    print_fn(f'总迭代数: {total_iters}, 预热: {warmup_iters}')

    it = 0
    is_mulsen = (args.input_mode == 'dual_mulsen')

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        loss_list = []

        for batch in train_dataloader:
            if is_mulsen:
                (ct_img, pet_img), label = batch
                ct_img = ct_img.to(device)
                pet_img = pet_img.to(device)
                img = torch.cat([ct_img, pet_img], dim=0)  # [2B, 3, H, W]
            else:
                img, label = batch
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

    if is_mulsen:
        results = evaluation_petct_mulsen(
            model, test_dataloader, device,
            max_ratio=args.max_ratio, fusion=args.mulsen_fusion,
        )
    else:
        results = evaluation_petct(model, test_dataloader, device, max_ratio=args.max_ratio)

    slice_metrics, patient_metrics = results
    save_path = os.path.join(args.save_dir, args.save_name, 'model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'slice_metrics': slice_metrics,
        'patient_metrics': patient_metrics,
        'input_mode': args.input_mode,
    }, save_path)

    print_fn('=' * 80)
    print_fn(f'第 {args.num_epochs} 轮模型已保存: {save_path}')
    print_fn('最终测试结果:\n' + format_petct_metrics(slice_metrics, patient_metrics))

    return results


def evaluate_checkpoint(args, print_fn, device):
    image_size = 256
    crop_size = 252
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

    encoder_name = 'dinov2reg_vit_base_14'
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    encoder = vit_encoder.load(encoder_name)
    embed_dim, num_heads = 768, 12
    model, _, _ = build_model(
        encoder, embed_dim, num_heads, target_layers,
        fuse_layer_encoder, fuse_layer_decoder,
        input_mode=args.input_mode,
    )
    model = model.to(device)
    model.init_weights()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)

    is_mulsen = (args.input_mode == 'dual_mulsen')
    if is_mulsen:
        results = evaluation_petct_mulsen(
            model, test_dataloader, device,
            max_ratio=args.max_ratio, fusion=args.mulsen_fusion,
        )
    else:
        results = evaluation_petct(model, test_dataloader, device, max_ratio=args.max_ratio)

    slice_metrics, patient_metrics = results
    print_fn(f'Checkpoint: {args.checkpoint}')
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print_fn(f"Checkpoint epoch: {checkpoint['epoch']}")
    print_fn(format_petct_metrics(slice_metrics, patient_metrics))
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dinomaly2 PET-CT Anomaly Detection')
    parser.add_argument('--data_path', type=str,
                        default='/data/cyf/shared_data/PET-CT/AutoPET/2d_equal/FDG')
    parser.add_argument('--input_mode', type=str, default='dual',
                        choices=['ct', 'pet', 'dual', 'dual_rgbd', 'dual_mulsen'],
                        help='ct/pet: 单模态伪RGB, dual: [CT,PET,PET]伪RGB, '
                             'dual_rgbd: 6通道DinomalyRGBD, dual_mulsen: MulSen式分数融合')
    parser.add_argument('--mulsen_fusion', type=str, default='mean',
                        choices=['mean', 'max'],
                        help='MulSen模式下异常图融合方式 (仅dual_mulsen有效)')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0', help='GPU 编号')
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default=None,
                        help='实验名称，不填则自动生成')
    parser.add_argument('--eval_only', action='store_true',
                        help='只加载 --checkpoint 在测试集上计算 PhyTwin 口径指标')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='用于 --eval_only 的 .pth 路径')
    parser.add_argument('--max_ratio', type=float, default=0.01,
                        help='image score 使用异常图 top ratio 均值；0 表示 max')
    args = parser.parse_args()

    if args.save_name is None:
        args.save_name = f'dinomaly2_petct_{args.input_mode}_ep{args.num_epochs}_s{args.seed}'

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    print_fn(f'Device: {device}')
    print_fn(f'Args: {args}')

    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError('--eval_only 需要指定 --checkpoint /path/to/model.pth')
        evaluate_checkpoint(args, print_fn, device)
    else:
        train(args, print_fn, device)
