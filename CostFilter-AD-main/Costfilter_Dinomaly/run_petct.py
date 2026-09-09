"""
CostFilter-AD adapted for PET-CT Medical Anomaly Detection

Supports three input modalities:
  - ct:    [CT, CT, CT]   pseudo-RGB
  - pet:   [PET, PET, PET] pseudo-RGB
  - petct: [CT, PET, PET]  pseudo-RGB (dual-modality)

Two-phase training pipeline:
  Phase 1 — Train Dinomaly (unsupervised reconstruction on normal images)
  Phase 2 — Train CostFilter 3D-UNet (with synthetic anomalies), then test once
            after epoch 30 and save that final checkpoint

Three-level evaluation:
  Slice image-level | Slice pixel-level | Patient-level

  # 进入工作目录
cd /data/cyf/codes/A-PET-CT/CostFilter-AD-main/Costfilter_Dinomaly

# ========== PET+CT 双模态（推荐） ==========
# 完整流水线：Phase 1 训练 Dinomaly 50轮 + Phase 2 训练 CostFilter 30轮
nohup python run_petct.py \
    --dataset psma \
    --modality petct \
    --gpu 0 \
    --dinomaly_epochs 50 \
    --costfilter_epochs 30 \
    > run_petct_petct.log 2>&1 &

# ========== 单独 CT ==========
nohup python run_petct.py \
    --dataset psma \
    --modality ct \
    --gpu 1 \
    --dinomaly_epochs 50 \
    --costfilter_epochs 30 \
    > run_petct_ct.log 2>&1 &

# ========== 单独 PET ==========
nohup python run_petct.py \
    --dataset psma \
    --modality pet \
    --gpu 2 \
    --dinomaly_epochs 50 \
    --costfilter_epochs 30 \
    > run_petct_pet.log 2>&1 &

# ========== 只跑 Dinomaly 基线（不训练 CostFilter） ==========
python run_petct.py \
    --dataset psma \
    --modality petct \
    --gpu 0 \
    --dinomaly_epochs 30 \
    --skip_phase2

# ========== 已有 Dinomaly 权重，只训练 CostFilter ==========
python run_petct.py \
    --dataset psma \
    --modality petct \
    --gpu 0 \
    --skip_phase1 \
    --costfilter_epochs 30
# ========== 只测试  ==========
只测试
nohup python run_petct.py \
  --dataset psma \
  --modality petct \
  --gpu 2 \
  --save_dir ./checkpoint_psma_petct \
  --costfilter_epochs 30 \
  --test_only \
  --test_stage costfilter \
  --ci_iters 500 \
  --ci_pixel_max_samples 0 \
  > test_petct_psma_ci.log 2>&1 &

nohup python run_petct.py \
  --dataset fdg \
  --modality petct \
  --gpu 4 \
  --save_dir ./checkpoint_fdg_petct \
  --costfilter_epochs 30 \
  --test_only \
  --test_stage costfilter \
  --ci_iters 500 \
  --ci_pixel_max_samples 0 \
  > test_costfilter_fdg_ci.log 2>&1 &

导出npz
nohup python run_petct.py \
  --dataset psma \
  --modality petct \
  --gpu 4 \
  --save_dir ./checkpoint_psma_petct \
  --costfilter_epochs 30 \
  --test_only \
  --test_stage costfilter \
  --ci_iters 0 \
  --eval_cache_path ./checkpoint_psma_petct/eval_cache/PSMA_petct_eval_cache.npz \
  > export_psma_eval_cache.log 2>&1 &

nohup python run_petct.py \
  --dataset fdg \
  --modality petct \
  --gpu 4 \
  --save_dir ./checkpoint_fdg_petct \
  --costfilter_epochs 30 \
  --test_only \
  --test_stage costfilter \
  --ci_iters 0 \
  --eval_cache_path ./checkpoint_fdg_petct/eval_cache/FDG_petct_eval_cache.npz \
  > fdg_npz.log 2>&1 &

"""
import numpy as np
# ---- NumPy 2.0 compatibility patch for imgaug ----
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'float': [np.float16, np.float32, np.float64],
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void],
    }
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.intp
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'object'):
    np.object = np.object_
if not hasattr(np, 'str'):
    np.str = np.str_

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import sys
import glob
import math
import time
import copy
import logging
import argparse
import warnings
from functools import partial
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from models.uad import ViTill
from models import vit_encoder
from models.vision_transformer import (
    Block as VitBlock, bMlp, LinearAttention2
)
from unet3d_att_dino_channel_test_48_min import DiscriminativeSubNetwork_3d_att_dino_channel
from loss import FocalLoss_gamma, SSIM, SoftIoULoss
from optimizers import StableAdamW
from utils import (
    cal_anomaly_maps, global_cosine_hm_percent, WarmCosineScheduler,
    get_gaussian_kernel
)
try:
    from perlin_noise import rand_perlin_2d_np
except ImportError:
    rand_perlin_2d_np = None

try:
    import imgaug.augmenters as iaa
except ImportError:
    iaa = None

from einops import rearrange

warnings.filterwarnings("ignore")

# Dataset presets keep paths and artifact names out of the code body.
DATASET_PROFILES = {
    'psma': {
        'data_path': '/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA',
        'tag': 'psma',
    },
    'fdg': {
        'data_path': '/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG',
        'tag': 'fdg',
    },
    'custom': {
        'data_path': None,
        'tag': 'custom',
    },
}

# =====================================================================
#  Utilities
# =====================================================================

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name, save_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def f1_score_max(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if y_true.size == 0:
        return 0.0
    if len(np.unique(y_true)) < 2:
        return 0.0
    precs, recs, _ = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    return float(f1s[:-1].max()) if len(f1s) > 1 else 0.0


def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if y_true.size == 0:
        return 0.0
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if y_true.size == 0:
        return 0.0
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def finite_binary_arrays(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y_score)
    return y_true[valid].astype(np.int32), y_score[valid]


def clip_ci(low, high):
    return float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0))


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=500, seed=20260707):
    y_true, y_score = finite_binary_arrays(y_true, y_score)
    value = float(metric_fn(y_true, y_score))
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return value, (value, value)

    if n_boot <= 0:
        return value, (value, value)

    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(float(metric_fn(y_true[idx], y_score[idx])))
    if not values:
        return value, (value, value)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return value, clip_ci(lo, hi)


def metrics_from_hist(pos_hist, neg_hist):
    tp = pos_hist[::-1].astype(np.float64, copy=False)
    fp = neg_hist[::-1].astype(np.float64, copy=False)
    total_pos = float(tp.sum())
    total_neg = float(fp.sum())
    if total_pos <= 0 or total_neg <= 0:
        return 0.0, 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    tpr = cum_tp / total_pos
    fpr = cum_fp / total_neg
    auroc = float(np.trapz(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)
    aupr = float(np.sum(precision * (tp / total_pos)))
    return auroc, aupr


def pixel_slice_bootstrap_ci(labels, masks, maps, n_boot=500, seed=20260717,
                             hist_bins=16384, logger=None):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    masks = np.asarray(masks, dtype=np.float32)
    maps = np.asarray(maps, dtype=np.float32)
    keep = labels == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        return (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0))

    if logger is not None:
        logger.info("Computing exact full-pixel point estimates with sklearn...")
    flat_y = masks[keep].reshape(-1).astype(bool)
    flat_score = maps[keep].reshape(-1).astype(np.float64)
    flat_y, flat_score = finite_binary_arrays(flat_y, flat_score)
    px_auroc_value = safe_auroc(flat_y, flat_score)
    px_aupr_value = safe_ap(flat_y, flat_score)

    if n_boot <= 0:
        return (px_auroc_value, (px_auroc_value, px_auroc_value)), (px_aupr_value, (px_aupr_value, px_aupr_value))

    score_min = float(np.nanmin(maps[keep]))
    score_max = float(np.nanmax(maps[keep]))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    scale = (hist_bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), hist_bins), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), hist_bins), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        scores = maps[idx].reshape(-1)
        mask = masks[idx].reshape(-1).astype(bool)
        bin_idx = np.floor((scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, hist_bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[mask], minlength=hist_bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~mask], minlength=hist_bins).astype(np.uint32)

    rng = np.random.default_rng(seed)
    aurocs, auprs = [], []
    n = len(abn_idx)
    for i in range(n_boot):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.float64)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if logger is not None and ((i + 1) % 50 == 0 or (i + 1) == n_boot):
            logger.info(f"Pixel histogram slice bootstrap: {i + 1}/{n_boot}")

    auroc_ci = clip_ci(*np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = clip_ci(*np.percentile(auprs, [2.5, 97.5]))
    return (px_auroc_value, auroc_ci), (px_aupr_value, aupr_ci)


def fmt_metric(value, ci):
    return f'{value * 100:.2f}% (95% CI {ci[0] * 100:.2f}-{ci[1] * 100:.2f}%)'


def save_eval_cache(path, labels, masks, maps, image_scores, paths, bootstrap_iters):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(
        path,
        labels=np.asarray(labels, dtype=np.int32),
        masks=np.asarray(masks, dtype=np.uint8),
        maps=np.asarray(maps, dtype=np.float32),
        image_scores=np.asarray(image_scores, dtype=np.float64),
        paths=np.asarray(paths, dtype=str),
        bootstrap_iters=np.asarray(bootstrap_iters, dtype=np.int32),
    )


def dataset_sample_paths(dataset, n_items):
    if hasattr(dataset, 'samples'):
        return [str(sample[0]) for sample in dataset.samples[:n_items]]
    return [f'unknown/unknown/{i:06d}.png' for i in range(n_items)]


def sensitivity_at_specificity(y_true, y_score, target_spec):
    """Compute sensitivity when specificity is fixed at target_spec."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if y_true.size == 0:
        return 0.0
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = fpr <= (1.0 - target_spec)
    return float(tpr[valid].max()) if valid.any() else 0.0


def weights_init_2d(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def normalize_map(score_map):
    score_map = np.asarray(score_map, dtype=np.float32)
    return (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)


def colorize_score_map(score_map):
    score_u8 = (normalize_map(score_map) * 255).astype(np.uint8)
    blue = np.zeros_like(score_u8)
    red = score_u8
    green = np.clip(255 - np.abs(score_u8.astype(np.int16) - 128) * 2, 0, 255).astype(np.uint8)
    return Image.merge('RGB', [Image.fromarray(red), Image.fromarray(green), Image.fromarray(blue)])


def load_pseudo_rgb_image(ct_path, pet_path, modality):
    if modality == 'ct':
        ct = Image.open(ct_path).convert('L')
        return Image.merge('RGB', [ct, ct, ct])
    if modality == 'pet':
        pet = Image.open(pet_path).convert('L')
        return Image.merge('RGB', [pet, pet, pet])
    ct = Image.open(ct_path).convert('L')
    pet = Image.open(pet_path).convert('L')
    return Image.merge('RGB', [ct, pet, pet])


def save_heatmap_triplet(input_image, score_map, save_prefix, size):
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    input_image = input_image.resize((size, size), Image.BILINEAR).convert('RGB')
    heat = colorize_score_map(score_map).resize((size, size), Image.BILINEAR)
    overlay = Image.blend(input_image, heat, alpha=0.35)
    input_image.save(f'{save_prefix}_input.png')
    heat.save(f'{save_prefix}_heat.png')
    overlay.save(f'{save_prefix}_overlay.png')


def resolve_dataset_profile(args):
    if args.dataset not in DATASET_PROFILES:
        raise ValueError(f'Unknown dataset profile: {args.dataset}')
    profile = DATASET_PROFILES[args.dataset]
    if args.data_path is None:
        if profile['data_path'] is None:
            raise ValueError('--data_path is required when --dataset custom')
        args.data_path = profile['data_path']
    args.dataset_tag = args.dataset_name or profile['tag']
    if args.save_dir is None:
        args.save_dir = os.path.join('.', f'checkpoint_{args.dataset_tag}_{args.modality}')
    args.dinomaly_ckpt = os.path.join(args.save_dir, f'dinomaly_{args.dataset_tag}_{args.modality}.pth')
    args.costfilter_ckpt = os.path.join(
        args.save_dir, f'costfilter_{args.dataset_tag}_{args.modality}_epoch{args.costfilter_epochs}.pth'
    )
    if args.dinomaly_ckpt_path:
        args.dinomaly_ckpt = args.dinomaly_ckpt_path
    if args.costfilter_ckpt_path:
        args.costfilter_ckpt = args.costfilter_ckpt_path
    args.heatmap_dir = args.heatmap_dir or os.path.join(args.save_dir, 'heatmaps', args.dataset_tag, args.modality)
    if args.eval_cache_path is None:
        args.eval_cache_path = os.path.join(
            args.save_dir, 'eval_cache', f'{args.dataset_tag}_{args.modality}_{args.test_stage}_eval_cache.npz'
        )


# =====================================================================
#  Transforms
# =====================================================================

def get_data_transforms(image_size, crop_size):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train),
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])
    return data_transforms, gt_transforms


# =====================================================================
#  Datasets
# =====================================================================

class PETCTTestDataset(Dataset):
    """Test dataset loading normal + abnormal PET-CT slices with labels."""

    def __init__(self, data_root, modality, transform, gt_transform):
        self.modality = modality
        self.transform = transform
        self.gt_transform = gt_transform
        self.samples = []  # (ct_path, pet_path, label_path_or_None, label_int, patient_id)

        for category in ['normal', 'abnormal']:
            cat_dir = os.path.join(data_root, 'test', category)
            if not os.path.isdir(cat_dir):
                continue
            label_val = 0 if category == 'normal' else 1
            for patient in sorted(os.listdir(cat_dir)):
                patient_dir = os.path.join(cat_dir, patient)
                ct_dir = os.path.join(patient_dir, 'ct')
                pet_dir = os.path.join(patient_dir, 'pet')
                label_dir = os.path.join(patient_dir, 'label') if category == 'abnormal' else None
                if not os.path.isdir(ct_dir):
                    continue
                for fname in sorted(os.listdir(ct_dir)):
                    if not fname.endswith('.png'):
                        continue
                    ct_path = os.path.join(ct_dir, fname)
                    pet_path = os.path.join(pet_dir, fname)
                    lbl_path = os.path.join(label_dir, fname) if label_dir and os.path.isdir(label_dir) else None
                    self.samples.append((ct_path, pet_path, lbl_path, label_val, patient))

    def _load_rgb(self, ct_path, pet_path):
        if self.modality == 'ct':
            ct = Image.open(ct_path).convert('L')
            return Image.merge('RGB', [ct, ct, ct])
        elif self.modality == 'pet':
            pet = Image.open(pet_path).convert('L')
            return Image.merge('RGB', [pet, pet, pet])
        else:
            ct = Image.open(ct_path).convert('L')
            pet = Image.open(pet_path).convert('L')
            return Image.merge('RGB', [ct, pet, pet])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_path, pet_path, lbl_path, label, patient_id = self.samples[idx]
        img = self._load_rgb(ct_path, pet_path)
        img = self.transform(img)

        if lbl_path and os.path.exists(lbl_path):
            gt = Image.open(lbl_path).convert('L')
            gt = self.gt_transform(gt)
            gt = (gt > 0.5).float()
        else:
            gt = torch.zeros(1, img.shape[1], img.shape[2])

        return img, gt, label, patient_id


class PETCTTrainDataset(Dataset):
    """Train dataset loading normal PET-CT slices (for Dinomaly unsupervised training)."""

    def __init__(self, data_root, modality, transform):
        self.modality = modality
        self.transform = transform
        self.samples = []

        normal_dir = os.path.join(data_root, 'train', 'normal')
        for patient in sorted(os.listdir(normal_dir)):
            ct_dir = os.path.join(normal_dir, patient, 'ct')
            pet_dir = os.path.join(normal_dir, patient, 'pet')
            if not os.path.isdir(ct_dir):
                continue
            for fname in sorted(os.listdir(ct_dir)):
                if not fname.endswith('.png'):
                    continue
                self.samples.append((
                    os.path.join(ct_dir, fname),
                    os.path.join(pet_dir, fname),
                ))

    def _load_rgb(self, ct_path, pet_path):
        if self.modality == 'ct':
            ct = Image.open(ct_path).convert('L')
            return Image.merge('RGB', [ct, ct, ct])
        elif self.modality == 'pet':
            pet = Image.open(pet_path).convert('L')
            return Image.merge('RGB', [pet, pet, pet])
        else:
            ct = Image.open(ct_path).convert('L')
            pet = Image.open(pet_path).convert('L')
            return Image.merge('RGB', [ct, pet, pet])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_path, pet_path = self.samples[idx]
        img = self._load_rgb(ct_path, pet_path)
        img = self.transform(img)
        return img


class PETCTTrainSynthDataset(Dataset):
    """Train dataset that generates synthetic anomalies for CostFilter supervision."""

    def __init__(self, data_root, modality, transform, gt_transform, input_size):
        self.modality = modality
        self.transform = transform
        self.gt_transform = gt_transform
        self.input_size = (input_size, input_size)
        self.samples = []

        normal_dir = os.path.join(data_root, 'train', 'normal')
        for patient in sorted(os.listdir(normal_dir)):
            ct_dir = os.path.join(normal_dir, patient, 'ct')
            pet_dir = os.path.join(normal_dir, patient, 'pet')
            if not os.path.isdir(ct_dir):
                continue
            for fname in sorted(os.listdir(ct_dir)):
                if not fname.endswith('.png'):
                    continue
                self.samples.append((
                    os.path.join(ct_dir, fname),
                    os.path.join(pet_dir, fname),
                ))

        self.augmenters = None
        if iaa is not None:
            self.augmenters = [
                iaa.GammaContrast((0.5, 2.0), per_channel=True),
                iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                iaa.pillike.EnhanceSharpness(),
                iaa.Solarize(0.5, threshold=(32, 128)),
                iaa.Posterize(),
                iaa.Invert(),
                iaa.pillike.Autocontrast(),
                iaa.pillike.Equalize(),
                iaa.Affine(rotate=(-45, 45)),
            ]
            self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.structure_grid_size = 8

    def _load_rgb(self, ct_path, pet_path):
        if self.modality == 'ct':
            ct = Image.open(ct_path).convert('L')
            return Image.merge('RGB', [ct, ct, ct])
        elif self.modality == 'pet':
            pet = Image.open(pet_path).convert('L')
            return Image.merge('RGB', [pet, pet, pet])
        else:
            ct = Image.open(ct_path).convert('L')
            pet = Image.open(pet_path).convert('L')
            return Image.merge('RGB', [ct, pet, pet])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_path, pet_path = self.samples[idx]
        img = self._load_rgb(ct_path, pet_path)
        img_resized = img.resize(self.input_size, Image.BILINEAR)
        img_np = np.array(img_resized)

        synth_img, synth_mask, is_normal = self._generate_anomaly(img_np)

        synth_img = self.transform(synth_img)
        synth_mask = self.gt_transform(synth_mask)
        is_normal_t = torch.tensor(is_normal, dtype=torch.float32)
        cls_label = 0
        return synth_img, synth_mask, is_normal_t, cls_label

    def _generate_anomaly(self, image):
        if self.augmenters is None or rand_perlin_2d_np is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            return Image.fromarray(image), Image.fromarray(mask), np.array([0.0], dtype=np.float32)

        aug_ind = np.random.choice(len(self.augmenters), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[i] for i in aug_ind])

        perlin_scale = 6
        min_perlin_scale = 0
        threshold = 0.3
        sx = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
        sy = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
        perlin_noise = rand_perlin_2d_np((image.shape[0], image.shape[1]), (sx, sy))
        perlin_noise = self.rot(image=perlin_noise)
        perlin_thr = np.where(perlin_noise > threshold, 1.0, 0.0).astype(np.float32)
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        anomaly_source = self._self_augment_source(image, aug)
        anomaly_source_thr = anomaly_source * perlin_thr
        beta = np.random.random() * 0.8
        augmented = image * (1 - perlin_thr) + (1 - beta) * anomaly_source_thr + beta * image * perlin_thr

        if np.random.random() > 0.5:
            augmented = augmented.astype(np.uint8)
            msk = (perlin_thr * 255).astype(np.uint8).squeeze()
            has_anomaly = 0.0 if np.sum(msk) == 0 else 1.0
            return Image.fromarray(augmented), Image.fromarray(msk), np.array([has_anomaly], dtype=np.float32)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            return Image.fromarray(image.astype(np.uint8)), Image.fromarray(mask), np.array([0.0], dtype=np.float32)

    def _self_augment_source(self, img, aug):
        h, w = self.input_size
        grid = self.structure_grid_size
        structure_source = aug(image=img)
        gw = w // grid
        gh = h // grid
        structure_source = rearrange(
            structure_source, '(h gh) (w gw) c -> (h w) gw gh c', gw=gw, gh=gh
        )
        order = np.arange(structure_source.shape[0])
        np.random.shuffle(order)
        return rearrange(
            structure_source[order], '(h w) gw gh c -> (h gh) (w gw) c', h=grid, w=grid
        ).astype(np.float32)


# =====================================================================
#  Model Builder
# =====================================================================

def build_dinomaly(device, encoder_name='dinov2reg_vit_base_14'):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture must be small, base or large.")

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])

    decoder = nn.ModuleList()
    for _ in range(8):
        blk = VitBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention2,
        )
        decoder.append(blk)

    model = ViTill(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder,
        target_layers=target_layers, mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    )
    model = model.to(device)
    return model, embed_dim


# =====================================================================
#  Three-Level Evaluation (Dinomaly-only)
# =====================================================================

def evaluate_dinomaly(model, test_loader, device, crop_size, sigma=4,
                      logger=None, prefix='', ci_iters=500, ci_seed=20260707,
                      ci_pixel_max_samples=200000, ci_hist_bins=16384,
                      eval_cache_path=None):
    model.eval()
    all_maps, all_gts, all_labels, all_pids = [], [], [], []

    with torch.no_grad():
        for img, gt, label, pid in tqdm(test_loader, desc='  Eval', leave=False):
            img = img.to(device)
            en, de = model(img)
            amap, _ = cal_anomaly_maps(en, de, crop_size)
            all_maps.append(amap.cpu())
            all_gts.append(gt)
            if isinstance(label, torch.Tensor):
                all_labels.extend(label.numpy().tolist())
            else:
                all_labels.extend(label)
            all_pids.extend(pid)

    amaps = torch.cat(all_maps, dim=0)  # [N,1,H,W]
    gts = torch.cat(all_gts, dim=0)
    labels = np.array(all_labels, dtype=int)
    pids = all_pids

    amaps_np = amaps[:, 0].numpy()
    for i in range(amaps_np.shape[0]):
        amaps_np[i] = gaussian_filter(amaps_np[i], sigma=sigma)

    results = _compute_three_level(
        amaps_np, gts[:, 0].numpy(), labels, pids,
        logger=logger, prefix=prefix, ci_iters=ci_iters,
        ci_seed=ci_seed, ci_pixel_max_samples=ci_pixel_max_samples,
        ci_hist_bins=ci_hist_bins
    )
    if eval_cache_path:
        paths = dataset_sample_paths(test_loader.dataset, len(labels))
        save_eval_cache(
            eval_cache_path, labels, gts[:, 0].numpy(), amaps_np,
            results['image_scores'], paths, ci_iters
        )
        if logger is not None:
            logger.info(f"Saved eval cache: {eval_cache_path}")
    return results


# =====================================================================
#  Three-Level Evaluation (CostFilter)
# =====================================================================

def evaluate_costfilter(model, model_unet, test_loader, device, crop_size,
                        feat_size, pre_min_dim, embed_dim, sigma=4, lamda=0.5,
                        heatmap_dir=None, heatmap_limit=128, modality='petct',
                        logger=None, prefix='', ci_iters=500, ci_seed=20260707,
                        ci_pixel_max_samples=200000, ci_hist_bins=16384,
                        eval_cache_path=None):
    model.eval()
    model_unet.eval()
    all_maps, all_gts, all_labels, all_pids = [], [], [], []
    unet_spatial = 64
    sample_offset = 0
    saved_heatmaps = 0

    with torch.no_grad():
        for img, gt, label, pid in tqdm(test_loader, desc='  Eval-CF', leave=False):
            img = img.to(device)
            en, de = model(img)
            B = en[0].shape[0]
            epsilon = 1e-8

            min_anomaly_map = []
            anomaly_map_all_bat = []
            for rec_feat, org_feat in zip(en, de):
                H, W = org_feat.shape[2:]
                pi = org_feat.reshape(B, -1, H * W).permute(0, 2, 1)
                pr = rec_feat.reshape(B, -1, H * W).permute(0, 2, 1)
                pi = pi / (torch.norm(pi, p=2, dim=-1, keepdim=True) + epsilon)
                pr = pr / (torch.norm(pr, p=2, dim=-1, keepdim=True) + epsilon)
                cos0 = torch.bmm(pi, pr.permute(0, 2, 1))
                a_map, _ = torch.min(1 - cos0, dim=-1)
                a_map = nn.UpsamplingBilinear2d(size=(unet_spatial, unet_spatial))(
                    a_map.reshape(B, 1, H, W))
                min_anomaly_map.append(a_map)
                one_minus = 1 - cos0
                cur_dim = min(pre_min_dim, one_minus.shape[-1])
                _, idx = torch.topk(one_minus, cur_dim, dim=-1, largest=False, sorted=False)
                sel = torch.gather(one_minus, dim=-1, index=torch.sort(idx, dim=-1)[0])
                am3d = sel.view(B, H, W, cur_dim).unsqueeze(1)
                anomaly_map_all_bat.append(am3d)

            n_layers = len(en)
            min_anomaly_map = torch.stack(min_anomaly_map, dim=0).permute(1, 0, 2, 3, 4).squeeze(0)
            anomaly_map_all_bat = torch.stack(anomaly_map_all_bat, dim=0).squeeze(2)
            anomaly_map_all_bat = nn.UpsamplingBilinear2d(size=(unet_spatial, unet_spatial))(
                anomaly_map_all_bat.permute(0, 1, 4, 2, 3).reshape(-1, pre_min_dim, feat_size, feat_size)
            )
            anomaly_map_all_bat = anomaly_map_all_bat.view(
                n_layers, B, pre_min_dim, unet_spatial, unet_spatial
            ).permute(1, 0, 2, 3, 4).permute(0, 2, 1, 3, 4)

            pred1, _ = cal_anomaly_maps(en, de, crop_size)
            ft = feat_size * feat_size
            dino_features = [
                feat.reshape(feat.shape[0], feat.shape[1], ft).permute(0, 2, 1)
                for feat in en
            ]

            min_sim = torch.zeros_like(min_anomaly_map.squeeze())
            if min_anomaly_map.dim() == 4:
                min_sim = min_sim.unsqueeze(0)
            pred1_small = nn.UpsamplingBilinear2d(size=(unet_spatial, unet_spatial))(pred1).squeeze(1)
            min_sim[:, 0, :, :] = pred1_small
            min_sim[:, 1, :, :] = pred1_small

            output, _ = model_unet(
                anomaly_map_all_bat.float().to(device), dino_features,
                min_sim.float().to(device)
            )
            output_focl = torch.softmax(output, dim=1)
            output_focl = F.interpolate(output_focl, size=crop_size, mode='bilinear', align_corners=True)
            cf_map = output_focl[:, 1, :, :].unsqueeze(1)
            anomaly_map_final = cf_map * lamda + pred1 * (1 - lamda)
            batch_maps = anomaly_map_final.cpu()

            if heatmap_dir and hasattr(test_loader.dataset, 'samples'):
                max_to_save = heatmap_limit if heatmap_limit and heatmap_limit > 0 else batch_maps.shape[0]
                for bi in range(batch_maps.shape[0]):
                    if heatmap_limit and heatmap_limit > 0 and saved_heatmaps >= max_to_save:
                        break
                    sample_idx = sample_offset + bi
                    if sample_idx >= len(test_loader.dataset.samples):
                        break
                    ct_path, pet_path, _, label_val, patient_id = test_loader.dataset.samples[sample_idx]
                    fname = os.path.splitext(os.path.basename(ct_path))[0]
                    cls = 'abnormal' if label_val else 'normal'
                    save_prefix = os.path.join(heatmap_dir, cls, patient_id, fname)
                    raw_img = load_pseudo_rgb_image(ct_path, pet_path, modality)
                    save_heatmap_triplet(raw_img, batch_maps[bi, 0].numpy(), save_prefix, crop_size)
                    saved_heatmaps += 1

            all_maps.append(batch_maps)
            all_gts.append(gt)
            if isinstance(label, torch.Tensor):
                all_labels.extend(label.numpy().tolist())
            else:
                all_labels.extend(label)
            all_pids.extend(pid)
            sample_offset += img.shape[0]

    amaps = torch.cat(all_maps, dim=0)[:, 0].numpy()
    gts = torch.cat(all_gts, dim=0)[:, 0].numpy()
    labels = np.array(all_labels, dtype=int)

    for i in range(amaps.shape[0]):
        amaps[i] = gaussian_filter(amaps[i], sigma=sigma)

    results = _compute_three_level(
        amaps, gts, labels, all_pids,
        logger=logger, prefix=prefix, ci_iters=ci_iters,
        ci_seed=ci_seed, ci_pixel_max_samples=ci_pixel_max_samples,
        ci_hist_bins=ci_hist_bins
    )
    if eval_cache_path:
        paths = dataset_sample_paths(test_loader.dataset, len(labels))
        save_eval_cache(eval_cache_path, labels, gts, amaps, results['image_scores'], paths, ci_iters)
        if logger is not None:
            logger.info(f"Saved eval cache: {eval_cache_path}")
    return results


# =====================================================================
#  Shared metric computation
# =====================================================================

def _compute_three_level(amaps_np, gts_np, labels, pids, logger=None, prefix='',
                         ci_iters=500, ci_seed=20260707, ci_pixel_max_samples=200000,
                         ci_hist_bins=16384):
    """
    amaps_np: [N, H, W]  anomaly maps
    gts_np:   [N, H, W]  ground truth masks (0/1)
    labels:   [N]         image labels (0=normal, 1=abnormal)
    pids:     list[str]   patient ids
    """
    nonfinite_count = int((~np.isfinite(amaps_np)).sum())
    if nonfinite_count:
        amaps_np = np.nan_to_num(amaps_np, nan=0.0, posinf=0.0, neginf=0.0)
    N = amaps_np.shape[0]
    flat = amaps_np.reshape(N, -1)
    k = max(1, int(flat.shape[1] * 0.01))
    slice_scores = np.array([np.sort(flat[i])[-k:].mean() for i in range(N)], dtype=np.float64)

    patient_scores = defaultdict(list)
    patient_labels = defaultdict(list)
    for pid, score, label_value in zip(pids, slice_scores, labels):
        patient_scores[pid].append(float(score))
        patient_labels[pid].append(int(label_value))
    pat_scores = np.asarray([np.max(patient_scores[pid]) for pid in patient_scores], dtype=np.float64)
    pat_labels = np.asarray([np.max(patient_labels[pid]) for pid in patient_scores], dtype=np.int32)

    if logger is not None:
        logger.info("Computing image-level metrics and 95CI...")
    img_auroc, img_auroc_ci = bootstrap_ci(labels, slice_scores, safe_auroc, ci_iters, ci_seed)
    img_ap, img_ap_ci = bootstrap_ci(labels, slice_scores, safe_ap, ci_iters, ci_seed + 1)
    img_f1, img_f1_ci = bootstrap_ci(labels, slice_scores, f1_score_max, ci_iters, ci_seed + 2)
    if logger is not None:
        logger.info(f'[Slice-Img]  AUROC={fmt_metric(img_auroc, img_auroc_ci)}  '
                    f'AP={fmt_metric(img_ap, img_ap_ci)}  F1={fmt_metric(img_f1, img_f1_ci)}')

    if logger is not None:
        logger.info("Computing pixel-level Slice-Px(abn) metrics and 95CI...")
    (px_auroc, px_auroc_ci), (px_aupr, px_aupr_ci) = pixel_slice_bootstrap_ci(
        labels, gts_np, amaps_np, n_boot=ci_iters, seed=ci_seed + 10,
        hist_bins=ci_hist_bins, logger=logger
    )

    if logger is not None:
        logger.info("Computing patient-level metrics and 95CI...")
    pat_auroc, pat_auroc_ci = bootstrap_ci(pat_labels, pat_scores, safe_auroc, ci_iters, ci_seed + 5)
    pat_ap, pat_ap_ci = bootstrap_ci(pat_labels, pat_scores, safe_ap, ci_iters, ci_seed + 6)
    pat_f1, pat_f1_ci = bootstrap_ci(pat_labels, pat_scores, f1_score_max, ci_iters, ci_seed + 7)

    results = {
        'img_auroc': img_auroc,
        'img_auroc_ci': img_auroc_ci,
        'img_ap': img_ap,
        'img_ap_ci': img_ap_ci,
        'img_f1': img_f1,
        'img_f1_ci': img_f1_ci,
        'px_auroc_abn': px_auroc,
        'px_auroc_abn_ci': px_auroc_ci,
        'px_aupr_abn': px_aupr,
        'px_aupr_abn_ci': px_aupr_ci,
        'pat_auroc': pat_auroc,
        'pat_auroc_ci': pat_auroc_ci,
        'pat_ap': pat_ap,
        'pat_ap_ci': pat_ap_ci,
        'pat_f1': pat_f1,
        'pat_f1_ci': pat_f1_ci,
        'n_patients': len(pat_labels),
        'n_abn_patients': int(pat_labels.sum()) if len(pat_labels) else 0,
        'nonfinite_count': nonfinite_count,
        'image_scores': slice_scores,
    }
    return results


def print_results(logger, results, prefix=''):
    logger.info(f'[Slice-Img]  AUROC={fmt_metric(results["img_auroc"], results["img_auroc_ci"])}  '
                f'AP={fmt_metric(results["img_ap"], results["img_ap_ci"])}  '
                f'F1={fmt_metric(results["img_f1"], results["img_f1_ci"])}')
    logger.info(f'[Slice-Px(abn)]  AUROC={fmt_metric(results["px_auroc_abn"], results["px_auroc_abn_ci"])}  '
                f'AUPR={fmt_metric(results["px_aupr_abn"], results["px_aupr_abn_ci"])}')
    logger.info(f'[Patient({results["n_abn_patients"]}/{results["n_patients"]}abn)]  '
                f'AUROC={fmt_metric(results["pat_auroc"], results["pat_auroc_ci"])}  '
                f'AP={fmt_metric(results["pat_ap"], results["pat_ap_ci"])}  '
                f'F1={fmt_metric(results["pat_f1"], results["pat_f1_ci"])}')
    if results.get('nonfinite_count', 0):
        logger.warning(f'{prefix}Replaced {results["nonfinite_count"]} non-finite anomaly-map values with 0.0 before metrics')


# =====================================================================
#  Phase 1: Train Dinomaly
# =====================================================================

def train_phase1(args, logger):
    logger.info('=' * 60)
    logger.info('Phase 1: Training Dinomaly (unsupervised reconstruction)')
    logger.info('=' * 60)

    device = args.device
    crop_size = args.crop_size

    data_transform, gt_transform = get_data_transforms(args.image_size, crop_size)
    train_dataset = PETCTTrainDataset(args.data_path, args.modality, data_transform)
    test_dataset = PETCTTestDataset(args.data_path, args.modality, data_transform, gt_transform)
    logger.info(f'Train slices: {len(train_dataset)}, Test slices: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model, embed_dim = build_dinomaly(device)
    trainable = nn.ModuleList([model.bottleneck, model.decoder])

    from dinov1.utils import trunc_normal_
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    total_iters = args.dinomaly_epochs * len(train_loader)
    optimizer = StableAdamW(
        [{'params': trainable.parameters()}],
        lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10
    )
    lr_scheduler = WarmCosineScheduler(
        optimizer, base_value=2e-3, final_value=2e-4,
        total_iters=total_iters, warmup_iters=min(100, total_iters // 5)
    )

    best_ap = -1
    best_path = args.dinomaly_ckpt

    for epoch in range(1, args.dinomaly_epochs + 1):
        model.train()
        loss_sum = 0
        for img in tqdm(train_loader, desc=f'  P1 Epoch {epoch}/{args.dinomaly_epochs}', leave=False):
            img = img.to(device)
            en, de = model(img)
            loss = global_cosine_hm_percent(en, de, p=0.9, factor=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_sum += loss.item()

        avg_loss = loss_sum / len(train_loader)
        logger.info(f'[P1] Epoch {epoch}/{args.dinomaly_epochs}  loss={avg_loss:.6f}  lr={optimizer.param_groups[0]["lr"]:.6f}')

        if epoch % args.dinomaly_eval_freq == 0 or epoch == args.dinomaly_epochs:
            eval_prefix = f'[P1 E{epoch}] '
            results = evaluate_dinomaly(
                model, test_loader, device, crop_size, sigma=4,
                logger=logger, prefix=eval_prefix,
                ci_iters=args.ci_iters, ci_seed=args.ci_seed,
                ci_pixel_max_samples=args.ci_pixel_max_samples,
                ci_hist_bins=args.ci_hist_bins,
                eval_cache_path=os.path.join(
                    args.save_dir, 'eval_cache',
                    f'{args.dataset_tag}_{args.modality}_dinomaly_eval_cache.npz'
                )
            )
            print_results(logger, results, prefix=f'[P1 E{epoch}] ')
            cur_ap = results['img_ap']
            if not math.isnan(cur_ap) and cur_ap > best_ap:
                best_ap = cur_ap
                torch.save(model.state_dict(), best_path)
                logger.info(f'  -> New best Dinomaly (Img-AP={best_ap:.4f}), saved to {best_path}')

    if best_ap < 0:
        torch.save(model.state_dict(), best_path)
    logger.info(f'Phase 1 done. Best Dinomaly Img-AP={best_ap:.4f}')
    return best_path


# =====================================================================
#  Phase 2: Train CostFilter + Test once after final epoch
# =====================================================================

def train_phase2(args, logger, dinomaly_path):
    logger.info('=' * 60)
    logger.info('Phase 2: Training CostFilter (with synthetic anomalies)')
    logger.info('=' * 60)

    device = args.device
    crop_size = args.crop_size
    feat_size = crop_size // 14
    feat_total = feat_size * feat_size
    pre_min_dim = min(768, feat_total)
    n_layers = 2
    unet_spatial = 64

    data_transform, gt_transform = get_data_transforms(args.image_size, crop_size)
    train_dataset = PETCTTrainSynthDataset(
        args.data_path, args.modality, data_transform, gt_transform, args.image_size
    )
    test_dataset = PETCTTestDataset(args.data_path, args.modality, data_transform, gt_transform)
    logger.info(f'Train slices (synth): {len(train_dataset)}, Test slices: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model, embed_dim = build_dinomaly(device)
    state_dict = torch.load(dinomaly_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f'Loaded pretrained Dinomaly from {dinomaly_path}')

    model_unet = DiscriminativeSubNetwork_3d_att_dino_channel(
        in_channels=pre_min_dim, out_channels=2, base_channels=args.base_channels
    ).to(device).float()
    model_unet.apply(weights_init_2d)

    optimizer = torch.optim.Adam([{"params": model_unet.parameters(), "lr": args.cf_lr}], betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    loss_focal = FocalLoss_gamma()
    loss_l2 = nn.MSELoss()
    loss_ssim = SSIM()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.costfilter_epochs + 1):
        model.eval()
        model_unet.train()
        loss_sum = 0
        cnt = 0

        for synth_img, synth_mask, is_normal, cls_label in tqdm(
                train_loader, desc=f'  P2 Epoch {epoch}/{args.costfilter_epochs}', leave=False):
            synth_img = synth_img.to(device)
            synth_mask = synth_mask.to(device)
            cls_label = cls_label.to(device)

            with torch.no_grad():
                en, de = model(synth_img)
                B = en[0].shape[0]
                epsilon = 1e-8
                min_anomaly_map = []
                anomaly_map_all_bat = []

                for rec_feat, org_feat in zip(en, de):
                    H, W = org_feat.shape[2:]
                    pi = org_feat.reshape(B, -1, H * W).permute(0, 2, 1)
                    pr = rec_feat.reshape(B, -1, H * W).permute(0, 2, 1)
                    pi = pi / (torch.norm(pi, p=2, dim=-1, keepdim=True) + epsilon)
                    pr = pr / (torch.norm(pr, p=2, dim=-1, keepdim=True) + epsilon)
                    cos0 = torch.bmm(pi, pr.permute(0, 2, 1))
                    a_map, _ = torch.min(1 - cos0, dim=-1)
                    a_map = nn.UpsamplingBilinear2d(size=(unet_spatial, unet_spatial))(
                        a_map.reshape(B, 1, H, W))
                    min_anomaly_map.append(a_map)

                    one_minus = 1 - cos0
                    cur_dim = min(pre_min_dim, one_minus.shape[-1])
                    _, idx = torch.topk(one_minus, cur_dim, dim=-1, largest=False, sorted=False)
                    sel = torch.gather(one_minus, dim=-1, index=torch.sort(idx, dim=-1)[0])
                    am3d = sel.view(B, H, W, cur_dim).unsqueeze(1)
                    anomaly_map_all_bat.append(am3d)

                min_anomaly_map = torch.stack(min_anomaly_map, dim=0).permute(1, 0, 2, 3, 4).squeeze(0)
                anomaly_map_all_bat = torch.stack(anomaly_map_all_bat, dim=0).squeeze(2)
                anomaly_map_all_bat = nn.UpsamplingBilinear2d(size=(unet_spatial, unet_spatial))(
                    anomaly_map_all_bat.permute(0, 1, 4, 2, 3).reshape(-1, cur_dim, H, W)
                )
                anomaly_map_all_bat = anomaly_map_all_bat.view(
                    n_layers, B, cur_dim, unet_spatial, unet_spatial
                ).permute(1, 0, 2, 3, 4).permute(0, 2, 1, 3, 4)

                pred1, _ = cal_anomaly_maps(en, de, crop_size)
                dino_features = [
                    feat.reshape(feat.shape[0], feat.shape[1], feat_total).permute(0, 2, 1)
                    for feat in en
                ]

                min_sim = torch.zeros_like(min_anomaly_map.squeeze())
                if min_anomaly_map.dim() == 4:
                    min_sim = min_sim.unsqueeze(0)
                pred1_small = nn.UpsamplingBilinear2d(size=(unet_spatial, unet_spatial))(pred1).squeeze(1)
                min_sim[:, 0, :, :] = pred1_small
                min_sim[:, 1, :, :] = pred1_small

            optimizer.zero_grad()
            output, cls_out = model_unet(
                anomaly_map_all_bat.float(), dino_features, min_sim.float()
            )

            output_focl = torch.softmax(output, dim=1)
            output_focl = F.interpolate(output_focl, size=crop_size, mode='bilinear', align_corners=True)
            anomaly_prob = output_focl[:, 1, :, :].unsqueeze(1)

            fl = loss_focal(output_focl.float(), synth_mask.float(), 2.0)
            sl = loss_ssim(anomaly_prob.float(), synth_mask.float())
            ll = loss_l2(anomaly_prob.float(), synth_mask.float())
            sl_iou = SoftIoULoss(anomaly_prob.float(), synth_mask.float())
            seg_loss = 1.0 * fl + 0.1 * sl + 1.0 * ll + 0.1 * sl_iou
            loss = seg_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_unet.parameters(), 2.0)
            optimizer.step()
            loss_sum += loss.item()
            cnt += 1

        avg_loss = loss_sum / max(cnt, 1)
        scheduler.step(avg_loss)
        logger.info(f'[P2] Epoch {epoch}/{args.costfilter_epochs}  loss={avg_loss:.6f}  '
                     f'lr={optimizer.param_groups[0]["lr"]:.6f}')

    torch.save(
        {
            'epoch': args.costfilter_epochs,
            'dataset': args.dataset_tag,
            'modality': args.modality,
            'model_state_dict': model_unet.state_dict(),
        },
        args.costfilter_ckpt,
    )
    logger.info(f'Saved final CostFilter epoch {args.costfilter_epochs} to {args.costfilter_ckpt}')

    results = evaluate_costfilter(
        model, model_unet, test_loader, device, crop_size,
        feat_size, pre_min_dim, embed_dim, sigma=4, lamda=args.lamda,
        heatmap_dir=args.heatmap_dir, heatmap_limit=args.heatmap_limit, modality=args.modality,
        logger=logger, prefix=f'[P2 Final E{args.costfilter_epochs}] ',
        ci_iters=args.ci_iters, ci_seed=args.ci_seed,
        ci_pixel_max_samples=args.ci_pixel_max_samples,
        ci_hist_bins=args.ci_hist_bins,
        eval_cache_path=args.eval_cache_path
    )
    print_results(logger, results, prefix=f'[P2 Final E{args.costfilter_epochs}] ')
    logger.info(f'Heatmaps saved to {args.heatmap_dir}')
    logger.info('Phase 2 done.')


# =====================================================================
#  Direct Test
# =====================================================================

def load_dinomaly_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)


def load_costfilter_checkpoint(model_unet, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model_unet.load_state_dict(state)


def test_only(args, logger):
    logger.info('=' * 60)
    logger.info(f'Test only: {args.test_stage}')
    logger.info('=' * 60)

    device = args.device
    crop_size = args.crop_size
    feat_size = crop_size // 14
    feat_total = feat_size * feat_size
    pre_min_dim = min(768, feat_total)

    data_transform, gt_transform = get_data_transforms(args.image_size, crop_size)
    test_dataset = PETCTTestDataset(args.data_path, args.modality, data_transform, gt_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logger.info(f'Test slices: {len(test_dataset)}')

    model, embed_dim = build_dinomaly(device)
    if not os.path.exists(args.dinomaly_ckpt):
        logger.error(f'Dinomaly checkpoint not found: {args.dinomaly_ckpt}')
        sys.exit(1)
    load_dinomaly_checkpoint(model, args.dinomaly_ckpt, device)
    logger.info(f'Loaded Dinomaly from {args.dinomaly_ckpt}')

    if args.test_stage == 'dinomaly':
        results = evaluate_dinomaly(
            model, test_loader, device, crop_size, sigma=4,
            logger=logger, prefix='[Test Dinomaly] ',
            ci_iters=args.ci_iters, ci_seed=args.ci_seed,
            ci_pixel_max_samples=args.ci_pixel_max_samples,
            ci_hist_bins=args.ci_hist_bins,
            eval_cache_path=args.eval_cache_path
        )
        print_results(logger, results, prefix='[Test Dinomaly] ')
        return

    if not os.path.exists(args.costfilter_ckpt):
        logger.error(f'CostFilter checkpoint not found: {args.costfilter_ckpt}')
        sys.exit(1)
    model_unet = DiscriminativeSubNetwork_3d_att_dino_channel(
        in_channels=pre_min_dim, out_channels=2, base_channels=args.base_channels
    ).to(device).float()
    load_costfilter_checkpoint(model_unet, args.costfilter_ckpt, device)
    logger.info(f'Loaded CostFilter from {args.costfilter_ckpt}')

    results = evaluate_costfilter(
        model, model_unet, test_loader, device, crop_size,
        feat_size, pre_min_dim, embed_dim, sigma=4, lamda=args.lamda,
        heatmap_dir=args.heatmap_dir, heatmap_limit=args.heatmap_limit, modality=args.modality,
        logger=logger, prefix='[Test CostFilter] ',
        ci_iters=args.ci_iters, ci_seed=args.ci_seed,
        ci_pixel_max_samples=args.ci_pixel_max_samples,
        ci_hist_bins=args.ci_hist_bins,
        eval_cache_path=args.eval_cache_path
    )
    print_results(logger, results, prefix='[Test CostFilter] ')
    logger.info(f'Heatmaps saved to {args.heatmap_dir}')


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='CostFilter-AD for PET-CT')
    parser.add_argument('--dataset', type=str, default='psma',
                        choices=sorted(DATASET_PROFILES.keys()),
                        help='Dataset profile for paths and artifact names')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Optional tag used in checkpoint and heatmap paths')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override dataset root; required for --dataset custom')
    parser.add_argument('--modality', type=str, default='petct',
                        choices=['ct', 'pet', 'petct'],
                        help='Input modality: ct / pet / petct')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=256,
                        help='Resize target (then center-crop to crop_size)')
    parser.add_argument('--crop_size', type=int, default=252,
                        help='Must be divisible by 14 for DINOv2 (252=18*14)')
    parser.add_argument('--dinomaly_epochs', type=int, default=50,
                        help='Phase 1 Dinomaly training epochs')
    parser.add_argument('--dinomaly_eval_freq', type=int, default=5,
                        help='Evaluate Dinomaly every N epochs')
    parser.add_argument('--costfilter_epochs', type=int, default=30,
                        help='Phase 2 CostFilter training epochs; final epoch is saved and tested')
    parser.add_argument('--cf_lr', type=float, default=1e-4,
                        help='CostFilter learning rate')
    parser.add_argument('--lamda', type=float, default=0.5,
                        help='Weight for CostFilter output blending')
    parser.add_argument('--base_channels', type=int, default=48,
                        help='3D UNet base channel width')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output directory; defaults to checkpoint_<dataset>_<modality>')
    parser.add_argument('--dinomaly_ckpt_path', type=str, default=None,
                        help='Override Dinomaly checkpoint path for skip/test')
    parser.add_argument('--costfilter_ckpt_path', type=str, default=None,
                        help='Override CostFilter checkpoint path for test')
    parser.add_argument('--heatmap_dir', type=str, default=None,
                        help='Directory for final CostFilter heatmaps')
    parser.add_argument('--heatmap_limit', type=int, default=128,
                        help='Max heatmap triplets to save; <=0 saves all')
    parser.add_argument('--eval_cache_path', type=str, default=None,
                        help='Path to save eval cache npz; defaults to save_dir/eval_cache/<dataset>_<modality>_<stage>_eval_cache.npz')
    parser.add_argument('--ci_iters', type=int, default=500,
                        help='Bootstrap iterations for all 95CI metrics')
    parser.add_argument('--ci_seed', type=int, default=20260707,
                        help='Random seed for bootstrap confidence intervals')
    parser.add_argument('--ci_pixel_max_samples', type=int, default=200000,
                        help='Deprecated; kept for old commands. Pixel CI now uses abnormal-slice histogram bootstrap.')
    parser.add_argument('--ci_hist_bins', type=int, default=16384,
                        help='Histogram bins for pixel abnormal-slice bootstrap CI')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_phase1', action='store_true',
                        help='Skip Dinomaly training (load from save_dir)')
    parser.add_argument('--skip_phase2', action='store_true',
                        help='Skip CostFilter training (Dinomaly-only eval)')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run evaluation from saved checkpoints')
    parser.add_argument('--test_stage', type=str, default='costfilter',
                        choices=['dinomaly', 'costfilter'],
                        help='Which checkpoint stack to evaluate with --test_only')
    args = parser.parse_args()
    resolve_dataset_profile(args)

    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    assert args.crop_size % 14 == 0, f'crop_size must be divisible by 14, got {args.crop_size}'

    os.makedirs(args.save_dir, exist_ok=True)
    setup_seed(args.seed)
    logger = get_logger('petct_ad', args.save_dir)
    logger.info(f'Args: {vars(args)}')
    logger.info(f'Device: {args.device}')

    if args.test_only:
        test_only(args, logger)
        return

    dinomaly_path = args.dinomaly_ckpt
    if not args.skip_phase1:
        dinomaly_path = train_phase1(args, logger)
    else:
        if not os.path.exists(dinomaly_path):
            logger.error(f'--skip_phase1 but {dinomaly_path} not found!')
            sys.exit(1)
        logger.info(f'Skipping Phase 1, using {dinomaly_path}')

    if not args.skip_phase2:
        train_phase2(args, logger, dinomaly_path)
    else:
        logger.info('Phase 2 skipped. Dinomaly-only pipeline complete.')


if __name__ == '__main__':
    main()
