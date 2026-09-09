"""MambaAD for PET-CT medical anomaly detection.

Examples:
  python run_petct.py --dataset psma --modality petct --gpu 0
  python run_petct.py --mode test --dataset fdg --modality petct --gpu 0
  python run_petct.py --mode test --dataset psma --save_heatmaps 50
  python run_petct.py --mode test --dataset psma --save_all_heatmaps
"""

import os
import sys
import random
import logging
import argparse
import warnings
import types
import importlib.util
from argparse import Namespace
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

warnings.filterwarnings("ignore")
"""
conda activate /nfsdata_a40/cyf/miniconda3/envs/py310
python -c "import mamba_ssm; print('mamba_ssm ok')"
python -c "import causal_conv1d; print('causal_conv1d ok')"


95ci
nohup python run_petct.py \
  --mode test \
  --dataset psma \
  --modality petct \
  --gpu 1 \
  --ci_bootstrap 500 \
  --hist_bins 16384 \
  --progress_every 50 \
  > test_psma_petct_metrics.log 2>&1 &

nohup python run_petct.py \
  --mode test \
  --dataset fdg \
  --modality petct \
  --gpu 0 \
  --ci_bootstrap 500 \
  --progress_every 50 \
  > fdg_npz.log 2>&1 &
"""
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS = {
    'fdg': '/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG',
    'psma': '/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA',
}


# =====================================================================
#  Load local MambaAD model
# =====================================================================

class Registry:
    def __init__(self, name):
        self.name = name
        self.modules = {}

    def register_module(self, fn):
        self.modules[fn.__name__] = fn
        return fn


def _load_mambaad_module():
    """Import the local mambaad.py without requiring the full ADer package."""
    old_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    sys.path.insert(0, SCRIPT_DIR)

    model_pkg = types.ModuleType('model')
    MODEL = Registry('Model')
    model_pkg.MODEL = MODEL

    import timm
    from timm.models._helpers import load_checkpoint
    from timm.models.layers import set_layer_config

    def get_model(cfg_model):
        model_name = cfg_model.name
        kwargs = {k: v for k, v in cfg_model.kwargs.items()}
        pretrained = kwargs.pop('pretrained')
        checkpoint_path = kwargs.pop('checkpoint_path')
        strict = kwargs.pop('strict')
        if model_name.startswith('timm_'):
            real_name = model_name[5:]
            with set_layer_config(scriptable=None, exportable=None, no_jit=None):
                model = timm.create_model(real_name, pretrained=pretrained, **kwargs)
            if not pretrained and checkpoint_path:
                load_checkpoint(model, checkpoint_path, strict=strict)
        return model

    model_pkg.get_model = get_model
    sys.modules['model'] = model_pkg

    spec = importlib.util.spec_from_file_location(
        'model.mambaad', os.path.join(SCRIPT_DIR, 'model', 'mambaad.py'))
    mambaad_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mambaad_mod)

    os.chdir(old_cwd)
    return mambaad_mod


_mambaad = _load_mambaad_module()


def build_mambaad(pretrained_path, depths_decoder, scan_type, num_direction):
    """Build MambaAD model using ADer's original code."""
    model_t = Namespace()
    model_t.name = 'timm_resnet34'
    model_t.kwargs = dict(
        pretrained=False, checkpoint_path=pretrained_path,
        strict=False, features_only=True, out_indices=[1, 2, 3])
    model_s = dict(
        depths_decoder=list(depths_decoder),
        scan_type=scan_type,
        num_direction=num_direction)
    return _mambaad.mambaad(model_t=model_t, model_s=model_s)


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
    logger.handlers.clear()
    fmt = logging.Formatter('%(message)s')
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
    if len(np.unique(y_true)) < 2:
        return 0.0
    precs, recs, _ = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    return float(f1s[:-1].max()) if len(f1s) > 1 else 0.0


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _clip_ci(lo, hi):
    return (float(max(0.0, lo)), float(min(1.0, hi)))


def _metric_value_ci(value, ci):
    return {'value': float(value), 'ci': _clip_ci(ci[0], ci[1])}


def _format_metric(metric):
    return f'{metric["value"] * 100:.2f}% (95% CI {metric["ci"][0] * 100:.2f}-{metric["ci"][1] * 100:.2f}%)'


def _bootstrap_metric(y_true, y_score, metric_fn, iters, seed):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    value = metric_fn(y_true, y_score)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_fn(y_true[idx], y_score[idx]))
    if not values:
        return _metric_value_ci(value, (value, value))
    return _metric_value_ci(value, np.percentile(values, [2.5, 97.5]))


def auroc_metric(y_true, y_score, args, seed_offset=0):
    return _bootstrap_metric(y_true, y_score, _safe_auroc, args.ci_bootstrap, args.ci_seed + seed_offset)


def aupr_metric(y_true, y_score, args, seed_offset=0):
    return _bootstrap_metric(y_true, y_score, _safe_ap, args.ci_bootstrap, args.ci_seed + seed_offset)


def f1_metric(y_true, y_score, args, seed_offset=0):
    return _bootstrap_metric(y_true, y_score, f1_score_max, args.ci_bootstrap, args.ci_seed + seed_offset)


def _metrics_from_hist(pos_hist, neg_hist):
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


def pixel_slice_bootstrap(labels, masks, maps, args, logger=None):
    keep = np.asarray(labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        zero = _metric_value_ci(0.0, (0.0, 0.0))
        return zero, zero

    if logger is not None:
        logger.info('Computing exact full-pixel point estimates with sklearn...')
    pixel_gt = masks[keep].reshape(-1).astype(int)
    pixel_score = maps[keep].reshape(-1).astype(np.float64)
    exact_auroc = _safe_auroc(pixel_gt, pixel_score)
    exact_aupr = _safe_ap(pixel_gt, pixel_score)

    bins = args.hist_bins
    score_min = float(np.min(maps[keep]))
    score_max = float(np.max(maps[keep]))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    scale = (bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        scores = maps[idx].reshape(-1)
        mask = masks[idx].reshape(-1).astype(bool)
        bin_idx = np.floor((scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[mask], minlength=bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~mask], minlength=bins).astype(np.uint32)

    rng = np.random.default_rng(args.ci_seed + 10)
    aurocs, auprs = [], []
    n = len(abn_idx)
    for i in range(args.ci_bootstrap):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum('i,ij->j', weights, pos_hists, optimize=True)
        neg = np.einsum('i,ij->j', weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if logger is not None and ((i + 1) % args.progress_every == 0 or i + 1 == args.ci_bootstrap):
            logger.info(f'Pixel histogram slice bootstrap: {i + 1}/{args.ci_bootstrap}')

    return (
        _metric_value_ci(exact_auroc, np.percentile(aurocs, [2.5, 97.5])),
        _metric_value_ci(exact_aupr, np.percentile(auprs, [2.5, 97.5])),
    )


def patient_id_from_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


# =====================================================================
#  Transforms
# =====================================================================

def get_data_transforms(image_size):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train),
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    return data_transforms, gt_transforms


# =====================================================================
#  Datasets
# =====================================================================

class PETCTTrainDataset(Dataset):
    def __init__(self, data_root, modality, transform):
        self.modality = modality
        self.transform = transform
        self.samples = []
        normal_dir = os.path.join(data_root, 'train', 'normal')
        if not os.path.isdir(normal_dir):
            raise FileNotFoundError(f'Train normal directory not found: {normal_dir}')
        for patient in sorted(os.listdir(normal_dir)):
            ct_dir = os.path.join(normal_dir, patient, 'ct')
            pet_dir = os.path.join(normal_dir, patient, 'pet')
            if not os.path.isdir(ct_dir) and not os.path.isdir(pet_dir):
                continue
            filenames = set()
            if os.path.isdir(ct_dir):
                filenames.update(f for f in os.listdir(ct_dir) if f.endswith('.png'))
            if os.path.isdir(pet_dir):
                filenames.update(f for f in os.listdir(pet_dir) if f.endswith('.png'))
            for fname in sorted(filenames):
                if not fname.endswith('.png'):
                    continue
                ct_path = os.path.join(ct_dir, fname)
                pet_path = os.path.join(pet_dir, fname)
                self.samples.append((ct_path, pet_path))

    def _load_rgb(self, ct_path, pet_path):
        ct_exists = os.path.exists(ct_path)
        pet_exists = os.path.exists(pet_path)
        if not ct_exists and not pet_exists:
            raise FileNotFoundError(f'Neither CT nor PET slice exists: {ct_path}, {pet_path}')
        ct = Image.open(ct_path).convert('L') if ct_exists else Image.open(pet_path).convert('L')
        pet = Image.open(pet_path).convert('L') if pet_exists else ct
        if self.modality == 'ct':
            return Image.merge('RGB', [ct, ct, ct])
        elif self.modality == 'pet':
            return Image.merge('RGB', [pet, pet, pet])
        else:
            return Image.merge('RGB', [ct, pet, pet])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_path, pet_path = self.samples[idx]
        return self.transform(self._load_rgb(ct_path, pet_path))


class PETCTTestDataset(Dataset):
    def __init__(self, data_root, modality, transform, gt_transform):
        self.modality = modality
        self.transform = transform
        self.gt_transform = gt_transform
        self.samples = []

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
                if not os.path.isdir(ct_dir) and not os.path.isdir(pet_dir):
                    continue
                filenames = set()
                if os.path.isdir(ct_dir):
                    filenames.update(f for f in os.listdir(ct_dir) if f.endswith('.png'))
                if os.path.isdir(pet_dir):
                    filenames.update(f for f in os.listdir(pet_dir) if f.endswith('.png'))
                for fname in sorted(filenames):
                    if not fname.endswith('.png'):
                        continue
                    ct_path = os.path.join(ct_dir, fname)
                    pet_path = os.path.join(pet_dir, fname)
                    lbl_path = (os.path.join(label_dir, fname)
                                if label_dir and os.path.isdir(label_dir) else None)
                    self.samples.append((ct_path, pet_path, lbl_path, label_val, patient))

    def _load_rgb(self, ct_path, pet_path):
        ct_exists = os.path.exists(ct_path)
        pet_exists = os.path.exists(pet_path)
        if not ct_exists and not pet_exists:
            raise FileNotFoundError(f'Neither CT nor PET slice exists: {ct_path}, {pet_path}')
        ct = Image.open(ct_path).convert('L') if ct_exists else Image.open(pet_path).convert('L')
        pet = Image.open(pet_path).convert('L') if pet_exists else ct
        if self.modality == 'ct':
            return Image.merge('RGB', [ct, ct, ct])
        elif self.modality == 'pet':
            return Image.merge('RGB', [pet, pet, pet])
        else:
            return Image.merge('RGB', [ct, pet, pet])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_path, pet_path, lbl_path, label, patient_id = self.samples[idx]
        rgb = self._load_rgb(ct_path, pet_path)
        img = self.transform(rgb)
        if lbl_path and os.path.exists(lbl_path):
            gt = Image.open(lbl_path).convert('L')
            gt = self.gt_transform(gt)
            gt = (gt > 0.5).float()
        else:
            gt = torch.zeros(1, img.shape[1], img.shape[2])
        return img, gt, label, patient_id, ct_path, pet_path


# =====================================================================
#  Anomaly Map & Three-Level Evaluation
# =====================================================================

def compute_anomaly_maps(feats_t, feats_s, out_size, sigma=4):
    """Multi-scale cosine similarity anomaly maps with Gaussian smoothing."""
    anomaly_map = torch.zeros(feats_t[0].shape[0], 1, out_size, out_size,
                              device=feats_t[0].device)
    for ft, fs in zip(feats_t, feats_s):
        a_map = 1 - F.cosine_similarity(ft, fs, dim=1)
        a_map = F.interpolate(a_map.unsqueeze(1), size=(out_size, out_size),
                              mode='bilinear', align_corners=True)
        anomaly_map += a_map

    amaps_np = anomaly_map[:, 0].detach().cpu().numpy()
    for i in range(amaps_np.shape[0]):
        amaps_np[i] = gaussian_filter(amaps_np[i], sigma=sigma)
    return amaps_np


def normalize01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def colorize_heatmap(amap):
    h = normalize01(amap)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)


def load_display_rgb(ct_path, pet_path, modality, image_size):
    ct_exists = os.path.exists(ct_path)
    pet_exists = os.path.exists(pet_path)
    if not ct_exists and not pet_exists:
        raise FileNotFoundError(f'Neither CT nor PET slice exists: {ct_path}, {pet_path}')
    ct = Image.open(ct_path).convert('L') if ct_exists else Image.open(pet_path).convert('L')
    pet = Image.open(pet_path).convert('L') if pet_exists else ct
    if modality == 'ct':
        rgb = Image.merge('RGB', [ct, ct, ct])
    elif modality == 'pet':
        rgb = Image.merge('RGB', [pet, pet, pet])
    else:
        rgb = Image.merge('RGB', [ct, pet, pet])
    return rgb.resize((image_size, image_size), Image.BILINEAR)


def save_heatmaps(output_dir, amaps_np, labels, ct_paths, pet_paths, modality, image_size, limit):
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for idx, (amap, label, ct_path, pet_path) in enumerate(zip(amaps_np, labels, ct_paths, pet_paths)):
        if limit is not None and saved >= limit:
            break
        base = np.asarray(load_display_rgb(ct_path, pet_path, modality, image_size), dtype=np.float32) / 255.0
        heat = colorize_heatmap(amap)
        overlay = np.clip(base * 0.55 + heat * 0.45, 0, 1)
        patient = patient_id_from_path(ct_path if os.path.exists(ct_path) else pet_path)
        name = os.path.splitext(os.path.basename(ct_path if os.path.exists(ct_path) else pet_path))[0]
        out_name = f'{idx:06d}_label{int(label)}_{patient}_{name}.png'
        Image.fromarray((overlay * 255).astype(np.uint8)).save(os.path.join(output_dir, out_name))
        saved += 1
    return saved


def compute_three_level(amaps_np, gts_np, labels, paths, args, logger=None, prefix=''):
    N = amaps_np.shape[0]
    flat = amaps_np.reshape(N, -1)
    k = max(1, int(flat.shape[1] * 0.01))
    slice_scores = np.array([np.sort(flat[i])[-k:].mean() for i in range(N)])

    if logger is not None:
        logger.info('Computing image-level metrics and 95CI...')
    image_metrics = {
        'img_auroc': auroc_metric(labels, slice_scores, args, seed_offset=0),
        'img_aupr': aupr_metric(labels, slice_scores, args, seed_offset=1),
        'img_f1': f1_metric(labels, slice_scores, args, seed_offset=2),
    }
    if logger is not None:
        print_image_results(logger, image_metrics, prefix=prefix)

    if logger is not None:
        logger.info('Computing pixel-level Slice-Px(abn) metrics and 95CI...')
    px_auroc, px_aupr = pixel_slice_bootstrap(labels, gts_np, amaps_np, args, logger=logger)
    pixel_metrics = {
        'px_auroc_abn': px_auroc,
        'px_aupr_abn': px_aupr,
    }
    if logger is not None:
        print_pixel_results(logger, pixel_metrics, prefix=prefix)

    if logger is not None:
        logger.info('Computing patient-level metrics and 95CI...')
    patient_dict = defaultdict(lambda: {'scores': [], 'label': 0})
    for i, path in enumerate(paths):
        pid = patient_id_from_path(path)
        patient_dict[pid]['scores'].append(slice_scores[i])
        patient_dict[pid]['label'] = max(patient_dict[pid]['label'], labels[i])
    pat_scores = np.array([max(v['scores']) for v in patient_dict.values()])
    pat_labels = np.array([v['label'] for v in patient_dict.values()])
    patient_metrics = {
        'pat_auroc': auroc_metric(pat_labels, pat_scores, args, seed_offset=5),
        'pat_aupr': aupr_metric(pat_labels, pat_scores, args, seed_offset=6),
        'pat_f1': f1_metric(pat_labels, pat_scores, args, seed_offset=7),
        'n_patients': len(pat_labels),
        'n_abn_patients': int(pat_labels.sum()),
    }

    results = {}
    results.update(image_metrics)
    results.update(pixel_metrics)
    results.update(patient_metrics)
    return results


def print_image_results(logger, results, prefix=''):
    logger.info(f'{prefix}[Slice-Img]  AUROC={_format_metric(results["img_auroc"])}  '
                f'AUPR={_format_metric(results["img_aupr"])}  '
                f'F1={_format_metric(results["img_f1"])}')


def print_pixel_results(logger, results, prefix=''):
    logger.info(f'{prefix}[Slice-Px(abn)]  AUROC={_format_metric(results["px_auroc_abn"])}  '
                f'AUPR={_format_metric(results["px_aupr_abn"])}')


def print_results(logger, results, prefix=''):
    print_image_results(logger, results, prefix=prefix)
    print_pixel_results(logger, results, prefix=prefix)
    logger.info(f'{prefix}[Patient({results["n_abn_patients"]}/{results["n_patients"]}abn)] '
                f'AUROC={_format_metric(results["pat_auroc"])}  '
                f'AUPR={_format_metric(results["pat_aupr"])}  '
                f'F1={_format_metric(results["pat_f1"])}')


# =====================================================================
#  Evaluate
# =====================================================================

@torch.no_grad()
def evaluate(model, test_loader, device, image_size, args, sigma=4, return_outputs=False, logger=None, prefix=''):
    model.eval()
    all_maps, all_gts, all_labels, all_pids, all_ct_paths, all_pet_paths = [], [], [], [], [], []

    for img, gt, label, pid, ct_path, pet_path in tqdm(test_loader, desc='  Eval', leave=False):
        img = img.to(device)
        feats_t, feats_s = model(img)
        amap = compute_anomaly_maps(feats_t, feats_s, image_size, sigma=sigma)
        all_maps.append(amap)
        all_gts.append(gt[:, 0].numpy())
        if isinstance(label, torch.Tensor):
            all_labels.extend(label.numpy().tolist())
        else:
            all_labels.extend(label)
        all_pids.extend(pid)
        all_ct_paths.extend(ct_path)
        all_pet_paths.extend(pet_path)

    amaps_np = np.concatenate(all_maps, axis=0)
    gts_np = np.concatenate(all_gts, axis=0)
    labels = np.array(all_labels, dtype=int)
    paths = [ct if os.path.exists(ct) else pet for ct, pet in zip(all_ct_paths, all_pet_paths)]
    results = compute_three_level(amaps_np, gts_np, labels, paths, args, logger=logger, prefix=prefix)
    if not return_outputs:
        return results
    return results, {
        'amaps': amaps_np,
        'gts': gts_np,
        'labels': labels,
        'pids': all_pids,
        'ct_paths': all_ct_paths,
        'pet_paths': all_pet_paths,
    }


# =====================================================================
#  Train + Eval Pipeline
# =====================================================================

def train_and_eval(args, logger):
    device = args.device
    image_size = args.image_size

    data_transform, gt_transform = get_data_transforms(image_size)
    train_dataset = PETCTTrainDataset(args.data_path, args.modality, data_transform)
    test_dataset = PETCTTestDataset(args.data_path, args.modality, data_transform, gt_transform)
    logger.info(f'Train slices: {len(train_dataset)},  Test slices: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_mambaad(
        pretrained_path=args.pretrained_encoder,
        depths_decoder=tuple(args.depths_decoder),
        scan_type=args.scan_type,
        num_direction=args.num_direction,
    ).to(device)
    model.train()

    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    logger.info(f'Trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M  '
                f'Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    optimizer = optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100)

    final_path = args.weight_path
    loss_lam = args.loss_lam

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0
        for img in tqdm(train_loader, desc=f'  Epoch {epoch}/{args.epochs}', leave=False):
            img = img.to(device)
            feats_t, feats_s = model(img)
            loss = sum(F.mse_loss(ft, fs) for ft, fs in zip(feats_t, feats_s)) * loss_lam
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        avg_loss = loss_sum / len(train_loader)
        lr_now = optimizer.param_groups[0]['lr']
        scheduler.step()
        logger.info(f'Epoch {epoch}/{args.epochs}  loss={avg_loss:.6f}  lr={lr_now:.6f}')

    eval_result = evaluate(model, test_loader, device, image_size, args, sigma=args.sigma,
                           return_outputs=args.should_save_heatmaps, logger=logger, prefix='')
    if args.should_save_heatmaps:
        results, outputs = eval_result
    else:
        results = eval_result
        outputs = None
    print_results(logger, results, prefix='')

    torch.save({
        'epoch': args.epochs,
        'dataset': args.dataset,
        'data_path': args.data_path,
        'modality': args.modality,
        'model_state_dict': model.state_dict(),
        'results': results,
    }, final_path)
    logger.info(f'Saved final epoch {args.epochs} checkpoint to {final_path}')

    maybe_save_heatmaps(args, logger, outputs)
    return final_path


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt


def run_test(args, logger):
    device = args.device
    image_size = args.image_size
    data_transform, gt_transform = get_data_transforms(image_size)
    test_dataset = PETCTTestDataset(args.data_path, args.modality, data_transform, gt_transform)
    logger.info(f'Test slices: {len(test_dataset)}')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_mambaad(
        pretrained_path='',
        depths_decoder=tuple(args.depths_decoder),
        scan_type=args.scan_type,
        num_direction=args.num_direction,
    ).to(device)
    logger.info('Test mode: skipped encoder pretrain loading; loading full trained checkpoint next.')
    ckpt = load_checkpoint(model, args.weight_path, device)
    logger.info(f'Loaded checkpoint from {args.weight_path} (epoch={ckpt.get("epoch", "unknown")})')

    results, outputs = evaluate(model, test_loader, device, image_size, args, sigma=args.sigma,
                                return_outputs=True, logger=logger, prefix='')
    print_results(logger, results, prefix='')
    maybe_save_heatmaps(args, logger, outputs)
    return results


def maybe_save_heatmaps(args, logger, outputs):
    if not args.should_save_heatmaps:
        return
    if outputs is None:
        logger.info('Heatmap outputs were not collected.')
        return
    limit = None if args.save_all_heatmaps else args.save_heatmaps
    saved = save_heatmaps(
        args.heatmap_dir,
        outputs['amaps'],
        outputs['labels'],
        outputs['ct_paths'],
        outputs['pet_paths'],
        args.modality,
        args.image_size,
        limit,
    )
    logger.info(f'Saved {saved} heatmap(s) to {args.heatmap_dir}')


def resolve_args(args):
    if args.dataset not in DEFAULT_DATASETS:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    if args.data_path is None:
        args.data_path = DEFAULT_DATASETS[args.dataset]

    if args.save_dir is None:
        args.save_dir = os.path.join(
            SCRIPT_DIR, 'checkpoints', f'{args.dataset}_{args.modality}')
    os.makedirs(args.save_dir, exist_ok=True)

    if args.weight_path is None:
        args.weight_path = os.path.join(
            args.save_dir, f'mambaad_{args.dataset}_{args.modality}_epoch{args.epochs}.pth')
    if args.heatmap_dir is None:
        args.heatmap_dir = os.path.join(args.save_dir, 'heatmaps')
    args.should_save_heatmaps = args.save_all_heatmaps or args.save_heatmaps > 0
    return args


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='MambaAD for PET-CT')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='psma', choices=sorted(DEFAULT_DATASETS))
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override dataset root. Defaults are selected by --dataset.')
    parser.add_argument('--modality', type=str, default='petct',
                        choices=['ct', 'pet', 'petct'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--loss_lam', type=float, default=5.0)
    parser.add_argument('--depths_decoder', type=int, nargs=4, default=[3, 4, 6, 3])
    parser.add_argument('--scan_type', type=str, default='hilbert',
                        choices=['sweep', 'scan', 'zorder', 'zigzag', 'hilbert'])
    parser.add_argument('--num_direction', type=int, default=8, choices=[2, 4, 8])
    parser.add_argument('--pretrained_encoder', type=str,
                        default=os.path.join(SCRIPT_DIR, 'model/pretrain/resnet34-b627a593.pth'))
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--weight_path', type=str, default=None,
                        help='Checkpoint to save in train mode or load in test mode.')
    parser.add_argument('--sigma', type=float, default=4)
    parser.add_argument('--save_heatmaps', type=int, default=0,
                        help='Number of heatmaps to save. Default 0 disables saving.')
    parser.add_argument('--save_all_heatmaps', action='store_true',
                        help='Save heatmaps for every test slice.')
    parser.add_argument('--heatmap_dir', type=str, default=None)
    parser.add_argument('--ci_bootstrap', type=int, default=500,
                        help='Bootstrap iterations for 95% confidence intervals.')
    parser.add_argument('--ci_seed', type=int, default=20260717,
                        help='Random seed for confidence interval resampling.')
    parser.add_argument('--ci_max_samples', type=int, default=200000,
                        help='Deprecated; ignored. CI follows slice/patient bootstrap protocol.')
    parser.add_argument('--hist_bins', type=int, default=16384,
                        help='Score histogram bins for pixel-level bootstrap acceleration.')
    parser.add_argument('--progress_every', type=int, default=50,
                        help='Log pixel histogram bootstrap progress every N iterations.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args = resolve_args(args)

    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    setup_seed(args.seed)

    logger = get_logger('mambaad_petct', args.save_dir)
    logger.info(f'Args: {vars(args)}')
    logger.info(f'Device: {args.device}')

    if args.mode == 'train':
        train_and_eval(args, logger)
    else:
        run_test(args, logger)


if __name__ == '__main__':
    main()
