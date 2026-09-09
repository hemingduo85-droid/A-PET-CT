
import os
import sys
import torch
from torch.utils.data import DataLoader

import numpy as np
from numpy import ndarray
import pandas as pd

from tqdm import tqdm

import time
import argparse
import yaml
from collections import defaultdict

from src.datasets import build_dataset
from src.denoiser import get_denoiser, Denoiser
from src.backbones import get_backbone, get_backbone_feature_shape

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from src.utils import AverageMeter

from torch.utils.data import ConcatDataset
from torch.nn import functional as F
import matplotlib.pyplot as plt

MAX_BATCH_SIZE = 64
NUM_WORKERS = 4
CI_BOOTSTRAP_ITERS = 500
CI_ALPHA = 0.95
CI_HIST_BINS = 16384

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def parse_args():
    parser = argparse.ArgumentParser(description="InversionAD Inference")
    
    parser.add_argument('--eval_strategy', type=str, default='inversion', choices=['inversion', 'reconstruction'], help='Evaluation strategy: inversion or reconstruction')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to the directory contais results')
    parser.add_argument('--eval_step', type=int, default=-1, help='Number of steps for evaluation')
    parser.add_argument('--noise_step', type=int, default=8, help='Number of noise steps for evaluation')
    parser.add_argument('--use_ema_model', action='store_true', help='Use EMA model for evaluation')
    parser.add_argument('--use_best_model', action='store_true', help='Use best model for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--category', type=str, default=None, help='Category to evaluate on')
    parser.add_argument('--visualize_samples', action='store_true', help='Visualize samples during evaluation')
    parser.add_argument("--save_heatmaps", type=int, default=None, help="Number of heatmaps to save; 0 disables, -1 saves all")
    parser.add_argument("--save_all_heatmaps", action="store_true", help="Save heatmaps for all evaluated samples")
    parser.add_argument("--bootstrap_iters", type=int, default=CI_BOOTSTRAP_ITERS, help="Bootstrap iterations for non-AUROC CI")
    parser.add_argument("--ci_pixel_max_samples", type=int, default=0, help="Deprecated compatibility option; pixel CI uses abnormal-slice histogram bootstrap")
    parser.add_argument("--ci_hist_bins", type=int, default=CI_HIST_BINS, help="Histogram bins for pixel slice-bootstrap CI")
    
    args = parser.parse_args()
    assert sum([args.use_best_model, args.use_ema_model]) < 2, "Please specify either --use_best_model or --use_ema_model"
    return args

def denormalize(x):
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = x * imagenet_std + imagenet_mean
    return x.clamp(0, 1)

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

def main(config, args):
    
    # For reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.save_dir is None:
        args.save_dir = config.get("logging", {}).get("save_dir")
    assert args.save_dir is not None, "Please provide a save directory or logging.save_dir in config"

    dataset_config = config['data']
    device = config['meta']['device']
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        dev_idx = torch.device(device).index
        if dev_idx is None:
            dev_idx = torch.cuda.current_device()
        torch.cuda.set_device(dev_idx)
        free_mem, total_mem = torch.cuda.mem_get_info(dev_idx)
        logger.info(
            f"Evaluation device: cuda:{dev_idx} ({torch.cuda.get_device_name(dev_idx)}), "
            f"free={free_mem / 1024**3:.2f}GiB / total={total_mem / 1024**3:.2f}GiB"
        )
    else:
        logger.info(f"Evaluation device: {device}")
    
    if args.category is not None:
        dataset_config['category'] = args.category
        dataset_config['dataset_name'] = dataset_config['dataset_name'].split('_all')[0]  # Remove '_all' suffix if exists
        logger.info(f"Evaluating on category: {args.category}")
    
    dataset_config['train'] = False
    dataset_config['normal_only'] = False
    dataset_config['anom_only'] = True
    anom_dataset = build_dataset(**dataset_config)
    dataset_config['anom_only'] = False
    dataset_config['normal_only'] = True
    normal_dataset = build_dataset(**dataset_config)
    
    is_multi_class = isinstance(normal_dataset, ConcatDataset) or isinstance(anom_dataset, ConcatDataset)
    if is_multi_class:
        logger.info(f"Using multi-class dataset")
        anom_loader = [
            DataLoader(
                anom_ds,
                batch_size=MAX_BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                drop_last=False
            )
            for anom_ds in anom_dataset.datasets
        ]
        normal_loader = [
            DataLoader(
                normal_ds,
                batch_size=MAX_BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                drop_last=False
            )
            for normal_ds in normal_dataset.datasets
        ]
    else:
        logger.info(f"Using single-class dataset: {anom_dataset.category}")
        anom_loader = [
            DataLoader(
                anom_dataset,
                batch_size=MAX_BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                drop_last=False,
            )
        ]
        normal_loader = [
            DataLoader(
                normal_dataset,
                batch_size=MAX_BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                drop_last=False,
            )
        ]
    
    diff_in_sh = get_backbone_feature_shape(model_type=config['backbone']['model_type'])
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    model.to(device).eval()

    backbone_kwargs = config['backbone']
    logger.info(f"Using feature space reconstruction with {backbone_kwargs['model_type']} backbone")
    
    feature_extractor = get_backbone(**backbone_kwargs)
    feature_extractor.to(device).eval()
    
    # Load the model
    if args.use_ema_model:
        checkpoint_path = os.path.join(args.save_dir, 'model_ema_latest.pth')
    elif args.use_best_model:
        checkpoint_path = os.path.join(args.save_dir, 'model_best.pth')
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Best model checkpoint not found at {checkpoint_path}. Using latest model instead.")
            checkpoint_path = os.path.join(args.save_dir, 'model_latest.pth')
    else:
        checkpoint_path = os.path.join(args.save_dir, 'model_latest.pth')
        
    model_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if 'module.' in list(model_ckpt.keys())[0]:
        model_ckpt = {k.replace('module.', ''): v for k, v in model_ckpt.items()}
    model.load_state_dict(model_ckpt, strict=True)
    logger.info(f"Loaded model from {checkpoint_path}")
    
    # Cout the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in the model: {num_params / 1e6:.2f}M")
    
    if args.eval_strategy == 'reconstruction':
        logger.info("Evaluating reconstruction performance")
        assert args.noise_step < args.eval_step, "Noise step should be less than evaluation step for reconstruction"
        metrics_dict = evaluate_recon(
            model,
            feature_extractor,
            anom_loader,
            normal_loader,
            config, 
            diff_in_sh,
            "Eval",
            args.eval_step if args.eval_step != -1 else config["evaluation"]["eval_step"],
            args.noise_step,
            device
        )
    elif args.eval_strategy == 'inversion':
        metrics_dict = evaluate_inv(
            model,
            feature_extractor,
            anom_loader,
            normal_loader,
            config, 
            diff_in_sh,
            "Eval",
            config["evaluation"]["eval_step"] if args.eval_step == -1 else args.eval_step,
            device,
            args
        )
            
    if is_multi_class:
        cats = list(metrics_dict.keys())
        avg = lambda k: float(np.mean([metrics_dict[c][k] for c in cats]))
        mean_metrics = {
            "img_auroc": avg("img_auroc"),
            "img_auroc_ci": tuple(np.mean([metrics_dict[c]["img_auroc_ci"] for c in cats], axis=0)),
            "img_ap": avg("img_ap"),
            "img_ap_ci": tuple(np.mean([metrics_dict[c]["img_ap_ci"] for c in cats], axis=0)),
            "img_f1": avg("img_f1"),
            "img_f1_ci": tuple(np.mean([metrics_dict[c]["img_f1_ci"] for c in cats], axis=0)),
            "px_auroc_abn": avg("px_auroc_abn"),
            "px_auroc_abn_ci": tuple(np.mean([metrics_dict[c]["px_auroc_abn_ci"] for c in cats], axis=0)),
            "px_aupr_abn": avg("px_aupr_abn"),
            "px_aupr_abn_ci": tuple(np.mean([metrics_dict[c]["px_aupr_abn_ci"] for c in cats], axis=0)),
            "pat_auroc": avg("pat_auroc"),
            "pat_auroc_ci": tuple(np.mean([metrics_dict[c]["pat_auroc_ci"] for c in cats], axis=0)),
            "pat_ap": avg("pat_ap"),
            "pat_ap_ci": tuple(np.mean([metrics_dict[c]["pat_ap_ci"] for c in cats], axis=0)),
            "pat_f1": avg("pat_f1"),
            "pat_f1_ci": tuple(np.mean([metrics_dict[c]["pat_f1_ci"] for c in cats], axis=0)),
            "n_patients": int(np.sum([metrics_dict[c]["n_patients"] for c in cats])),
            "n_abn_patients": int(np.sum([metrics_dict[c]["n_abn_patients"] for c in cats])),
        }
        logger.info("\n=== Multi-class Average Metrics ===")
        logger.info(f"\n{format_protocol_metrics(mean_metrics)}")
        
    
def init_denoiser(num_inference_steps, device, config, in_sh, inherit_model=None):
    config["diffusion"]["num_sampling_steps"] = str(num_inference_steps)
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=in_sh)
    
    if inherit_model is not None:
        for p, p_inherit in zip(model.parameters(), inherit_model.parameters()):
            p.data.copy_(p_inherit.data)
    model.to(device).eval()
    return model

def calculate_log_pdf(x):
    ll = -0.5 * (x ** 2 + np.log(2 * np.pi))
    ll = ll.sum(dim=(1, 2, 3))
    return ll

def calculate_log_pdf_spatial(x):
    # Calculate log pdf for each spatial dimension
    ll = -0.5 * (x ** 2 + np.log(2 * np.pi))
    ll = ll.sum(dim=1)  # Sum over the channel dimension
    return ll


def top1pct_score(anomaly_map_2d: np.ndarray) -> float:
    """Image-level anomaly score: mean of top-1% pixels in the anomaly map."""
    flat = anomaly_map_2d.flatten()
    k = max(1, int(len(flat) * 0.01))
    return float(np.partition(flat, -k)[-k:].mean())


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Best F1 on the precision-recall curve (F1-Max)."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)
    f1 = f1[:-1]
    return float(np.max(f1)) if len(f1) else 0.0


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _ci_tuple(value: float) -> tuple[float, float]:
    return (float(value), float(value))


def _format_metric(value: float, ci: tuple[float, float], scale: float) -> str:
    return f"{value * scale:.2f}% (95% CI {ci[0] * scale:.2f}-{ci[1] * scale:.2f}%)"


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_iters: int = CI_BOOTSTRAP_ITERS, seed: int = 42):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    value = float(metric_fn(y_true, y_score))
    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        return value, _ci_tuple(value)

    rng = np.random.default_rng(seed)
    values = []
    indices = np.arange(len(y_true))
    for _ in range(n_iters):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        sampled_true = y_true[sample_idx]
        if len(np.unique(sampled_true)) < 2:
            continue
        values.append(float(metric_fn(sampled_true, y_score[sample_idx])))
    if not values:
        return value, _ci_tuple(value)
    lower, upper = np.percentile(values, [2.5, 97.5])
    return value, (float(lower), float(upper))


def pixel_metrics_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> tuple[float, float]:
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


def pixel_slice_histogram_bootstrap_ci(
    labels: np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
    iters: int,
    seed: int,
    bins: int,
    progress_callback=None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    keep = np.asarray(labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        return (0.0, 0.0), (0.0, 0.0)

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

    rng = np.random.default_rng(seed)
    aurocs, auprs = [], []
    n = len(abn_idx)
    for i in range(int(iters)):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc_value, aupr_value = pixel_metrics_from_hist(pos, neg)
        aurocs.append(auroc_value)
        auprs.append(aupr_value)
        done = i + 1
        if progress_callback is not None and (done % 50 == 0 or done == int(iters)):
            progress_callback(done, int(iters))

    return (
        tuple(float(v) for v in np.percentile(aurocs, [2.5, 97.5])),
        tuple(float(v) for v in np.percentile(auprs, [2.5, 97.5])),
    )


def patient_id_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


def compute_image_metrics(gt_labels, image_scores, seed: int = 42, bootstrap_iters: int = CI_BOOTSTRAP_ITERS) -> dict:
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    image_scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    img_auroc, img_auroc_ci = bootstrap_ci(gt_labels, image_scores, safe_auroc, n_iters=bootstrap_iters, seed=seed)
    img_ap, img_ap_ci = bootstrap_ci(gt_labels, image_scores, safe_ap, n_iters=bootstrap_iters, seed=seed + 1)
    img_f1, img_f1_ci = bootstrap_ci(gt_labels, image_scores, best_f1, n_iters=bootstrap_iters, seed=seed + 2)
    return {
        "img_auroc": img_auroc,
        "img_auroc_ci": img_auroc_ci,
        "img_ap": img_ap,
        "img_ap_ci": img_ap_ci,
        "img_f1": img_f1,
        "img_f1_ci": img_f1_ci,
    }


def compute_protocol_metrics(
    gt_labels,
    gt_masks,
    anomaly_maps,
    image_scores,
    paths,
    image_metrics=None,
    seed: int = 42,
    bootstrap_iters: int = CI_BOOTSTRAP_ITERS,
    ci_pixel_max_samples: int = 200000,
    ci_hist_bins: int = CI_HIST_BINS,
    progress_callback=None,
) -> dict:
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    image_scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    if image_metrics is None:
        image_metrics = compute_image_metrics(gt_labels, image_scores, seed=seed, bootstrap_iters=bootstrap_iters)

    gt_masks_abn = gt_masks[gt_labels == 1]
    maps_abn = maps[gt_labels == 1]
    gt_px_abn = gt_masks_abn.reshape(-1).astype(bool)
    pr_px_abn = maps_abn.reshape(-1).astype(np.float64)

    patient_scores = defaultdict(list)
    patient_labels = defaultdict(list)
    for path, score, label in zip(paths, image_scores, gt_labels):
        pid = patient_id_from_path(path)
        patient_scores[pid].append(float(score))
        patient_labels[pid].append(int(label))

    pat_score = np.asarray([max(patient_scores[pid]) for pid in patient_scores], dtype=np.float64)
    pat_label = np.asarray([max(patient_labels[pid]) for pid in patient_scores], dtype=np.int32)

    px_auroc_abn = safe_auroc(gt_px_abn, pr_px_abn)
    px_aupr_abn = safe_ap(gt_px_abn, pr_px_abn)
    px_auroc_abn_ci, px_aupr_abn_ci = pixel_slice_histogram_bootstrap_ci(
        gt_labels,
        gt_masks,
        maps,
        bootstrap_iters,
        seed + 3,
        ci_hist_bins,
        progress_callback=progress_callback,
    )
    pat_auroc, pat_auroc_ci = bootstrap_ci(pat_label, pat_score, safe_auroc, n_iters=bootstrap_iters, seed=seed + 5)
    pat_ap, pat_ap_ci = bootstrap_ci(pat_label, pat_score, safe_ap, n_iters=bootstrap_iters, seed=seed + 6)
    pat_f1, pat_f1_ci = bootstrap_ci(pat_label, pat_score, best_f1, n_iters=bootstrap_iters, seed=seed + 7)

    metrics = {
        **image_metrics,
        "px_auroc_abn": px_auroc_abn,
        "px_auroc_abn_ci": px_auroc_abn_ci,
        "px_aupr_abn": px_aupr_abn,
        "px_aupr_abn_ci": px_aupr_abn_ci,
        "pat_auroc": pat_auroc,
        "pat_auroc_ci": pat_auroc_ci,
        "pat_ap": pat_ap,
        "pat_ap_ci": pat_ap_ci,
        "pat_f1": pat_f1,
        "pat_f1_ci": pat_f1_ci,
        "n_patients": int(len(pat_label)),
        "n_abn_patients": int(pat_label.sum()),
    }
    return metrics


def format_image_metrics(metrics: dict, as_percent: bool = True) -> str:
    s = 100.0 if as_percent else 1.0
    return (
        "[Slice-Img]  "
        f"AUROC={_format_metric(metrics['img_auroc'], metrics['img_auroc_ci'], s)}  "
        f"AUPR={_format_metric(metrics['img_ap'], metrics['img_ap_ci'], s)}  "
        f"F1={_format_metric(metrics['img_f1'], metrics['img_f1_ci'], s)}"
    )


def format_pixel_metrics(metrics: dict, as_percent: bool = True) -> str:
    s = 100.0 if as_percent else 1.0
    return (
        "[Slice-Px(abn)]  "
        f"AUROC={_format_metric(metrics['px_auroc_abn'], metrics['px_auroc_abn_ci'], s)}  "
        f"AUPR={_format_metric(metrics['px_aupr_abn'], metrics['px_aupr_abn_ci'], s)}"
    )


def format_patient_metrics(metrics: dict, as_percent: bool = True) -> str:
    s = 100.0 if as_percent else 1.0
    return (
        f"[Patient({metrics['n_abn_patients']}/{metrics['n_patients']}abn)]  "
        f"AUROC={_format_metric(metrics['pat_auroc'], metrics['pat_auroc_ci'], s)}  "
        f"AUPR={_format_metric(metrics['pat_ap'], metrics['pat_ap_ci'], s)}  "
        f"F1={_format_metric(metrics['pat_f1'], metrics['pat_f1_ci'], s)}"
    )


def format_protocol_metrics(metrics: dict, as_percent: bool = True) -> str:
    return "\n".join([
        format_image_metrics(metrics, as_percent=as_percent),
        format_pixel_metrics(metrics, as_percent=as_percent),
        format_patient_metrics(metrics, as_percent=as_percent),
    ])


def heatmap_save_limit(args, config) -> int:
    if getattr(args, "save_all_heatmaps", False):
        return -1
    if getattr(args, "save_heatmaps", None) is not None:
        return int(args.save_heatmaps)
    return int(config.get("evaluation", {}).get("save_heatmaps", 0))


@torch.no_grad()
def evaluate_recon(denoiser, feature_extractor, anom_loaders, normal_loaders, config, in_sh, epoch, eval_step, noise_step, device):
    denoiser.eval()
    feature_extractor.eval()
    
    eval_denoiser = init_denoiser(eval_step, device, config, in_sh, inherit_model=denoiser)
    roc_dict = {}
    for normal_loader, anom_loader in zip(normal_loaders, anom_loaders):
        category = anom_loader.dataset.category if hasattr(anom_loader.dataset, 'category') else 'unknown'
        logger.info(f"[{category}] Evaluating on {len(anom_loader)} anomalous samples and {len(normal_loader)} normal samples")
        logger.info(f"[{category}] Evaluation step: {eval_step}")
        logger.info(f"[{category}] Epoch: {epoch}")
        
        losses = []
        mses = []
        mses_sp = []
        noise_steps = torch.tensor([noise_step] * 1, device=device, dtype=torch.long)
        noise = torch.randn((1, *in_sh), device=device, dtype=torch.float32)
        for i, batch in enumerate(normal_loader):
            images = batch["samples"].to(device)
            labels = batch["clslabels"].to(device)
            
            features, features_list = feature_extractor(images)
            loss = denoiser(features, labels)
            losses.append(loss.cpu().numpy())
            
            # Perturb to x_t
            x_t = eval_denoiser.q_sample(features, noise_steps, noise=noise)
            
            # Reconstruct
            x_rec = eval_denoiser.denoise_from_intermediate(x_t, noise_steps, labels, sampler="ddim")
            
            mse = torch.mean((x_rec - features) ** 2, dim=(1, 2, 3))  # (bs, )
            min_mse_spatial = mse.view(mse.shape[0], -1).min(dim=1)[0]  # (bs, )
            max_mse_spatial = mse.view(mse.shape[0], -1).max(dim=1)[0]  # (bs, )
            mse_sp = torch.abs(min_mse_spatial - max_mse_spatial)  # (bs, )
            mses_sp.extend(mse_sp.cpu().numpy())
            mses.extend(mse.cpu().numpy())
        
        for i, batch in enumerate(anom_loader):
            images = batch["samples"].to(device)
            labels = batch["clslabels"].to(device)
            
            features, features_list = feature_extractor(images)
            loss = denoiser(features, labels)
            losses.append(loss.cpu().numpy())
            
            # Perturb to x_t
            x_t = eval_denoiser.q_sample(features, noise_steps, noise=noise)
            
            # Reconstruct
            x_rec = eval_denoiser.denoise_from_intermediate(x_t, noise_steps, labels, sampler="ddim")
            
            mse = torch.mean((x_rec - features) ** 2, dim=(1, 2, 3))
            min_mse_spatial = mse.view(mse.shape[0], -1).min(dim=1)[0]  # (bs, )
            max_mse_spatial = mse.view(mse.shape[0], -1).max(dim=1)[0]  # (bs, )
            mse_sp = torch.abs(min_mse_spatial - max_mse_spatial)  # (bs, )
            mses_sp.extend(mse_sp.cpu().numpy())
            mses.extend(mse.cpu().numpy())
            
        losses = np.array(losses)
        logger.info(f"[{category}] Loss: {losses.mean()} at epoch {epoch}")
        mses = np.array(mses)
        logger.info(f"[{category}] MSE: {mses.mean()} at epoch {epoch}")
        
        normal_mses = mses[:len(normal_loader.dataset)]
        anomaly_mses = mses[len(normal_loader.dataset):]
        normal_mses_sp = mses_sp[:len(normal_loader.dataset)]
        anomaly_mses_sp = mses_sp[len(normal_loader.dataset):]
        normal_mses = np.array(normal_mses)
        anomaly_mses = np.array(anomaly_mses)
        normal_mses_sp = np.array(normal_mses_sp)
        anomaly_mses_sp = np.array(anomaly_mses_sp)
        mses_min = np.min([normal_mses.min(), anomaly_mses.min()])
        mses_max = np.max([normal_mses.max(), anomaly_mses.max()])
        mses_sp_min = np.min([normal_mses_sp.min(), anomaly_mses_sp.min()])
        mses_sp_max = np.max([normal_mses_sp.max(), anomaly_mses_sp.max()])
        eps = 1e-8
        normal_mses = (normal_mses - mses_min) / (mses_max - mses_min + eps)
        anomaly_mses = (anomaly_mses - mses_min) / (mses_max - mses_min + eps)
        normal_mses_sp = (normal_mses_sp - mses_sp_min) / (mses_sp_max - mses_sp_min + eps)
        anomaly_mses_sp = (anomaly_mses_sp - mses_sp_min) / (mses_sp_max - mses_sp_min + eps)

        y_true = np.concatenate([np.zeros(len(normal_mses)), np.ones(len(anomaly_mses))])
        normal_scores = normal_mses + normal_mses_sp
        anomaly_scores = anomaly_mses + anomaly_mses_sp
        y_score = np.concatenate([normal_scores, anomaly_scores])
        from sklearn.metrics import roc_auc_score
    
        roc_auc = roc_auc_score(y_true, y_score)
        roc_dict[category] = roc_auc
        
        logger.info(f"[{category}] AUC: {roc_auc} at epoch {epoch}")
        
    logger.info(f"Evaluation completed for all categories.")
    return roc_dict

@torch.no_grad()
def evaluate_inv(denoiser, feature_extractor, anom_loaders, normal_loaders, config, in_sh, epoch, eval_step, device, args):
    """Evaluate using DDIM inversion.

    Evaluation protocol
    -------------------
    * Pixel anomaly map  : upsampled L2 norm of DDIM-inverted feature latents.
    * Image-level score  : mean of top-1% pixels of the anomaly map.
    * AUROC / AP         : computed on raw (non-normalised) scores.
    * F1-Max             : pixel map globally min-max normalised across the full
                           test set before finding the best threshold; image F1
                           uses raw image scores.
    * Best-model metric  : Image AP.
    """
    denoiser.eval()
    feature_extractor.eval()

    eval_denoiser = init_denoiser(eval_step, device, config, in_sh, inherit_model=denoiser)
    metrics_dict = {}

    for normal_loader, anom_loader in zip(normal_loaders, anom_loaders):
        category = anom_loader.dataset.category if hasattr(anom_loader.dataset, 'category') else 'unknown'
        logger.info(f"[{category}] Evaluating — epoch {epoch}, eval_step {eval_step}")
        logger.info(f"[{category}] Normal samples: {len(normal_loader.dataset)}  |  Anomaly samples: {len(anom_loader.dataset)}")

        org_h, org_w = None, None

        # Collect per-image anomaly maps, gt masks, and image-level labels
        all_maps: list[np.ndarray] = []       # (H, W) float32 per image
        all_img_labels: list[int] = []        # 0 = normal, 1 = anomaly
        all_gt_masks: list[np.ndarray] = []   # (H, W) uint8 binary per image
        all_paths: list[str] = []
        samples_dict: dict = {}               # for optional visualisation
        save_heatmaps = heatmap_save_limit(args, config)

        for loader, img_label in [(normal_loader, 0), (anom_loader, 1)]:
            for batch in tqdm(loader, total=len(loader),
                              desc=f"[{category}] {'normal' if img_label == 0 else 'anomaly'}"):
                images = batch["samples"].to(device)
                if org_h is None:
                    org_h, org_w = images.shape[2], images.shape[3]
                cls_labels = batch["clslabels"].to(device)

                features, _ = feature_extractor(images)
                start_t = torch.zeros(images.shape[0], device=device, dtype=torch.long)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                    latents_last = eval_denoiser.ddim_reverse_sample(
                        features, start_t, cls_labels, eta=0.0
                    )

                # L2 norm over channel dim → (B, H', W')
                latents_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
                # Upsample to original image size → (B, H, W)
                anom_maps_batch = F.interpolate(
                    latents_l2.unsqueeze(0),
                    size=(org_h, org_w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0).cpu().numpy()   # (B, H, W)

                for i in range(images.shape[0]):
                    amap = anom_maps_batch[i]
                    path = batch["filenames"][i]
                    all_maps.append(amap)
                    all_img_labels.append(img_label)
                    all_paths.append(path)

                    if "masks" in batch:
                        gt = batch["masks"][i].cpu().numpy()
                        gt_bin = (gt > 0).astype(np.uint8)
                    else:
                        gt_bin = np.zeros((org_h, org_w), dtype=np.uint8)
                    all_gt_masks.append(gt_bin)

                    if save_heatmaps != 0:
                        org_img = denormalize(images[i:i+1].cpu()).squeeze(0)
                        samples_dict[path] = {
                            "image": convert2image(org_img),
                            "anomaly_map": amap,
                            "gt_mask": gt_bin,
                        }

        # ── stack everything ───────────────────────────────────────────────
        all_maps_np = np.stack(all_maps, axis=0)    # (N, H, W)
        all_masks_np = np.stack(all_gt_masks, axis=0)  # (N, H, W) uint8
        img_labels = np.array(all_img_labels)

        # ── image-level score: mean of top-1% pixels per image ─────────────
        img_scores = np.array([top1pct_score(m) for m in all_maps_np])
        bootstrap_iters = int(getattr(args, "bootstrap_iters", CI_BOOTSTRAP_ITERS))
        ci_pixel_max_samples = int(getattr(args, "ci_pixel_max_samples", 0))
        ci_hist_bins = int(getattr(args, "ci_hist_bins", CI_HIST_BINS))
        logger.info("Computing image-level metrics and 95CI...")
        image_metrics = compute_image_metrics(
            img_labels,
            img_scores,
            seed=config["meta"].get("seed", 42),
            bootstrap_iters=bootstrap_iters,
        )
        logger.info(format_image_metrics(image_metrics))
        logger.info("Computing pixel-level Slice-Px(abn) metrics and 95CI...")
        logger.info("Computing exact full-pixel point estimates with sklearn...")

        # ── global min-max normalisation for heatmap display only ──────────
        p_min, p_max = all_maps_np.min(), all_maps_np.max()
        maps_norm = (all_maps_np - p_min) / (p_max - p_min + 1e-8)

        def pixel_progress(done, total):
            logger.info(f"Pixel histogram slice bootstrap: {done}/{total}")

        metrics_dict[category] = compute_protocol_metrics(
            img_labels,
            all_masks_np,
            all_maps_np,
            img_scores,
            all_paths,
            image_metrics=image_metrics,
            seed=config["meta"].get("seed", 42),
            bootstrap_iters=bootstrap_iters,
            ci_pixel_max_samples=ci_pixel_max_samples,
            ci_hist_bins=ci_hist_bins,
            progress_callback=pixel_progress,
        )
        logger.info(format_pixel_metrics(metrics_dict[category]))
        logger.info("Computing patient-level metrics and 95CI...")
        logger.info(f"\n{format_protocol_metrics(metrics_dict[category])}")

        torch.cuda.empty_cache()

        if save_heatmaps != 0:
            _save_visualizations(category, samples_dict, maps_norm, args.save_dir, epoch, save_heatmaps)

    logger.info("Evaluation completed for all categories.")
    return metrics_dict


def _save_visualizations(category: str, samples_dict: dict, maps_norm: np.ndarray, save_dir: str, epoch, save_heatmaps: int):
    map_min, map_max = maps_norm.min(), maps_norm.max()
    heatmap_dir = os.path.join(save_dir, "heatmaps", f"epoch_{epoch}", category)
    os.makedirs(heatmap_dir, exist_ok=True)
    max_items = len(samples_dict) if save_heatmaps == -1 else max(0, save_heatmaps)
    for idx, (path, sample) in enumerate(tqdm(samples_dict.items(), desc="Saving visualisations")):
        if idx >= max_items:
            break
        amap_norm = maps_norm[idx]
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(sample["image"]); ax[0].set_title("Image"); ax[0].axis('off')
        ax[1].imshow(amap_norm, cmap='viridis', vmin=map_min, vmax=map_max)
        ax[1].set_title("Anomaly Map"); ax[1].axis('off')
        ax[2].imshow(sample["gt_mask"], cmap='gray'); ax[2].set_title("GT Mask"); ax[2].axis('off')
        plt.tight_layout()
        save_path = os.path.join(heatmap_dir, path.replace("/", "_").replace(".png", "") + ".png")
        plt.savefig(save_path); plt.close(fig)
    logger.info(f"Saved visualisations to {heatmap_dir}")

import torch.distributed as dist
@torch.no_grad()
def concat_all_gather(array, world_size):
    world_size = dist.get_world_size()
    gather_list = [None] * world_size
    dist.all_gather_object(gather_list, array)  # Gather the arrays from all processes
    return np.concatenate(gather_list, axis=0)

@torch.no_grad()
def evaluate_dist(denoiser, feature_extractor, anom_loader, normal_loader, config, in_sh, epoch, eval_step, device, world_size, rank):
    denoiser.eval()
    feature_extractor.eval()
    category = anom_loader.dataset.category
    
    eval_denoiser = init_denoiser(eval_step, device, config, in_sh, inherit_model=denoiser)
    
    logger.info(f"[{category}] Evaluating on {len(anom_loader.dataset)} anomalous samples and {len(normal_loader.dataset)} normal samples")
    logger.info(f"[{category}] Evaluation step: {eval_step}")
    logger.info(f"[{category}] Epoch: {epoch}")
    
    start_t = torch.tensor([0] * 8, device=device, dtype=torch.long)
    normal_diffs = []
    normal_nlls = []
    normal_maps = []
    normal_gt_masks = []
    normal_paths = []
    losses = []
    for i, batch in enumerate(normal_loader):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        normal_gt_masks.append(batch["masks"])
        normal_paths.extend(list(batch["filenames"]))
        
        features, _ = feature_extractor(images)
        loss = denoiser(features, labels)
        losses.append(loss.cpu().numpy())
        
        latents_last = eval_denoiser.ddim_reverse_sample(
            features, start_t, labels, eta=0.0
        )
        latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
        min_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).min(dim=1)[0]  # (bs, )
        max_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).max(dim=1)[0]  # (bs, )
        diffs = min_diffs_spatial - max_diffs_spatial  # (bs, )
        nll = calculate_log_pdf(latents_last) * -1
        
        normal_map = F.interpolate(latents_last_l2.unsqueeze(0), size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False).squeeze(0)
        normal_maps.append(normal_map.cpu())
    
        normal_nlls.append(nll.cpu())
        normal_diffs.append(diffs.cpu())
    dist.barrier()  # Ensure all processes have completed the normal data processing
        
    anomaly_diffs = []
    anomaly_nlls = []
    anomaly_maps = []
    anomaly_gt_masks = []
    anomaly_paths = []
    for i, batch in enumerate(anom_loader):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        anomaly_gt_masks.append(batch["masks"])
        anomaly_paths.extend(list(batch["filenames"]))
        
        features, _ = feature_extractor(images)
        loss = denoiser(features, labels)
        losses.append(loss.cpu().numpy())
        latents_last = eval_denoiser.ddim_reverse_sample(
            features, start_t, labels, eta=0.0
        )
        latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
        min_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).min(dim=1)[0]
        max_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).max(dim=1)[0]
        diffs = min_diffs_spatial - max_diffs_spatial
        nll = calculate_log_pdf(latents_last) * -1
        
        anomaly_map = F.interpolate(latents_last_l2.unsqueeze(0), size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False).squeeze(0)
        anomaly_maps.append(anomaly_map.cpu())
        anomaly_nlls.append(nll.cpu())
        anomaly_diffs.append(diffs.cpu())
        del latents_last, latents_last_l2, diffs, nll
        torch.cuda.empty_cache()                     
    dist.barrier()  # Ensure all processes have completed the anomaly data processing
    
    losses = np.array(losses)
    logger.info(f"[{category}] Loss: {losses.mean()} at epoch {epoch}")

    normal_maps = torch.cat(normal_maps, dim=0)
    anomaly_maps = torch.cat(anomaly_maps, dim=0)
    normal_gt_masks = torch.cat(normal_gt_masks, dim=0).squeeze(1)
    anomaly_gt_masks = torch.cat(anomaly_gt_masks, dim=0).squeeze(1)

    # Gather results from all processes
    normal_maps = concat_all_gather(normal_maps, world_size)
    anomaly_maps = concat_all_gather(anomaly_maps, world_size)
    normal_gt_masks = concat_all_gather(normal_gt_masks, world_size)
    anomaly_gt_masks = concat_all_gather(anomaly_gt_masks, world_size)
    gathered_normal_paths = [None] * world_size
    gathered_anomaly_paths = [None] * world_size
    dist.all_gather_object(gathered_normal_paths, normal_paths)
    dist.all_gather_object(gathered_anomaly_paths, anomaly_paths)
    
    if rank != 0:
        return None

    logger.info(f"[{category}] Normal samples: {len(normal_maps)}  |  Anomaly samples: {len(anomaly_maps)}")

    # Stack maps and masks
    all_maps_np = np.concatenate([normal_maps, anomaly_maps], axis=0)   # (N, H, W)
    all_masks_np = np.concatenate([
        (normal_gt_masks > 0).astype(np.uint8),
        (anomaly_gt_masks > 0).astype(np.uint8),
    ], axis=0)
    img_labels = np.concatenate([
        np.zeros(len(normal_maps), dtype=np.int32),
        np.ones(len(anomaly_maps), dtype=np.int32),
    ])

    # Image-level score: top-1% pixel mean
    img_scores = np.array([top1pct_score(m) for m in all_maps_np])
    bootstrap_iters = int(config.get("evaluation", {}).get("bootstrap_iters", CI_BOOTSTRAP_ITERS))
    ci_pixel_max_samples = int(config.get("evaluation", {}).get("ci_pixel_max_samples", 0))
    ci_hist_bins = int(config.get("evaluation", {}).get("ci_hist_bins", CI_HIST_BINS))
    logger.info("Computing image-level metrics and 95CI...")
    image_metrics = compute_image_metrics(
        img_labels,
        img_scores,
        seed=config["meta"].get("seed", 42),
        bootstrap_iters=bootstrap_iters,
    )
    logger.info(format_image_metrics(image_metrics))
    logger.info("Computing pixel-level Slice-Px(abn) metrics and 95CI...")
    logger.info("Computing exact full-pixel point estimates with sklearn...")

    all_paths = [
        path
        for group in gathered_normal_paths + gathered_anomaly_paths
        for path in group
    ]
    def pixel_progress(done, total):
        logger.info(f"Pixel histogram slice bootstrap: {done}/{total}")

    metrics = compute_protocol_metrics(
        img_labels,
        all_masks_np,
        all_maps_np,
        img_scores,
        all_paths,
        image_metrics=image_metrics,
        seed=config["meta"].get("seed", 42),
        bootstrap_iters=bootstrap_iters,
        ci_pixel_max_samples=ci_pixel_max_samples,
        ci_hist_bins=ci_hist_bins,
        progress_callback=pixel_progress,
    )
    logger.info(format_pixel_metrics(metrics))
    logger.info("Computing patient-level metrics and 95CI...")
    logger.info(f"\n{format_protocol_metrics(metrics)}")

    return {category: metrics}

if __name__ == "__main__":
    args = parse_args()
    def load_config(config_path):
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return config
    
    config = load_config(os.path.join(args.save_dir, "config.yaml"))
    main(config, args)
