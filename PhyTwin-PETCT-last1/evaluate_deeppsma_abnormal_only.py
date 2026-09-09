#!/usr/bin/env python3
"""Slice-level evaluation for DeepPSMA.

This evaluator is intentionally separate from train.py because the rebuilt
DeepPSMA set is slice-level: test/normal and test/abnormal contain flat
pet/ct/label modality folders rather than patient folders. Patient-level
metrics are not reported.

cd /data/cyf/codes/A-PET-CT/PhyTwin-PETCT-last1

python evaluate_deeppsma_abnormal_only.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 \
  --experiment_name PhyTwin_PETCT_PSMA \
  --score_mode lesion_z \
  --adaptive_physio \
  --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 \
  --gpu 0 > deeppsma1_psma.log 2>&1 &


python evaluate_deeppsma_abnormal_only.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 \
  --experiment_name PhyTwin_PETCT_FDG \
  --score_mode lesion_z \
  --adaptive_physio \
  --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 \
  --gpu 1 > deeppsma1_fdg.log 2>&1 &

95ci
nohup python evaluate_deeppsma_abnormal_only.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 \
  --experiment_name PhyTwin_PETCT_PSMA \
  --ckpt_dir ./ckpts \
  --score_mode lesion_z \
  --adaptive_physio \
  --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 \
  --batch_size 1 \
  --image_size 256 \
  --gpu 7 \
  --bootstrap_iters 500 \
  --save_dir ./saved_results_deeppsma_npz_psma \
  > deeppsma_npz_psma.log 2>&1 &

nohup python evaluate_deeppsma_abnormal_only.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 \
  --experiment_name PhyTwin_PETCT_FDG \
  --ckpt_dir ./ckpts \
  --score_mode lesion_z \
  --adaptive_physio \
  --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 \
  --batch_size 1 \
  --image_size 256 \
  --gpu 7 \
  --bootstrap_iters 500 \
  --save_dir ./saved_results_deeppsma_npz_fdg \
  > deeppsma_npz_fdg.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label as cc_label
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from phytwin_petct.data import IMAGENET_MEAN, IMAGENET_STD
from phytwin_petct.dynamic_physio import DynamicPhysioSuppressor
from phytwin_petct.eval_protocol import compute_metrics
from phytwin_petct.models.normal_twin import NormalTwinUNet, residual_map
from phytwin_petct.physio import pet_tensor_to_gray
from phytwin_petct.scoring import fuse_maps, normal_calibrated_score, predict_mask, topk_score
from phytwin_petct.utils import get_logger, limited, setup_seed
from train import (
    apply_adaptive_physio,
    build_adaptive_physio,
    compute_lesion_score,
    image_refinement_config,
    lesion_component_score,
    load_full_checkpoint,
    memory_z_map,
    parse_float_list,
    positive_zscore_map,
    safe_name,
)


DEFAULT_DEEPPSMA_ROOT = "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma"


class DeepPSMASliceDataset(torch.utils.data.Dataset):
    """Collect flat test/normal and test/abnormal PET/CT slices."""

    def __init__(self, data_root: str, image_size: int = 256):
        self.data_root = os.path.abspath(os.path.expanduser(data_root))
        self.image_size = int(image_size)
        self.img_transform = T.Compose([
            T.Resize((self.image_size, self.image_size), InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.mask_transform = T.Compose([
            T.Resize((self.image_size, self.image_size), InterpolationMode.NEAREST),
            T.ToTensor(),
        ])
        self.samples = []
        self._collect()
        if not self.samples:
            raise RuntimeError(
                f"No DeepPSMA test samples found under {self.data_root}. "
                "Expected flat test/{normal,abnormal}/{pet,ct,label}/*.png"
            )

    @staticmethod
    def _rgb(gray: Image.Image) -> Image.Image:
        return Image.merge("RGB", [gray, gray, gray])

    def _collect_flat_group(self, group: str, label_value: int, has_mask: bool) -> None:
        group_root = os.path.join(self.data_root, "test", group)
        pet_dir = os.path.join(group_root, "pet")
        ct_dir = os.path.join(group_root, "ct")
        label_dir = os.path.join(group_root, "label")
        if not (os.path.isdir(pet_dir) and os.path.isdir(ct_dir)):
            return
        if has_mask and not os.path.isdir(label_dir):
            return
        for name in sorted(os.listdir(pet_dir)):
            if not name.lower().endswith(".png"):
                continue
            pet_path = os.path.join(pet_dir, name)
            ct_path = os.path.join(ct_dir, name)
            mask_path = os.path.join(label_dir, name) if has_mask else None
            if os.path.isfile(ct_path) and (not has_mask or os.path.isfile(mask_path)):
                self.samples.append((pet_path, ct_path, mask_path, label_value, group))

    def _collect_legacy_group(self, group: str, label_value: int, has_mask: bool) -> None:
        group_root = os.path.join(self.data_root, "test", group)
        if not os.path.isdir(group_root):
            return
        for patient in sorted(os.listdir(group_root)):
            patient_dir = os.path.join(group_root, patient)
            pet_dir = os.path.join(patient_dir, "pet")
            ct_dir = os.path.join(patient_dir, "ct")
            label_dir = os.path.join(patient_dir, "label")
            if not (os.path.isdir(pet_dir) and os.path.isdir(ct_dir)):
                continue
            if has_mask and not os.path.isdir(label_dir):
                continue
            for name in sorted(os.listdir(pet_dir)):
                if not name.lower().endswith(".png"):
                    continue
                pet_path = os.path.join(pet_dir, name)
                ct_path = os.path.join(ct_dir, name)
                mask_path = os.path.join(label_dir, name) if has_mask else None
                if os.path.isfile(ct_path) and (not has_mask or os.path.isfile(mask_path)):
                    self.samples.append((pet_path, ct_path, mask_path, label_value, group))

    def _collect(self) -> None:
        before = len(self.samples)
        self._collect_flat_group("normal", label_value=0, has_mask=False)
        self._collect_flat_group("abnormal", label_value=1, has_mask=True)
        if len(self.samples) == before:
            self._collect_legacy_group("normal", label_value=0, has_mask=False)
            self._collect_legacy_group("abnormal", label_value=1, has_mask=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        pet_path, ct_path, mask_path, label_value, _group = self.samples[index]
        pet_img = Image.open(pet_path).convert("L")
        ct_img = Image.open(ct_path).convert("L")
        if mask_path is None:
            mask = Image.new("L", pet_img.size, 0)
        else:
            mask = Image.open(mask_path).convert("L")
        return {
            "pet": self.img_transform(self._rgb(pet_img)),
            "ct": self.img_transform(self._rgb(ct_img)),
            "mask": (self.mask_transform(mask) > 0.5).float(),
            "path": pet_path,
            "label": int(label_value),
        }


def slice_group_path(path: str, label_value: int) -> str:
    """Make compute_metrics bootstrap each flat slice independently."""
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    return os.path.join("__deeppsma_slice_groups__", f"label{label_value}_{stem}", "pet", os.path.basename(str(path)))


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    recall = tp / (gt_sum + 1e-8)
    precision = tp / (pred_sum + 1e-8)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "pixel_recall": float(recall),
        "pixel_precision": float(precision),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "gt_pixels": float(gt_sum),
        "pred_pixels": float(pred_sum),
    }


def lesion_recall(pred: np.ndarray, gt: np.ndarray, min_overlap_pixels: int = 1) -> tuple[int, int]:
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)
    cc, num = cc_label(gt)
    if num == 0:
        return 0, 0
    hit = 0
    for lesion_idx in range(1, num + 1):
        region = cc == lesion_idx
        if int((pred & region).sum()) >= int(min_overlap_pixels):
            hit += 1
    return hit, int(num)


def describe(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def image_score_from_mode(final, pet_gray, ct_gray, prior, memory, residual, calibration, args):
    prior_arr = None if prior is None else prior.prior
    if args.score_mode == "component":
        return lesion_component_score(
            final,
            physio_prior=prior_arr,
            threshold_quantile=args.component_quantile,
            min_area=args.component_min_area,
            prior_penalty=args.prior_penalty,
        )
    if args.score_mode == "normal_z":
        return normal_calibrated_score(
            final,
            calibration["normal_image_mean"],
            calibration["normal_image_std"],
            physio_prior=prior_arr,
            prior_penalty=args.prior_penalty,
        )
    if args.score_mode == "lesion_z":
        raw_score = compute_lesion_score(
            final,
            pet_gray,
            ct_gray,
            prior,
            args,
            normal_hotspot_memory=calibration.get("normal_hotspot_memory") if args.use_hotspot_memory else None,
        )
        normal_mean = calibration.get(
            "normal_lesion_ms_mean",
            calibration.get("normal_lesion_mean", calibration["normal_image_mean"]),
        )
        normal_std = calibration.get(
            "normal_lesion_ms_std",
            calibration.get("normal_lesion_std", calibration["normal_image_std"]),
        )
        return float((raw_score - normal_mean) / (normal_std + 1e-8))
    return topk_score(final, fraction=0.01)


@torch.no_grad()
def evaluate_slice_level(model, memory, prior, dynamic_suppressor, calibration, loader, device, args, logger=None):
    model.eval()
    if memory is not None:
        memory = memory.to(device)
    adaptive_suppressor = build_adaptive_physio(args)
    labels, masks, metric_paths = [], [], []
    final_maps, image_scores = [], []
    records = []

    for _idx, batch in limited(loader, args.max_test_batches):
        pet = batch["pet"].to(device)
        ct = batch["ct"].to(device)
        pred, logvar = model(ct)
        residual = residual_map(pet, pred, logvar if args.uncertainty else None)
        residual_np = residual[0].detach().cpu().numpy().astype(np.float32)
        residual_z = positive_zscore_map(residual_np, calibration["residual_mean"], calibration["residual_std"])
        memory_z = memory_z_map(memory, residual, calibration, args)
        final = fuse_maps(
            residual_z,
            memory_z,
            residual_weight=args.residual_weight,
            memory_weight=args.memory_weight,
            physio_prior=prior,
            sigma=args.gaussian_sigma,
        )
        pet_gray = pet_tensor_to_gray(batch["pet"])[0].numpy().astype(np.float32)
        ct_gray = pet_tensor_to_gray(batch["ct"])[0].numpy().astype(np.float32)
        if dynamic_suppressor is not None:
            final, _dynamic_mask = dynamic_suppressor.suppress(
                final,
                pet_gray,
                static_prior=None if prior is None else prior.prior,
            )
        final, _adaptive_mask = apply_adaptive_physio(final, pet_gray, ct_gray, prior, adaptive_suppressor)
        image_score = image_score_from_mode(final, pet_gray, ct_gray, prior, memory, residual, calibration, args)
        pred_mask = predict_mask(
            final,
            threshold_quantile=args.pred_mask_quantile,
            min_area=args.pred_mask_min_area,
        )
        label_value = int(batch["label"].item())
        path = batch["path"][0] if isinstance(batch["path"], (list, tuple)) else str(batch["path"])
        labels.append(label_value)
        masks.append(batch["mask"].numpy())
        metric_paths.append(slice_group_path(path, label_value))
        final_maps.append(final)
        image_scores.append(float(image_score))
        records.append({
            "path": path,
            "label": label_value,
            "score": float(image_score),
            "pred_pixels": int(np.asarray(pred_mask).astype(bool).sum()),
            "gt_pixels": int(batch["mask"][0, 0].numpy().astype(bool).sum()),
        })
    masks_arr = np.squeeze(np.concatenate(masks, axis=0), axis=1)
    final_maps_arr = np.stack(final_maps, axis=0)
    slice_metrics, _pat_metrics_unused = compute_metrics(
        labels,
        masks_arr,
        final_maps_arr,
        image_scores,
        metric_paths,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
        hist_bins=args.hist_bins,
        progress_callback=logger.info if logger is not None else None,
    )
    labels_arr = np.asarray(labels, dtype=np.int32)

    return {
        "dataset": "DeepPSMA slice-level",
        "n_slices": int(len(records)),
        "n_normal_slices": int((labels_arr == 0).sum()),
        "n_abnormal_slices": int((labels_arr == 1).sum()),
        "score_mode": args.score_mode,
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.bootstrap_seed),
        "slice_metrics": slice_metrics,
        "records": records,
    }


def format_metrics_only(metrics: dict) -> str:
    def metric_line(metric_dict: dict, name: str, label: str) -> str:
        value = 100.0 * float(metric_dict[name])
        ci = metric_dict.get(f"{name}_ci")
        if ci is None:
            return f"{label}={value:.2f}%"
        return f"{label}={value:.2f}% (95% CI {100.0 * ci[0]:.2f}-{100.0 * ci[1]:.2f}%)"

    m = metrics["slice_metrics"]
    lines = [
        f"[DeepPSMA] slices={metrics['n_slices']} normal={metrics['n_normal_slices']} abnormal={metrics['n_abnormal_slices']} score_mode={metrics['score_mode']}",
        "[Slice-Img]  " + "  ".join([
            metric_line(m, "img_auroc", "AUROC"),
            metric_line(m, "img_ap", "AUPR"),
            metric_line(m, "img_f1", "F1"),
        ]),
        "[Slice-Px(abn)]  " + "  ".join([
            metric_line(m, "px_auroc_abn", "AUROC"),
            metric_line(m, "px_aupr_abn", "AUPR"),
        ]),
    ]
    return "\n".join(lines)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepPSMA slice-level evaluator")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--no_uncertainty", action="store_true", default=False)
    parser.add_argument("--score_mode", type=str, default="lesion_z", choices=["top1pct", "component", "normal_z", "lesion_z"])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./saved_results_deeppsma_abnormal_only")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts")
    parser.add_argument("--experiment_name", type=str, default="PhyTwin_PETCT_PSMA",
                        help="Checkpoint name to load. Usually reuse the PSMA normal-trained checkpoint.")
    parser.add_argument("--seed", type=int, default=1203)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--bootstrap_iters", type=int, default=500,
                        help="Number of bootstrap resamples for 95% confidence intervals. Set 0 to disable CI.")
    parser.add_argument("--bootstrap_seed", type=int, default=None,
                        help="Random seed for bootstrap confidence intervals. Defaults to --seed.")
    parser.add_argument("--hist_bins", type=int, default=16384,
                        help="Number of score bins for accelerated pixel-level bootstrap CI.")

    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--memory_max_patches", type=int, default=30000)
    parser.add_argument("--memory_k", type=int, default=3)
    parser.add_argument("--residual_weight", type=float, default=1.0)
    parser.add_argument("--memory_weight", type=float, default=0.0)
    parser.add_argument("--gaussian_sigma", type=float, default=3.0)
    parser.add_argument("--no_physio", action="store_true", default=False)
    parser.add_argument("--adaptive_physio", action="store_true", default=False)
    parser.add_argument("--adaptive_alpha", type=float, default=0.50)
    parser.add_argument("--adaptive_pet_quantile", type=float, default=0.985)
    parser.add_argument("--adaptive_anomaly_quantile", type=float, default=0.990)
    parser.add_argument("--adaptive_min_area", type=int, default=24)
    parser.add_argument("--adaptive_lesion_protect", type=float, default=0.70)
    parser.add_argument("--dynamic_physio", action="store_true", default=False)
    parser.add_argument("--dynamic_alpha", type=float, default=0.45)
    parser.add_argument("--dynamic_pet_quantile", type=float, default=0.985)
    parser.add_argument("--dynamic_min_area", type=int, default=24)
    parser.add_argument("--component_quantile", type=float, default=0.995)
    parser.add_argument("--component_min_area", type=int, default=4)
    parser.add_argument("--pred_mask_quantile", type=float, default=0.985)
    parser.add_argument("--pred_mask_min_area", type=int, default=8)
    parser.add_argument("--lesion_hit_min_pixels", type=int, default=1)
    parser.add_argument("--lesion_quantile", type=float, default=0.992)
    parser.add_argument("--lesion_quantiles", type=str, default="0.985,0.992,0.997")
    parser.add_argument("--lesion_min_area", type=int, default=3)
    parser.add_argument("--lesion_max_components", type=int, default=6)
    parser.add_argument("--prior_penalty", type=float, default=0.65)
    parser.add_argument("--edge_penalty", type=float, default=0.55)
    parser.add_argument("--large_area_penalty", type=float, default=0.70)
    parser.add_argument("--fov_penalty", type=float, default=0.65)
    parser.add_argument("--fov_band_fraction", type=float, default=0.06)
    parser.add_argument("--organ_prior_threshold", type=float, default=0.45)
    parser.add_argument("--organ_area_fraction", type=float, default=0.0035)
    parser.add_argument("--use_hotspot_memory", action="store_true", default=False)
    parser.add_argument("--hotspot_memory_max", type=int, default=6000)
    parser.add_argument("--hotspot_penalty", type=float, default=0.35)
    parser.add_argument("--hotspot_sigma", type=float, default=0.35)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.uncertainty = not args.no_uncertainty
    args.lesion_quantiles_list = parse_float_list(args.lesion_quantiles)
    if args.bootstrap_seed is None:
        args.bootstrap_seed = args.seed
    args.use_dynamic_physio = bool(args.dynamic_physio)
    args.method_name = safe_name(f"{args.experiment_name}_DeepPSMA_slice_level")
    return args


def write_outputs(metrics: dict, out_dir: str) -> tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "deeppsma_slice_level_metrics.json")
    txt_path = os.path.join(out_dir, "deeppsma_slice_level_metrics.txt")
    csv_path = os.path.join(out_dir, "deeppsma_slice_level_scores.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(jsonable(metrics), f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_metrics_only(metrics) + "\n")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path", "label", "score", "gt_pixels", "pred_pixels",
            ],
        )
        writer.writeheader()
        for row in metrics["records"]:
            writer.writerow(row)
    return json_path, txt_path, csv_path


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(args.save_dir, args.method_name)
    logger = get_logger(args.method_name, out_dir)
    logger.info(f"DeepPSMA slice-level evaluation | data_root={args.data_root}")
    logger.info(f"Device: {device}")

    dataset = DeepPSMASliceDataset(args.data_root, image_size=args.image_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    n_normal = sum(1 for sample in dataset.samples if sample[3] == 0)
    n_abnormal = sum(1 for sample in dataset.samples if sample[3] == 1)
    logger.info(f"Loaded slices={len(dataset)} normal={n_normal} abnormal={n_abnormal}")

    model = NormalTwinUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        uncertainty=args.uncertainty,
    ).to(device)
    ckpt_full = os.path.join(args.ckpt_dir, safe_name(args.experiment_name), "BEST_PHYTWIN.pth")
    logger.info(f"Loading checkpoint from {ckpt_full}")
    model, memory, prior, calibration = load_full_checkpoint(ckpt_full, model)
    model = model.to(device)
    if args.no_physio:
        prior = None
        logger.info("Static PHYSIO prior disabled by --no_physio.")
    if args.score_mode == "lesion_z":
        has_lesion_calibration = (
            "normal_lesion_ms_mean" in calibration
            or "normal_lesion_mean" in calibration
            or "normal_image_mean" in calibration
        )
        if not has_lesion_calibration:
            raise RuntimeError(
                "Checkpoint lacks lesion/image normal calibration statistics. "
                "Run train.py once on a dataset with train/normal, then rerun this evaluator."
            )
        if calibration.get("image_refinement_config") != image_refinement_config(args):
            logger.info(
                "Checkpoint lesion_z refinement config differs from current args; "
                "DeepPSMA slice-level evaluation cannot recalibrate without train/normal, "
                "so it will reuse the checkpoint's available normal calibration statistics."
            )

    dynamic_suppressor = None
    if args.use_dynamic_physio:
        dynamic_suppressor = DynamicPhysioSuppressor(
            alpha=args.dynamic_alpha,
            pet_quantile=args.dynamic_pet_quantile,
            min_area=args.dynamic_min_area,
    )
    metrics = evaluate_slice_level(model, memory, prior, dynamic_suppressor, calibration, loader, device, args, logger=logger)
    json_path, txt_path, csv_path = write_outputs(metrics, out_dir)
    metrics_only = format_metrics_only(metrics)
    logger.info("[Final metrics]\n" + metrics_only)
    logger.info(f"Saved: {json_path} | {txt_path} | {csv_path}")


if __name__ == "__main__":
    main()
