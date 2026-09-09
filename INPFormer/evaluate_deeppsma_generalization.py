#!/usr/bin/env python3
"""
DeepPSMA1 direct generalization test for INPFormer.

Data root:
  /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1

Run with existing PSMA weights:
  python evaluate_deeppsma_generalization.py --source_tracer PSMA --modality ct,pet --gpu 0

Run with existing FDG weights:
  python evaluate_deeppsma_generalization.py --source_tracer FDG --modality ct,pet --gpu 5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace


def _set_cuda_visible_devices_from_argv():
    for idx, arg in enumerate(os.sys.argv):
        if arg == "--gpu" and idx + 1 < len(os.sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = os.sys.argv[idx + 1]
            return
        if arg.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]
            return


_set_cuda_visible_devices_from_argv()

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from train import build_model
from utils import build_anomaly_map, setup_seed


DEFAULT_DEEPPSMA_ROOT = "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
REPO_DIR = Path(__file__).resolve().parent


def parse_modalities(value):
    modalities = [m.strip().lower() for m in str(value).replace("+", ",").split(",") if m.strip()]
    invalid = sorted(set(modalities) - {"ct", "pet"})
    if invalid:
        raise ValueError(f"Invalid modalities: {invalid}. Use ct, pet, or ct,pet.")
    if not modalities:
        raise ValueError("At least one modality is required.")
    return modalities


def build_transforms(input_size, crop_size):
    image_steps = [transforms.Resize((input_size, input_size))]
    mask_steps = [transforms.Resize((input_size, input_size))]
    if crop_size > 0 and crop_size != input_size:
        image_steps.append(transforms.CenterCrop(crop_size))
        mask_steps.append(transforms.CenterCrop(crop_size))
    image_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_steps.append(transforms.ToTensor())
    return transforms.Compose(image_steps), transforms.Compose(mask_steps)


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_aupr(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _max_f1(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    f1 = f1[:-1]
    return float(f1.max()) if len(f1) else 0.0


def _bootstrap_ci(y_true, y_score, metric_fn, iters, seed):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    value = metric_fn(y_true, y_score)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(int(iters)):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_fn(y_true[idx], y_score[idx]))
    if not values:
        return value, (value, value)
    low, high = np.percentile(values, [2.5, 97.5])
    return value, (float(low), float(high))


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
    integrate = getattr(np, "trapezoid", np.trapz)
    auroc = float(integrate(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)
    aupr = float(np.sum(precision * (tp / total_pos)))
    return auroc, aupr


def _pixel_slice_bootstrap(labels, masks, maps, iters, seed, bins, exact_auroc, exact_aupr):
    keep = np.asarray(labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        return (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0))

    score_min = float(np.min(maps[keep]))
    score_max = float(np.max(maps[keep]))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    bins = int(bins)
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
    for i in range(1, int(iters) + 1):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if i % 50 == 0 or i == int(iters):
            print(f"Pixel histogram slice bootstrap: {i}/{int(iters)}", flush=True)

    auroc_ci = tuple(float(v) for v in np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = tuple(float(v) for v in np.percentile(auprs, [2.5, 97.5]))
    return (float(exact_auroc), auroc_ci), (float(exact_aupr), aupr_ci)


def _fmt(name, value_ci):
    value, ci = value_ci
    return f"{name}={value * 100:.2f}% (95% CI {ci[0] * 100:.2f}-{ci[1] * 100:.2f}%)"


def top_percent_mean(score_maps, percent=1.0):
    score_maps = np.asarray(score_maps, dtype=np.float32)
    if score_maps.ndim == 2:
        score_maps = score_maps[None]
    flat = score_maps.reshape(score_maps.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * percent / 100.0)))
    return np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)


def compute_slice_only_metrics(labels, masks, maps, scores, bootstrap_iters=500, hist_bins=16384, seed=2):
    labels = np.asarray(labels, dtype=np.int32)
    masks = (np.asarray(masks) > 0).astype(np.uint8)
    maps = np.asarray(maps, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float64)

    print(
        f"Cache loaded: labels={labels.shape}, masks={masks.shape}, maps={maps.shape}, "
        f"bootstrap_iters={int(bootstrap_iters)}",
        flush=True,
    )
    print("Computing image-level metrics and 95CI...", flush=True)
    img_auroc = _bootstrap_ci(labels, scores, _safe_auroc, bootstrap_iters, seed)
    img_aupr = _bootstrap_ci(labels, scores, _safe_aupr, bootstrap_iters, seed + 1)
    img_f1 = _bootstrap_ci(labels, scores, _max_f1, bootstrap_iters, seed + 2)
    image_line = "[Slice-Img]  " + "  ".join(
        [_fmt("AUROC", img_auroc), _fmt("AUPR", img_aupr), _fmt("F1", img_f1)]
    )
    print(image_line, flush=True)

    print("Computing pixel-level Slice-Px(abn) metrics and 95CI...", flush=True)
    print("Computing exact full-pixel point estimates with sklearn...", flush=True)
    keep = labels == 1
    px_true = masks[keep].reshape(-1).astype(np.int32)
    px_score = maps[keep].reshape(-1).astype(np.float64)
    px_auroc, px_aupr = _pixel_slice_bootstrap(
        labels,
        masks,
        maps,
        bootstrap_iters,
        seed + 10,
        hist_bins,
        _safe_auroc(px_true, px_score),
        _safe_aupr(px_true, px_score),
    )
    pixel_line = "[Slice-Px(abn)]  " + "  ".join([_fmt("AUROC", px_auroc), _fmt("AUPR", px_aupr)])
    print(pixel_line, flush=True)
    return "\n".join([image_line, pixel_line])


class DeepPSMAFlatDataset(Dataset):
    def __init__(self, data_root, modalities, transform, gt_transform, channel_fill="mean_modalities"):
        self.root = Path(data_root).expanduser().resolve()
        self.modalities = [m.lower() for m in modalities]
        self.transform = transform
        self.gt_transform = gt_transform
        self.channel_fill = channel_fill
        self.samples = []
        self.skipped = 0
        self._collect()
        if not self.samples:
            raise RuntimeError(
                f"No DeepPSMA samples found under {self.root}. "
                "Expected test/{normal,abnormal}/{pet,ct,label}/*.png"
            )

    @staticmethod
    def _is_image(path):
        return path.is_file() and path.suffix.lower() in IMG_EXTS and path.stat().st_size > 0

    def _collect_group(self, group, label):
        group_root = self.root / "test" / group
        primary_dir = group_root / self.modalities[0]
        label_dir = group_root / "label"
        if not primary_dir.is_dir():
            return
        for path in sorted(p for p in primary_dir.iterdir() if self._is_image(p)):
            mod_paths = [group_root / mod / path.name for mod in self.modalities]
            if not all(self._is_image(p) for p in mod_paths):
                self.skipped += 1
                continue
            mask_path = None
            if label:
                candidate = label_dir / path.name
                if not self._is_image(candidate):
                    self.skipped += 1
                    continue
                mask_path = candidate
            self.samples.append((mod_paths, mask_path, int(label), str(path)))

    def _collect(self):
        self._collect_group("normal", 0)
        self._collect_group("abnormal", 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mod_paths, mask_path, label, sample_id = self.samples[idx]
        modal_images = []
        for mod_path in mod_paths:
            img = Image.open(mod_path).convert("L")
            modal_images.append(np.asarray(img, dtype=np.float32) / 255.0)

        if len(modal_images) == 1:
            modal_images = [modal_images[0], modal_images[0], modal_images[0]]
        while len(modal_images) < 3:
            modal_images.append(self._make_fill_channel(modal_images))

        fused = np.stack(modal_images[:3], axis=-1)
        img_tensor = self.transform(Image.fromarray((fused * 255).astype(np.uint8)))
        if mask_path is not None:
            mask = self.gt_transform(Image.open(mask_path).convert("L"))
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros((1, img_tensor.shape[-2], img_tensor.shape[-1]), dtype=torch.float32)
        return img_tensor, mask, torch.tensor(label, dtype=torch.long), sample_id

    def _make_fill_channel(self, modal_images):
        if self.channel_fill == "zero":
            return np.zeros_like(modal_images[0], dtype=np.float32)
        if self.channel_fill == "mean_modalities":
            return np.mean(np.stack(modal_images, axis=0), axis=0).astype(np.float32)
        return np.full_like(modal_images[0], 0.406, dtype=np.float32)

    def distribution(self):
        normal = sum(1 for _, _, label, _ in self.samples if label == 0)
        abnormal = sum(1 for _, _, label, _ in self.samples if label == 1)
        return {"normal": normal, "abnormal": abnormal, "skipped_unpaired_or_unmasked": self.skipped}


def find_checkpoint(args, modalities):
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()
    base = (REPO_DIR / args.checkpoint_dir).resolve() if not Path(args.checkpoint_dir).is_absolute() else Path(args.checkpoint_dir)
    ckpt_dir = base / args.source_tracer.upper() / "_".join(modalities)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoint directory found: {ckpt_dir}")
    pth_files = sorted(ckpt_dir.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return pth_files[-1]


def load_checkpoint(model, checkpoint, device):
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    try:
        state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()


@torch.no_grad()
def evaluate_deeppsma(model, loader, device, args, eval_size):
    labels, masks, maps = [], [], []
    for img, gt, label, _path in tqdm(loader, ncols=80, desc="DeepPSMA eval"):
        img = img.to(device)
        output = model(img)
        en, de = output[0], output[1]
        anomaly_map = build_anomaly_map(model, en, de, eval_size, args.anomaly_source)
        if gt.shape[-2:] != anomaly_map.shape[-2:]:
            gt = nn.functional.interpolate(gt, size=anomaly_map.shape[-2:], mode="nearest")
        if gt.shape[1] > 1:
            gt = torch.max(gt, dim=1, keepdim=True)[0]
        maps.extend(anomaly_map[:, 0].detach().cpu().numpy().astype(np.float32))
        masks.extend((gt[:, 0].detach().cpu().numpy() > 0.5).astype(np.uint8))
        labels.extend(label.detach().cpu().view(-1).numpy().astype(np.int32).tolist())

    maps = np.stack(maps).astype(np.float32)
    masks = np.stack(masks).astype(np.uint8)
    labels = np.asarray(labels, dtype=np.int32)
    if args.image_score_mode == "max":
        scores = maps.reshape(maps.shape[0], -1).max(axis=1)
    else:
        scores = top_percent_mean(maps, args.max_ratio * 100.0)
    return compute_slice_only_metrics(
        labels,
        masks,
        maps,
        scores,
        bootstrap_iters=args.bootstrap_iters,
        hist_bins=args.hist_bins,
        seed=args.ci_seed,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="INPFormer direct DeepPSMA1 generalization evaluator")
    parser.add_argument("--data_root", default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--source_tracer", default="PSMA", choices=["PSMA", "FDG", "psma", "fdg"])
    parser.add_argument("--modality", default="ct,pet", help="ct, pet, or ct,pet")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--input_size", default=448, type=int)
    parser.add_argument("--crop_size", default=392, type=int)
    parser.add_argument("--INP_num", default=6, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--max_ratio", default=0.01, type=float)
    parser.add_argument("--image_score_mode", default="topk_mean", choices=["topk_mean", "max"])
    parser.add_argument("--channel_fill", default="mean_modalities", choices=["zero", "imagenet_mean", "mean_modalities"])
    parser.add_argument("--anomaly_source", default="reconstruction", choices=["prototype", "reconstruction", "fused"])
    parser.add_argument("--bootstrap_iters", default=500, type=int)
    parser.add_argument("--hist_bins", default=16384, type=int)
    parser.add_argument("--ci_seed", default=2, type=int)
    parser.add_argument("--seed", default=2, type=int)
    return parser


def main():
    args = build_parser().parse_args()
    args.source_tracer = args.source_tracer.upper()
    args.modalities = parse_modalities(args.modality)
    setup_seed(args.seed)

    device_name = args.device or ("cpu" if str(args.gpu).lower() in {"cpu", "-1", "none"} else "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    data_transform, gt_transform = build_transforms(args.input_size, args.crop_size)
    eval_size = args.input_size if args.crop_size <= 0 else args.crop_size
    dataset = DeepPSMAFlatDataset(args.data_root, args.modalities, data_transform, gt_transform, args.channel_fill)
    print(f"DeepPSMA root: {Path(args.data_root).expanduser().resolve()}", flush=True)
    print(f"Source weights: {args.source_tracer}", flush=True)
    print(f"Modalities: {args.modalities}", flush=True)
    print(f"Distribution: {dataset.distribution()}", flush=True)
    print(f"Device: {device}", flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_args = SimpleNamespace(
        encoder=args.encoder,
        INP_num=args.INP_num,
    )
    model, _trainable = build_model(model_args, device)
    checkpoint = find_checkpoint(args, args.modalities)
    load_checkpoint(model, checkpoint, device)
    formatted = evaluate_deeppsma(model, loader, device, args, eval_size)

    out_dir = REPO_DIR / "checkpoints" / "deeppsma1_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.source_tracer}_{'_'.join(args.modalities)}_slice_only_metrics.txt"
    out_path.write_text(
        "\n".join([
            f"checkpoint={checkpoint}",
            f"data_root={Path(args.data_root).expanduser().resolve()}",
            formatted,
            "",
        ]),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
