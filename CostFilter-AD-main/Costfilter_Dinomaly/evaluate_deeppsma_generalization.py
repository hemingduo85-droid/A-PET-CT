#!/usr/bin/env python3
# DeepPSMA1 generalization with existing PSMA PETCT weights:
#   python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer PSMA --modality petct --gpu 0
# DeepPSMA1 generalization with existing FDG PETCT weights:
#   python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer FDG --modality petct --gpu 1
# Single-modality examples:
#   python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer PSMA --modality ct --gpu 0
#   python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer PSMA --modality pet --gpu 0
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path


DEFAULT_DEEPPSMA_ROOT = "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1"
REPO_DIR = Path(__file__).resolve().parent


def _set_visible_gpu_from_argv() -> None:
    argv = sys.argv[1:]
    gpu = None
    device = None
    for idx, item in enumerate(argv):
        if item == "--gpu" and idx + 1 < len(argv):
            gpu = argv[idx + 1]
        elif item.startswith("--gpu="):
            gpu = item.split("=", 1)[1]
        elif item == "--device" and idx + 1 < len(argv):
            device = argv[idx + 1]
        elif item.startswith("--device="):
            device = item.split("=", 1)[1]
    if gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    elif device and device.startswith("cuda:") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]


_set_visible_gpu_from_argv()
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import run_petct as cf
from unet3d_att_dino_channel_test_48_min import DiscriminativeSubNetwork_3d_att_dino_channel
from utils import cal_anomaly_maps


def _fmt_metric(name: str, point: float, ci: tuple[float, float] | None = None) -> str:
    if ci is None:
        return f"{name}={point * 100:.2f}%"
    return f"{name}={point * 100:.2f}% (95% CI {ci[0] * 100:.2f}-{ci[1] * 100:.2f}%)"


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.uint8)
    scores = np.asarray(scores).astype(np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def _safe_aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.uint8)
    scores = np.asarray(scores).astype(np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def _best_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.uint8)
    scores = np.asarray(scores).astype(np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(np.max(f1)) if f1.size else 0.0


def _image_bootstrap(labels: np.ndarray, scores: np.ndarray, iters: int, seed: int) -> dict[str, tuple[float, float]]:
    labels = np.asarray(labels).astype(np.uint8)
    scores = np.asarray(scores).astype(np.float64)
    rng = np.random.default_rng(seed)
    values = {"auroc": [], "aupr": [], "f1": []}
    n = len(labels)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        boot_labels = labels[idx]
        if np.unique(boot_labels).size < 2:
            continue
        boot_scores = scores[idx]
        values["auroc"].append(_safe_roc_auc(boot_labels, boot_scores))
        values["aupr"].append(_safe_aupr(boot_labels, boot_scores))
        values["f1"].append(_best_f1(boot_labels, boot_scores))
    return {
        key: tuple(np.percentile(vals, [2.5, 97.5]).astype(float)) if vals else (0.0, 0.0)
        for key, vals in values.items()
    }


def _hist_metrics(pos_hist: np.ndarray, neg_hist: np.ndarray) -> tuple[float, float]:
    pos_total = float(pos_hist.sum())
    neg_total = float(neg_hist.sum())
    if pos_total <= 0 or neg_total <= 0:
        return 0.0, 0.0
    tp = np.cumsum(pos_hist[::-1])
    fp = np.cumsum(neg_hist[::-1])
    tpr = tp / pos_total
    fpr = fp / neg_total
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tpr
    trapz = getattr(np, "trapezoid", np.trapz)
    auroc = float(trapz(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    order = np.argsort(recall)
    aupr = float(trapz(precision[order], recall[order]))
    return auroc, aupr


def _pixel_hist_bootstrap(
    masks: np.ndarray,
    maps: np.ndarray,
    labels: np.ndarray,
    bins: int,
    iters: int,
    seed: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    abn_idx = np.flatnonzero(labels.astype(np.uint8) == 1)
    if abn_idx.size == 0:
        return (0.0, 0.0), (0.0, 0.0)
    abn_maps = maps[abn_idx].astype(np.float32, copy=False)
    vmin = float(np.nanmin(abn_maps))
    vmax = float(np.nanmax(abn_maps))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return (0.0, 0.0), (0.0, 0.0)
    edges = np.linspace(vmin, vmax, bins + 1, dtype=np.float64)
    pos_hists = []
    neg_hists = []
    for idx in abn_idx:
        mask_flat = masks[idx].reshape(-1) > 0.5
        map_flat = maps[idx].reshape(-1)
        pos_hists.append(np.histogram(map_flat[mask_flat], bins=edges)[0].astype(np.float64))
        neg_hists.append(np.histogram(map_flat[~mask_flat], bins=edges)[0].astype(np.float64))
    pos_hists = np.stack(pos_hists, axis=0)
    neg_hists = np.stack(neg_hists, axis=0)
    rng = np.random.default_rng(seed)
    aurocs = []
    auprs = []
    for i in range(1, iters + 1):
        sample = rng.integers(0, len(abn_idx), size=len(abn_idx))
        auroc, aupr = _hist_metrics(pos_hists[sample].sum(axis=0), neg_hists[sample].sum(axis=0))
        aurocs.append(auroc)
        auprs.append(aupr)
        if i % 50 == 0 or i == iters:
            print(f"Pixel histogram slice bootstrap: {i}/{iters}", flush=True)
    return (
        tuple(np.percentile(aurocs, [2.5, 97.5]).astype(float)),
        tuple(np.percentile(auprs, [2.5, 97.5]).astype(float)),
    )


def compute_slice_only_metrics(
    labels: np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
    scores: np.ndarray,
    bootstrap_iters: int,
    ci_seed: int,
    hist_bins: int,
) -> str:
    labels = np.asarray(labels).astype(np.uint8)
    masks = np.asarray(masks).astype(np.float32)
    maps = np.asarray(maps).astype(np.float32)
    scores = np.asarray(scores).astype(np.float64)

    print(f"Cache loaded: labels={labels.shape}, masks={masks.shape}, maps={maps.shape}, bootstrap_iters={bootstrap_iters}", flush=True)
    print("Computing image-level metrics and 95CI...", flush=True)
    img_auroc = _safe_roc_auc(labels, scores)
    img_aupr = _safe_aupr(labels, scores)
    img_f1 = _best_f1(labels, scores)
    img_ci = _image_bootstrap(labels, scores, bootstrap_iters, ci_seed)
    img_line = (
        "[Slice-Img]  "
        + _fmt_metric("AUROC", img_auroc, img_ci["auroc"])
        + "  "
        + _fmt_metric("AUPR", img_aupr, img_ci["aupr"])
        + "  "
        + _fmt_metric("F1", img_f1, img_ci["f1"])
    )
    print(img_line, flush=True)

    print("Computing pixel-level Slice-Px(abn) metrics and 95CI...", flush=True)
    print("Computing exact full-pixel point estimates with sklearn...", flush=True)
    abn_idx = np.flatnonzero(labels == 1)
    if abn_idx.size:
        px_labels = masks[abn_idx].reshape(-1) > 0.5
        px_scores = maps[abn_idx].reshape(-1)
        px_auroc = _safe_roc_auc(px_labels, px_scores)
        px_aupr = _safe_aupr(px_labels, px_scores)
        px_auroc_ci, px_aupr_ci = _pixel_hist_bootstrap(masks, maps, labels, hist_bins, bootstrap_iters, ci_seed + 1009)
    else:
        px_auroc = px_aupr = 0.0
        px_auroc_ci = px_aupr_ci = (0.0, 0.0)
    px_line = (
        "[Slice-Px(abn)]  "
        + _fmt_metric("AUROC", px_auroc, px_auroc_ci)
        + "  "
        + _fmt_metric("AUPR", px_aupr, px_aupr_ci)
    )
    print(px_line, flush=True)
    return img_line + "\n" + px_line


class DeepPSMAFlatDataset(Dataset):
    def __init__(self, data_root: str | Path, modality: str, transform, gt_transform):
        self.data_root = Path(data_root).expanduser().resolve()
        self.modality = modality
        self.transform = transform
        self.gt_transform = gt_transform
        self.samples, self.distribution = self._collect_samples()
        if not self.samples:
            raise RuntimeError(
                f"No DeepPSMA samples found under {self.data_root}. "
                "Expected test/{normal,abnormal}/{ct,pet,label}/*.png"
            )

    def _collect_samples(self):
        samples = []
        stats = {"normal": 0, "abnormal": 0, "skipped_unpaired_or_unmasked": 0}
        for category, label_val in (("normal", 0), ("abnormal", 1)):
            base = self.data_root / "test" / category
            ct_dir = base / "ct"
            pet_dir = base / "pet"
            label_dir = base / "label"
            ct_paths = {p.name: p for p in sorted(ct_dir.glob("*.png"))} if ct_dir.is_dir() else {}
            pet_paths = {p.name: p for p in sorted(pet_dir.glob("*.png"))} if pet_dir.is_dir() else {}
            label_paths = {p.name: p for p in sorted(label_dir.glob("*.png"))} if label_dir.is_dir() else {}
            for name in sorted(set(ct_paths) | set(pet_paths)):
                if name not in ct_paths or name not in pet_paths:
                    stats["skipped_unpaired_or_unmasked"] += 1
                    continue
                lbl_path = label_paths.get(name)
                if label_val == 1 and lbl_path is None:
                    stats["skipped_unpaired_or_unmasked"] += 1
                    continue
                samples.append((str(ct_paths[name]), str(pet_paths[name]), str(lbl_path) if lbl_path else None, label_val, name))
                stats[category] += 1
        return samples, stats

    def _load_rgb(self, ct_path: str, pet_path: str) -> Image.Image:
        if self.modality == "ct":
            ct = Image.open(ct_path).convert("L")
            return Image.merge("RGB", [ct, ct, ct])
        if self.modality == "pet":
            pet = Image.open(pet_path).convert("L")
            return Image.merge("RGB", [pet, pet, pet])
        ct = Image.open(ct_path).convert("L")
        pet = Image.open(pet_path).convert("L")
        return Image.merge("RGB", [ct, pet, pet])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ct_path, pet_path, lbl_path, label, image_id = self.samples[idx]
        img = self.transform(self._load_rgb(ct_path, pet_path))
        if lbl_path and os.path.exists(lbl_path):
            gt = self.gt_transform(Image.open(lbl_path).convert("L"))
            gt = (gt > 0.5).float()
        else:
            gt = torch.zeros(1, img.shape[1], img.shape[2])
        return img, gt, int(label), image_id


@torch.no_grad()
def evaluate_costfilter_slice_only(
    model,
    model_unet,
    test_loader,
    device,
    crop_size: int,
    feat_size: int,
    pre_min_dim: int,
    sigma: float,
    lamda: float,
):
    model.eval()
    model_unet.eval()
    all_maps = []
    all_gts = []
    all_labels = []
    unet_spatial = 64

    for img, gt, label, _ in tqdm(test_loader, desc="DeepPSMA eval", leave=True):
        img = img.to(device)
        en, de = model(img)
        batch_size = en[0].shape[0]
        epsilon = 1e-8

        min_anomaly_map = []
        anomaly_map_all_bat = []
        for rec_feat, org_feat in zip(en, de):
            h, w = org_feat.shape[2:]
            pi = org_feat.reshape(batch_size, -1, h * w).permute(0, 2, 1)
            pr = rec_feat.reshape(batch_size, -1, h * w).permute(0, 2, 1)
            pi = pi / (torch.norm(pi, p=2, dim=-1, keepdim=True) + epsilon)
            pr = pr / (torch.norm(pr, p=2, dim=-1, keepdim=True) + epsilon)
            cos0 = torch.bmm(pi, pr.permute(0, 2, 1))
            a_map, _ = torch.min(1 - cos0, dim=-1)
            a_map = nn.UpsamplingBilinear2d(size=(unet_spatial, unet_spatial))(a_map.reshape(batch_size, 1, h, w))
            min_anomaly_map.append(a_map)
            one_minus = 1 - cos0
            cur_dim = min(pre_min_dim, one_minus.shape[-1])
            _, top_idx = torch.topk(one_minus, cur_dim, dim=-1, largest=False, sorted=False)
            sel = torch.gather(one_minus, dim=-1, index=torch.sort(top_idx, dim=-1)[0])
            anomaly_map_all_bat.append(sel.view(batch_size, h, w, cur_dim).unsqueeze(1))

        n_layers = len(en)
        min_anomaly_map = torch.stack(min_anomaly_map, dim=0).permute(1, 0, 2, 3, 4).squeeze(0)
        anomaly_map_all_bat = torch.stack(anomaly_map_all_bat, dim=0).squeeze(2)
        anomaly_map_all_bat = nn.UpsamplingBilinear2d(size=(unet_spatial, unet_spatial))(
            anomaly_map_all_bat.permute(0, 1, 4, 2, 3).reshape(-1, pre_min_dim, feat_size, feat_size)
        )
        anomaly_map_all_bat = anomaly_map_all_bat.view(
            n_layers, batch_size, pre_min_dim, unet_spatial, unet_spatial
        ).permute(1, 0, 2, 3, 4).permute(0, 2, 1, 3, 4)

        pred1, _ = cal_anomaly_maps(en, de, crop_size)
        ft = feat_size * feat_size
        dino_features = [feat.reshape(feat.shape[0], feat.shape[1], ft).permute(0, 2, 1) for feat in en]

        min_sim = torch.zeros_like(min_anomaly_map.squeeze())
        if min_anomaly_map.dim() == 4:
            min_sim = min_sim.unsqueeze(0)
        pred1_small = nn.UpsamplingBilinear2d(size=(unet_spatial, unet_spatial))(pred1).squeeze(1)
        min_sim[:, 0, :, :] = pred1_small
        min_sim[:, 1, :, :] = pred1_small

        output, _ = model_unet(anomaly_map_all_bat.float().to(device), dino_features, min_sim.float().to(device))
        output_focl = torch.softmax(output, dim=1)
        output_focl = F.interpolate(output_focl, size=crop_size, mode="bilinear", align_corners=True)
        cf_map = output_focl[:, 1, :, :].unsqueeze(1)
        anomaly_map_final = cf_map * lamda + pred1 * (1 - lamda)

        all_maps.append(anomaly_map_final.cpu())
        all_gts.append(gt)
        if isinstance(label, torch.Tensor):
            all_labels.extend(label.numpy().tolist())
        else:
            all_labels.extend(label)

    maps = torch.cat(all_maps, dim=0)[:, 0].numpy()
    masks = torch.cat(all_gts, dim=0)[:, 0].numpy()
    labels = np.asarray(all_labels, dtype=np.uint8)
    for i in range(maps.shape[0]):
        maps[i] = gaussian_filter(maps[i], sigma=sigma)
    flat = maps.reshape(maps.shape[0], -1)
    topk = max(1, int(np.ceil(flat.shape[1] * 0.01)))
    scores = np.partition(flat, -topk, axis=1)[:, -topk:].mean(axis=1)
    return labels, masks, maps, scores


def _torch_load(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_dinomaly_checkpoint(model, ckpt_path: str | Path, device: torch.device) -> None:
    state = _torch_load(ckpt_path, device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)


def _load_costfilter_checkpoint(model_unet, ckpt_path: str | Path, device: torch.device) -> None:
    state = _torch_load(ckpt_path, device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model_unet.load_state_dict(state)


def _default_save_dir(source: str, modality: str) -> Path:
    return REPO_DIR / f"checkpoint_{source.lower()}_{modality}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct DeepPSMA1 generalization evaluator for CostFilter-AD")
    parser.add_argument("--data_root", default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--source_tracer", default="PSMA", choices=["PSMA", "FDG", "psma", "fdg"])
    parser.add_argument("--modality", default="petct", choices=["ct", "pet", "petct"])
    parser.add_argument("--gpu", default="0", help="Physical GPU id; internally mapped to cuda:0")
    parser.add_argument("--device", default=None, help="Optional device string, e.g. cuda:4 or cpu")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--dinomaly_ckpt_path", default=None)
    parser.add_argument("--costfilter_ckpt_path", default=None)
    parser.add_argument("--checkpoint", default=None, help="Alias for --costfilter_ckpt_path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=252)
    parser.add_argument("--costfilter_epochs", type=int, default=30)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--bootstrap_iters", "--ci_iters", dest="bootstrap_iters", type=int, default=500)
    parser.add_argument("--ci_seed", type=int, default=20260707)
    parser.add_argument("--hist_bins", "--ci_hist_bins", dest="hist_bins", type=int, default=16384)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.source_tracer.upper()
    data_root = Path(args.data_root).expanduser().resolve()
    if args.device == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else _default_save_dir(source, args.modality)
    dinomaly_ckpt = Path(args.dinomaly_ckpt_path).expanduser().resolve() if args.dinomaly_ckpt_path else save_dir / f"dinomaly_{source.lower()}_{args.modality}.pth"
    cf_ckpt_arg = args.costfilter_ckpt_path or args.checkpoint
    costfilter_ckpt = Path(cf_ckpt_arg).expanduser().resolve() if cf_ckpt_arg else save_dir / f"costfilter_{source.lower()}_{args.modality}_epoch{args.costfilter_epochs}.pth"

    print(f"DeepPSMA root: {data_root}", flush=True)
    print(f"Source weights: {source}", flush=True)
    print(f"Modality: {args.modality}", flush=True)
    print(f"Device: {device}", flush=True)

    data_transform, gt_transform = cf.get_data_transforms(args.image_size, args.crop_size)
    test_dataset = DeepPSMAFlatDataset(data_root, args.modality, data_transform, gt_transform)
    print(f"Distribution: {test_dataset.distribution}", flush=True)
    print(f"Test slices: {len(test_dataset)}", flush=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if not dinomaly_ckpt.exists():
        raise FileNotFoundError(f"Dinomaly checkpoint not found: {dinomaly_ckpt}")
    if not costfilter_ckpt.exists():
        raise FileNotFoundError(f"CostFilter checkpoint not found: {costfilter_ckpt}")

    crop_size = int(args.crop_size)
    feat_size = crop_size // 14
    feat_total = feat_size * feat_size
    pre_min_dim = min(768, feat_total)

    model, _ = cf.build_dinomaly(device)
    _load_dinomaly_checkpoint(model, dinomaly_ckpt, device)
    print(f"Loaded Dinomaly from {dinomaly_ckpt}", flush=True)

    model_unet = DiscriminativeSubNetwork_3d_att_dino_channel(
        in_channels=pre_min_dim,
        out_channels=2,
        base_channels=args.base_channels,
    ).to(device).float()
    _load_costfilter_checkpoint(model_unet, costfilter_ckpt, device)
    print(f"Loaded CostFilter from {costfilter_ckpt}", flush=True)

    labels, masks, maps, scores = evaluate_costfilter_slice_only(
        model,
        model_unet,
        test_loader,
        device,
        crop_size=crop_size,
        feat_size=feat_size,
        pre_min_dim=pre_min_dim,
        sigma=4,
        lamda=args.lamda,
    )
    compute_slice_only_metrics(
        labels=labels,
        masks=masks,
        maps=maps,
        scores=scores,
        bootstrap_iters=args.bootstrap_iters,
        ci_seed=args.ci_seed,
        hist_bins=args.hist_bins,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
