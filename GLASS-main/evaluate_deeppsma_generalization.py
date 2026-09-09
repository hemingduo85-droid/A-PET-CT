#!/usr/bin/env python3
# DeepPSMA1 generalization with existing PSMA PETCT weights:
#   python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer PSMA --modality petct --gpu 4
# DeepPSMA1 generalization with existing FDG PETCT weights:
#   python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer FDG --modality petct --gpu 5
# Single-modality examples:
#   python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer PSMA --modality ct --gpu 4
#   python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer PSMA --modality pet --gpu 4
from __future__ import annotations

import argparse
import os
import sys
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

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import backbones
import glass
import petct_config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    def __init__(self, data_root: str | Path, modality: str, resize: int = 256, imagesize: int = 256):
        self.data_root = Path(data_root).expanduser().resolve()
        self.modality = modality
        self.resize = int(resize)
        self.imgsize = int(imagesize)
        self.imagesize = (3, self.imgsize, self.imgsize)
        self.transform_img = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
        ])
        self.samples, self.distribution = self._collect_samples()
        if not self.samples:
            raise RuntimeError(
                f"No DeepPSMA samples found under {self.data_root}. "
                "Expected test/{normal,abnormal}/{ct,pet,label}/*.png"
            )

    @staticmethod
    def _normalize_gray(arr: np.ndarray) -> np.ndarray:
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax > vmin:
            return ((arr - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    def _collect_samples(self):
        samples = []
        stats = {"normal": 0, "abnormal": 0, "skipped_unpaired_or_unmasked": 0}
        for subset, label in (("normal", 0), ("abnormal", 1)):
            base = self.data_root / "test" / subset
            ct_dir = base / "ct"
            pet_dir = base / "pet"
            label_dir = base / "label"
            ct_paths = {p.name: p for p in sorted(ct_dir.glob("*.png"))} if ct_dir.is_dir() else {}
            pet_paths = {p.name: p for p in sorted(pet_dir.glob("*.png"))} if pet_dir.is_dir() else {}
            label_paths = {p.name: p for p in sorted(label_dir.glob("*.png"))} if label_dir.is_dir() else {}
            names = sorted(set(ct_paths) | set(pet_paths))
            for name in names:
                if name not in ct_paths or name not in pet_paths:
                    stats["skipped_unpaired_or_unmasked"] += 1
                    continue
                mask_path = label_paths.get(name)
                if label == 1 and mask_path is None:
                    stats["skipped_unpaired_or_unmasked"] += 1
                    continue
                samples.append({
                    "ct": ct_paths[name],
                    "pet": pet_paths[name],
                    "mask": mask_path,
                    "label": label,
                    "path": str(pet_paths[name] if self.modality == "pet" else ct_paths[name]),
                })
                stats[subset] += 1
        return samples, stats

    def _load_image(self, pet_path: Path, ct_path: Path) -> Image.Image:
        if self.modality == "pet":
            gray = np.array(Image.open(pet_path).convert("L"), dtype=np.float32)
            return Image.fromarray(self._normalize_gray(gray)).convert("RGB")
        if self.modality == "ct":
            gray = np.array(Image.open(ct_path).convert("L"), dtype=np.float32)
            return Image.fromarray(self._normalize_gray(gray)).convert("RGB")
        pet = self._normalize_gray(np.array(Image.open(pet_path).convert("L"), dtype=np.float32))
        ct = self._normalize_gray(np.array(Image.open(ct_path).convert("L"), dtype=np.float32))
        return Image.fromarray(np.stack([ct, pet, pet], axis=-1), mode="RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = self._load_image(sample["pet"], sample["ct"])
        if sample["mask"] is None:
            mask = Image.fromarray(np.zeros(image.size[::-1], dtype=np.uint8), mode="L")
        else:
            mask = Image.open(sample["mask"]).convert("L")
        mask_tensor = (self.transform_mask(mask) > 0.5).float()
        image_tensor = self.transform_img(image)
        return {
            "image": image_tensor,
            "mask_gt": mask_tensor,
            "is_anomaly": torch.tensor(int(sample["label"]), dtype=torch.long),
            "image_path": sample["path"],
        }


def _load_checkpoint(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _resolve_checkpoint(model: glass.GLASS, override: str | None) -> str:
    if override:
        path = str(Path(override).expanduser().resolve())
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    candidates = model._find_checkpoint()
    if not candidates:
        raise FileNotFoundError(f"No GLASS checkpoint found under {model.ckpt_dir}")
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct DeepPSMA1 generalization evaluator for GLASS")
    parser.add_argument("--data_root", default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--source_tracer", default="PSMA", choices=["PSMA", "FDG", "psma", "fdg"])
    parser.add_argument("--modality", default="petct", choices=["ct", "pet", "petct"])
    parser.add_argument("--gpu", default="0", help="Physical GPU id; internally mapped to cuda:0")
    parser.add_argument("--device", default=None, help="Optional device string, e.g. cuda:4 or cpu")
    parser.add_argument("--checkpoint", default=None, help="Optional direct checkpoint file")
    parser.add_argument("--save_dir", default=None, help="Directory containing models/backbone_0/petct_<modality>")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--bootstrap_iters", type=int, default=500)
    parser.add_argument("--ci_seed", type=int, default=42)
    parser.add_argument("--hist_bins", type=int, default=16384)
    parser.add_argument("--backbone", default="wideresnet50")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1536)
    parser.add_argument("--target_embed_dimension", type=int, default=1536)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--meta_epochs", type=int, default=30)
    parser.add_argument("--eval_epochs", type=int, default=30)
    parser.add_argument("--dsc_layers", type=int, default=2)
    parser.add_argument("--dsc_hidden", type=int, default=1024)
    parser.add_argument("--pre_proj", type=int, default=1)
    parser.add_argument("--mining", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--radius", type=float, default=0.75)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--limit", type=int, default=1840)
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

    print(f"DeepPSMA root: {data_root}", flush=True)
    print(f"Source weights: {source}", flush=True)
    print(f"Modality: {args.modality}", flush=True)
    print(f"Device: {device}", flush=True)

    dataset = DeepPSMAFlatDataset(data_root, args.modality, resize=args.resize, imagesize=args.imagesize)
    print(f"Distribution: {dataset.distribution}", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    backbone = backbones.load(args.backbone)
    backbone.name = args.backbone
    backbone.seed = None
    model = glass.GLASS(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=args.layers,
        device=device,
        input_shape=dataset.imagesize,
        pretrain_embed_dimension=args.pretrain_embed_dimension,
        target_embed_dimension=args.target_embed_dimension,
        patchsize=args.patchsize,
        meta_epochs=args.meta_epochs,
        eval_epochs=args.eval_epochs,
        dsc_layers=args.dsc_layers,
        dsc_hidden=args.dsc_hidden,
        pre_proj=args.pre_proj,
        mining=args.mining,
        noise=args.noise,
        radius=args.radius,
        p=args.p,
        step=args.step,
        limit=args.limit,
    )
    model = model.to(device)

    save_dir = Path(args.save_dir or petct_config.default_save_dir("results", source.lower(), args.modality)).expanduser()
    if not save_dir.is_absolute():
        save_dir = REPO_DIR / save_dir
    model.set_model_dir(os.path.join(str(save_dir), "models", "backbone_0"), f"petct_{args.modality}")
    checkpoint = _resolve_checkpoint(model, args.checkpoint)
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    state = _load_checkpoint(checkpoint, device)
    if isinstance(state, dict) and "discriminator" in state:
        model.discriminator.load_state_dict(state["discriminator"])
        if "pre_projection" in state and hasattr(model, "pre_projection"):
            model.pre_projection.load_state_dict(state["pre_projection"])
    else:
        model.load_state_dict(state, strict=False)

    _, scores, maps, labels, masks, _ = model.predict(loader)
    compute_slice_only_metrics(
        labels=np.asarray(labels),
        masks=np.asarray(masks)[:, 0] if np.asarray(masks).ndim == 4 else np.asarray(masks),
        maps=np.asarray(maps),
        scores=np.asarray(scores),
        bootstrap_iters=args.bootstrap_iters,
        ci_seed=args.ci_seed,
        hist_bins=args.hist_bins,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
