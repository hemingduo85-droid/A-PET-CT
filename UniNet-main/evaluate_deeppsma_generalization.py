#!/usr/bin/env python3
"""
DeepPSMA1 direct generalization test for UniNet.

Data root:
  /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1

Run with existing PSMA weights:
  python evaluate_deeppsma_generalization.py --source_tracer PSMA --modality petct --gpu 1 > deeppsma_psma.log 2>&1 &

Run with existing FDG weights:
  python evaluate_deeppsma_generalization.py --source_tracer FDG --modality petct --gpu 2 > deeppsma_fdg.log 2>&1 &
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
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from main_petct import build_model, final_ckpt_suffix


DEFAULT_DEEPPSMA_ROOT = "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
REPO_DIR = Path(__file__).resolve().parent


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


def compute_slice_only_metrics(labels, masks, maps, scores, bootstrap_iters=500, hist_bins=16384, seed=1203):
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
    def __init__(self, data_root, modality="petct", image_size=256):
        self.root = Path(data_root).expanduser().resolve()
        self.modality = modality.lower()
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
        primary = "pet" if self.modality == "petct" else self.modality
        primary_dir = group_root / primary
        label_dir = group_root / "label"
        if not primary_dir.is_dir():
            return

        for path in sorted(p for p in primary_dir.iterdir() if self._is_image(p)):
            ct_path = group_root / "ct" / path.name
            pet_path = group_root / "pet" / path.name
            if self.modality == "petct" and not (self._is_image(ct_path) and self._is_image(pet_path)):
                self.skipped += 1
                continue

            mask_path = None
            if label:
                candidate = label_dir / path.name
                if not self._is_image(candidate):
                    self.skipped += 1
                    continue
                mask_path = candidate
            self.samples.append((ct_path, pet_path, path, mask_path, int(label), str(path)))

    def _collect(self):
        self._collect_group("normal", 0)
        self._collect_group("abnormal", 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_path, pet_path, primary_path, mask_path, label, sample_id = self.samples[idx]
        if self.modality == "petct":
            ct = Image.open(ct_path).convert("L")
            pet = Image.open(pet_path).convert("L")
            image = Image.merge("RGB", (ct, pet, pet))
        else:
            gray = Image.open(primary_path).convert("L")
            image = Image.merge("RGB", (gray, gray, gray))

        img = self.img_transform(image)
        if label and mask_path is not None:
            mask = self.mask_transform(Image.open(mask_path).convert("L"))
            mask = (mask > 0).float()
        else:
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        return img, torch.tensor(label, dtype=torch.long), mask, sample_id

    def distribution(self):
        normal = sum(1 for _, _, _, _, label, _ in self.samples if label == 0)
        abnormal = sum(1 for _, _, _, _, label, _ in self.samples if label == 1)
        return {"normal": normal, "abnormal": abnormal, "skipped_unpaired_or_unmasked": self.skipped}


def build_config(args):
    source = args.source_tracer.strip().replace(" ", "_").upper()
    modality = args.modality.lower()
    dataset = f"PETCT_{source}_{modality.upper()}"
    return SimpleNamespace(
        data_root=str(Path(args.data_root).expanduser().resolve()),
        dataset_name=source,
        modality=modality,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr_s=5e-3,
        lr_t=1e-6,
        T=args.temperature,
        weighted_decision_mechanism=True,
        alpha=0.01,
        beta=0.00003,
        default=0.3,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        ckpt_dir=args.ckpt_dir,
        load_ckpts=True,
        ckpt_suffix=args.ckpt_suffix,
        save_heatmaps=False,
        image_agg=args.image_agg,
        ci_bootstraps=args.bootstrap_iters,
        ci_pixel_max_samples=200000,
        pixel_metric_max_samples=0,
        hist_bins=args.hist_bins,
        ci_seed=args.ci_seed,
        gpu=str(args.gpu),
        domain="medical",
        setting="oc",
        dataset=dataset,
        _class_=dataset,
    )


def load_uninet_checkpoint(model, checkpoint, ckpt_dir, suffix, device):
    if checkpoint is None:
        checkpoint = Path(ckpt_dir) / f"{suffix}.pth"
    else:
        checkpoint = Path(checkpoint).expanduser()
        if checkpoint.is_dir():
            checkpoint = checkpoint / f"{suffix}.pth"
    checkpoint = checkpoint.resolve()
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    state = torch.load(str(checkpoint), map_location=device)
    modules = {"tt": model.t.t_t, "bn": model.bn.bn, "st": model.s.s1, "dfs": model.dfs}
    for key, module in modules.items():
        if module is not None and key in state and state[key] is not None:
            module.load_state_dict(state[key])
            module.to(device)
            module.eval()
    return checkpoint


@torch.no_grad()
def evaluate_deeppsma(model, loader, device, args):
    model.train_or_eval(type="eval")
    labels, masks, maps, scores = [], [], [], []

    for img, label, mask, _paths in tqdm(loader, desc="DeepPSMA eval", leave=True):
        img = img.to(device)
        t_tf, de_features = model(img)
        batch_map = None
        for t, s in zip(t_tf, de_features):
            layer_map = 1 - F.cosine_similarity(t, s)
            layer_map = F.interpolate(
                layer_map.unsqueeze(1),
                size=args.image_size,
                mode="bilinear",
                align_corners=True,
            )[:, 0, :, :]
            batch_map = layer_map if batch_map is None else batch_map + layer_map

        batch_map = batch_map.detach().cpu().numpy().astype(np.float32)
        batch_map = np.stack([gaussian_filter(batch_map[i], sigma=4) for i in range(batch_map.shape[0])], axis=0)

        flat = batch_map.reshape(batch_map.shape[0], -1)
        if args.image_agg == "top1pct":
            k = max(1, int(flat.shape[1] * 0.01))
            score = np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)
        else:
            score = flat.max(axis=1)

        labels.extend(label.numpy().astype(np.int32).tolist())
        masks.extend(mask.squeeze(1).numpy().astype(np.uint8))
        maps.extend(batch_map)
        scores.extend(score.astype(np.float64).tolist())

    return compute_slice_only_metrics(
        np.asarray(labels, dtype=np.int32),
        np.asarray(masks, dtype=np.uint8),
        np.asarray(maps, dtype=np.float32),
        np.asarray(scores, dtype=np.float64),
        bootstrap_iters=args.bootstrap_iters,
        hist_bins=args.hist_bins,
        seed=args.ci_seed,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="UniNet direct DeepPSMA1 generalization evaluator")
    parser.add_argument("--data_root", default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--source_tracer", default="PSMA", choices=["PSMA", "FDG", "psma", "fdg"])
    parser.add_argument("--modality", default="petct", choices=["ct", "pet", "petct"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Full checkpoint path, or a directory containing the checkpoint suffix")
    parser.add_argument("--ckpt_dir", default="./ckpts")
    parser.add_argument("--ckpt_suffix", default=None)
    parser.add_argument("--save_dir", default="./saved_results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_agg", default="top1pct", choices=["top1pct", "max"])
    parser.add_argument("--bootstrap_iters", type=int, default=500)
    parser.add_argument("--ci_seed", type=int, default=1203)
    parser.add_argument("--hist_bins", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=2.0)
    return parser


def main():
    args = build_parser().parse_args()
    source = args.source_tracer.upper()
    modality = args.modality.lower()
    c = build_config(args)

    device_name = args.device or ("cpu" if str(args.gpu).lower() in {"cpu", "-1", "none"} else "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset = DeepPSMAFlatDataset(args.data_root, modality, args.image_size)
    print(f"DeepPSMA root: {Path(args.data_root).expanduser().resolve()}", flush=True)
    print(f"Source weights: {source}", flush=True)
    print(f"Modality: {modality}", flush=True)
    print(f"Distribution: {dataset.distribution()}", flush=True)
    print(f"Device: {device}", flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, _bn, _student, _dfs, _target_teacher = build_model(c, str(device))
    ckpt_root = Path(args.ckpt_dir).expanduser() / c.dataset
    suffix = final_ckpt_suffix(c)
    checkpoint = load_uninet_checkpoint(model, args.checkpoint, ckpt_root, suffix, device)
    formatted = evaluate_deeppsma(model, loader, device, args)

    out_dir = REPO_DIR / "saved_results" / "deeppsma1_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source}_{modality}_slice_only_metrics.txt"
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
