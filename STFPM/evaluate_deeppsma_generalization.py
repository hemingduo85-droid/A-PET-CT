#!/usr/bin/env python3
"""
DeepPSMA1 direct generalization test for STFPM.

Data root:
  /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1

Run with existing PSMA weights:
  python evaluate_deeppsma_generalization.py --source_tracer PSMA --modality petct --gpu 0

Run with existing FDG weights:
  python evaluate_deeppsma_generalization.py --source_tracer FDG --modality petct --gpu 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from main import ResNet18_MS3


DEFAULT_DEEPPSMA_ROOT = "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
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


def compute_slice_only_metrics(labels, masks, maps, scores, bootstrap_iters=500, hist_bins=16384, seed=20260717):
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
    image_line = "[Slice-Img]  " + "  ".join([_fmt("AUROC", img_auroc), _fmt("AUPR", img_aupr), _fmt("F1", img_f1)])
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
        self.transform = transforms.Compose([
            transforms.Resize([self.image_size, self.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
            self.samples.append((ct_path, pet_path, mask_path, int(label), str(ct_path)))

    def _collect(self):
        self._collect_group("normal", 0)
        self._collect_group("abnormal", 1)

    def __len__(self):
        return len(self.samples)

    def _load_image(self, ct_path, pet_path):
        ct_img = Image.open(ct_path).convert("L")
        pet_img = Image.open(pet_path).convert("L")
        if self.modality == "ct":
            return ct_img.convert("RGB")
        if self.modality == "pet":
            return pet_img.convert("RGB")
        return Image.merge("RGB", (ct_img, pet_img, pet_img))

    def __getitem__(self, idx):
        ct_path, pet_path, mask_path, label, sample_id = self.samples[idx]
        image = self.transform(self._load_image(ct_path, pet_path))
        if label and mask_path is not None:
            mask_img = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)
            mask = torch.from_numpy((np.asarray(mask_img, dtype=np.float32) > 127.5).astype(np.float32)).unsqueeze(0)
            mask = (mask > 0).float()
        else:
            mask = torch.zeros((1, image.shape[-2], image.shape[-1]), dtype=torch.float32)
        return image, mask, label, sample_id

    def distribution(self):
        normal = sum(1 for _, _, _, label, _ in self.samples if label == 0)
        abnormal = sum(1 for _, _, _, label, _ in self.samples if label == 1)
        return {"normal": normal, "abnormal": abnormal, "skipped_unpaired_or_unmasked": self.skipped}


@torch.no_grad()
def evaluate(teacher, student, loader, device, args):
    teacher.eval()
    student.eval()
    labels, masks, maps, scores = [], [], [], []
    for img, gt, label, paths in tqdm(loader, desc="DeepPSMA eval", leave=True):
        img = img.to(device)
        t_feat = teacher(img)
        s_feat = student(img)
        score_map = 1.0
        for t, s in zip(t_feat, s_feat):
            t = F.normalize(t, dim=1)
            s = F.normalize(s, dim=1)
            sm = torch.sum((t - s) ** 2, dim=1, keepdim=True)
            sm = F.interpolate(sm, size=(64, 64), mode="bilinear", align_corners=False)
            score_map = score_map * sm
        anomaly_map = F.interpolate(score_map, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        flat = anomaly_map.flatten(1)
        score = flat.max(dim=1)[0]
        maps.extend(anomaly_map.squeeze(1).cpu().numpy().astype(np.float32))
        masks.extend(gt.squeeze(1).cpu().numpy().astype(np.uint8))
        labels.extend(label.cpu().numpy().astype(np.int32).tolist())
        scores.extend(score.cpu().numpy().astype(np.float64).tolist())

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
    parser = argparse.ArgumentParser(description="STFPM direct DeepPSMA1 generalization evaluator")
    parser.add_argument("--data_root", default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--source_tracer", default="PSMA", choices=["PSMA", "FDG", "psma", "fdg"])
    parser.add_argument("--modality", default="petct", choices=["ct", "pet", "petct"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model_save_path", default="snapshots")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap_iters", type=int, default=500)
    parser.add_argument("--ci_seed", type=int, default=20260717)
    parser.add_argument("--hist_bins", type=int, default=16384)
    parser.add_argument("--no_pretrained_teacher", action="store_true")
    parser.add_argument("--teacher_weights", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    source = args.source_tracer.upper()
    device_name = args.device or ("cpu" if str(args.gpu).lower() in {"cpu", "-1", "none"} else f"cuda:{args.gpu}")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset = DeepPSMAFlatDataset(args.data_root, args.modality, image_size=256)
    print(f"DeepPSMA root: {Path(args.data_root).expanduser().resolve()}")
    print(f"Source weights: {source}")
    print(f"Modality: {args.modality}")
    print(f"Distribution: {dataset.distribution()}")
    print(f"Device: {device}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    in_channels = 3
    teacher = ResNet18_MS3(
        pretrained=not args.no_pretrained_teacher,
        in_channels=in_channels,
        weights_path=args.teacher_weights,
    ).to(device)
    student = ResNet18_MS3(pretrained=False, in_channels=in_channels).to(device)
    modality_tag = "ct+pet" if args.modality == "petct" else args.modality
    input_mode = "pseudo_rgb" if args.modality == "petct" else "concat"
    checkpoint = args.checkpoint or str(REPO_DIR / args.model_save_path / f"{source.lower()}_{modality_tag}_{input_mode}_epoch30.pth.tar")
    print(f"Loading checkpoint: {checkpoint}")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = saved["state_dict"] if isinstance(saved, dict) and "state_dict" in saved else saved
    student.load_state_dict(state_dict)
    formatted = evaluate(teacher, student, loader, device, args)

    out_dir = REPO_DIR / "results" / "deeppsma1_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source}_{args.modality}_slice_only_metrics.txt"
    out_path.write_text(formatted + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
