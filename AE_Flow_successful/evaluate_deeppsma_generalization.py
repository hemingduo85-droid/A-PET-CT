"""
Run examples:
  python evaluate_deeppsma_generalization.py --source_tracer PSMA --modality ct,pet --device cuda:4
  python evaluate_deeppsma_generalization.py --source_tracer FDG --modality ct,pet --device cuda:5
  python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer PSMA --modality ct --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import ae_flow
from train import find_checkpoint, make_dataloader, parse_modalities, resolve_device
from paper_eval_utils import top_percent_mean

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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
    print(f"Cache loaded: labels={labels.shape}, masks={masks.shape}, maps={maps.shape}, bootstrap_iters={int(bootstrap_iters)}")
    print("Computing image-level metrics and 95CI...")
    img_auroc = _bootstrap_ci(labels, scores, _safe_auroc, bootstrap_iters, seed)
    img_aupr = _bootstrap_ci(labels, scores, _safe_aupr, bootstrap_iters, seed + 1)
    img_f1 = _bootstrap_ci(labels, scores, _max_f1, bootstrap_iters, seed + 2)
    image_line = "[Slice-Img]  " + "  ".join([_fmt("AUROC", img_auroc), _fmt("AUPR", img_aupr), _fmt("F1", img_f1)])
    print(image_line)
    print("Computing pixel-level Slice-Px(abn) metrics and 95CI...")
    print("Computing exact full-pixel point estimates with sklearn...")
    keep = np.asarray(labels) == 1
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
    print(pixel_line)
    return "\n".join([image_line, pixel_line])


class ResizeToTensorNormalize:
    def __init__(self, image_size):
        self.image_size = int(image_size)
        self.mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image):
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
        return (tensor - self.mean) / self.std


class DeepPSMAFlatDataset(Dataset):
    def __init__(self, data_root, modalities=("ct", "pet"), image_size=256):
        self.data_root = Path(data_root).expanduser().resolve()
        self.modalities = [m.lower() for m in modalities]
        self.image_size = int(image_size)
        self.transform = ResizeToTensorNormalize(image_size)
        self.samples = []
        self._load_samples()
        if not self.samples:
            raise FileNotFoundError(f"No valid DeepPSMA samples found under {self.data_root}")

    def _list_images(self, folder):
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)

    def _load_samples(self):
        for class_name, label in (("normal", 0), ("abnormal", 1)):
            class_dir = self.data_root / "test" / class_name
            base_dir = class_dir / self.modalities[0]
            for base_path in self._list_images(base_dir):
                paths = {mod: class_dir / mod / base_path.name for mod in self.modalities}
                if not all(path.is_file() for path in paths.values()):
                    continue
                mask_path = class_dir / "label" / base_path.name if label else None
                if label and not (mask_path and mask_path.is_file()):
                    continue
                self.samples.append({
                    "id": f"{class_name}__deeppsma1__{base_path.stem}",
                    "label": float(label),
                    "paths": paths,
                    "mask": mask_path,
                })

    def _read_gray(self, path, is_mask=False):
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        with Image.open(path) as img:
            img = img.convert("L").resize((self.image_size, self.image_size), resample=resample)
            arr = np.asarray(img, dtype=np.float32)
        if is_mask:
            return (arr > 127.5).astype(np.float32)
        return arr / 255.0

    def _load_and_fuse_modalities(self, sample):
        channels = [self._read_gray(sample["paths"][mod], is_mask=False) for mod in self.modalities]
        if len(channels) == 1:
            channels = channels * 3
        elif len(channels) == 2:
            fused = (channels[0] + channels[1]) / 2.0
            channels.append(fused)
        else:
            channels = channels[:3]
        return Image.fromarray((np.stack(channels, axis=-1) * 255).astype(np.uint8))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self.transform(self._load_and_fuse_modalities(sample))
        if sample["label"]:
            mask = self._read_gray(sample["mask"], is_mask=True)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        return image, torch.tensor(sample["label"], dtype=torch.float32), torch.from_numpy(mask).unsqueeze(0), sample["id"]


def load_checkpoint(model, checkpoint_path, device):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    print(f"Loaded checkpoint from {checkpoint_path}")


@torch.no_grad()
def evaluate_deeppsma(args, model, device):
    dataset = DeepPSMAFlatDataset(args.data_root, args.modalities, image_size=args.image_size)
    loader = make_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    filenames, labels, maps, masks = [], [], [], []
    for img, label, mask, filename in tqdm(loader, desc="AE cache", ncols=80):
        img = img.to(device)
        rec_img, _, _ = model(img)
        anomap = torch.abs(img - rec_img).mean(dim=1).detach().cpu().numpy().astype(np.float32)
        mask_np = mask.detach().cpu().numpy()
        if mask_np.ndim == 4:
            mask_np = mask_np.max(axis=1)
        maps.extend(list(anomap))
        masks.extend(list((mask_np > 0.5).astype(np.uint8)))
        labels.extend(label.detach().cpu().view(-1).numpy().astype(np.int64).tolist())
        filenames.extend(list(filename))

    maps = np.stack(maps).astype(np.float32)
    masks = np.stack(masks).astype(np.uint8)
    labels = np.asarray(labels, dtype=np.int64)
    scores = top_percent_mean(maps, args.topk_percent)
    print(f"Cache loaded: labels={labels.shape}, masks={masks.shape}, maps={maps.shape}, bootstrap_iters={args.bootstrap_iters}")
    return compute_slice_only_metrics(
        labels,
        masks,
        maps,
        scores,
        bootstrap_iters=args.bootstrap_iters,
        hist_bins=args.hist_bins,
        seed=args.ci_seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct DeepPSMA1 generalization evaluator for AE_Flow.")
    parser.add_argument("--data_root", default="/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1")
    parser.add_argument("--source_tracer", choices=["PSMA", "FDG", "psma", "fdg"], default="PSMA")
    parser.add_argument("--modality", default="ct,pet", help="ct, pet, or ct,pet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--subnet", default="conv_type", choices=["conv_type", "resnet_type"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--topk_percent", type=float, default=1.0)
    parser.add_argument("--bootstrap_iters", type=int, default=500)
    parser.add_argument("--ci_seed", type=int, default=42)
    parser.add_argument("--ci_pixel_max_samples", type=int, default=200000)
    parser.add_argument("--hist_bins", type=int, default=16384)
    args = parser.parse_args()

    args.modalities = parse_modalities(args.modality)
    args.tracer = args.source_tracer.upper()
    device = resolve_device(args.device)
    model = ae_flow.AE_FLOW(subnet=args.subnet).to(device)
    checkpoint = find_checkpoint(args, args.modalities)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    print(evaluate_deeppsma(args, model, device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
