"""
Run examples:
  python evaluate_deeppsma_generalization.py --source_tracer PSMA --modality petct --gpu 0
  python evaluate_deeppsma_generalization.py --source_tracer FDG --modality petct --gpu 0
  python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer PSMA --modality petct --gpu 0

 cd /data/cyf/codes/A-PET-CT/GatingAno

python evaluate_deeppsma_generalization.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 \
  --source_tracer PSMA \
  --modality petct \
  --dual_input_mode ctctpet \
  --gpu 5 \
  --checkpoint /data/cyf/codes/A-PET-CT/GatingAno/checkpoints/ctctpet/PSMA_petct_epoch30.pth \
  --bootstrap_iters 500 \
  --hist_bins 16384 

python evaluate_deeppsma_generalization.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 \
  --source_tracer FDG \
  --modality petct \
  --dual_input_mode ctctpet \
  --gpu 7 \
  --checkpoint /data/cyf/codes/A-PET-CT/GatingAno/checkpoints/ctctpetfdg30/FDG_petct_epoch30.pth \
  --bootstrap_iters 500 \
  --hist_bins 16384 
  """

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

REPO_DIR = Path(__file__).resolve().parent
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_DEEPPSMA_ROOT = "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1"


class DeepPSMAFlatDataset(Dataset):
    """Direct reader for deeppsma1/test/{normal,abnormal}/{pet,ct,label}."""

    def __init__(
        self,
        data_root: str | Path,
        modality: str,
        transform,
        image_size: int,
        dual_input_mode: str = "pseudo_rgb",
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.modality = modality.lower()
        self.transform = transform
        self.image_size = int(image_size)
        self.dual_input_mode = dual_input_mode
        if self.modality not in {"ct", "pet", "petct"}:
            raise ValueError("--modality must be ct, pet, or petct")
        if self.modality == "petct" and self.dual_input_mode not in {"pseudo_rgb", "ctctpet"}:
            raise ValueError("--dual_input_mode must be pseudo_rgb or ctctpet")
        self.samples = []
        self.skipped_unpaired_or_unmasked = 0
        self._load_samples()
        if not self.samples:
            raise FileNotFoundError(f"No valid DeepPSMA samples found under {self.data_root}")

    def _list_images(self, folder: Path):
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)

    def _load_samples(self):
        normal_count = 0
        abnormal_count = 0
        primary = "pet" if self.modality in {"pet", "petct"} else "ct"
        for class_name, label in (("normal", 0), ("abnormal", 1)):
            class_dir = self.data_root / "test" / class_name
            for primary_path in self._list_images(class_dir / primary):
                paths = {
                    "pet": class_dir / "pet" / primary_path.name,
                    "ct": class_dir / "ct" / primary_path.name,
                }
                needed = ("pet", "ct") if self.modality == "petct" else (self.modality,)
                if not all(paths[mod].is_file() for mod in needed):
                    self.skipped_unpaired_or_unmasked += 1
                    continue
                mask_path = class_dir / "label" / primary_path.name if label else None
                if label and not (mask_path and mask_path.is_file()):
                    self.skipped_unpaired_or_unmasked += 1
                    continue
                self.samples.append({
                    "id": f"{class_name}__{primary_path.stem}__{primary_path.stem}",
                    "label": label,
                    "paths": paths,
                    "mask": mask_path,
                    "name": primary_path.name,
                    "stem": primary_path.stem,
                    "class_name": class_name,
                })
                normal_count += int(label == 0)
                abnormal_count += int(label == 1)
        print(
            f"Test [{self.modality.upper()}]: {normal_count} normal + "
            f"{abnormal_count} abnormal = {len(self.samples)} total | "
            f"skipped_unpaired_or_unmasked={self.skipped_unpaired_or_unmasked}"
        )

    def _read_gray(self, path: Path, is_mask: bool = False):
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        with Image.open(path) as img:
            img = img.convert("L").resize((self.image_size, self.image_size), resample=resample)
            arr = np.asarray(img, dtype=np.float32)
        if is_mask:
            return (arr > 127.5).astype(np.float32)
        return arr.astype(np.uint8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.modality == "petct":
            ct = self._read_gray(sample["paths"]["ct"], is_mask=False)
            pet = self._read_gray(sample["paths"]["pet"], is_mask=False)
            if self.dual_input_mode == "ctctpet":
                image = Image.fromarray(np.stack([ct, ct, pet], axis=-1), mode="RGB")
            else:
                image = Image.fromarray(np.stack([ct, pet, pet], axis=-1), mode="RGB")
        else:
            gray = self._read_gray(sample["paths"][self.modality], is_mask=False)
            image = Image.fromarray(gray, mode="L").convert("RGB")

        image_tensor = self.transform(image)
        if sample["label"]:
            mask = self._read_gray(sample["mask"], is_mask=True)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        metric_path = str(Path("__deeppsma_flat__") / sample["class_name"] / self.modality / sample["name"])
        return (
            image_tensor,
            torch.tensor(float(sample["label"]), dtype=torch.float32),
            torch.from_numpy(mask).unsqueeze(0),
            metric_path,
        )


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score):
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
    auroc = float(np.trapz(np.r_[0.0, tpr], np.r_[0.0, fpr]))
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
    scale = (int(bins) - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), int(bins)), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), int(bins)), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        scores = maps[idx].reshape(-1)
        mask = masks[idx].reshape(-1).astype(bool)
        bin_idx = np.floor((scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, int(bins) - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[mask], minlength=int(bins)).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~mask], minlength=int(bins)).astype(np.uint32)

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


def compute_cache_metrics(cache_path, bootstrap_iters=500, seed=20260717, hist_bins=16384):
    print(f"Loading eval cache from {cache_path}", flush=True)
    cache = np.load(cache_path, allow_pickle=True)
    labels = cache["labels"].astype(np.int32)
    image_scores = cache["image_scores"].astype(np.float64)
    paths = cache["paths"]
    masks = cache["masks"]
    maps = cache["maps"]
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)
    print(
        f"Cache loaded: labels={labels.shape}, masks={masks.shape}, "
        f"maps={maps.shape}, bootstrap_iters={int(bootstrap_iters)}",
        flush=True,
    )

    print("Computing image-level metrics and 95CI...", flush=True)
    img_auroc = _bootstrap_ci(labels, image_scores, _safe_auroc, bootstrap_iters, seed)
    img_ap = _bootstrap_ci(labels, image_scores, _safe_ap, bootstrap_iters, seed + 1)
    img_f1 = _bootstrap_ci(labels, image_scores, _max_f1, bootstrap_iters, seed + 2)
    image_line = "[Slice-Img]  " + "  ".join([_fmt("AUROC", img_auroc), _fmt("AUPR", img_ap), _fmt("F1", img_f1)])
    print(image_line, flush=True)

    print("Computing pixel-level Slice-Px(abn) metrics and 95CI...", flush=True)
    print("Computing exact full-pixel point estimates with sklearn...", flush=True)
    keep = labels == 1
    px_true = masks[keep].reshape(-1).astype(np.int32)
    px_score = maps[keep].reshape(-1).astype(np.float64)
    lesion_prevalence = float(px_true.mean()) if px_true.size else 0.0
    print(
        f"Pixel AUPR random baseline (lesion-pixel prevalence): "
        f"{lesion_prevalence * 100:.4f}%",
        flush=True,
    )
    exact_auroc = _safe_auroc(px_true, px_score)
    exact_aupr = _safe_ap(px_true, px_score)
    px_auroc, px_aupr = _pixel_slice_bootstrap(
        labels,
        masks,
        maps,
        bootstrap_iters,
        seed + 10,
        hist_bins,
        exact_auroc,
        exact_aupr,
    )
    pixel_line = "[Slice-Px(abn)]  " + "  ".join([_fmt("AUROC", px_auroc), _fmt("AUPR", px_aupr)])
    print(pixel_line, flush=True)

    text = "\n".join([image_line, pixel_line])
    print(text, flush=True)
    return text


def load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct DeepPSMA1 generalization evaluator for GatingAno.")
    parser.add_argument("--data_root", default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--source_tracer", choices=["PSMA", "FDG", "psma", "fdg"], default="PSMA")
    parser.add_argument("--modality", choices=["ct", "pet", "petct"], default="petct")
    parser.add_argument(
        "--dual_input_mode",
        choices=["pseudo_rgb", "ctctpet"],
        default="pseudo_rgb",
        help="petct input: pseudo_rgb=[CT,PET,PET], ctctpet=[CT,CT,PET]",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--metrics_path", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--bootstrap_iters", type=int, default=500)
    parser.add_argument("--ci_seed", type=int, default=20260717)
    parser.add_argument("--hist_bins", type=int, default=16384)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from models import GatingAno
    from train import Config, evaluate, validate_checkpoint_metadata

    source = args.source_tracer.upper()
    config = Config(
        modality=args.modality,
        gpu=str(args.gpu),
        dataset=source,
        data_root=str(Path(args.data_root).expanduser().resolve()),
        dual_input_mode=args.dual_input_mode,
        bootstrap_iters=args.bootstrap_iters,
    )
    config.dataset_name = "deeppsma1"
    mode_suffix = "" if args.dual_input_mode == "pseudo_rgb" else f"_{args.dual_input_mode}"
    config.save_dir = str(
        REPO_DIR / "saved_results" / f"deeppsma1_{source.lower()}_{args.modality}{mode_suffix}"
    )
    config.heatmap_dir = str(Path(config.save_dir) / "heatmaps")

    cache_path = Path(args.cache_path) if args.cache_path else Path(config.save_dir) / f"{source}_{args.modality}_eval_cache.npz"
    metrics_path = Path(args.metrics_path) if args.metrics_path else Path(config.save_dir) / f"{source}_{args.modality}_metrics.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = DeepPSMAFlatDataset(
        args.data_root,
        args.modality,
        transform,
        config.image_size,
        dual_input_mode=args.dual_input_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    generator = GatingAno(
        n_channels=config.input_channels,
        n_classes=config.output_channels,
    ).to(config.device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else REPO_DIR / "checkpoints" / f"{source}_{args.modality}_epoch30.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = load_checkpoint(checkpoint, config.device)
    validate_checkpoint_metadata(state, source, args.modality, config)
    generator.load_state_dict(state["state_dict"])
    if isinstance(state, dict) and "alpha" in state:
        config.alpha = state["alpha"]

    evaluate(
        epoch="deeppsma1",
        config=config,
        generator=generator,
        test_loader=loader,
        save_heatmaps=False,
        cache_path=str(cache_path),
        compute_ci=False,
    )
    text = compute_cache_metrics(
        cache_path=str(cache_path),
        bootstrap_iters=args.bootstrap_iters,
        seed=args.ci_seed,
        hist_bins=args.hist_bins,
    )
    metrics_path.write_text(text + "\n", encoding="utf-8")
    print(f"Metrics saved to: {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
