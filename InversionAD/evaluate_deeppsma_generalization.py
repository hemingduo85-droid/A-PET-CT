#!/usr/bin/env python3
"""
DeepPSMA1 direct generalization test for InversionAD.

Data root:
  /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1

Run with existing PSMA weights:
  python evaluate_deeppsma_generalization.py --source_tracer PSMA --modality petct --gpu 6 > deeppsma_psma.log 2>&1 &

Run with existing FDG weights:
  python evaluate_deeppsma_generalization.py --source_tracer FDG --modality petct --gpu 7 > deeppsma_fdg.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


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
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.backbones import get_backbone, get_backbone_feature_shape
from src.config import resolve_petct_config
from src.denoiser import get_denoiser
from src.datasets import build_transforms
from src.evaluate import init_denoiser, top1pct_score


DEFAULT_DEEPPSMA_ROOT = "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
REPO_DIR = Path(__file__).resolve().parent


def modality_to_input_mode(modality):
    value = str(modality).lower().replace("+", ",")
    if value in {"petct", "dual", "ct,pet", "pet,ct"}:
        return "dual"
    if value in {"ct", "pet"}:
        return value
    raise ValueError("modality must be ct, pet, petct, dual, or ct,pet")


def default_config_for_mode(input_mode):
    if input_mode == "dual":
        return "configs/exp_dit_petct/petct_dual.yml"
    return f"configs/exp_dit_petct/petct_{input_mode}.yml"


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


def compute_slice_only_metrics(labels, masks, maps, scores, bootstrap_iters=500, hist_bins=16384, seed=42):
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
    def __init__(self, data_root, input_res, transform, input_mode="dual"):
        self.root = Path(data_root).expanduser().resolve()
        self.input_res = int(input_res)
        self.transform = transform
        self.input_mode = input_mode
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
        primary = "pet" if self.input_mode == "pet" else "ct"
        primary_dir = group_root / primary
        label_dir = group_root / "label"
        if not primary_dir.is_dir():
            return
        for path in sorted(p for p in primary_dir.iterdir() if self._is_image(p)):
            ct_path = group_root / "ct" / path.name
            pet_path = group_root / "pet" / path.name
            if self.input_mode == "dual" and not (self._is_image(ct_path) and self._is_image(pet_path)):
                self.skipped += 1
                continue
            if self.input_mode == "ct" and not self._is_image(ct_path):
                self.skipped += 1
                continue
            if self.input_mode == "pet" and not self._is_image(pet_path):
                self.skipped += 1
                continue

            mask_path = None
            if label:
                candidate = label_dir / path.name
                if not self._is_image(candidate):
                    self.skipped += 1
                    continue
                mask_path = candidate
            self.samples.append((ct_path, pet_path, int(label), mask_path, str(path)))

    def _collect(self):
        self._collect_group("normal", 0)
        self._collect_group("abnormal", 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_path, pet_path, label, mask_path, sample_id = self.samples[idx]
        ct_img = Image.open(ct_path).convert("L")
        pet_img = Image.open(pet_path).convert("L")
        if self.input_mode == "ct":
            image = Image.merge("RGB", (ct_img, ct_img, ct_img))
        elif self.input_mode == "pet":
            image = Image.merge("RGB", (pet_img, pet_img, pet_img))
        else:
            image = Image.merge("RGB", (ct_img, pet_img, pet_img))

        mask = Image.open(mask_path).convert("L") if mask_path is not None else Image.new("L", (self.input_res, self.input_res), 0)
        mask = mask.resize((self.input_res, self.input_res), resample=Image.NEAREST)
        return {
            "samples": self.transform(image),
            "clslabels": torch.tensor(0, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
            "masks": torch.from_numpy((np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)),
            "filenames": sample_id,
        }

    def distribution(self):
        normal = sum(1 for _, _, label, _, _ in self.samples if label == 0)
        abnormal = sum(1 for _, _, label, _, _ in self.samples if label == 1)
        return {"normal": normal, "abnormal": abnormal, "skipped_unpaired_or_unmasked": self.skipped}


def load_config(args, input_mode):
    fname = Path(args.fname or default_config_for_mode(input_mode))
    if not fname.is_absolute():
        fname = REPO_DIR / fname
    with fname.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    config["data"]["petct_dataset"] = args.source_tracer.lower()
    config["data"]["category"] = args.source_tracer.lower()
    config["data"]["input_mode"] = input_mode
    config = resolve_petct_config(config)
    config["data"]["data_root"] = str(Path(args.data_root).expanduser().resolve())
    config.setdefault("meta", {})["device"] = args.device_name
    return config, fname


def load_state_dict(path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


def resolve_checkpoint(args, save_dir):
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()
    if args.use_ema_model:
        return Path(save_dir) / "model_ema_latest.pth"
    if args.use_best_model and (Path(save_dir) / "model_best.pth").exists():
        return Path(save_dir) / "model_best.pth"
    return Path(save_dir) / "model_latest.pth"


@torch.no_grad()
def evaluate_deeppsma(denoiser, feature_extractor, eval_denoiser, loader, device, args):
    labels, masks, maps = [], [], []
    for batch in tqdm(loader, desc="DeepPSMA eval", leave=True):
        images = batch["samples"].to(device)
        cls_labels = batch["clslabels"].to(device)
        features, _ = feature_extractor(images)
        start_t = torch.zeros(images.shape[0], device=device, dtype=torch.long)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            latents_last = eval_denoiser.ddim_reverse_sample(
                features,
                start_t,
                cls_labels,
                eta=0.0,
            )
        latents_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
        batch_maps = F.interpolate(
            latents_l2.unsqueeze(0),
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).detach().cpu().numpy().astype(np.float32)
        maps.extend(batch_maps)
        masks.extend(batch["masks"].detach().cpu().numpy().astype(np.uint8))
        labels.extend(batch["labels"].detach().cpu().view(-1).numpy().astype(np.int32).tolist())

    maps_np = np.stack(maps).astype(np.float32)
    masks_np = np.stack(masks).astype(np.uint8)
    labels_np = np.asarray(labels, dtype=np.int32)
    scores = np.asarray([top1pct_score(m) for m in maps_np], dtype=np.float64)
    return compute_slice_only_metrics(
        labels_np,
        masks_np,
        maps_np,
        scores,
        bootstrap_iters=args.bootstrap_iters,
        hist_bins=args.ci_hist_bins,
        seed=args.seed,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="InversionAD direct DeepPSMA1 generalization evaluator")
    parser.add_argument("--data_root", default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--source_tracer", default="PSMA", choices=["PSMA", "FDG", "psma", "fdg"])
    parser.add_argument("--modality", default="petct", help="ct, pet, petct, dual, or ct,pet")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fname", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval_step", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema_model", action="store_true")
    parser.add_argument("--use_best_model", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--bootstrap_iters", type=int, default=500)
    parser.add_argument("--ci_hist_bins", type=int, default=16384)
    return parser


def main():
    args = build_parser().parse_args()
    args.source_tracer = args.source_tracer.upper()
    input_mode = modality_to_input_mode(args.modality)
    args.device_name = args.device or ("cpu" if str(args.gpu).lower() in {"cpu", "-1", "none"} else "cuda:0")
    device = torch.device(args.device_name if torch.cuda.is_available() or args.device_name == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config, config_path = load_config(args, input_mode)
    if args.save_dir is None:
        args.save_dir = config.get("logging", {}).get("save_dir")
    if args.save_dir is None:
        raise ValueError("Provide --save_dir or set logging.save_dir in the config.")

    batch_size = args.batch_size or int(config["data"].get("batch_size", 8))
    num_workers = args.num_workers if args.num_workers is not None else int(config["data"].get("num_workers", 4))
    transform = build_transforms(config["data"]["img_size"], config["data"]["transform_type"])
    dataset = DeepPSMAFlatDataset(args.data_root, config["data"]["img_size"], transform, input_mode=input_mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    print(f"DeepPSMA root: {Path(args.data_root).expanduser().resolve()}", flush=True)
    print(f"Source weights: {args.source_tracer}", flush=True)
    print(f"Input mode: {input_mode}", flush=True)
    print(f"Distribution: {dataset.distribution()}", flush=True)
    print(f"Device: {device}", flush=True)

    in_shape = get_backbone_feature_shape(model_type=config["backbone"]["model_type"])
    denoiser = get_denoiser(**config["diffusion"], input_shape=in_shape).to(device).eval()
    feature_extractor = get_backbone(**config["backbone"]).to(device).eval()

    checkpoint = resolve_checkpoint(args, args.save_dir)
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    state = load_state_dict(checkpoint, device)
    if state and "module." in next(iter(state.keys())):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    denoiser.load_state_dict(state, strict=True)
    eval_denoiser = init_denoiser(args.eval_step, device, config, in_shape, inherit_model=denoiser)

    formatted = evaluate_deeppsma(denoiser, feature_extractor, eval_denoiser, loader, device, args)
    out_dir = REPO_DIR / "results" / "deeppsma1_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.source_tracer}_{input_mode}_slice_only_metrics.txt"
    out_path.write_text(
        "\n".join([
            f"config={config_path}",
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
