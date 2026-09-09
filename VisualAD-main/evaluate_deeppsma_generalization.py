#!/usr/bin/env python3
"""
DeepPSMA1 direct generalization test for VisualAD.

Data root:
  /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1

Run with existing PSMA weights:
  python evaluate_deeppsma_generalization.py --source_tracer PSMA --modality petct --gpu 0

Run with existing FDG weights:
  python evaluate_deeppsma_generalization.py --source_tracer FDG --modality petct --gpu 4
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import VisualAD_lib
from utils.anomaly_detection import generate_anomaly_map_from_tokens
from utils.feature_transform import create_feature_transform
from utils.petct_config import resolve_checkpoint_path
from utils.scoring import DEFAULT_TOPK_RATIO, reduce_anomaly_map
from utils.transforms import get_transform


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
    def __init__(self, data_root, transform, target_transform, modality="petct"):
        self.root = Path(data_root).expanduser().resolve()
        self.transform = transform
        self.target_transform = target_transform
        self.modality = modality
        self.samples = []
        self.skipped = 0
        self._collect()
        if not self.samples:
            raise RuntimeError(
                f"No DeepPSMA samples found under {self.root}. "
                "Expected test/{normal,abnormal}/{pet,ct,label}/*.png"
            )
        self.obj_list = ["petct"]
        self.class_name_map_class_id = {"petct": 0}

    @staticmethod
    def _is_image(path):
        return path.is_file() and path.suffix.lower() in IMG_EXTS and path.stat().st_size > 0

    def _collect_group(self, group, label):
        group_root = self.root / "test" / group
        primary = "pet" if self.modality in {"pet", "petct"} else "ct"
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
            if self.modality == "ct" and not self._is_image(ct_path):
                self.skipped += 1
                continue
            if self.modality == "pet" and not self._is_image(pet_path):
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

    def _build_image(self, pet_img, ct_img):
        if self.modality == "pet":
            return Image.merge("RGB", [pet_img, pet_img, pet_img])
        if self.modality == "ct":
            return Image.merge("RGB", [ct_img, ct_img, ct_img])
        return Image.merge("RGB", [ct_img, pet_img, pet_img])

    def __getitem__(self, idx):
        ct_path, pet_path, label, mask_path, sample_id = self.samples[idx]
        pet = Image.open(pet_path).convert("L")
        ct = Image.open(ct_path).convert("L")
        if pet.size != ct.size:
            ct = ct.resize(pet.size, Image.BILINEAR)
        image = self._build_image(pet, ct)

        if mask_path is not None:
            mask_arr = np.array(Image.open(mask_path).convert("L")) > 0
            mask = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L")
        else:
            mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8), mode="L")

        return {
            "img": self.transform(image),
            "img_mask": self.target_transform(mask),
            "anomaly": torch.tensor(label, dtype=torch.long),
            "cls_name": "petct",
            "img_path": sample_id,
            "cls_id": 0,
        }

    def distribution(self):
        normal = sum(1 for _, _, label, _, _ in self.samples if label == 0)
        abnormal = sum(1 for _, _, label, _, _ in self.samples if label == 1)
        return {"normal": normal, "abnormal": abnormal, "skipped_unpaired_or_unmasked": self.skipped}


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_visualad_from_checkpoint(args, device):
    checkpoint = load_checkpoint(args.checkpoint_path, device)
    args.backbone = checkpoint.get("backbone", "ViT-L/14@336px")
    args.image_size = checkpoint.get("image_size", args.image_size)
    args.features_list = checkpoint.get("features_list", [6, 12, 18, 24])

    model, _ = VisualAD_lib.load(args.backbone, device=device)
    model.eval()
    model.to(device)

    feature_dim = model.visual.embed_dim
    model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
    model.visual.normal_token.data = checkpoint["normal_token"].to(device)
    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None and checkpoint.get("ln_post_weight") is not None:
        ln_post.weight.data = checkpoint["ln_post_weight"].to(device)
        ln_post.bias.data = checkpoint["ln_post_bias"].to(device)

    layer_transforms = nn.ModuleDict()
    if "layer_transforms" in checkpoint:
        for layer_name, state_dict in checkpoint["layer_transforms"].items():
            hidden_dim = state_dict["mlp.0.weight"].shape[0]
            layer_transforms[layer_name] = create_feature_transform(
                transform_type="mlp",
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
                dropout=0.0,
            ).to(device)
            layer_transforms[layer_name].load_state_dict(state_dict)
            layer_transforms[layer_name].eval()

    cross_attn = None
    if "cross_attn" in checkpoint:
        from utils.spatial_cross_attention import build_layer_adaptive_cross_attention

        config = checkpoint.get("cross_attn_config", {})
        cross_attn = build_layer_adaptive_cross_attention(
            layers=args.features_list,
            embed_dim=feature_dim,
            num_anchors=config.get("num_anchors", 4),
            dropout=config.get("dropout", 0.1),
            res_scale_init=config.get("res_scale_init", 0.01),
        ).to(device)
        cross_attn.load_state_dict(checkpoint["cross_attn"])
        cross_attn.eval()

    return model, layer_transforms, cross_attn


@torch.no_grad()
def evaluate_deeppsma(args, model, layer_transforms, cross_attn, loader, device):
    labels, masks, maps, scores = [], [], [], []
    for items in tqdm(loader, desc="DeepPSMA eval", leave=True):
        image = items["img"].to(device)
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        vision_output = model.encode_image(image, args.features_list)
        anomaly_features = vision_output["anomaly_features"]
        normal_features = vision_output["normal_features"]
        patch_tokens = vision_output["patch_tokens"]
        patch_start_idx = vision_output["patch_start_idx"]
        patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]

        if cross_attn is not None:
            adapted_list = cross_attn(anomaly_features, normal_features, patch_features_list, args.features_list)
            anomaly_features_list = [a["anomaly"] for a in adapted_list]
            normal_features_list = [a["normal"] for a in adapted_list]
        else:
            anomaly_features_list = [anomaly_features] * len(patch_tokens)
            normal_features_list = [normal_features] * len(patch_tokens)

        anomaly_map_list = []
        for idx, patch_feature in enumerate(patch_tokens):
            af_norm = F.normalize(anomaly_features_list[idx], dim=1, eps=1e-8)
            nf_norm = F.normalize(normal_features_list[idx], dim=1, eps=1e-8)
            tk = f"layer_{args.features_list[idx]}"
            if tk in layer_transforms:
                batch, tokens, dim = patch_feature.shape
                patch_feature = layer_transforms[tk](patch_feature.view(-1, dim)).view(batch, tokens, dim)
            anomaly_map = generate_anomaly_map_from_tokens(
                af_norm,
                nf_norm,
                patch_feature[:, patch_start_idx:, :],
                args.image_size,
            )
            anomaly_map_list.append(anomaly_map)

        final_map = torch.stack(anomaly_map_list).sum(dim=0).cpu()
        filtered = np.stack([gaussian_filter(final_map[i].numpy(), sigma=args.sigma) for i in range(final_map.shape[0])])
        final_map = torch.from_numpy(filtered).float()
        score = reduce_anomaly_map(final_map, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO)

        maps.extend(final_map.numpy().astype(np.float32))
        masks.extend(gt_mask[:, 0].cpu().numpy().astype(np.uint8))
        labels.extend(items["anomaly"].detach().cpu().view(-1).numpy().astype(np.int32).tolist())
        scores.extend(score.detach().cpu().view(-1).numpy().astype(np.float64).tolist())

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
    parser = argparse.ArgumentParser(description="VisualAD direct DeepPSMA1 generalization evaluator")
    parser.add_argument("--data_root", default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--source_tracer", default="PSMA", choices=["PSMA", "FDG", "psma", "fdg"])
    parser.add_argument("--modality", default="petct", choices=["ct", "pet", "petct"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_root", default="./experiments")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--save_path", default="./test_results_deeppsma1")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--bootstrap_iters", type=int, default=500)
    parser.add_argument("--ci_seed", type=int, default=42)
    parser.add_argument("--hist_bins", type=int, default=16384)
    return parser


def main():
    args = build_parser().parse_args()
    args.source_tracer = args.source_tracer.upper()
    dataset = args.source_tracer.lower()
    device_name = args.device or ("cpu" if str(args.gpu).lower() in {"cpu", "-1", "none"} else "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    args.checkpoint_path = resolve_checkpoint_path(
        args.checkpoint,
        args.checkpoint_root,
        dataset,
        args.modality,
        args.epoch,
    )
    print(f"DeepPSMA root: {Path(args.data_root).expanduser().resolve()}", flush=True)
    print(f"Source weights: {args.source_tracer}", flush=True)
    print(f"Modality: {args.modality}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Loading checkpoint: {args.checkpoint_path}", flush=True)

    model, layer_transforms, cross_attn = build_visualad_from_checkpoint(args, device)
    preprocess, target_transform = get_transform(args)
    test_data = DeepPSMAFlatDataset(args.data_root, preprocess, target_transform, modality=args.modality)
    print(f"Distribution: {test_data.distribution()}", flush=True)
    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    formatted = evaluate_deeppsma(args, model, layer_transforms, cross_attn, loader, device)

    out_dir = REPO_DIR / args.save_path / dataset / args.modality
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "deeppsma1_slice_only_metrics.txt"
    out_path.write_text(
        "\n".join([
            f"checkpoint={args.checkpoint_path}",
            f"data_root={Path(args.data_root).expanduser().resolve()}",
            formatted,
            "",
        ]),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
