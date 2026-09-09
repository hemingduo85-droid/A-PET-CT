#!/usr/bin/env python3
"""
Run example:
cd /data/cyf/codes/A-PET-CT/VisualAD-main
python export_fixed_lesion_case_figure.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 \
  --modality petct \
  --device cuda:4 \
  --checkpoint_root experiments \
  --epoch 30 \
  --output_dir fixed_lesion_case_figures
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import VisualAD_lib
from utils.anomaly_detection import generate_anomaly_map_from_tokens
from utils.feature_transform import create_feature_transform
from utils.petct_config import resolve_checkpoint_path
from utils.transforms import get_transform


METHOD = "VisualAD"
PROJECT_DIR = Path(__file__).resolve().parent
GROUPS = ("small", "medium", "large")
TRACERS = ("PSMA", "FDG")
PANEL_NAMES = ("PET", "CT", "GT", "Pred", "VisualAD")

FIXED_CASES = {
     "small": {
           "PSMA": ("psma_1ca3af29b7df127d_2020-06-27", "0009"),
           "FDG": ("fdg_cd2ef932b5_04-27-2003-NA-PET-CT Ganzkoerper  primaer mit KM-12737", "0035"),
       },
       "medium": {
           "PSMA": ("psma_ec45934c2fa23c76_2019-08-05", "0027"),
           "FDG": ("fdg_2b60c8135a_12-09-2005-NA-PET-CT Ganzkoerper  primaer mit KM-49696", "0033"),
       },
       "large": {
           "PSMA": ("psma_18eba3b35ee1ddac_2020-09-05", "0036"),
           "FDG": ("fdg_ad7cd4a9d2_10-02-2003-NA-Unspecified CT ABDOMEN-67897", "0020"),
       },
}



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    for candidate in (PROJECT_DIR / path, PROJECT_DIR.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_DIR / path).resolve()


def tracer_root(data_root: Path, tracer: str) -> Path:
    return data_root / tracer if (data_root / tracer / "test").exists() else data_root


def case_paths(data_root: Path, tracer: str, patient: str, slice_id: str) -> dict[str, Path]:
    base = tracer_root(data_root, tracer) / "test" / "abnormal" / patient
    return {
        "pet": base / "pet" / f"{slice_id}.png",
        "ct": base / "ct" / f"{slice_id}.png",
        "mask": base / "label" / f"{slice_id}.png",
    }


def check_paths(paths):
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing fixed case files:\n" + "\n".join(missing))


def normalize01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.5])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def read_gray(path: Path, image_size: int, is_mask=False):
    image = Image.open(path).convert("L")
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    image = image.resize((image_size, image_size), resample=resample)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.uint8) if is_mask else arr


def colorize(arr, cmap_name="magma"):
    import matplotlib.cm as cm

    rgb = cm.get_cmap(cmap_name)(normalize01(arr))[..., :3]
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def gray_image(arr):
    return Image.fromarray((normalize01(arr) * 255).astype(np.uint8)).convert("RGB")


def mask_image(mask, color=(0, 255, 80)):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask > 0] = np.array(color, dtype=np.uint8)
    return Image.fromarray(rgb).convert("RGB")


def mask_outline(mask, size, color=(0, 255, 80, 255), width=2):
    gt = np.squeeze(np.asarray(mask)) > 0.5
    m = Image.fromarray((gt.astype(np.uint8) * 255)).resize(size, resample=Image.Resampling.NEAREST)
    arr = np.asarray(m) > 0
    edge = arr.copy()
    for _ in range(max(1, int(width))):
        inner = edge.copy()
        inner[1:-1, 1:-1] = arr[1:-1, 1:-1] & (
            arr[:-2, 1:-1] & arr[2:, 1:-1] & arr[1:-1, :-2] & arr[1:-1, 2:]
        )
        edge = arr & ~inner
        arr = inner
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    rgba[edge] = np.array(color, dtype=np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def overlay_gt(base_img, gt_mask):
    return Image.alpha_composite(base_img.convert("RGBA"), mask_outline(gt_mask, base_img.size)).convert("RGB")


def save_image(path: Path, image, size: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((size, size), resample=Image.Resampling.LANCZOS).save(path)


def best_f1_threshold(gt_mask, score_map):
    from sklearn.metrics import precision_recall_curve

    y_true = (np.asarray(gt_mask).reshape(-1) > 0).astype(np.uint8)
    y_score = np.asarray(score_map, dtype=np.float32).reshape(-1)
    if y_true.sum() == 0:
        return 0.5, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    f1 = f1[:-1]
    if len(f1) == 0:
        return 0.5, 0.0
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def build_visualad_image(paths, preprocess, image_size, modality):
    pet = Image.open(paths["pet"]).convert("L")
    ct = Image.open(paths["ct"]).convert("L")
    if pet.size != ct.size:
        ct = ct.resize(pet.size, Image.Resampling.BILINEAR)
    if modality == "pet":
        image = Image.merge("RGB", [pet, pet, pet])
    elif modality == "ct":
        image = Image.merge("RGB", [ct, ct, ct])
    else:
        image = Image.merge("RGB", [ct, pet, pet])
    arrays = {"pet": read_gray(paths["pet"], image_size), "ct": read_gray(paths["ct"], image_size)}
    return preprocess(image).unsqueeze(0), arrays


def load_model_for_tracer(args, tracer, device):
    ckpt_arg = args.psma_checkpoint if tracer == "PSMA" else args.fdg_checkpoint
    ckpt_path = resolve_checkpoint_path(
        ckpt_arg,
        str(resolve_path(args.checkpoint_root)),
        tracer.lower(),
        args.modality,
        args.epoch,
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    ns = argparse.Namespace(**vars(args))
    ns.backbone = checkpoint.get("backbone", "ViT-L/14@336px")
    ns.image_size = checkpoint.get("image_size", args.image_size)
    ns.features_list = checkpoint.get("features_list", [6, 12, 18, 24])
    preprocess, _ = get_transform(ns)

    model, _ = VisualAD_lib.load(ns.backbone, device=device)
    model.eval().to(device)
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
            layers=ns.features_list,
            embed_dim=feature_dim,
            num_anchors=config.get("num_anchors", 4),
            dropout=config.get("dropout", 0.1),
            res_scale_init=config.get("res_scale_init", 0.01),
        ).to(device)
        cross_attn.load_state_dict(checkpoint["cross_attn"])
        cross_attn.eval()

    print(f"Loaded {tracer} checkpoint: {ckpt_path}")
    return {
        "model": model,
        "layer_transforms": layer_transforms,
        "cross_attn": cross_attn,
        "preprocess": preprocess,
        "image_size": int(ns.image_size),
        "features_list": list(ns.features_list),
        "checkpoint": str(ckpt_path),
    }


def infer_raw_map(bundle, image, device, sigma):
    from scipy.ndimage import gaussian_filter

    model = bundle["model"]
    layer_transforms = bundle["layer_transforms"]
    cross_attn = bundle["cross_attn"]
    features_list = bundle["features_list"]
    with torch.no_grad():
        vision_output = model.encode_image(image, features_list)
        anomaly_features = vision_output["anomaly_features"]
        normal_features = vision_output["normal_features"]
        patch_tokens = vision_output["patch_tokens"]
        patch_start_idx = vision_output["patch_start_idx"]
        patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
        if cross_attn is not None:
            adapted_list = cross_attn(anomaly_features, normal_features, patch_features_list, features_list)
            anomaly_features_list = [a["anomaly"] for a in adapted_list]
            normal_features_list = [a["normal"] for a in adapted_list]
        else:
            anomaly_features_list = [anomaly_features] * len(patch_tokens)
            normal_features_list = [normal_features] * len(patch_tokens)

        anomaly_map_list = []
        for layer_idx, patch_feature in enumerate(patch_tokens):
            af_norm = F.normalize(anomaly_features_list[layer_idx], dim=1, eps=1e-8)
            nf_norm = F.normalize(normal_features_list[layer_idx], dim=1, eps=1e-8)
            layer_key = f"layer_{features_list[layer_idx]}"
            if layer_key in layer_transforms:
                bsz, num_tokens, dim = patch_feature.shape
                patch_feature = layer_transforms[layer_key](patch_feature.view(-1, dim)).view(bsz, num_tokens, dim)
            anomaly_map_list.append(
                generate_anomaly_map_from_tokens(
                    af_norm,
                    nf_norm,
                    patch_feature[:, patch_start_idx:, :],
                    bundle["image_size"],
                )
            )
        final_map = torch.stack(anomaly_map_list).sum(dim=0).cpu()
        return gaussian_filter(final_map[0].numpy(), sigma=sigma).astype(np.float32)


def render_case(args, bundle, device, data_root, group, tracer, patient, slice_id):
    paths = case_paths(data_root, tracer, patient, slice_id)
    check_paths(paths)
    image, arrays = build_visualad_image(paths, bundle["preprocess"], bundle["image_size"], args.modality)
    raw_map = infer_raw_map(bundle, image.to(device), device, args.sigma)
    mask = read_gray(paths["mask"], raw_map.shape[0], is_mask=True)
    threshold, best_f1 = best_f1_threshold(mask, raw_map)
    pred = (raw_map >= threshold).astype(np.uint8)
    case_name = f"{group}_{tracer}_{patient}__{slice_id}"
    case_dir = args.output_dir / "cases" / group / case_name
    pet_img = colorize(arrays["pet"], "magma")
    ct_img = gray_image(arrays["ct"])
    panels = {
        "PET": pet_img,
        "CT": ct_img,
        "GT": mask_image(mask),
        "Pred": mask_image(pred, color=(255, 128, 0)),
        "VisualAD": colorize(raw_map, "jet"),
    }
    save_image(case_dir / "PET_enhanced.png", panels["PET"], args.image_size)
    save_image(case_dir / "CT.png", panels["CT"], args.image_size)
    save_image(case_dir / "GT.png", panels["GT"], args.image_size)
    save_image(case_dir / "GT_on_PET.png", overlay_gt(pet_img, mask), args.image_size)
    save_image(case_dir / "GT_on_CT.png", overlay_gt(ct_img, mask), args.image_size)
    save_image(case_dir / "Pred.png", panels["Pred"], args.image_size)
    save_image(case_dir / "VisualAD_heatmap.png", panels["VisualAD"], args.image_size)
    meta = {
        "method": METHOD,
        "group": group,
        "tracer": tracer,
        "patient": patient,
        "slice": slice_id,
        "score": float(np.max(raw_map)),
        "score_mode": "max pixel score",
        "threshold": threshold,
        "best_f1": best_f1,
        "checkpoint": bundle["checkpoint"],
        "case_dir": str(case_dir),
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "case_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"meta": meta, "panels": panels}


def load_font(size, bold=False):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if path and Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def make_plate(records, save_path: Path, title: str):
    tile, gap, left, top, header_h, meta_h = 160, 8, 18, 46, 30, 26
    row_h = meta_h + tile + gap
    width = left * 2 + len(PANEL_NAMES) * tile + (len(PANEL_NAMES) - 1) * gap
    height = top + header_h + len(records) * row_h + 10
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 12), title, font=load_font(22, True), fill=(0, 0, 0), anchor="ma")
    y = top
    for col, name in enumerate(PANEL_NAMES):
        draw.text((left + col * (tile + gap) + 4, y), name, font=load_font(15, True), fill=(0, 0, 0))
    y += header_h
    for row_idx, record in enumerate(records, 1):
        meta = record["meta"]
        meta_text = (
            f"{meta['group'].capitalize()} #{row_idx} | score={meta['score']:.4f} | "
            f"F1={meta['best_f1']:.3f} | {meta['tracer']} | {meta['patient']}__{meta['slice']}"
        )
        draw.rectangle([left - 4, y, width - left + 4, y + meta_h - 2], fill=(245, 245, 245))
        draw.text((left, y + 4), meta_text, font=load_font(13), fill=(0, 0, 0))
        y += meta_h
        for col, name in enumerate(PANEL_NAMES):
            x = left + col * (tile + gap)
            canvas.paste(record["panels"][name].resize((tile, tile), resample=Image.Resampling.LANCZOS), (x, y))
            draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(255, 255, 255), width=2)
        y += tile + gap
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
    print(f"Saved {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/data/cyf/shared_data/A-PETCT/2d_equal_mask50")
    parser.add_argument("--modality", default="petct", choices=["pet", "ct", "petct"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint_root", default="experiments")
    parser.add_argument("--psma_checkpoint", default=None)
    parser.add_argument("--fdg_checkpoint", default=None)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "fixed_lesion_case_figures")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    args.output_dir = resolve_path(args.output_dir)
    data_root = resolve_path(args.data_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundles = {tracer: load_model_for_tracer(args, tracer, device) for tracer in TRACERS}
    all_records = []
    for group in GROUPS:
        group_records = []
        for tracer in TRACERS:
            patient, slice_id = FIXED_CASES[group][tracer]
            record = render_case(args, bundles[tracer], device, data_root, group, tracer, patient, slice_id)
            group_records.append(record)
            all_records.append(record)
        make_plate(group_records, args.output_dir / f"{group}.png", f"VisualAD {group} lesion fixed cases")
    make_plate(all_records, args.output_dir / "all_fixed_lesion_cases.png", "VisualAD fixed lesion cases")


if __name__ == "__main__":
    main()
