#!/usr/bin/env python3
"""
Run example:
cd /data/cyf/codes/A-PET-CT/CostFilter-AD-main
python export_fixed_lesion_case_figure.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 \
  --modality petct \
  --gpu 4 \
  --checkpoint_root Costfilter_Dinomaly \
  --output_dir fixed_lesion_case_figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
COSTFILTER_DIR = PROJECT_DIR / "Costfilter_Dinomaly"
if str(COSTFILTER_DIR) not in sys.path:
    sys.path.insert(0, str(COSTFILTER_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from run_petct import (
    DiscriminativeSubNetwork_3d_att_dino_channel,
    build_dinomaly,
    cal_anomaly_maps,
    get_data_transforms,
    load_costfilter_checkpoint,
    load_dinomaly_checkpoint,
    setup_seed,
)


METHOD = "CostFilter-AD"
GROUPS = ("small", "medium", "large")
TRACERS = ("PSMA", "FDG")
PANEL_NAMES = ("PET", "CT", "GT", "Pred", "CostFilter-AD")

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
    return {"pet": base / "pet" / f"{slice_id}.png", "ct": base / "ct" / f"{slice_id}.png", "mask": base / "label" / f"{slice_id}.png"}


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
        inner[1:-1, 1:-1] = arr[1:-1, 1:-1] & (arr[:-2, 1:-1] & arr[2:, 1:-1] & arr[1:-1, :-2] & arr[1:-1, 2:])
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


def load_rgb(paths, modality):
    if modality == "ct":
        ct = Image.open(paths["ct"]).convert("L")
        return Image.merge("RGB", [ct, ct, ct])
    if modality == "pet":
        pet = Image.open(paths["pet"]).convert("L")
        return Image.merge("RGB", [pet, pet, pet])
    ct = Image.open(paths["ct"]).convert("L")
    pet = Image.open(paths["pet"]).convert("L")
    return Image.merge("RGB", [ct, pet, pet])


def checkpoint_paths(args, tracer):
    tag = tracer.lower()
    base = resolve_path(args.checkpoint_root) / f"checkpoint_{tag}_{args.modality}"
    dinomaly = args.psma_dinomaly_checkpoint if tracer == "PSMA" else args.fdg_dinomaly_checkpoint
    costfilter = args.psma_costfilter_checkpoint if tracer == "PSMA" else args.fdg_costfilter_checkpoint
    dinomaly = resolve_path(dinomaly) if dinomaly else base / f"dinomaly_{tag}_{args.modality}.pth"
    costfilter = resolve_path(costfilter) if costfilter else base / f"costfilter_{tag}_{args.modality}_epoch{args.costfilter_epochs}.pth"
    return dinomaly, costfilter


def load_models_for_tracer(args, tracer, device):
    crop_size = args.crop_size
    feat_size = crop_size // 14
    feat_total = feat_size * feat_size
    pre_min_dim = min(768, feat_total)
    model, embed_dim = build_dinomaly(device)
    dinomaly_ckpt, costfilter_ckpt = checkpoint_paths(args, tracer)
    if not dinomaly_ckpt.exists():
        raise FileNotFoundError(f"Dinomaly checkpoint not found: {dinomaly_ckpt}")
    if not costfilter_ckpt.exists():
        raise FileNotFoundError(f"CostFilter checkpoint not found: {costfilter_ckpt}")
    load_dinomaly_checkpoint(model, str(dinomaly_ckpt), device)
    model.eval()
    model_unet = DiscriminativeSubNetwork_3d_att_dino_channel(
        in_channels=pre_min_dim,
        out_channels=2,
        base_channels=args.base_channels,
    ).to(device).float()
    load_costfilter_checkpoint(model_unet, str(costfilter_ckpt), device)
    model_unet.eval()
    print(f"Loaded {tracer} Dinomaly checkpoint: {dinomaly_ckpt}")
    print(f"Loaded {tracer} CostFilter checkpoint: {costfilter_ckpt}")
    return {
        "model": model,
        "model_unet": model_unet,
        "embed_dim": embed_dim,
        "feat_size": feat_size,
        "pre_min_dim": pre_min_dim,
        "dinomaly_checkpoint": str(dinomaly_ckpt),
        "costfilter_checkpoint": str(costfilter_ckpt),
    }


def infer_raw_map(bundle, image, args, device):
    model = bundle["model"]
    model_unet = bundle["model_unet"]
    crop_size = args.crop_size
    feat_size = bundle["feat_size"]
    pre_min_dim = bundle["pre_min_dim"]
    unet_spatial = 64
    with torch.no_grad():
        en, de = model(image)
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
            _, idx = torch.topk(one_minus, cur_dim, dim=-1, largest=False, sorted=False)
            sel = torch.gather(one_minus, dim=-1, index=torch.sort(idx, dim=-1)[0])
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
        anomaly_map_final = cf_map * args.lamda + pred1 * (1 - args.lamda)
        return anomaly_map_final[0, 0].detach().cpu().numpy().astype(np.float32)


def render_case(args, bundle, transform, device, data_root, group, tracer, patient, slice_id):
    paths = case_paths(data_root, tracer, patient, slice_id)
    check_paths(paths)
    image = transform(load_rgb(paths, args.modality)).unsqueeze(0).to(device)
    raw_map = infer_raw_map(bundle, image, args, device)
    mask = read_gray(paths["mask"], raw_map.shape[0], is_mask=True)
    threshold, best_f1 = best_f1_threshold(mask, raw_map)
    pred = (raw_map >= threshold).astype(np.uint8)
    arrays = {"pet": read_gray(paths["pet"], raw_map.shape[0]), "ct": read_gray(paths["ct"], raw_map.shape[0])}
    case_name = f"{group}_{tracer}_{patient}__{slice_id}"
    case_dir = args.output_dir / "cases" / group / case_name
    pet_img = colorize(arrays["pet"], "magma")
    ct_img = gray_image(arrays["ct"])
    panels = {"PET": pet_img, "CT": ct_img, "GT": mask_image(mask), "Pred": mask_image(pred, color=(255, 128, 0)), "CostFilter-AD": colorize(raw_map, "jet")}
    save_image(case_dir / "PET_enhanced.png", panels["PET"], args.crop_size)
    save_image(case_dir / "CT.png", panels["CT"], args.crop_size)
    save_image(case_dir / "GT.png", panels["GT"], args.crop_size)
    save_image(case_dir / "GT_on_PET.png", overlay_gt(pet_img, mask), args.crop_size)
    save_image(case_dir / "GT_on_CT.png", overlay_gt(ct_img, mask), args.crop_size)
    save_image(case_dir / "Pred.png", panels["Pred"], args.crop_size)
    save_image(case_dir / "CostFilter-AD_heatmap.png", panels["CostFilter-AD"], args.crop_size)
    meta = {"method": METHOD, "group": group, "tracer": tracer, "patient": patient, "slice": slice_id, "score": float(np.max(raw_map)), "score_mode": "max pixel score", "threshold": threshold, "best_f1": best_f1, "dinomaly_checkpoint": bundle["dinomaly_checkpoint"], "costfilter_checkpoint": bundle["costfilter_checkpoint"], "case_dir": str(case_dir)}
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "case_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"meta": meta, "panels": panels}


def load_font(size, bold=False):
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
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
        meta_text = f"{meta['group'].capitalize()} #{row_idx} | score={meta['score']:.4f} | F1={meta['best_f1']:.3f} | {meta['tracer']} | {meta['patient']}__{meta['slice']}"
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
    parser.add_argument("--modality", choices=["ct", "pet", "petct"], default="petct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint_root", default="Costfilter_Dinomaly")
    parser.add_argument("--psma_dinomaly_checkpoint", default=None)
    parser.add_argument("--fdg_dinomaly_checkpoint", default=None)
    parser.add_argument("--psma_costfilter_checkpoint", default=None)
    parser.add_argument("--fdg_costfilter_checkpoint", default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=252)
    parser.add_argument("--costfilter_epochs", type=int, default=30)
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "fixed_lesion_case_figures")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.crop_size % 14 != 0:
        raise ValueError(f"crop_size must be divisible by 14, got {args.crop_size}")
    setup_seed(args.seed)
    args.output_dir = resolve_path(args.output_dir)
    data_root = resolve_path(args.data_root)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    transform, _gt_transform = get_data_transforms(args.image_size, args.crop_size)
    bundles = {tracer: load_models_for_tracer(args, tracer, device) for tracer in TRACERS}
    all_records = []
    for group in GROUPS:
        group_records = []
        for tracer in TRACERS:
            patient, slice_id = FIXED_CASES[group][tracer]
            record = render_case(args, bundles[tracer], transform, device, data_root, group, tracer, patient, slice_id)
            group_records.append(record)
            all_records.append(record)
        make_plate(group_records, args.output_dir / f"{group}.png", f"CostFilter-AD {group} lesion fixed cases")
    make_plate(all_records, args.output_dir / "all_fixed_lesion_cases.png", "CostFilter-AD fixed lesion cases")


if __name__ == "__main__":
    main()
