#!/usr/bin/env python3
"""
Run example:
cd /data/cyf/codes/A-PET-CT/RD4AD
python export_fixed_lesion_case_figure.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 \
  --modalities ct,pet \
  --input-mode ct_pet_pet \
  --gpu 4 \
  --output_dir fixed_lesion_case_figures
"""
"""Render fixed PSMA/FDG lesion-size cases for RD4AD comparison figures."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from config import checkpoint_path, get_dataset_config, parse_modalities, resolve_input_mode
from test_modified import cal_anomaly_map, load_checkpoint, make_model


METHOD = "RD4AD"
PROJECT_DIR = Path(__file__).resolve().parent
GROUPS = ("small", "medium", "large")
TRACERS = ("PSMA", "FDG")
PANEL_NAMES = ("PET", "CT", "GT", "Pred", "RD4AD")

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
    base = base_img.convert("RGBA")
    return Image.alpha_composite(base, mask_outline(gt_mask, base.size)).convert("RGB")


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


def checkpoint_for_tracer(args, tracer: str, modalities, input_mode) -> Path:
    explicit = args.psma_checkpoint if tracer == "PSMA" else args.fdg_checkpoint
    if explicit:
        return resolve_path(explicit)
    dataset_config = get_dataset_config(tracer.lower())
    return Path(checkpoint_path(dataset_config, modalities, input_mode, checkpoint_dir=str(resolve_path(args.checkpoint_dir))))


def load_models_for_tracer(args, tracer: str, modalities, input_mode, device):
    ckpt_path = checkpoint_for_tracer(args, tracer, modalities, input_mode)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {tracer}: {ckpt_path}")
    encoder, bn, decoder = make_model(device, modalities, input_mode)
    load_checkpoint(bn, decoder, str(ckpt_path), device)
    encoder.eval()
    bn.eval()
    decoder.eval()
    return encoder, bn, decoder, ckpt_path


def make_gray_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])


def build_input(paths, gray_transform):
    ct = gray_transform(Image.open(paths["ct"]).convert("L"))
    pet = gray_transform(Image.open(paths["pet"]).convert("L"))
    return torch.cat([ct, pet, pet], dim=0).unsqueeze(0)


def render_case(args, encoder, bn, decoder, ckpt_path, gray_transform, device, data_root, group, tracer, patient, slice_id):
    paths = case_paths(data_root, tracer, patient, slice_id)
    check_paths(paths)
    image = build_input(paths, gray_transform).to(device)
    with torch.no_grad():
        inputs = encoder(image)
        outputs = decoder(bn(inputs))
        raw_map, _ = cal_anomaly_map(inputs, outputs, image.shape[-1], amap_mode="a")
    raw_map = gaussian_filter(raw_map, sigma=4).astype(np.float32)
    pet_arr = read_gray(paths["pet"], raw_map.shape[0])
    ct_arr = read_gray(paths["ct"], raw_map.shape[0])
    mask = read_gray(paths["mask"], raw_map.shape[0], is_mask=True)
    threshold, best_f1 = best_f1_threshold(mask, raw_map)
    pred = (raw_map >= threshold).astype(np.uint8)
    score = float(np.max(raw_map))

    case_name = f"{group}_{tracer}_{patient}__{slice_id}"
    case_dir = args.output_dir / "cases" / group / case_name
    pet_img = colorize(pet_arr, "magma")
    ct_img = gray_image(ct_arr)
    gt_img = mask_image(mask)
    pred_img = mask_image(pred, color=(255, 128, 0))
    heatmap_img = colorize(raw_map, "jet")

    save_image(case_dir / "PET_enhanced.png", pet_img, args.image_size)
    save_image(case_dir / "CT.png", ct_img, args.image_size)
    save_image(case_dir / "GT.png", gt_img, args.image_size)
    save_image(case_dir / "GT_on_PET.png", overlay_gt(pet_img, mask), args.image_size)
    save_image(case_dir / "GT_on_CT.png", overlay_gt(ct_img, mask), args.image_size)
    save_image(case_dir / "Pred.png", pred_img, args.image_size)
    save_image(case_dir / "RD4AD_heatmap.png", heatmap_img, args.image_size)

    meta = {
        "method": METHOD,
        "group": group,
        "tracer": tracer,
        "patient": patient,
        "slice": slice_id,
        "score": score,
        "score_mode": "max pixel score",
        "threshold": threshold,
        "best_f1": best_f1,
        "checkpoint": str(ckpt_path),
        "case_dir": str(case_dir),
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "case_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"meta": meta, "panels": {"PET": pet_img, "CT": ct_img, "GT": gt_img, "Pred": pred_img, "RD4AD": heatmap_img}}


def load_font(size, bold=False):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_plate(records, save_path: Path, tile: int, title: str):
    margin_x, margin_top, gap, header_h, col_h = 22, 18, 6, 28, 22
    row_h = header_h + tile + gap
    width = margin_x * 2 + len(PANEL_NAMES) * tile + (len(PANEL_NAMES) - 1) * gap
    height = margin_top + col_h + len(records) * row_h + 14
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_title, font_col, font_meta = load_font(18, True), load_font(13, True), load_font(12)
    y = margin_top
    draw.text((margin_x, y), f"{title} (n={len(records)})", fill=(0, 0, 0), font=font_title)
    y += col_h
    x = margin_x
    for name in PANEL_NAMES:
        draw.text((x, y), name, fill=(0, 0, 0), font=font_col)
        x += tile + gap
    y += col_h
    for record in records:
        meta = record["meta"]
        label = f"{meta['group'].capitalize()} | {meta['tracer']} | score={meta['score']:.4f} thr={meta['threshold']:.4f} | {meta['patient']} / {meta['slice']}"
        draw.rectangle((margin_x - 2, y - 1, width - margin_x + 2, y + header_h - 4), fill=(245, 245, 245))
        draw.text((margin_x + 2, y + 6), label, fill=(0, 0, 0), font=font_meta)
        y_img = y + header_h
        x = margin_x
        for name in PANEL_NAMES:
            panel = record["panels"][name].resize((tile, tile), resample=Image.Resampling.LANCZOS)
            canvas.paste(panel, (x, y_img))
            draw.rectangle((x, y_img, x + tile - 1, y_img + tile - 1), outline=(60, 60, 60), width=1)
            x += tile + gap
        y += row_h
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
    print(f"Saved {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/data/cyf/shared_data/A-PETCT/2d_equal_mask50")
    parser.add_argument("--modalities", default="ct,pet")
    parser.add_argument("--input-mode", dest="input_mode", default="ct_pet_pet")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default="./checkpoints")
    parser.add_argument("--psma_checkpoint", default=None)
    parser.add_argument("--fdg_checkpoint", default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--tile", type=int, default=160)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "fixed_lesion_case_figures")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.output_dir = resolve_path(args.output_dir)
    data_root = resolve_path(args.data_root)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    modalities = parse_modalities("pet", args.modalities)
    input_mode = resolve_input_mode(modalities, args.input_mode)
    if input_mode != "ct_pet_pet":
        raise ValueError("This fixed-case script expects --input-mode ct_pet_pet.")
    gray_transform = make_gray_transform(args.image_size)
    models = {tracer: load_models_for_tracer(args, tracer, modalities, input_mode, device) for tracer in TRACERS}
    records_by_group = {group: [] for group in GROUPS}
    for group in GROUPS:
        for tracer in TRACERS:
            patient, slice_id = FIXED_CASES[group][tracer]
            encoder, bn, decoder, ckpt_path = models[tracer]
            records_by_group[group].append(render_case(args, encoder, bn, decoder, ckpt_path, gray_transform, device, data_root, group, tracer, patient, slice_id))
    all_records = []
    for group in GROUPS:
        draw_plate(records_by_group[group], args.output_dir / f"{group}.png", args.tile, f"{METHOD} {group} fixed lesion cases")
        all_records.extend(records_by_group[group])
    draw_plate(all_records, args.output_dir / "all_fixed_lesion_cases.png", args.tile, f"{METHOD} fixed lesion cases")


if __name__ == "__main__":
    main()
