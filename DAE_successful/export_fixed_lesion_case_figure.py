#!/usr/bin/env python3
"""Render fixed PSMA/FDG lesion-size cases for DAE comparison figures.

python export_fixed_lesion_case_figure.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 \
  --modalities ct,pet \
  --device cuda:4 \
  --image_size 256 \
  --output_dir fixed_lesion_case_figures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sys

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infer import infer_map, load_models  # noqa: E402
from paper_eval_utils import (  # noqa: E402
    best_f1_threshold,
    load_mask,
    read_gray,
    top_percent_mean,
)


METHOD = "DAE"
GROUPS = ("small", "medium", "large")
PANEL_NAMES = ("PET", "CT", "GT", "Pred", "DAE")

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


def parse_modalities(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def resolve_project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    for candidate in (PROJECT_DIR / path, PROJECT_DIR.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_DIR / path).resolve()


def tracer_root(data_root: Path, tracer: str) -> Path:
    return data_root if (data_root / "test").exists() else data_root / tracer


def case_paths(data_root: Path, tracer: str, patient: str, slice_id: str) -> dict[str, Path]:
    base = tracer_root(data_root, tracer) / "test" / "abnormal" / patient
    return {
        "pet": base / "pet" / f"{slice_id}.png",
        "ct": base / "ct" / f"{slice_id}.png",
        "mask": base / "label" / f"{slice_id}.png",
    }


def check_paths(paths: dict[str, Path]) -> None:
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


def to_gray01(arr):
    return normalize01(arr)


def colorize(arr, cmap_name="magma"):
    import matplotlib.cm as cm

    arr = to_gray01(arr)
    rgb = cm.get_cmap(cmap_name)(arr)[..., :3]
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def gray_image(arr):
    gray = to_gray01(arr)
    return Image.fromarray((gray * 255).astype(np.uint8)).convert("RGB")


def image_from_array(arr, cmap="gray"):
    import matplotlib.cm as cm

    arr = normalize01(arr)
    rgb = cm.get_cmap(cmap)(arr)[..., :3]
    return Image.fromarray((rgb * 255).astype(np.uint8)).convert("RGB")


def mask_image(mask):
    arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return Image.fromarray(arr).convert("RGB")


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


def overlay_gt(base_img, gt_mask, color=(0, 255, 80, 255), width=2):
    base = base_img.convert("RGBA")
    outline = mask_outline(gt_mask, base.size, color=color, width=width)
    return Image.alpha_composite(base, outline).convert("RGB")


def save_image(path: Path, image, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((size, size), resample=Image.Resampling.LANCZOS).save(path)


def remove_legacy_case_outputs(case_dir: Path) -> None:
    for name in (
        "pet_original.png",
        "ct_original.png",
        "gt_mask.png",
        "pred_mask.png",
        "pred_heatmap.png",
        "anomaly_map.png",
        "threshold_map.png",
        "overview.png",
        "raw_pred_mask.png",
    ):
        path = case_dir / name
        if path.exists():
            path.unlink()


def threshold_pred(raw_map, threshold, roi, min_component_pixels):
    from paper_eval_utils import remove_small_components

    pred = (raw_map >= threshold).astype(np.uint8)
    if roi is not None:
        pred = (pred & (roi > 0)).astype(np.uint8)
    return remove_small_components(pred, min_component_pixels)


def load_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_case(args, models, modalities, device, data_root: Path, group: str, tracer: str, patient: str, slice_id: str):
    paths = case_paths(data_root, tracer, patient, slice_id)
    check_paths(paths)
    images = {mod: read_gray(paths[mod], args.image_size) for mod in modalities}
    mask = load_mask(paths["mask"], args.image_size)
    raw_map = infer_map(models, images, modalities, device)
    roi = np.zeros_like(raw_map, dtype=np.uint8)
    for image in images.values():
        roi |= (image > args.roi_min).astype(np.uint8)
    threshold = args.thr
    best_f1 = None
    if args.threshold_mode == "gt_best_f1":
        threshold, best_f1 = best_f1_threshold(mask, raw_map)
    score = float(top_percent_mean(raw_map, args.topk_percent)[0])
    pred = threshold_pred(raw_map, threshold, roi, args.min_component_pixels)
    case_name = f"{group}_{tracer}_{patient}__{slice_id}"
    case_dir = args.output_dir / "cases" / group / case_name
    remove_legacy_case_outputs(case_dir)
    pet_img = colorize(images.get("pet", next(iter(images.values()))), "magma")
    ct_img = gray_image(images.get("ct", next(iter(images.values()))))
    gt_img = mask_image(mask)
    gt_pet_img = overlay_gt(pet_img, mask, color=(0, 255, 80, 255), width=2)
    gt_ct_img = overlay_gt(ct_img, mask, color=(0, 255, 80, 255), width=2)
    dae_heatmap_img = colorize(raw_map, "jet")

    save_image(case_dir / "PET_enhanced.png", pet_img, args.image_size)
    save_image(case_dir / "CT.png", ct_img, args.image_size)
    save_image(case_dir / "GT.png", gt_img, args.image_size)
    save_image(case_dir / "GT_on_PET.png", gt_pet_img, args.image_size)
    save_image(case_dir / "GT_on_CT.png", gt_ct_img, args.image_size)
    save_image(case_dir / "DAE_heatmap.png", dae_heatmap_img, args.image_size)
    save_image(case_dir / "Pred.png", mask_image(pred), args.image_size)
    meta = {
        "group": group,
        "tracer": tracer,
        "patient": patient,
        "slice": slice_id,
        "score": score,
        "threshold": threshold,
        "best_f1": best_f1,
        "case_dir": str(case_dir),
    }
    with (case_dir / "case_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    panels = {
        "PET": pet_img,
        "CT": ct_img,
        "GT": gt_img,
        "Pred": mask_image(pred),
        "DAE": dae_heatmap_img,
    }
    return {"meta": meta, "panels": panels}


def draw_plate(records, save_path: Path, tile: int, title: str) -> None:
    margin_x = 18
    margin_top = 18
    gap = 6
    header_h = 24
    meta_h = 28
    row_h = meta_h + tile + gap
    width = margin_x * 2 + len(PANEL_NAMES) * tile + (len(PANEL_NAMES) - 1) * gap
    height = margin_top + header_h + len(records) * row_h + 12
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_title = load_font(18, bold=True)
    font_col = load_font(13, bold=True)
    font_meta = load_font(12)
    y = margin_top
    draw.text((margin_x, y), title, fill=(0, 0, 0), font=font_title)
    y += header_h
    x = margin_x
    for name in PANEL_NAMES:
        draw.text((x, y), name, fill=(0, 0, 0), font=font_col)
        x += tile + gap
    y += header_h
    for record in records:
        meta = record["meta"]
        label = (
            f"{meta['group'].capitalize()} | {meta['tracer']} | "
            f"score={meta['score']:.4f} thr={meta['threshold']:.4f} | "
            f"{meta['patient']} / {meta['slice']}"
        )
        draw.rectangle((margin_x - 2, y - 1, width - margin_x + 2, y + meta_h - 4), fill=(245, 245, 245))
        draw.text((margin_x + 2, y + 6), label, fill=(0, 0, 0), font=font_meta)
        y_img = y + meta_h
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
    parser.add_argument("--data_root", default="A_data/2d_equal_mask50")
    parser.add_argument("--modalities", default="ct,pet")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--noise_res", type=int, default=16)
    parser.add_argument("--roi_min", type=float, default=0.03)
    parser.add_argument("--score_q_low", type=float, default=0.01)
    parser.add_argument("--score_q_high", type=float, default=0.995)
    parser.add_argument("--heatmap_norm", default="percentile", choices=["percentile", "minmax", "none"])
    parser.add_argument("--threshold_mode", default="gt_best_f1", choices=["fixed", "gt_best_f1"])
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--min_component_pixels", type=int, default=0)
    parser.add_argument("--topk_percent", type=float, default=1.0)
    parser.add_argument("--tile", type=int, default=156)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "fixed_lesion_case_figures")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir = resolve_project_path(args.output_dir)
    data_root = resolve_project_path(args.data_root)
    modalities = parse_modalities(args.modalities)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models_by_tracer = {}
    records_by_group = {group: [] for group in GROUPS}
    for tracer in ("PSMA", "FDG"):
        args.tracer = tracer
        args.checkpoint = None
        models_by_tracer[tracer] = load_models(args, modalities, device)
    for group in GROUPS:
        for tracer in ("PSMA", "FDG"):
            patient, slice_id = FIXED_CASES[group][tracer]
            record = render_case(
                args,
                models_by_tracer[tracer],
                modalities,
                device,
                data_root,
                group,
                tracer,
                patient,
                slice_id,
            )
            records_by_group[group].append(record)
    all_records = []
    for group in GROUPS:
        draw_plate(records_by_group[group], args.output_dir / f"{group}.png", args.tile, f"{METHOD} {group} fixed lesion cases")
        all_records.extend(records_by_group[group])
    draw_plate(all_records, args.output_dir / "all_fixed_lesion_cases.png", args.tile, f"{METHOD} fixed lesion cases")


if __name__ == "__main__":
    main()
