#!/usr/bin/env python3
"""
Run example:
cd /data/cyf/codes/A-PET-CT/GLASS-main
python export_fixed_lesion_case_figure.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 \
  --modality petct \
  --gpu 4 \
  --save_root results \
  --output_dir fixed_lesion_case_figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import backbones
import glass
import petct_config
import utils
from datasets.petct import DatasetSplit, PETCTDataset


METHOD = "GLASS"
PROJECT_DIR = Path(__file__).resolve().parent
GROUPS = ("small", "medium", "large")
TRACERS = ("PSMA", "FDG")
PANEL_NAMES = ("PET", "CT", "GT", "Pred", "GLASS")

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


def make_dataset(data_root, tracer, modality, resize, imagesize):
    dataset = PETCTDataset(
        str(tracer_root(data_root, tracer)),
        "",
        dataset_name="petct",
        classname=modality,
        resize=resize,
        imagesize=imagesize,
        split=DatasetSplit.TEST,
        seed=0,
    )
    return dataset


def fixed_dataset_subset(dataset, data_root, tracer, patient, slice_id):
    paths = case_paths(data_root, tracer, patient, slice_id)
    check_paths(paths)
    target_pet = str(paths["pet"])
    for item in dataset.data_to_iterate:
        classname, anomaly, image_ref, mask_path = item
        display_path = image_ref[0] if dataset.modality == "petct" else image_ref
        if str(display_path) == target_pet:
            dataset.data_to_iterate = [(classname, anomaly, image_ref, mask_path)]
            return dataset
    raise FileNotFoundError(f"Fixed case not found in GLASS dataset scan: {target_pet}")


def load_model_for_tracer(args, tracer, dataset, device):
    save_dir = resolve_path(args.psma_save_dir if tracer == "PSMA" and args.psma_save_dir else args.fdg_save_dir if tracer == "FDG" and args.fdg_save_dir else petct_config.default_save_dir(args.save_root, tracer.lower(), args.modality))
    backbone = backbones.load(args.backbone)
    backbone.name, backbone.seed = args.backbone, None
    model = glass.GLASS(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=args.layers,
        device=device,
        input_shape=dataset.imagesize,
        pretrain_embed_dimension=args.pretrain_embed_dimension,
        target_embed_dimension=args.target_embed_dimension,
        patchsize=args.patchsize,
        meta_epochs=args.meta_epochs,
        eval_epochs=args.eval_epochs,
        dsc_layers=args.dsc_layers,
        dsc_hidden=args.dsc_hidden,
        dsc_margin=args.dsc_margin,
        pre_proj=args.pre_proj,
        mining=args.mining,
        noise=args.noise,
        radius=args.radius,
        p=args.p,
        lr=args.lr,
        svd=args.svd,
        step=args.step,
        limit=args.limit,
    )
    model = model.to(device)
    model.set_model_dir(os.path.join(str(save_dir), "models", "backbone_0"), f"petct_{args.modality}")
    ckpt = resolve_path(args.psma_checkpoint if tracer == "PSMA" and args.psma_checkpoint else args.fdg_checkpoint if tracer == "FDG" and args.fdg_checkpoint else model._find_checkpoint()[0])
    state = torch.load(ckpt, map_location=device)
    if "discriminator" in state:
        model.discriminator.load_state_dict(state["discriminator"])
        if "pre_projection" in state:
            model.pre_projection.load_state_dict(state["pre_projection"])
    else:
        model.load_state_dict(state, strict=False)
    print(f"Loaded {tracer} checkpoint: {ckpt}")
    return model.eval(), ckpt


def render_case(args, model, ckpt_path, device, data_root, group, tracer, patient, slice_id):
    dataset = make_dataset(data_root, tracer, args.modality, args.resize, args.imagesize)
    dataset = fixed_dataset_subset(dataset, data_root, tracer, patient, slice_id)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    _images, scores, maps, _labels, _masks, _paths = model.predict(loader)
    raw_map = np.asarray(maps[0], dtype=np.float32)
    paths = case_paths(data_root, tracer, patient, slice_id)
    mask = read_gray(paths["mask"], raw_map.shape[0], is_mask=True)
    threshold, best_f1 = best_f1_threshold(mask, raw_map)
    pred = (raw_map >= threshold).astype(np.uint8)
    arrays = {"pet": read_gray(paths["pet"], raw_map.shape[0]), "ct": read_gray(paths["ct"], raw_map.shape[0])}
    case_name = f"{group}_{tracer}_{patient}__{slice_id}"
    case_dir = args.output_dir / "cases" / group / case_name
    pet_img = colorize(arrays["pet"], "magma")
    ct_img = gray_image(arrays["ct"])
    panels = {"PET": pet_img, "CT": ct_img, "GT": mask_image(mask), "Pred": mask_image(pred, color=(255, 128, 0)), "GLASS": colorize(raw_map, "jet")}
    save_image(case_dir / "PET_enhanced.png", panels["PET"], args.imagesize)
    save_image(case_dir / "CT.png", panels["CT"], args.imagesize)
    save_image(case_dir / "GT.png", panels["GT"], args.imagesize)
    save_image(case_dir / "GT_on_PET.png", overlay_gt(pet_img, mask), args.imagesize)
    save_image(case_dir / "GT_on_CT.png", overlay_gt(ct_img, mask), args.imagesize)
    save_image(case_dir / "Pred.png", panels["Pred"], args.imagesize)
    save_image(case_dir / "GLASS_heatmap.png", panels["GLASS"], args.imagesize)
    meta = {"method": METHOD, "group": group, "tracer": tracer, "patient": patient, "slice": slice_id, "score": float(scores[0]), "score_mode": "GLASS top-1% mean score", "threshold": threshold, "best_f1": best_f1, "checkpoint": str(ckpt_path), "case_dir": str(case_dir)}
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
    parser.add_argument("--modality", choices=["pet", "ct", "petct"], default="petct")
    parser.add_argument("--save_root", default="results")
    parser.add_argument("--psma_save_dir", default=None)
    parser.add_argument("--fdg_save_dir", default=None)
    parser.add_argument("--psma_checkpoint", default=None)
    parser.add_argument("--fdg_checkpoint", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--backbone", default="wideresnet50")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1536)
    parser.add_argument("--target_embed_dimension", type=int, default=1536)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--meta_epochs", type=int, default=30)
    parser.add_argument("--eval_epochs", type=int, default=30)
    parser.add_argument("--dsc_layers", type=int, default=2)
    parser.add_argument("--dsc_hidden", type=int, default=1024)
    parser.add_argument("--dsc_margin", type=float, default=0.5)
    parser.add_argument("--pre_proj", type=int, default=1)
    parser.add_argument("--mining", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--radius", type=float, default=0.75)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--svd", type=int, default=0)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--limit", type=int, default=1840)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "fixed_lesion_case_figures")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir = resolve_path(args.output_dir)
    data_root = resolve_path(args.data_root)
    device = utils.set_torch_device([args.gpu])
    utils.fix_seeds(args.seed, device)
    datasets = {tracer: make_dataset(data_root, tracer, args.modality, args.resize, args.imagesize) for tracer in TRACERS}
    bundles = {tracer: load_model_for_tracer(args, tracer, datasets[tracer], device) for tracer in TRACERS}
    all_records = []
    for group in GROUPS:
        group_records = []
        for tracer in TRACERS:
            patient, slice_id = FIXED_CASES[group][tracer]
            model, ckpt = bundles[tracer]
            record = render_case(args, model, ckpt, device, data_root, group, tracer, patient, slice_id)
            group_records.append(record)
            all_records.append(record)
        make_plate(group_records, args.output_dir / f"{group}.png", f"GLASS {group} lesion fixed cases")
    make_plate(all_records, args.output_dir / "all_fixed_lesion_cases.png", "GLASS fixed lesion cases")


if __name__ == "__main__":
    main()
