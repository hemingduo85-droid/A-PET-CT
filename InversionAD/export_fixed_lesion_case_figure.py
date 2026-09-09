#!/usr/bin/env python3
"""
Run example:
cd /data/cyf/codes/A-PET-CT/InversionAD1
python export_fixed_lesion_case_figure.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 \
  --modality dual \
  --device cuda:4 \
  --fname configs/exp_dit_petct/petct_dual.yml \
  --eval_step 3 \
  --output_dir fixed_lesion_case_figures
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont

from src.backbones import get_backbone, get_backbone_feature_shape
from src.config import resolve_petct_config
from src.datasets import build_transforms
from src.denoiser import get_denoiser
from src.export_lifecycle import export_tracers_sequentially


METHOD = "InvAD"
PROJECT_DIR = Path(__file__).resolve().parent
GROUPS = ("small", "medium", "large")
TRACERS = ("PSMA", "FDG")
PANEL_NAMES = ("PET", "CT", "GT", "Pred", "InvAD")

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


def load_config(args, tracer):
    fname = resolve_path(args.fname)
    with fname.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    config["data"]["petct_dataset"] = tracer.lower()
    config["data"]["category"] = tracer.lower()
    config["data"]["input_mode"] = args.modality
    config = resolve_petct_config(config)
    config["data"]["data_root"] = str(tracer_root(resolve_path(args.data_root), tracer))
    config.setdefault("meta", {})["device"] = args.device
    return config


def checkpoint_for_tracer(args, config, tracer):
    explicit = args.psma_checkpoint if tracer == "PSMA" else args.fdg_checkpoint
    if explicit:
        return resolve_path(explicit)
    save_dir = Path(args.save_dir).expanduser() if args.save_dir else Path(config["logging"]["save_dir"])
    if not save_dir.is_absolute():
        save_dir = PROJECT_DIR / save_dir
    if args.use_ema_model:
        return save_dir / "model_ema_latest.pth"
    if args.use_best_model and (save_dir / "model_best.pth").exists():
        return save_dir / "model_best.pth"
    return save_dir / "model_latest.pth"


def load_state_dict(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_model_for_tracer(args, tracer, device):
    config = load_config(args, tracer)
    in_shape = get_backbone_feature_shape(model_type=config["backbone"]["model_type"])
    config["diffusion"]["num_sampling_steps"] = str(args.eval_step)
    eval_denoiser = get_denoiser(
        **config["diffusion"], input_shape=in_shape
    )
    feature_extractor = get_backbone(**config["backbone"]).to(device).eval()
    ckpt = checkpoint_for_tracer(args, config, tracer)
    state = load_state_dict(ckpt)
    if state and "module." in next(iter(state.keys())):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    eval_denoiser.load_state_dict(state, strict=True)
    del state
    eval_denoiser.to(device).eval()
    print(f"Loaded {tracer} checkpoint: {ckpt}")
    return feature_extractor, eval_denoiser, config, ckpt


def build_input(paths, transform, image_size, input_mode):
    ct_img = Image.open(paths["ct"]).convert("L")
    pet_img = Image.open(paths["pet"]).convert("L")
    if input_mode == "ct":
        image = Image.merge("RGB", (ct_img, ct_img, ct_img))
    elif input_mode == "pet":
        image = Image.merge("RGB", (pet_img, pet_img, pet_img))
    else:
        image = Image.merge("RGB", (ct_img, pet_img, pet_img))
    arrays = {"pet": read_gray(paths["pet"], image_size), "ct": read_gray(paths["ct"], image_size)}
    return transform(image).unsqueeze(0), arrays


def infer_raw_map(feature_extractor, eval_denoiser, image, device):
    with torch.no_grad():
        features, _ = feature_extractor(image)
        start_t = torch.zeros(image.shape[0], device=device, dtype=torch.long)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            latents_last = eval_denoiser.ddim_reverse_sample(features, start_t, torch.zeros(image.shape[0], device=device, dtype=torch.long), eta=0.0)
        latents_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
        return F.interpolate(latents_l2.unsqueeze(0), size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False).squeeze().detach().cpu().numpy().astype(np.float32)


def render_case(args, bundle, device, data_root, group, tracer, patient, slice_id):
    feature_extractor, eval_denoiser, config, ckpt = bundle
    paths = case_paths(data_root, tracer, patient, slice_id)
    check_paths(paths)
    image_size = int(config["data"]["img_size"])
    transform = build_transforms(image_size, config["data"]["transform_type"])
    image, arrays = build_input(paths, transform, image_size, args.modality)
    image = image.to(device)
    raw_map = infer_raw_map(feature_extractor, eval_denoiser, image, device)
    mask = read_gray(paths["mask"], raw_map.shape[0], is_mask=True)
    threshold, best_f1 = best_f1_threshold(mask, raw_map)
    pred = (raw_map >= threshold).astype(np.uint8)
    case_name = f"{group}_{tracer}_{patient}__{slice_id}"
    case_dir = args.output_dir / "cases" / group / case_name
    pet_img = colorize(arrays["pet"], "magma")
    ct_img = gray_image(arrays["ct"])
    panels = {"PET": pet_img, "CT": ct_img, "GT": mask_image(mask), "Pred": mask_image(pred, color=(255, 128, 0)), "InvAD": colorize(raw_map, "jet")}
    save_image(case_dir / "PET_enhanced.png", panels["PET"], args.image_size)
    save_image(case_dir / "CT.png", panels["CT"], args.image_size)
    save_image(case_dir / "GT.png", panels["GT"], args.image_size)
    save_image(case_dir / "GT_on_PET.png", overlay_gt(pet_img, mask), args.image_size)
    save_image(case_dir / "GT_on_CT.png", overlay_gt(ct_img, mask), args.image_size)
    save_image(case_dir / "Pred.png", panels["Pred"], args.image_size)
    save_image(case_dir / "InvAD_heatmap.png", panels["InvAD"], args.image_size)
    meta = {"method": METHOD, "group": group, "tracer": tracer, "patient": patient, "slice": slice_id, "score": float(np.max(raw_map)), "score_mode": "max pixel score", "threshold": threshold, "best_f1": best_f1, "checkpoint": str(ckpt), "case_dir": str(case_dir)}
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "case_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"meta": meta, "panels": panels}


def release_model(bundle, device):
    feature_extractor, eval_denoiser, _, _ = bundle
    feature_extractor.to("cpu")
    eval_denoiser.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
    parser.add_argument("--modality", default="dual", choices=["ct", "pet", "dual"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fname", default="configs/exp_dit_petct/petct_dual.yml")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--psma_checkpoint", default=None)
    parser.add_argument("--fdg_checkpoint", default=None)
    parser.add_argument("--eval_step", type=int, default=3)
    parser.add_argument("--use_ema_model", action="store_true")
    parser.add_argument("--use_best_model", action="store_true")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "fixed_lesion_case_figures")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir = resolve_path(args.output_dir)
    data_root = resolve_path(args.data_root)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f"Using {device}: free={free_bytes / 1024**3:.2f} GiB / "
            f"total={total_bytes / 1024**3:.2f} GiB"
        )
    records_by_group = export_tracers_sequentially(
        tracers=TRACERS,
        groups=GROUPS,
        fixed_cases=FIXED_CASES,
        load_model=lambda tracer: load_model_for_tracer(args, tracer, device),
        render_case=lambda bundle, group, tracer, patient, slice_id: render_case(
            args,
            bundle,
            device,
            data_root,
            group,
            tracer,
            patient,
            slice_id,
        ),
        release_model=lambda bundle: release_model(bundle, device),
    )
    for group in GROUPS:
        make_plate(records_by_group[group], args.output_dir / f"{group}.png", f"{METHOD} {group} fixed lesion cases")
    make_plate([r for group in GROUPS for r in records_by_group[group]], args.output_dir / "all_fixed_lesion_cases.png", f"{METHOD} fixed lesion cases")


if __name__ == "__main__":
    main()
