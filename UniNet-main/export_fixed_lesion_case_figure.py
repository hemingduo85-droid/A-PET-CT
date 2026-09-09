#!/usr/bin/env python3
"""
Run example:
cd /data/cyf/codes/A-PET-CT/UniNet-main
python export_fixed_lesion_case_figure.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 \
  --modality petct \
  --gpu 4 \
  --epochs 30 \
  --ckpt_dir ckpts \
  --output_dir fixed_lesion_case_figures
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

import main_petct
from UniNet_lib.mechanism import weighted_decision_mechanism
from datasets_petct import IMAGENET_MEAN, IMAGENET_STD
from utils import load_weights


METHOD = "UniNet"
PROJECT_DIR = Path(__file__).resolve().parent
GROUPS = ("small", "medium", "large")
TRACERS = ("PSMA", "FDG")
PANEL_NAMES = ("PET", "CT", "GT", "Pred", "UniNet")

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


def final_ckpt_suffix(args):
    return args.ckpt_suffix or f"EPOCH_{args.epochs:03d}"


def prepare_tracer_args(args, tracer: str):
    dataset_tag = tracer.upper()
    args.dataset_name = dataset_tag
    args.dataset = f"PETCT_{dataset_tag}_{args.modality.upper()}"
    args._class_ = args.dataset
    args.domain = "medical"
    args.setting = "oc"
    return args


def checkpoint_for_tracer(args, tracer: str) -> Path:
    explicit = args.psma_checkpoint if tracer == "PSMA" else args.fdg_checkpoint
    if explicit:
        return resolve_path(explicit)
    tracer_args = argparse.Namespace(**vars(args))
    prepare_tracer_args(tracer_args, tracer)
    return resolve_path(args.ckpt_dir) / tracer_args.dataset / f"{final_ckpt_suffix(args)}.pth"


def load_model_for_tracer(args, tracer: str, device):
    ckpt_path = checkpoint_for_tracer(args, tracer)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {tracer}: {ckpt_path}")
    tracer_args = argparse.Namespace(**vars(args))
    prepare_tracer_args(tracer_args, tracer)
    model, _bn, _student, dfs, _target_teacher = main_petct.build_model(tracer_args, device)
    modules = [model.t.t_t, model.bn.bn, model.s.s1, dfs]
    state = load_weights(modules, str(ckpt_path.parent), ckpt_path.stem, device=device)
    model.t.t_t = state["tt"]
    model.bn.bn = state["bn"]
    model.s.s1 = state["st"]
    model.dfs = state["dfs"]
    model.train_or_eval(type="eval")
    print(f"Loaded {tracer} checkpoint: {ckpt_path}")
    return model, ckpt_path


def make_transform(image_size):
    return T.Compose([
        T.Resize((image_size, image_size), InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_input(paths, transform):
    pet_img = Image.open(paths["pet"]).convert("L")
    ct_img = Image.open(paths["ct"]).convert("L")
    rgb = Image.merge("RGB", [ct_img, pet_img, pet_img])
    return transform(rgb).unsqueeze(0)


def infer_raw_map(args, model, image):
    n = model.n
    output_list = [[] for _ in range(n * 3)]
    with torch.no_grad():
        t_tf, de_features = model(image)
        for l, (t, s) in enumerate(zip(t_tf, de_features)):
            output_list[l].append(1 - F.cosine_similarity(t, s))
    _unused, anomaly_map = weighted_decision_mechanism(1, output_list, args.alpha, args.beta, out_size=args.image_size)
    return gaussian_filter(anomaly_map[0], sigma=4).astype(np.float32)


def render_case(args, model, ckpt_path, transform, device, data_root, group, tracer, patient, slice_id):
    paths = case_paths(data_root, tracer, patient, slice_id)
    check_paths(paths)
    image = build_input(paths, transform).to(device)
    raw_map = infer_raw_map(args, model, image)
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
    save_image(case_dir / "UniNet_heatmap.png", heatmap_img, args.image_size)

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
    return {"meta": meta, "panels": {"PET": pet_img, "CT": ct_img, "GT": gt_img, "Pred": pred_img, "UniNet": heatmap_img}}


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
    title_font = load_font(22, bold=True)
    header_font = load_font(15, bold=True)
    meta_font = load_font(13)
    draw.text((width // 2, 12), title, font=title_font, fill=(0, 0, 0), anchor="ma")
    y = top
    for col, name in enumerate(PANEL_NAMES):
        x = left + col * (tile + gap)
        draw.text((x + 4, y), name, font=header_font, fill=(0, 0, 0))
    y += header_h
    for row_idx, record in enumerate(records, 1):
        meta = record["meta"]
        meta_text = (
            f"{meta['group'].capitalize()} #{row_idx} | score={meta['score']:.4f} | "
            f"F1={meta['best_f1']:.3f} | {meta['tracer']} | {meta['patient']}__{meta['slice']}"
        )
        draw.rectangle([left - 4, y, width - left + 4, y + meta_h - 2], fill=(245, 245, 245))
        draw.text((left, y + 4), meta_text, font=meta_font, fill=(0, 0, 0))
        y += meta_h
        for col, name in enumerate(PANEL_NAMES):
            x = left + col * (tile + gap)
            img = record["panels"][name].resize((tile, tile), resample=Image.Resampling.LANCZOS)
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(255, 255, 255), width=2)
        y += tile + gap
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
    print(f"Saved {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/data/cyf/shared_data/A-PETCT/2d_equal_mask50")
    parser.add_argument("--modality", default="petct", choices=["petct"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr_s", type=float, default=5e-3)
    parser.add_argument("--lr_t", type=float, default=1e-6)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--weighted_decision_mechanism", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.00003)
    parser.add_argument("--default", type=float, default=0.3)
    parser.add_argument("--ckpt_dir", default="ckpts")
    parser.add_argument("--ckpt_suffix", default=None)
    parser.add_argument("--psma_checkpoint", default=None)
    parser.add_argument("--fdg_checkpoint", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "fixed_lesion_case_figures")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.output_dir = resolve_path(args.output_dir)
    data_root = resolve_path(args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = make_transform(args.image_size)
    models = {tracer: load_model_for_tracer(args, tracer, device) for tracer in TRACERS}
    records_by_group = {group: [] for group in GROUPS}
    for group in GROUPS:
        for tracer in TRACERS:
            patient, slice_id = FIXED_CASES[group][tracer]
            model, ckpt_path = models[tracer]
            records_by_group[group].append(render_case(args, model, ckpt_path, transform, device, data_root, group, tracer, patient, slice_id))
        make_plate(records_by_group[group], args.output_dir / f"{group}.png", f"{METHOD} {group} fixed lesion cases")
    all_records = [record for group in GROUPS for record in records_by_group[group]]
    make_plate(all_records, args.output_dir / "all_fixed_lesion_cases.png", f"{METHOD} fixed lesion cases")


if __name__ == "__main__":
    main()
