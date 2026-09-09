#!/usr/bin/env python3
# Example:
#   python export_selected_lesion_case_figure.py \
#     --group_root saved_results_psma_eval/PhyTwin_PETCT_PSMA/figure_scores/lesion_size_groups \
#     --pixel_metrics_csv saved_results_psma_eval/PhyTwin_PETCT_PSMA/figure_scores/pixel_slice_metrics.csv \
#     --output_dir saved_results_psma_eval/PhyTwin_PETCT_PSMA/figure_scores/ours_selected_lesion_cases \
#     --top_k 50 \
#     --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA \
#     --load_ckpt --experiment_name PhyTwin_PETCT_PSMA \
#     --score_mode lesion_z --adaptive_physio --adaptive_alpha 0.50 \
#     --adaptive_lesion_protect 0.70 --image_size 256 --gpu 0
"""Render Ours selected lesion-size case plates from lesion_size_groups CSVs.

This script is meant to run on the server that has the dataset and checkpoint.
It does not use previously saved `visualizations/` PNG files. Instead, it reads
the ranking CSVs produced by export_lesion_volume_groups.py, reruns PhyTwin only
for the selected slices, and renders PET / CT / GT / Pred / Ours panels.

    The row annotation reports heatmap quality measures:
- pixel AUROC/AUPR from continuous Ours heatmap vs GT mask
- Hotspot recall/precision: overlap between GT and the hottest pixels whose
  area equals the GT lesion area
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


GROUPS = ("small", "medium", "large")
PANEL_NAMES = ("PET", "CT", "GT", "Pred", "Ours", "Ours-ref")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_slice_id(value: object) -> str:
    value = str(value)
    try:
        return f"{int(float(value)):04d}"
    except Exception:
        return Path(value).stem


def case_id_from_path(path_value: object, fallback: object = "") -> str:
    if fallback:
        return str(fallback)
    p = Path(str(path_value))
    if p.parent.name.lower() in {"pet", "ct", "label", "labels", "mask", "masks", "gt"}:
        return p.parent.parent.name
    return p.parent.name


def slice_id_from_path(path_value: object, fallback: object = "") -> str:
    if fallback:
        return normalize_slice_id(fallback)
    return normalize_slice_id(Path(str(path_value)).stem)


def row_key(row: dict[str, object]) -> tuple[str, str]:
    return (
        case_id_from_path(row.get("path", ""), row.get("case_id", "")),
        slice_id_from_path(row.get("path", ""), row.get("slice_id", "")),
    )


def load_selected_rows(group_root: Path, groups: list[str], top_k: int | None) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for group in groups:
        csv_path = group_root / group / "slices.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing group CSV: {csv_path}")
        rows = [
            row for row in read_csv_rows(csv_path)
            if int(float(row.get("true_label", 0) or 0)) == 1
        ]
        rows.sort(
            key=lambda r: (
                float(r.get("ours_score", r.get("anomaly_score", 0.0)) or 0.0),
                int(float(r.get("lesion_pixels", 0) or 0)),
            ),
            reverse=True,
        )
        if top_k is not None and top_k > 0:
            rows = rows[:top_k]
        for rank, row in enumerate(rows, start=1):
            out = dict(row)
            out["requested_group"] = group
            out["requested_rank"] = rank
            selected.append(out)
    return selected


def load_pixel_metrics(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if not path or not path.exists():
        return {}
    out = {}
    for row in read_csv_rows(path):
        out[row_key(row)] = row
    return out


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def mask_metrics(gt_mask, pred_mask) -> dict[str, float]:
    import numpy as np

    gt = np.asarray(gt_mask)
    pred = np.asarray(pred_mask)
    gt = np.squeeze(gt) > 0.5
    pred = np.squeeze(pred) > 0.5
    if pred.shape != gt.shape:
        import cv2

        pred = cv2.resize(pred.astype("uint8"), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    tp = float((gt & pred).sum())
    gt_sum = float(gt.sum())
    pred_sum = float(pred.sum())
    union = float((gt | pred).sum())
    return {
        "dice": safe_div(2.0 * tp, gt_sum + pred_sum),
        "iou": safe_div(tp, union),
        "recall": safe_div(tp, gt_sum),
        "precision": safe_div(tp, pred_sum),
        "gt_pixels": int(gt_sum),
        "pred_pixels": int(pred_sum),
    }


def heatmap_overlap_metrics(final_map, gt_mask) -> dict[str, float | int]:
    import numpy as np

    scores = np.asarray(final_map, dtype=np.float32).reshape(-1)
    gt = (np.squeeze(np.asarray(gt_mask)).reshape(-1) > 0.5)
    n_gt = int(gt.sum())
    if n_gt <= 0:
        return {"hotspot_recall": 0.0, "hotspot_precision": 0.0, "hotspot_pixels": 0}
    n_hot = min(n_gt, scores.size)
    # Hottest region with the same pixel budget as the GT lesion.
    hot_idx = np.argpartition(scores, -n_hot)[-n_hot:]
    hot = np.zeros(scores.size, dtype=bool)
    hot[hot_idx] = True
    tp = float((hot & gt).sum())
    return {
        "hotspot_recall": safe_div(tp, float(n_gt)),
        "hotspot_precision": safe_div(tp, float(n_hot)),
        "hotspot_pixels": int(n_hot),
    }


def pixel_metrics_from_map(final_map, gt_mask) -> dict[str, float | str]:
    import cv2
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    amap = np.asarray(final_map, dtype=np.float32)
    gt = np.squeeze(np.asarray(gt_mask))
    if gt.shape != amap.shape:
        gt = cv2.resize(gt.astype(np.float32), (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_NEAREST)
    y_true = (gt.reshape(-1) > 0.5).astype(np.uint8)
    y_score = np.nan_to_num(amap.reshape(-1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return {"pixel_auroc": "", "pixel_aupr": ""}
    return {
        "pixel_auroc": float(roc_auc_score(y_true, y_score)),
        "pixel_aupr": float(average_precision_score(y_true, y_score)),
    }


def fmt_metric(value: object) -> str:
    if value == "" or value is None:
        return "NA"
    return f"{float(value):.3f}"


def to_gray01(x):
    import numpy as np

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        arr = arr * std + mean
        arr = arr[0]
    arr = np.squeeze(arr).astype(np.float32)
    lo, hi = np.percentile(arr[np.isfinite(arr)], [1.0, 99.5])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    arr = (arr - lo) / (hi - lo + 1e-8)
    return np.clip(arr, 0.0, 1.0)


def colorize(arr, cmap_name="magma"):
    import numpy as np
    from PIL import Image

    arr = to_gray01(arr)
    try:
        import matplotlib.cm as cm

        rgb = cm.get_cmap(cmap_name)(arr)[..., :3]
    except Exception:
        rgb = np.stack([arr ** 0.7, arr ** 1.6, 0.35 * arr], axis=-1)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def gray_image(arr):
    import numpy as np
    from PIL import Image

    gray = to_gray01(arr)
    return Image.fromarray((gray * 255).astype(np.uint8)).convert("RGB")


def mask_image(mask):
    import numpy as np
    from PIL import Image

    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    arr = np.squeeze(np.asarray(mask)) > 0.5
    return Image.fromarray((arr.astype(np.uint8) * 255)).convert("RGB")


def mask_outline(mask, size, color=(0, 255, 80, 255), width=2):
    import numpy as np
    from PIL import Image

    gt = np.squeeze(np.asarray(mask.detach().cpu().numpy() if hasattr(mask, "detach") else mask)) > 0.5
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
    from PIL import Image

    base = base_img.convert("RGBA")
    outline = mask_outline(gt_mask, base.size, color=color, width=width)
    return Image.alpha_composite(base, outline).convert("RGB")


def save_image(path: Path, image, size: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((size, size), resample=Image.Resampling.LANCZOS).save(path)


def top_individual_rows_by_group(group_root: Path, groups: list[str], top_k: int) -> list[dict[str, str]]:
    rows = load_selected_rows(group_root, groups, top_k=top_k)
    for row in rows:
        row["requested_rank"] = row.get("rank_in_size_group", row.get("requested_rank", ""))
    return rows


def render_ours_heatmap(train_mod, final_map):
    import numpy as np
    from PIL import Image

    heatmap = train_mod._heatmap(final_map)
    return Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8))


def render_reference_style_heatmap(final_map):
    return colorize(final_map, "jet")


def export_individual_images(train_mod, rows, records, pixel_lookup, output_dir: Path, image_size: int):
    summary = []
    for row in rows:
        group = row.get("requested_group", row.get("lesion_size_group", "unknown"))
        key = row_key(row)
        if key not in records:
            summary.append({
                "group": group,
                "case_id": key[0],
                "slice_id": key[1],
                "status": "missing_in_test_loader",
            })
            continue
        record = records[key]
        case_dir = output_dir / group / f"{key[0]}_{key[1]}"
        pet_img = colorize(record["pet"], "magma")
        ct_img = gray_image(record["ct"])
        gt_img = mask_image(record["gt"])
        gt_pet_img = overlay_gt(pet_img, record["gt"], color=(0, 255, 80, 255), width=2)
        gt_ct_img = overlay_gt(ct_img, record["gt"], color=(0, 255, 80, 255), width=2)
        heatmap_img = render_ours_heatmap(train_mod, record["final"])
        reference_heatmap_img = render_reference_style_heatmap(record["final"])

        save_image(case_dir / "PET_enhanced.png", pet_img, image_size)
        save_image(case_dir / "CT.png", ct_img, image_size)
        save_image(case_dir / "GT.png", gt_img, image_size)
        save_image(case_dir / "GT_on_PET.png", gt_pet_img, image_size)
        save_image(case_dir / "GT_on_CT.png", gt_ct_img, image_size)
        save_image(case_dir / "Ours_heatmap.png", heatmap_img, image_size)
        save_image(case_dir / "Ours_heatmap_reference_style.png", reference_heatmap_img, image_size)

        mm = record["mask_metrics"]
        pm = dict(record["pixel_metrics"])
        pm.update(pixel_lookup.get(key, {}))
        hm = record["heatmap_metrics"]
        meta = {
            "group": group,
            "case_id": key[0],
            "slice_id": key[1],
            "rank_in_size_group": row.get("rank_in_size_group", ""),
            "lesion_pixels": row.get("lesion_pixels", mm["gt_pixels"]),
            "lesion_area_ratio": row.get("lesion_area_ratio", ""),
            "image_score": record["image_score"],
            "pixel_auroc": pm.get("pixel_auroc", ""),
            "pixel_aupr": pm.get("pixel_aupr", ""),
            "hotspot_recall": hm["hotspot_recall"],
            "hotspot_precision": hm["hotspot_precision"],
            "pred_dice_reference": mm["dice"],
            "pred_iou_reference": mm["iou"],
            "path": record["path"],
            "status": "saved",
        }
        with (case_dir / "metrics.json").open("w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        write_csv(case_dir / "metrics.csv", [meta], list(meta.keys()))
        summary.append(meta)
    fields = [
        "group", "case_id", "slice_id", "rank_in_size_group", "lesion_pixels",
        "lesion_area_ratio", "image_score", "pixel_auroc", "pixel_aupr",
        "hotspot_recall", "hotspot_precision", "pred_dice_reference",
        "pred_iou_reference", "path", "status",
    ]
    write_csv(output_dir / "individual_summary.csv", summary, fields)
    return summary


def import_train_args(train_argv: list[str]):
    import train as train_mod
    from phytwin_petct.data import resolve_data_root

    old_argv = sys.argv[:]
    try:
        sys.argv = ["train.py"] + train_argv
        args = train_mod.parse_args()
    finally:
        sys.argv = old_argv
    args.data_root = resolve_data_root(args.data_root)
    args.batch_size = 1
    return train_mod, args


def load_phytwin(train_mod, args):
    import torch
    from phytwin_petct.data import build_dataloaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _train_loader, memory_loader, test_loader = build_dataloaders(
        args.data_root,
        image_size=args.image_size,
        batch_size=1,
        num_workers=args.num_workers,
    )
    model = train_mod.NormalTwinUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        uncertainty=args.uncertainty,
    ).to(device)
    ckpt_full = os.path.join(args.ckpt_dir, args.method_name, "BEST_PHYTWIN.pth")
    if not os.path.exists(ckpt_full):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_full}")
    model, memory, prior, calibration = train_mod.load_full_checkpoint(ckpt_full, model)
    model = model.to(device)
    if args.no_physio:
        prior = None
    if args.score_mode == "lesion_z" and (
        "normal_lesion_ms_mean" not in calibration
        or calibration.get("image_refinement_config") != train_mod.image_refinement_config(args)
    ):
        calibration = train_mod.calibrate_normal_lesion_scores(
            model, memory, prior, memory_loader, device, args, calibration, logger=None
        )
    dynamic_suppressor = None
    if args.use_dynamic_physio:
        dynamic_suppressor = train_mod.DynamicPhysioSuppressor(
            alpha=args.dynamic_alpha,
            pet_quantile=args.dynamic_pet_quantile,
            min_area=args.dynamic_min_area,
        )
    return model, memory, prior, dynamic_suppressor, calibration, test_loader, device, ckpt_full


def infer_selected(train_mod, model, memory, prior, dynamic_suppressor, calibration, loader, device, args, selected_by_key):
    import numpy as np
    import torch

    model.eval()
    if memory is not None:
        memory = memory.to(device)
    adaptive_suppressor = train_mod.build_adaptive_physio(args)
    prior_arr = None if prior is None else prior.prior
    records = {}

    with torch.no_grad():
        for idx, batch in train_mod.limited(loader, args.max_test_batches):
            path = batch["path"][0] if isinstance(batch["path"], (list, tuple)) else str(batch["path"])
            key = row_key({"path": path})
            if key not in selected_by_key:
                continue

            pet = batch["pet"].to(device)
            ct = batch["ct"].to(device)
            pred, logvar = model(ct)
            residual = train_mod.residual_map(pet, pred, logvar if args.uncertainty else None)
            residual_np = residual[0].detach().cpu().numpy().astype(np.float32)
            residual_z = train_mod.positive_zscore_map(
                residual_np, calibration["residual_mean"], calibration["residual_std"]
            )
            memory_z = train_mod.memory_z_map(memory, residual, calibration, args)
            final = train_mod.fuse_maps(
                residual_z,
                memory_z,
                residual_weight=args.residual_weight,
                memory_weight=args.memory_weight,
                physio_prior=prior,
                sigma=args.gaussian_sigma,
            )
            pet_gray = train_mod.pet_tensor_to_gray(batch["pet"])[0].numpy().astype(np.float32)
            ct_gray = train_mod.pet_tensor_to_gray(batch["ct"])[0].numpy().astype(np.float32)
            if dynamic_suppressor is not None:
                final, _dynamic_mask = dynamic_suppressor.suppress(
                    final,
                    pet_gray,
                    static_prior=None if prior is None else prior.prior,
                )
            final, _adaptive_mask = train_mod.apply_adaptive_physio(final, pet_gray, ct_gray, prior, adaptive_suppressor)

            if args.score_mode == "component":
                image_score = train_mod.lesion_component_score(
                    final,
                    physio_prior=prior_arr,
                    threshold_quantile=args.component_quantile,
                    min_area=args.component_min_area,
                    prior_penalty=args.prior_penalty,
                )
            elif args.score_mode == "normal_z":
                image_score = train_mod.normal_calibrated_score(
                    final,
                    calibration["normal_image_mean"],
                    calibration["normal_image_std"],
                    physio_prior=prior_arr,
                    prior_penalty=args.prior_penalty,
                )
            elif args.score_mode == "lesion_z":
                raw_score = train_mod.compute_lesion_score(
                    final,
                    pet_gray,
                    ct_gray,
                    prior,
                    args,
                    normal_hotspot_memory=calibration.get("normal_hotspot_memory") if args.use_hotspot_memory else None,
                )
                normal_mean = calibration.get(
                    "normal_lesion_ms_mean",
                    calibration.get("normal_lesion_mean", calibration["normal_image_mean"]),
                )
                normal_std = calibration.get(
                    "normal_lesion_ms_std",
                    calibration.get("normal_lesion_std", calibration["normal_image_std"]),
                )
                image_score = (raw_score - normal_mean) / (normal_std + 1e-8)
            else:
                image_score = train_mod.topk_score(final, fraction=0.01)

            pred_mask = train_mod.predict_mask(
                final,
                threshold_quantile=args.pred_mask_quantile,
                min_area=args.pred_mask_min_area,
            )
            gt_mask = batch["mask"][0].cpu()
            records[key] = {
                "path": path,
                "pet": batch["pet"][0].cpu(),
                "ct": batch["ct"][0].cpu(),
                "gt": gt_mask,
                "pred": pred_mask,
                "residual": residual_np,
                "final": final,
                "image_score": float(image_score),
                "mask_metrics": mask_metrics(gt_mask.numpy(), pred_mask),
                "pixel_metrics": pixel_metrics_from_map(final, gt_mask.numpy()),
                "heatmap_metrics": heatmap_overlap_metrics(final, gt_mask.numpy()),
            }
            if len(records) == len(selected_by_key):
                break
    return records


def load_font(size: int, bold: bool = False):
    from PIL import ImageFont

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


def panel_images(train_mod, record, tile: int):
    heatmap_img = render_ours_heatmap(train_mod, record["final"]).resize((256, 256))
    reference_heatmap_img = render_reference_style_heatmap(record["final"]).resize((256, 256))
    panels = [
        train_mod._panel(record["pet"], "PET", gt_mask=record["gt"], pred_mask=record["pred"]),
        train_mod._panel(record["ct"], "CT"),
        train_mod._panel(record["gt"], "GT", mask_fill=True),
        train_mod._panel(record["pred"], "Pred", pred_fill=True),
        heatmap_img,
        reference_heatmap_img,
    ]
    return [panel.resize((tile, tile)) for panel in panels]


def draw_plate(train_mod, rows, records, pixel_lookup, save_path: Path, title: str, tile: int):
    from PIL import Image, ImageDraw

    drawable = [(row, records[row_key(row)]) for row in rows if row_key(row) in records]
    if not drawable:
        return 0
    margin_x = 22
    margin_top = 18
    gap = 6
    header_h = 28
    col_h = 22
    row_h = header_h + tile + gap
    width = margin_x * 2 + len(PANEL_NAMES) * tile + (len(PANEL_NAMES) - 1) * gap
    height = margin_top + col_h + len(drawable) * row_h + 14
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_title = load_font(18, bold=True)
    font_col = load_font(13, bold=True)
    font_meta = load_font(12)

    y = margin_top
    draw.text((margin_x, y), f"{title} (n={len(drawable)})", fill=(0, 0, 0), font=font_title)
    y += col_h
    x = margin_x
    for name in PANEL_NAMES:
        draw.text((x, y), name, fill=(0, 0, 0), font=font_col)
        x += tile + gap
    y += col_h

    for row, record in drawable:
        key = row_key(row)
        mm = record["mask_metrics"]
        pm = dict(record["pixel_metrics"])
        pm.update(pixel_lookup.get(key, {}))
        hm = record["heatmap_metrics"]
        lesion_pixels = int(float(row.get("lesion_pixels", mm["gt_pixels"]) or 0))
        area = 100.0 * float(row.get("lesion_area_ratio", 0.0) or 0.0)
        meta = (
            f"{row.get('requested_group', row.get('lesion_size_group', '')).capitalize()} #{row.get('requested_rank', row.get('rank_in_size_group', ''))}  |  "
            f"pAUROC={fmt_metric(pm.get('pixel_auroc'))}  pAUPR={fmt_metric(pm.get('pixel_aupr'))}  |  "
            f"HotRecall={hm['hotspot_recall']:.3f}  HotPrecision={hm['hotspot_precision']:.3f}  |  "
            f"Lesion={lesion_pixels} px ({area:.3f}%)  |  {key[0]} / {key[1]}"
        )
        draw.rectangle((margin_x - 2, y - 1, width - margin_x + 2, y + header_h - 4), fill=(245, 245, 245))
        draw.text((margin_x + 2, y + 6), meta, fill=(0, 0, 0), font=font_meta)
        y_img = y + header_h
        x = margin_x
        for panel in panel_images(train_mod, record, tile):
            canvas.paste(panel, (x, y_img))
            draw.rectangle((x, y_img, x + tile - 1, y_img + tile - 1), outline=(60, 60, 60), width=1)
            x += tile + gap
        y += row_h

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
    return len(drawable)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_root", type=Path, required=True)
    parser.add_argument("--pixel_metrics_csv", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    parser.add_argument("--top_k", type=int, default=50, help="Top slices per lesion-size group. Use <=0 for all.")
    parser.add_argument("--tile", type=int, default=160)
    parser.add_argument("--individual_top_k", type=int, default=5, help="Top slices per size group exported as separate folders.")
    parser.add_argument("--individual_size", type=int, default=256, help="Pixel size for each separately exported best-case image.")
    args, train_argv = parser.parse_known_args(argv)
    return args, train_argv


def main(argv=None):
    args, train_argv = parse_args(sys.argv[1:] if argv is None else argv)
    train_mod, train_args = import_train_args(train_argv)
    from phytwin_petct import visualization as vis_mod

    # Reuse the exact panel renderer from the repo visualization module.
    train_mod._panel = vis_mod._panel
    train_mod._heatmap = vis_mod._heatmap

    top_k = None if args.top_k <= 0 else args.top_k
    pixel_lookup = load_pixel_metrics(args.pixel_metrics_csv)
    selected_rows = load_selected_rows(args.group_root, args.groups, top_k)
    individual_rows = top_individual_rows_by_group(
        args.group_root,
        args.groups,
        max(1, int(args.individual_top_k)),
    )
    inference_rows = selected_rows + individual_rows
    selected_by_key = {row_key(row): row for row in inference_rows}

    model, memory, prior, dynamic_suppressor, calibration, test_loader, device, ckpt_full = load_phytwin(train_mod, train_args)
    records = infer_selected(
        train_mod,
        model,
        memory,
        prior,
        dynamic_suppressor,
        calibration,
        test_loader,
        device,
        train_args,
        selected_by_key,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    individual_summary = export_individual_images(
        train_mod,
        individual_rows,
        records,
        pixel_lookup,
        args.output_dir / "individual_top_cases",
        args.individual_size,
    )
    summary = []
    for group in args.groups:
        rows = [row for row in selected_rows if row.get("requested_group") == group]
        n_drawn = draw_plate(
            train_mod,
            rows,
            records,
            pixel_lookup,
            args.output_dir / f"ours_{group}_top_cases.png",
            f"Ours {group} lesion top cases",
            args.tile,
        )
        summary.append({"group": group, "n_selected": len(rows), "n_drawn": n_drawn})

    n_drawn = draw_plate(
        train_mod,
        selected_rows,
        records,
        pixel_lookup,
        args.output_dir / "ours_all_size_groups_top_cases.png",
        "Ours lesion top cases by size group",
        args.tile,
    )
    summary.append({"group": "all", "n_selected": len(selected_rows), "n_drawn": n_drawn})

    missing = []
    for key, row in selected_by_key.items():
        if key not in records:
            miss = dict(row)
            miss["missing_reason"] = "selected case/slice was not found in test dataloader"
            missing.append(miss)
    write_csv(args.output_dir / "missing_selected_cases.csv", missing, list(missing[0].keys()) if missing else ["missing_reason"])
    write_csv(args.output_dir / "summary.csv", summary, ["group", "n_selected", "n_drawn"])
    with (args.output_dir / "manifest.json").open("w") as f:
        json.dump(
            {
                "group_root": str(args.group_root),
                "pixel_metrics_csv": str(args.pixel_metrics_csv) if args.pixel_metrics_csv else None,
                "output_dir": str(args.output_dir),
                "top_k_per_group": args.top_k,
                "individual_top_k_per_group": args.individual_top_k,
                "individual_output": str(args.output_dir / "individual_top_cases"),
                "n_individual_saved": int(sum(1 for row in individual_summary if row.get("status") == "saved")),
                "checkpoint": ckpt_full,
                "data_root": train_args.data_root,
                "quality_metrics": ["pixel_auroc", "pixel_aupr", "hotspot_recall", "hotspot_precision"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    for row in summary:
        print(row)
    print(f"Saved selected case figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
