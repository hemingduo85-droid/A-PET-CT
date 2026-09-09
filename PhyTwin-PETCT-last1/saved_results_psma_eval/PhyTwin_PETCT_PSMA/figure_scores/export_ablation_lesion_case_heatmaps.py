#!/usr/bin/env python3
"""
Run example:
cd /data/cyf/codes/A-PET-CT/PhyTwin-PETCT-last1
python export_ablation_lesion_case_heatmaps.py \
  --output_dir saved_results_psma_eval/PhyTwin_PETCT_PSMA/figure_scores/ours_ablation_lesion_heatmaps \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA \
  --load_ckpt --experiment_name PhyTwin_PETCT_PSMA \
  --score_mode lesion_z --adaptive_physio --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 --image_size 256 --gpu 4

FDG example:
cd /data/cyf/codes/A-PET-CT/PhyTwin-PETCT-last1
python export_ablation_lesion_case_heatmaps.py \
  --output_dir saved_results_fdg_eval/PhyTwin_PETCT_FDG/figure_scores/ours_ablation_lesion_heatmaps \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG \
  --load_ckpt --experiment_name PhyTwin_PETCT_FDG \
  --score_mode lesion_z --adaptive_physio --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 --image_size 256 --gpu 4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path


GROUPS = ("small", "medium", "large")
PANEL_NAMES = ("PET", "CT", "GT", "Pred", "w/o A-PHYSIO", "w/o MS-LAI", "MS-LAI evidence")
FIXED_CASES = {
    "small": {
        "PSMA": ("psma_1ca3af29b7df127d_2020-06-27", "0009"),
        "FDG": ("fdg_791ec15924_04-16-2001-NA-PET-CT Ganzkoerper  primaer mit KM-50308", "0053"),
    },
    "medium": {
        "PSMA": ("psma_ec45934c2fa23c76_2019-08-05", "0027"),
        "FDG": ("fdg_19b68a666b_04-14-2005-NA-PET-CT Ganzkoerper  primaer mit KM-01313", "0036"),
    },
    "large": {
        "PSMA": ("psma_18eba3b35ee1ddac_2020-09-05", "0036"),
        "FDG": ("fdg_ea0fd89f0f_10-25-2003-NA-PET-CT Ganzkoerper  primaer mit KM-32502", "0038"),
    },
}
EVIDENCE_COLORS = {
    0: (0, 255, 255, 255),
    1: (255, 255, 255, 255),
    2: (255, 0, 255, 255),
}


def import_case_helpers():
    import export_selected_lesion_case_figure as base

    return base


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def infer_tracer_from_args(train_args) -> str:
    text = " ".join(
        str(x)
        for x in [
            getattr(train_args, "data_root", ""),
            getattr(train_args, "experiment_name", ""),
            getattr(train_args, "method_name", ""),
        ]
    ).upper()
    if "FDG" in text:
        return "FDG"
    if "PSMA" in text:
        return "PSMA"
    raise ValueError("Cannot infer PSMA/FDG from --data_root or --experiment_name. Please include PSMA or FDG.")


def fixed_case_rows(groups: list[str], tracer: str) -> list[dict[str, str]]:
    rows = []
    for rank, group in enumerate(groups, start=1):
        patient, slice_id = FIXED_CASES[group][tracer]
        rows.append(
            {
                "requested_group": group,
                "lesion_size_group": group,
                "requested_rank": "1",
                "rank_in_size_group": "fixed",
                "case_id": patient,
                "slice_id": slice_id,
                "tracer": tracer,
                "fixed_case_order": str(rank),
                "lesion_pixels": "",
                "lesion_area_ratio": "",
            }
        )
    return rows


def no_adaptive_physio_args(train_args):
    args = deepcopy(train_args)
    args.no_physio = True
    args.adaptive_physio = False
    args.dynamic_physio = False
    args.use_dynamic_physio = False
    args.score_mode = "lesion_z"
    return args


def compute_map_and_score(train_mod, args, residual_z, memory_z, pet_gray, ct_gray, prior, calibration):
    prior_for_map = None if getattr(args, "no_physio", False) else prior
    final = train_mod.fuse_maps(
        residual_z,
        memory_z,
        residual_weight=args.residual_weight,
        memory_weight=args.memory_weight,
        physio_prior=prior_for_map,
        sigma=args.gaussian_sigma,
    )
    if bool(getattr(args, "dynamic_physio", False)):
        dynamic_suppressor = train_mod.DynamicPhysioSuppressor(
            alpha=args.dynamic_alpha,
            pet_quantile=args.dynamic_pet_quantile,
            min_area=args.dynamic_min_area,
        )
        final, _dynamic_mask = dynamic_suppressor.suppress(
            final,
            pet_gray,
            static_prior=None if prior_for_map is None else prior_for_map.prior,
        )
    adaptive_suppressor = train_mod.build_adaptive_physio(args)
    final, _adaptive_mask = train_mod.apply_adaptive_physio(
        final,
        pet_gray,
        ct_gray,
        prior_for_map,
        adaptive_suppressor,
    )

    prior_arr = None if prior_for_map is None else prior_for_map.prior
    raw_score = train_mod.compute_lesion_score(
        final,
        pet_gray,
        ct_gray,
        prior_for_map,
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
    pred_mask = train_mod.predict_mask(
        final,
        threshold_quantile=args.pred_mask_quantile,
        min_area=args.pred_mask_min_area,
    )
    return final, float(raw_score), float(image_score), pred_mask, prior_arr


def _normalize01(arr):
    import numpy as np

    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - float(arr.min())
    return arr / (float(arr.max()) + 1e-8)


def _body_and_surface(score, pet, ct):
    import numpy as np
    from scipy.ndimage import binary_erosion, gaussian_filter

    body_source = np.zeros_like(score, dtype=np.float32)
    if pet is not None:
        body_source = np.maximum(body_source, gaussian_filter(pet, sigma=2.0))
    if ct is not None:
        body_source = np.maximum(body_source, gaussian_filter(ct, sigma=2.0))
    body = body_source > max(0.03, float(np.quantile(body_source.reshape(-1), 0.20)))
    if not body.any():
        return None, None
    h, w = score.shape
    surface = body & ~binary_erosion(body, iterations=max(2, int(round(min(h, w) * 0.025))))
    return body, surface


def _component_feature(region, score, pet, ct, prior_overlap, boundary_overlap):
    import numpy as np

    h, w = score.shape
    yy, xx = np.nonzero(region)
    cy = float(yy.mean() / max(1, h - 1))
    cx = float(xx.mean() / max(1, w - 1))
    log_area = float(np.log1p(region.sum()) / np.log1p(score.size))
    pet_mean = 0.0 if pet is None else float(pet[region].mean())
    ct_mean = 0.0 if ct is None else float(ct[region].mean())
    peak = float(score[region].max() / (float(score.max()) + 1e-8))
    return np.asarray([cy, cx, log_area, prior_overlap, boundary_overlap, pet_mean, ct_mean, peak], dtype=np.float32)


def _hotspot_memory_factor(feature, memory, penalty=0.35, sigma=0.35):
    import numpy as np

    if memory is None:
        return 1.0
    mem = np.asarray(memory, dtype=np.float32)
    if mem.size == 0 or mem.ndim != 2:
        return 1.0
    dist = np.linalg.norm(mem - feature.reshape(1, -1), axis=1)
    nearest = float(dist.min())
    similarity = np.exp(-(nearest * nearest) / (2.0 * float(sigma) * float(sigma) + 1e-8))
    return max(0.15, 1.0 - float(penalty) * similarity)


def ms_lai_evidence_components(score_map, pet_map, ct_map, physio_prior, args, normal_hotspot_memory=None):
    """Return the same multi-scale candidate components used by lesion_z evidence aggregation."""
    import numpy as np
    from scipy.ndimage import binary_dilation, label

    score = np.asarray(score_map, dtype=np.float32)
    pet = _normalize01(pet_map) if pet_map is not None else None
    ct = _normalize01(ct_map) if ct_map is not None else None
    prior = None if physio_prior is None else np.asarray(physio_prior, dtype=np.float32)
    quantiles = getattr(args, "lesion_quantiles_list", None) or [args.lesion_quantile]
    quantiles = [float(q) for q in quantiles]
    _body, surface = _body_and_surface(score, pet, ct) if (pet is not None or ct is not None) else (None, None)
    h, w = score.shape
    by = max(1, int(round(h * float(args.fov_band_fraction))))
    bx = max(1, int(round(w * float(args.fov_band_fraction))))
    boundary = np.zeros_like(score, dtype=bool)
    boundary[:by, :] = True
    boundary[-by:, :] = True
    boundary[:, :bx] = True
    boundary[:, -bx:] = True

    out = []
    for scale_idx, q in enumerate(quantiles):
        threshold = float(np.quantile(score.reshape(-1), q))
        cc, num = label(score >= threshold)
        candidates = []
        for comp_idx in range(1, num + 1):
            region = cc == comp_idx
            area = int(region.sum())
            if area < int(args.lesion_min_area):
                continue
            area_fraction = area / float(score.size)
            mean_score = float(score[region].mean())
            peak_score = float(score[region].max())
            intensity = 0.65 * mean_score + 0.35 * peak_score

            compact_bonus = 1.0 + 0.35 * np.exp(-area / 80.0)
            size_gain = np.log1p(min(area, 180)) / np.log1p(180)
            mismatch_bonus = 1.0
            if pet is not None and ct is not None:
                ring = binary_dilation(region, iterations=6) & ~binary_dilation(region, iterations=1)
                if ring.any():
                    pet_contrast = float(pet[region].mean() - pet[ring].mean())
                    ct_contrast = abs(float(ct[region].mean() - ct[ring].mean()))
                    mismatch = max(0.0, pet_contrast - 0.5 * ct_contrast)
                    mismatch_bonus += min(0.45, 1.8 * mismatch)

            penalty = 1.0
            prior_overlap = 0.0 if prior is None else float(prior[region].mean())
            if prior is not None:
                penalty *= max(0.08, 1.0 - float(args.prior_penalty) * prior_overlap)
            if surface is not None and surface.any():
                surface_overlap = float(surface[region].mean())
                penalty *= max(0.20, 1.0 - float(args.edge_penalty) * surface_overlap)
            if area_fraction > 0.006:
                excess = min(1.0, (area_fraction - 0.006) / 0.05)
                penalty *= max(0.20, 1.0 - float(args.large_area_penalty) * excess)

            boundary_overlap = float(boundary[region].mean())
            if boundary_overlap > 0.0:
                penalty *= max(0.12, 1.0 - float(args.fov_penalty) * boundary_overlap)
            if prior_overlap >= float(args.organ_prior_threshold) and area_fraction >= float(args.organ_area_fraction):
                organ_excess = min(1.0, area_fraction / max(float(args.organ_area_fraction), 1e-8))
                penalty *= max(0.08, 1.0 - 0.55 * organ_excess)

            feature = _component_feature(region, score, pet, ct, prior_overlap, boundary_overlap)
            penalty *= _hotspot_memory_factor(
                feature,
                normal_hotspot_memory,
                penalty=args.hotspot_penalty,
                sigma=args.hotspot_sigma,
            )
            evidence_score = intensity * compact_bonus * (0.55 + 0.45 * size_gain) * mismatch_bonus * penalty
            candidates.append(
                {
                    "scale_idx": scale_idx,
                    "quantile": q,
                    "region": region,
                    "area": area,
                    "evidence_score": float(evidence_score),
                    "mean_score": mean_score,
                    "peak_score": peak_score,
                    "prior_overlap": prior_overlap,
                    "boundary_overlap": boundary_overlap,
                }
            )
        candidates.sort(key=lambda item: item["evidence_score"], reverse=True)
        out.extend(candidates[: int(args.lesion_max_components)])
    return out


def evidence_overlay_image(base, heatmap_img, components, max_components=5):
    import numpy as np
    from PIL import Image, ImageDraw, ImageEnhance

    # Circle MS-LAI evidence on the post-A-PHYSIO full A* map. The background is
    # deliberately darkened so this panel reads as evidence selection, not a new
    # anomaly heatmap.
    overlay = ImageEnhance.Brightness(heatmap_img.convert("RGB")).enhance(0.35).convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    shown = sorted(components, key=lambda x: x["evidence_score"], reverse=True)[: max(1, int(max_components))]
    for rank, item in enumerate(shown, start=1):
        region = np.asarray(item["region"], dtype=bool)
        ys0, xs0 = np.nonzero(region)
        if len(xs0) == 0:
            continue
        x0 = int(round(xs0.min() * (w - 1) / max(1, region.shape[1] - 1)))
        x1 = int(round(xs0.max() * (w - 1) / max(1, region.shape[1] - 1)))
        y0 = int(round(ys0.min() * (h - 1) / max(1, region.shape[0] - 1)))
        y1 = int(round(ys0.max() * (h - 1) / max(1, region.shape[0] - 1)))
        min_box = 34
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        half_w = max(min_box // 2, int((x1 - x0 + 1) * 0.75) + 10)
        half_h = max(min_box // 2, int((y1 - y0 + 1) * 0.75) + 10)
        x0, x1 = max(0, cx - half_w), min(w - 1, cx + half_w)
        y0, y1 = max(0, cy - half_h), min(h - 1, cy + half_h)
        color = EVIDENCE_COLORS.get(int(item["scale_idx"]), (0, 255, 255, 255))

        # Thick concentric ellipse: visible on every jet color.
        draw.ellipse((x0, y0, x1, y1), outline=(255, 255, 255, 255), width=10)
        draw.ellipse((x0, y0, x1, y1), outline=(0, 0, 0, 255), width=6)
        draw.ellipse((x0, y0, x1, y1), outline=color, width=3)

        # Add a small center marker for tiny lesions.
        r = 4
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 255))
        draw.ellipse((cx - r + 1, cy - r + 1, cx + r - 1, cy + r - 1), fill=(0, 0, 0, 255))
        draw.text((x0 + 3, max(0, y0 - 16)), str(rank), fill=(255, 255, 255, 255))
    return overlay.convert("RGB")


def infer_records(train_mod, model, memory, prior, calibration, loader, device, train_args, selected_by_key):
    import numpy as np
    import torch

    model.eval()
    if memory is not None:
        memory = memory.to(device)
    records = {}
    no_physio = no_adaptive_physio_args(train_args)

    with torch.no_grad():
        for _idx, batch in train_mod.limited(loader, train_args.max_test_batches):
            path = batch["path"][0] if isinstance(batch["path"], (list, tuple)) else str(batch["path"])
            key = base.row_key({"path": path})
            if key not in selected_by_key:
                continue

            pet = batch["pet"].to(device)
            ct = batch["ct"].to(device)
            pred, logvar = model(ct)
            residual = train_mod.residual_map(pet, pred, logvar if train_args.uncertainty else None)
            residual_np = residual[0].detach().cpu().numpy().astype(np.float32)
            residual_z = train_mod.positive_zscore_map(
                residual_np,
                calibration["residual_mean"],
                calibration["residual_std"],
            )
            memory_z = train_mod.memory_z_map(memory, residual, calibration, train_args)
            pet_gray = train_mod.pet_tensor_to_gray(batch["pet"])[0].numpy().astype(np.float32)
            ct_gray = train_mod.pet_tensor_to_gray(batch["ct"])[0].numpy().astype(np.float32)
            gt_mask = batch["mask"][0].cpu()

            full_map, raw_score, image_score, pred_mask, prior_arr = compute_map_and_score(
                train_mod,
                train_args,
                residual_z,
                memory_z,
                pet_gray,
                ct_gray,
                prior,
                calibration,
            )
            no_physio_map, no_physio_raw, no_physio_score, no_physio_pred, _ = compute_map_and_score(
                train_mod,
                no_physio,
                residual_z,
                memory_z,
                pet_gray,
                ct_gray,
                prior,
                calibration,
            )
            components = ms_lai_evidence_components(
                full_map,
                pet_gray,
                ct_gray,
                prior_arr,
                train_args,
                normal_hotspot_memory=calibration.get("normal_hotspot_memory") if train_args.use_hotspot_memory else None,
            )
            records[key] = {
                "path": path,
                "pet": batch["pet"][0].cpu(),
                "ct": batch["ct"][0].cpu(),
                "gt": gt_mask,
                "pred": pred_mask,
                "full": full_map,
                "full_raw_score": raw_score,
                "full_image_score": image_score,
                "without_a_physio": no_physio_map,
                "without_a_physio_raw_score": no_physio_raw,
                "without_a_physio_image_score": no_physio_score,
                "without_a_physio_pred": no_physio_pred,
                "components": components,
                "mask_metrics": base.mask_metrics(gt_mask.numpy(), pred_mask),
                "pixel_metrics": base.pixel_metrics_from_map(full_map, gt_mask.numpy()),
                "heatmap_metrics": base.heatmap_overlap_metrics(full_map, gt_mask.numpy()),
            }
            if len(records) == len(selected_by_key):
                break
    return records


def render_panels(train_mod, base, record, tile: int):
    ucr_a_physio_heatmap = base.render_reference_style_heatmap(record["full"]).resize((256, 256))
    no_physio_heatmap = base.render_reference_style_heatmap(record["without_a_physio"]).resize((256, 256))
    evidence = evidence_overlay_image(base, ucr_a_physio_heatmap, record["components"]).resize((256, 256))
    panels = [
        train_mod._panel(record["pet"], "PET", gt_mask=record["gt"], pred_mask=record["pred"]),
        train_mod._panel(record["ct"], "CT"),
        train_mod._panel(record["gt"], "GT", mask_fill=True),
        train_mod._panel(record["pred"], "Pred", pred_fill=True),
        no_physio_heatmap,
        ucr_a_physio_heatmap,
        evidence,
    ]
    return [panel.resize((tile, tile)) for panel in panels]


def export_individual_images(train_mod, base, rows, records, pixel_lookup, output_dir: Path, image_size: int):
    summary = []
    for row in rows:
        group = row.get("requested_group", row.get("lesion_size_group", "unknown"))
        key = base.row_key(row)
        if key not in records:
            summary.append({"group": group, "case_id": key[0], "slice_id": key[1], "status": "missing_in_test_loader"})
            continue
        record = records[key]
        case_dir = output_dir / group / f"{key[0]}_{key[1]}"
        pet_img = base.colorize(record["pet"], "magma")
        ct_img = base.gray_image(record["ct"])
        gt_img = base.mask_image(record["gt"])
        pred_img = train_mod._panel(record["pred"], "Pred", pred_fill=True)
        ucr_a_physio_heatmap = base.render_reference_style_heatmap(record["full"])
        no_physio_heatmap = base.render_reference_style_heatmap(record["without_a_physio"])
        evidence_img = evidence_overlay_image(base, ucr_a_physio_heatmap.resize((256, 256)), record["components"])

        base.save_image(case_dir / "PET_enhanced.png", pet_img, image_size)
        base.save_image(case_dir / "CT.png", ct_img, image_size)
        base.save_image(case_dir / "GT.png", gt_img, image_size)
        base.save_image(case_dir / "Pred.png", pred_img, image_size)
        base.save_image(case_dir / "Full_heatmap_jet.png", ucr_a_physio_heatmap, image_size)
        base.save_image(case_dir / "UCR_A_PHYSIO_heatmap_jet.png", ucr_a_physio_heatmap, image_size)
        base.save_image(case_dir / "without_A_PHYSIO_heatmap_jet.png", no_physio_heatmap, image_size)
        base.save_image(case_dir / "MS_LAI_evidence_overlay.png", evidence_img, image_size)

        mm = record["mask_metrics"]
        pm = dict(record["pixel_metrics"])
        pm.update(pixel_lookup.get(key, {}))
        hm = record["heatmap_metrics"]
        scale_counts = {}
        for component in record["components"]:
            q = f"{component['quantile']:.3f}"
            scale_counts[q] = scale_counts.get(q, 0) + 1
        meta = {
            "group": group,
            "case_id": key[0],
            "slice_id": key[1],
            "rank_in_size_group": row.get("rank_in_size_group", row.get("requested_rank", "")),
            "full_raw_lesion_evidence": record["full_raw_score"],
            "full_image_score_z": record["full_image_score"],
            "without_a_physio_raw_lesion_evidence": record["without_a_physio_raw_score"],
            "without_a_physio_image_score_z": record["without_a_physio_image_score"],
            "pixel_auroc": pm.get("pixel_auroc", ""),
            "pixel_aupr": pm.get("pixel_aupr", ""),
            "hotspot_recall": hm["hotspot_recall"],
            "hotspot_precision": hm["hotspot_precision"],
            "pred_dice_reference": mm["dice"],
            "pred_iou_reference": mm["iou"],
            "ms_lai_evidence_components": len(record["components"]),
            "ms_lai_components_by_quantile": json.dumps(scale_counts, sort_keys=True),
            "path": record["path"],
            "status": "saved",
        }
        with (case_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        write_csv(case_dir / "metrics.csv", [meta], list(meta.keys()))
        summary.append(meta)
    return summary


def draw_plate(train_mod, base, rows, records, pixel_lookup, save_path: Path, title: str, tile: int):
    from PIL import Image, ImageDraw

    drawable = [(row, records[base.row_key(row)]) for row in rows if base.row_key(row) in records]
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
    font_title = base.load_font(18, bold=True)
    font_col = base.load_font(13, bold=True)
    font_meta = base.load_font(12)

    y = margin_top
    draw.text((margin_x, y), f"{title} (n={len(drawable)})", fill=(0, 0, 0), font=font_title)
    y += col_h
    x = margin_x
    for name in PANEL_NAMES:
        draw.text((x, y), name, fill=(0, 0, 0), font=font_col)
        x += tile + gap
    y += col_h

    for row, record in drawable:
        key = base.row_key(row)
        mm = record["mask_metrics"]
        pm = dict(record["pixel_metrics"])
        pm.update(pixel_lookup.get(key, {}))
        hm = record["heatmap_metrics"]
        lesion_pixels = int(float(row.get("lesion_pixels", mm["gt_pixels"]) or 0))
        area = 100.0 * float(row.get("lesion_area_ratio", 0.0) or 0.0)
        meta = (
            f"{row.get('requested_group', row.get('lesion_size_group', '')).capitalize()} #{row.get('requested_rank', row.get('rank_in_size_group', ''))}  |  "
            f"Full z={record['full_image_score']:.3f}  w/o A-PHYSIO z={record['without_a_physio_image_score']:.3f}  |  "
            f"pAUROC={base.fmt_metric(pm.get('pixel_auroc'))}  pAUPR={base.fmt_metric(pm.get('pixel_aupr'))}  |  "
            f"HotRecall={hm['hotspot_recall']:.3f}  HotPrecision={hm['hotspot_precision']:.3f}  |  "
            f"MS-LAI comps={len(record['components'])}  |  Lesion={lesion_pixels} px ({area:.3f}%)  |  {key[0]} / {key[1]}"
        )
        draw.rectangle((margin_x - 2, y - 1, width - margin_x + 2, y + header_h - 4), fill=(245, 245, 245))
        draw.text((margin_x + 2, y + 6), meta, fill=(0, 0, 0), font=font_meta)
        y_img = y + header_h
        x = margin_x
        for panel in render_panels(train_mod, base, record, tile):
            canvas.paste(panel, (x, y_img))
            draw.rectangle((x, y_img, x + tile - 1, y_img + tile - 1), outline=(60, 60, 60), width=1)
            x += tile + gap
        y += row_h

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
    return len(drawable)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group_root", type=Path, default=None, help="Optional lesion_size_groups root. If omitted, fixed paper cases are used.")
    parser.add_argument("--pixel_metrics_csv", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tile", type=int, default=160)
    parser.add_argument("--individual_top_k", type=int, default=5)
    parser.add_argument("--individual_size", type=int, default=256)
    args, train_argv = parser.parse_known_args(argv)
    return args, train_argv


def main(argv=None):
    global base
    base = import_case_helpers()
    args, train_argv = parse_args(sys.argv[1:] if argv is None else argv)
    train_mod, train_args = base.import_train_args(train_argv)
    from phytwin_petct import visualization as vis_mod

    train_mod._panel = vis_mod._panel
    train_mod._heatmap = vis_mod._heatmap
    train_args.score_mode = "lesion_z"
    train_args.use_dynamic_physio = bool(getattr(train_args, "dynamic_physio", False))

    pixel_lookup = base.load_pixel_metrics(args.pixel_metrics_csv)
    if args.group_root is None:
        tracer = infer_tracer_from_args(train_args)
        selected_rows = fixed_case_rows(args.groups, tracer)
        individual_rows = list(selected_rows)
    else:
        top_k = None if args.top_k <= 0 else args.top_k
        selected_rows = base.load_selected_rows(args.group_root, args.groups, top_k)
        individual_rows = base.top_individual_rows_by_group(
            args.group_root,
            args.groups,
            max(1, int(args.individual_top_k)),
        )
    selected_by_key = {base.row_key(row): row for row in (selected_rows + individual_rows)}

    model, memory, prior, _dynamic_suppressor, calibration, test_loader, device, ckpt_full = base.load_phytwin(
        train_mod,
        train_args,
    )
    records = infer_records(
        train_mod,
        model,
        memory,
        prior,
        calibration,
        test_loader,
        device,
        train_args,
        selected_by_key,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    individual_summary = export_individual_images(
        train_mod,
        base,
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
            base,
            rows,
            records,
            pixel_lookup,
            args.output_dir / f"ablation_{group}_top_cases.png",
            f"Ablation and MS-LAI evidence {group} lesion top cases",
            args.tile,
        )
        summary.append({"group": group, "n_selected": len(rows), "n_drawn": n_drawn})
    n_drawn = draw_plate(
        train_mod,
        base,
        selected_rows,
        records,
        pixel_lookup,
        args.output_dir / "ablation_all_size_groups_top_cases.png",
        "Ablation and MS-LAI evidence lesion top cases by size group",
        args.tile,
    )
    summary.append({"group": "all", "n_selected": len(selected_rows), "n_drawn": n_drawn})

    write_csv(args.output_dir / "individual_summary.csv", individual_summary, list(individual_summary[0].keys()) if individual_summary else ["status"])
    write_csv(args.output_dir / "summary.csv", summary, ["group", "n_selected", "n_drawn"])
    missing = []
    for key, row in selected_by_key.items():
        if key not in records:
            miss = dict(row)
            miss["missing_reason"] = "selected case/slice was not found in test dataloader"
            missing.append(miss)
    write_csv(args.output_dir / "missing_selected_cases.csv", missing, list(missing[0].keys()) if missing else ["missing_reason"])
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "group_root": str(args.group_root),
                "pixel_metrics_csv": str(args.pixel_metrics_csv) if args.pixel_metrics_csv else None,
                "output_dir": str(args.output_dir),
                "top_k_per_group": args.top_k,
                "individual_top_k_per_group": args.individual_top_k,
                "case_selection": "fixed_paper_cases" if args.group_root is None else "lesion_size_group_csv",
                "checkpoint": ckpt_full,
                "panels": list(PANEL_NAMES),
                "ms_lai_evidence_note": (
                    "The w/o MS-LAI panel shows the post-UCR and post-A-PHYSIO A* heatmap before lesion-aware "
                    "evidence aggregation. MS-LAI evidence overlays the selected multi-scale candidate components "
                    "on that same A* map. It is an interpretability visualization, not a new anomaly heatmap and "
                    "not a top-1% replacement score."
                ),
                "evidence_outline_colors": {
                    "low_quantile": "green",
                    "middle_quantile": "yellow",
                    "high_quantile": "red",
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved ablation and MS-LAI evidence figures to {args.output_dir}")


if __name__ == "__main__":
    main()
