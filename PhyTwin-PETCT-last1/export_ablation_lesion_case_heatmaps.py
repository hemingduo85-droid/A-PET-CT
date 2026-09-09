#!/usr/bin/env python3
"""
Run example:
cd /data/cyf/codes/A-PET-CT/PhyTwin-PETCT-last1
python export_ablation_lesion_case_heatmaps.py \
  --output_dir saved_results_psma_eval/PhyTwin_PETCT_PSMA/figure_scores/ours_ablation_lesion_heatmaps \
  --without_ucr_experiment_name PhyTwin_PETCT_PSMA_no_uncertainty_residual \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA \
  --load_ckpt --experiment_name PhyTwin_PETCT_PSMA \
  --score_mode lesion_z --adaptive_physio --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 --image_size 256 --gpu 4

FDG example:
cd /data/cyf/codes/A-PET-CT/PhyTwin-PETCT-last1
python export_ablation_lesion_case_heatmaps.py \
  --group_root saved_results_fdg_eval/PhyTwin_PETCT_FDG/figure_scores/lesion_size_groups \
  --output_dir saved_results_fdg_eval/PhyTwin_PETCT_FDG/figure_scores/ours_ablation_lesion_heatmaps_visible \
  --select_a_physio_gain --top_k 300 --selected_per_group 5 \
  --without_ucr_experiment_name PhyTwin_PETCT_FDG_no_uncertainty_residual \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG \
  --load_ckpt --experiment_name PhyTwin_PETCT_FDG \
  --score_mode lesion_z --adaptive_physio --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 --image_size 256 --gpu 4

python export_ablation_lesion_case_heatmaps.py \
  --output_dir saved_results_fdg_eval/PhyTwin_PETCT_FDG/figure_scores/ours_ablation_fixed_cases \
  --without_ucr_experiment_name PhyTwin_PETCT_FDG_no_uncertainty_residual \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG \
  --load_ckpt \
  --experiment_name PhyTwin_PETCT_FDG \
  --score_mode lesion_z \
  --adaptive_physio \
  --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 \
  --image_size 256 \
  --gpu 5

python export_ablation_lesion_case_heatmaps.py \
  --output_dir saved_results_psma_eval/PhyTwin_PETCT_PSMA/figure_scores/ours_ablation_fixed_cases \
  --without_ucr_experiment_name PhyTwin_PETCT_PSMA_no_uncertainty_residual \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA \
  --load_ckpt \
  --experiment_name PhyTwin_PETCT_PSMA \
  --score_mode lesion_z \
  --adaptive_physio \
  --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 \
  --image_size 256 \
  --gpu 4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path


GROUPS = ("small", "medium", "large")
PANEL_NAMES = (
    "PET",
    "CT",
    "GT",
    "Pred",
    "w/o A-PHYSIO",
    "w/o MS-LAI",
    "A-PHYSIO effect",
    "MS-LAI evidence",
)
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


def select_a_physio_gain_rows(
    rows,
    records,
    key_fn,
    groups,
    min_aupr_gain=0.0,
    max_hotspot_drop=0.05,
    top_per_group=1,
    min_visible_suppression_fraction=0.0,
    min_lesion_retention=0.0,
    min_pred_dice=0.0,
    min_pred_precision=0.0,
):
    """Select positive A-PHYSIO examples, prioritizing clean final localization."""

    audit = []
    eligible_by_group = {group: [] for group in groups}
    for row in rows:
        key = key_fn(row)
        group = row.get("requested_group", row.get("lesion_size_group", "unknown"))
        record = records.get(key)
        entry = {
            "group": group,
            "case_id": key[0],
            "slice_id": key[1],
            "full_pixel_auroc": "",
            "without_a_physio_pixel_auroc": "",
            "full_pixel_aupr": "",
            "without_a_physio_pixel_aupr": "",
            "pixel_aupr_gain": "",
            "full_hotspot_recall": "",
            "without_a_physio_hotspot_recall": "",
            "hotspot_recall_gain": "",
            "outside_suppression_ratio": "",
            "visible_suppression_fraction": "",
            "lesion_retention": "",
            "visual_effect_score": "",
            "pred_dice": "",
            "pred_precision": "",
            "final_localization_quality": "",
            "balanced_selection_score": "",
            "selection_rank": "",
            "selection_status": "missing_record",
        }
        if record is None:
            audit.append(entry)
            continue

        full_pm = record.get("pixel_metrics", {})
        without_pm = record.get("without_a_physio_pixel_metrics", {})
        full_hm = record.get("heatmap_metrics", {})
        without_hm = record.get("without_a_physio_heatmap_metrics", {})
        effect = record.get("a_physio_effect_metrics", {})
        mask_metrics = record.get("mask_metrics", {})
        try:
            full_auroc = float(full_pm["pixel_auroc"])
            without_auroc = float(without_pm["pixel_auroc"])
            full_aupr = float(full_pm["pixel_aupr"])
            without_aupr = float(without_pm["pixel_aupr"])
            full_hot = float(full_hm["hotspot_recall"])
            without_hot = float(without_hm["hotspot_recall"])
            outside_ratio = float(effect.get("outside_suppression_ratio", 0.0))
            visible_fraction = float(effect.get("visible_suppression_fraction", 0.0))
            lesion_retention = float(effect.get("lesion_retention", 1.0))
            pred_dice = float(mask_metrics.get("dice", 0.0))
            pred_precision = float(mask_metrics.get("precision", 0.0))
        except (KeyError, TypeError, ValueError):
            entry["selection_status"] = "rejected_missing_metrics"
            audit.append(entry)
            continue

        aupr_gain = full_aupr - without_aupr
        hot_gain = full_hot - without_hot
        visual_effect_score = outside_ratio * (visible_fraction ** 0.5)
        localization_quality = (
            0.60 * full_aupr
            + 0.25 * full_hot
            + 0.075 * pred_dice
            + 0.075 * pred_precision
        )
        balanced_selection_score = (
            localization_quality
            + 0.10 * min(1.0, max(0.0, visual_effect_score))
            + 0.10 * min(1.0, max(0.0, aupr_gain))
        )
        entry.update(
            {
                "full_pixel_auroc": full_auroc,
                "without_a_physio_pixel_auroc": without_auroc,
                "full_pixel_aupr": full_aupr,
                "without_a_physio_pixel_aupr": without_aupr,
                "pixel_aupr_gain": aupr_gain,
                "full_hotspot_recall": full_hot,
                "without_a_physio_hotspot_recall": without_hot,
                "hotspot_recall_gain": hot_gain,
                "outside_suppression_ratio": outside_ratio,
                "visible_suppression_fraction": visible_fraction,
                "lesion_retention": lesion_retention,
                "visual_effect_score": visual_effect_score,
                "pred_dice": pred_dice,
                "pred_precision": pred_precision,
                "final_localization_quality": localization_quality,
                "balanced_selection_score": balanced_selection_score,
            }
        )
        if aupr_gain <= float(min_aupr_gain):
            entry["selection_status"] = "rejected_nonpositive_aupr_gain"
        elif hot_gain < -float(max_hotspot_drop):
            entry["selection_status"] = "rejected_hotspot_drop"
        elif visible_fraction < float(min_visible_suppression_fraction):
            entry["selection_status"] = "rejected_weak_visual_effect"
        elif lesion_retention < float(min_lesion_retention):
            entry["selection_status"] = "rejected_lesion_suppression"
        elif pred_dice < float(min_pred_dice) or pred_precision < float(min_pred_precision):
            entry["selection_status"] = "rejected_poor_final_localization"
        else:
            entry["selection_status"] = "eligible_not_selected"
            eligible_by_group.setdefault(group, []).append((row, entry))
        audit.append(entry)

    selected = []
    for group in groups:
        candidates = eligible_by_group.get(group, [])
        if not candidates:
            continue
        candidates = sorted(
            candidates,
            key=lambda item: (
                float(item[1]["balanced_selection_score"]),
                float(item[1]["final_localization_quality"]),
                float(item[1]["visual_effect_score"]),
                float(item[1]["outside_suppression_ratio"]),
                float(item[1]["pixel_aupr_gain"]),
            ),
            reverse=True,
        )
        for rank, (row, entry) in enumerate(candidates[: max(1, int(top_per_group))], start=1):
            entry["selection_status"] = "selected"
            entry["selection_rank"] = rank
            selected_row = dict(row)
            selected_row["requested_group"] = group
            selected_row["requested_rank"] = str(rank)
            selected_row["selection_reason"] = (
                "positive A-PHYSIO pixel-AUPR gain with stable hotspot recall, "
                "ranked by final heatmap localization quality"
            )
            selected.append(selected_row)
    return selected, audit


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


def without_ucr_args(train_mod, train_args, experiment_name):
    args = deepcopy(train_args)
    args.experiment_name = str(experiment_name)
    args.method_name = train_mod.safe_name(experiment_name)
    args.no_uncertainty = True
    args.uncertainty = False
    args.load_ckpt = True
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


def normalize_maps_shared(map_a, map_b, lower_percentile=1.0, upper_percentile=99.5):
    """Normalize two maps with one joint display range."""
    return tuple(
        normalize_maps_joint(
            map_a,
            map_b,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
    )


def normalize_maps_joint(*maps, lower_percentile=1.0, upper_percentile=99.5):
    """Normalize any number of maps with one joint display range."""
    import numpy as np

    arrays = [np.asarray(item, dtype=np.float32) for item in maps]
    if not arrays:
        return []
    finite_parts = [arr[np.isfinite(arr)] for arr in arrays]
    finite = np.concatenate([part for part in finite_parts if part.size]) if any(part.size for part in finite_parts) else np.array([])
    if finite.size == 0:
        return [np.zeros_like(arr, dtype=np.float32) for arr in arrays]
    lo, hi = np.percentile(finite, [float(lower_percentile), float(upper_percentile)])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0

    def normalize(arr):
        arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    return [normalize(arr) for arr in arrays]


def compute_a_physio_effect_metrics(full_map, without_map, gt_mask):
    """Quantify visible off-lesion suppression and lesion response retention."""
    import numpy as np

    full = np.asarray(full_map, dtype=np.float32)
    without = np.asarray(without_map, dtype=np.float32)
    gt = np.squeeze(np.asarray(gt_mask)) > 0.5
    if gt.shape != full.shape:
        import cv2

        gt = cv2.resize(
            gt.astype(np.uint8),
            (full.shape[1], full.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    diff = a_physio_suppression_map(full, without)
    finite = np.concatenate([full[np.isfinite(full)], without[np.isfinite(without)]])
    if finite.size:
        lo, hi = np.percentile(finite, [1.0, 99.5])
        span = max(float(hi - lo), 1e-8)
    else:
        span = 1.0
    outside = ~gt
    outside_energy = float(np.clip(without[outside], 0.0, None).sum()) if outside.any() else 0.0
    removed_outside = float(diff[outside].sum()) if outside.any() else 0.0
    visible_fraction = (
        float((diff[outside] >= 0.05 * span).mean())
        if outside.any()
        else 0.0
    )
    lesion_before = float(np.clip(without[gt], 0.0, None).sum()) if gt.any() else 0.0
    lesion_after = float(np.clip(full[gt], 0.0, None).sum()) if gt.any() else 0.0
    lesion_retention = lesion_after / max(lesion_before, 1e-8) if gt.any() else 0.0
    outside_ratio = removed_outside / max(outside_energy, 1e-8)
    visual_effect_score = outside_ratio * (visible_fraction ** 0.5)
    return {
        "outside_suppression_ratio": outside_ratio,
        "visible_suppression_fraction": visible_fraction,
        "lesion_retention": lesion_retention,
        "visual_effect_score": visual_effect_score,
    }


def a_physio_suppression_map(full_map, without_map):
    """Return only anomaly response removed by A-PHYSIO."""
    import numpy as np

    full = np.asarray(full_map, dtype=np.float32)
    without = np.asarray(without_map, dtype=np.float32)
    return np.clip(without - full, 0.0, None).astype(np.float32)


def render_shared_jet_pair(full_map, without_map):
    """Render full and ablated maps with an identical Jet display transform."""
    return tuple(render_shared_jet_maps(full_map, without_map))


def render_shared_jet_maps(*maps):
    """Render comparable maps with one identical Jet display transform."""
    import numpy as np
    from PIL import Image

    normalized = normalize_maps_joint(*maps)

    def render(arr):
        try:
            from matplotlib import colormaps

            rgb = colormaps["jet"](arr)[..., :3]
        except Exception:
            r = np.clip(1.5 - np.abs(4.0 * arr - 3.0), 0.0, 1.0)
            g = np.clip(1.5 - np.abs(4.0 * arr - 2.0), 0.0, 1.0)
            b = np.clip(1.5 - np.abs(4.0 * arr - 1.0), 0.0, 1.0)
            rgb = np.stack([r, g, b], axis=-1)
        return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))

    return [render(arr) for arr in normalized]


def render_a_physio_effect_image(base, full_map, without_map, gt_mask):
    effect = a_physio_suppression_map(full_map, without_map)
    image = base.colorize(effect, "jet")
    return base.overlay_gt(image, gt_mask, color=(255, 255, 255, 255), width=2)


def compose_a_physio_comparison(base, without_img, full_img, effect_img, record, tile=256):
    from PIL import Image, ImageDraw

    labels = ("w/o A-PHYSIO", "with A-PHYSIO", "Suppressed response")
    gap = 8
    label_h = 30
    meta_h = 28
    width = 3 * tile + 2 * gap
    canvas = Image.new("RGB", (width, label_h + tile + meta_h), "white")
    draw = ImageDraw.Draw(canvas)
    font_label = base.load_font(18, bold=True)
    font_meta = base.load_font(14)
    for idx, (label, image) in enumerate(zip(labels, (without_img, full_img, effect_img))):
        x = idx * (tile + gap)
        draw.text((x + 5, 4), label, fill=(0, 0, 0), font=font_label)
        canvas.paste(image.resize((tile, tile)), (x, label_h))
    full_aupr = record["pixel_metrics"].get("pixel_aupr", "")
    without_aupr = record["without_a_physio_pixel_metrics"].get("pixel_aupr", "")
    gain = (
        float(full_aupr) - float(without_aupr)
        if full_aupr not in ("", None) and without_aupr not in ("", None)
        else None
    )
    effect = record["a_physio_effect_metrics"]
    meta = (
        f"AUPR full/w/o={base.fmt_metric(full_aupr)}/{base.fmt_metric(without_aupr)}  "
        f"delta={'NA' if gain is None else f'{gain:+.3f}'}  "
        f"outside suppression={effect['outside_suppression_ratio']:.3f}  "
        f"lesion retention={effect['lesion_retention']:.3f}"
    )
    draw.text((5, label_h + tile + 5), meta, fill=(0, 0, 0), font=font_meta)
    return canvas


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


def merge_overlapping_evidence_components(components, overlap_threshold=0.5, max_components=5):
    """Merge multi-scale components that describe the same spatial hotspot."""
    import numpy as np

    clusters = []
    ordered = sorted(components, key=lambda item: float(item["evidence_score"]), reverse=True)
    for item in ordered:
        region = np.asarray(item["region"], dtype=bool)
        if not region.any():
            continue

        matched = []
        for idx, cluster in enumerate(clusters):
            cluster_region = cluster["region"]
            intersection = int(np.logical_and(region, cluster_region).sum())
            smaller_area = min(int(region.sum()), int(cluster_region.sum()))
            containment = intersection / float(max(1, smaller_area))
            if containment >= float(overlap_threshold):
                matched.append(idx)

        merged = dict(item)
        merged["region"] = region.copy()
        merged["scale_indices"] = {int(item.get("scale_idx", 0))}
        merged["member_count"] = 1
        if matched:
            first = matched[0]
            for idx in reversed(matched):
                cluster = clusters.pop(idx)
                merged["region"] = np.logical_or(merged["region"], cluster["region"])
                merged["evidence_score"] = max(
                    float(merged["evidence_score"]),
                    float(cluster["evidence_score"]),
                )
                merged["scale_indices"].update(cluster["scale_indices"])
                merged["member_count"] += int(cluster["member_count"])
            clusters.insert(first, merged)
        else:
            clusters.append(merged)

    clusters.sort(key=lambda item: float(item["evidence_score"]), reverse=True)
    return clusters[: max(1, int(max_components))]


def _intersection_area(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    return max(0, right - left) * max(0, bottom - top)


def _label_box_candidates(cx, cy, half_w, half_h, tw, th, canvas_w, canvas_h):
    gap = 8
    raw = [
        (cx - tw // 2, cy - half_h - th - gap),
        (cx + half_w + gap, cy - th // 2),
        (cx - half_w - tw - gap, cy - th // 2),
        (cx - tw // 2, cy + half_h + gap),
        (cx + half_w + gap, cy - half_h - th // 2),
        (cx - half_w - tw - gap, cy + half_h - th // 2),
        (cx + half_w + gap, cy + half_h - th // 2),
        (cx - half_w - tw - gap, cy - half_h - th // 2),
    ]
    boxes = []
    for x, y in raw:
        x = min(max(4, int(x)), max(4, canvas_w - tw - 5))
        y = min(max(4, int(y)), max(4, canvas_h - th - 5))
        box = (x - 4, y - 3, x + tw + 5, y + th + 5)
        if box not in boxes:
            boxes.append(box)
    return boxes


def evidence_overlay_image(base, heatmap_img, components, max_components=5):
    import numpy as np
    from PIL import Image, ImageDraw, ImageEnhance

    # Circle MS-LAI evidence on the post-A-PHYSIO full A* map. The background is
    # deliberately darkened so this panel reads as evidence selection, not a new
    # anomaly heatmap.
    overlay = ImageEnhance.Brightness(heatmap_img.convert("RGB")).enhance(0.35).convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    label_font = base.load_font(28, bold=True)
    shown = merge_overlapping_evidence_components(components, max_components=max_components)
    annotations = []
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
        label = str(rank)
        try:
            bbox = draw.textbbox((0, 0), label, font=label_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 18, 24

        annotations.append(
            {
                "rank": rank,
                "label": label,
                "text_size": (tw, th),
                "center": (cx, cy),
                "half_size": (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)),
                "circle_box": (x0, y0, x1, y1),
            }
        )

    # Draw every region first so later circles cannot cover earlier labels.
    for annotation in annotations:
        draw.ellipse(annotation["circle_box"], outline=(255, 255, 255, 255), width=5)

    occupied = [annotation["circle_box"] for annotation in annotations]
    label_boxes = []
    for annotation in annotations:
        cx, cy = annotation["center"]
        half_w, half_h = annotation["half_size"]
        tw, th = annotation["text_size"]
        candidates = _label_box_candidates(cx, cy, half_w, half_h, tw, th, w, h)
        label_box = min(
            candidates,
            key=lambda box: sum(_intersection_area(box, other) for other in occupied + label_boxes),
        )
        label_boxes.append(label_box)

        label_cx = (label_box[0] + label_box[2]) / 2.0
        label_cy = (label_box[1] + label_box[3]) / 2.0
        dx = label_cx - cx
        dy = label_cy - cy
        denom = np.sqrt((dx / max(1, half_w)) ** 2 + (dy / max(1, half_h)) ** 2)
        scale = 1.0 / max(1.0, float(denom))
        anchor = (int(round(cx + dx * scale)), int(round(cy + dy * scale)))
        endpoint = (int(round(label_cx)), int(round(label_cy)))
        draw.line((anchor, endpoint), fill=(0, 0, 0, 255), width=5)
        draw.line((anchor, endpoint), fill=(255, 255, 255, 255), width=2)

        tx, ty = label_box[0] + 4, label_box[1] + 3
        draw.rounded_rectangle(label_box, radius=3, fill=(0, 0, 0, 220))
        draw.text((tx, ty), annotation["label"], fill=(255, 255, 255, 255), font=label_font)
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
            full_pixel_metrics = base.pixel_metrics_from_map(full_map, gt_mask.numpy())
            without_pixel_metrics = base.pixel_metrics_from_map(no_physio_map, gt_mask.numpy())
            full_heatmap_metrics = base.heatmap_overlap_metrics(full_map, gt_mask.numpy())
            without_heatmap_metrics = base.heatmap_overlap_metrics(no_physio_map, gt_mask.numpy())
            a_physio_effect_metrics = compute_a_physio_effect_metrics(
                full_map,
                no_physio_map,
                gt_mask.numpy(),
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
                "pixel_metrics": full_pixel_metrics,
                "without_a_physio_pixel_metrics": without_pixel_metrics,
                "heatmap_metrics": full_heatmap_metrics,
                "without_a_physio_heatmap_metrics": without_heatmap_metrics,
                "a_physio_effect_metrics": a_physio_effect_metrics,
            }
            if len(records) == len(selected_by_key):
                break
    return records


def attach_without_ucr_records(records, without_ucr_records):
    for key, record in records.items():
        ablated = without_ucr_records.get(key)
        if ablated is None:
            continue
        record.update(
            {
                "without_ucr": ablated["full"],
                "without_ucr_raw_score": ablated["full_raw_score"],
                "without_ucr_image_score": ablated["full_image_score"],
                "without_ucr_pixel_metrics": ablated["pixel_metrics"],
                "without_ucr_heatmap_metrics": ablated["heatmap_metrics"],
            }
        )


def panel_names_for_record(record):
    names = list(PANEL_NAMES)
    if "without_ucr" in record:
        names.insert(4, "w/o UCR")
    return tuple(names)


def render_panels(train_mod, base, record, tile: int):
    comparable_maps = []
    if "without_ucr" in record:
        comparable_maps.append(record["without_ucr"])
    comparable_maps.extend([record["without_a_physio"], record["full"]])
    rendered_maps = render_shared_jet_maps(*comparable_maps)
    if "without_ucr" in record:
        without_ucr_heatmap, no_physio_heatmap, ucr_a_physio_heatmap = rendered_maps
    else:
        no_physio_heatmap, ucr_a_physio_heatmap = rendered_maps
    ucr_a_physio_heatmap = ucr_a_physio_heatmap.resize((256, 256))
    no_physio_heatmap = no_physio_heatmap.resize((256, 256))
    effect_heatmap = render_a_physio_effect_image(
        base,
        record["full"],
        record["without_a_physio"],
        record["gt"],
    ).resize((256, 256))
    evidence = evidence_overlay_image(base, ucr_a_physio_heatmap, record["components"]).resize((256, 256))
    panels = [
        train_mod._panel(record["pet"], "PET", gt_mask=record["gt"], pred_mask=record["pred"]),
        train_mod._panel(record["ct"], "CT"),
        train_mod._panel(record["gt"], "GT", mask_fill=True),
        train_mod._panel(record["pred"], "Pred", pred_fill=True),
    ]
    if "without_ucr" in record:
        panels.append(without_ucr_heatmap.resize((256, 256)))
    panels.extend([no_physio_heatmap, ucr_a_physio_heatmap, effect_heatmap, evidence])
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
        rank = int(row.get("requested_rank", row.get("rank_in_size_group", 0)) or 0)
        rank_prefix = f"rank_{rank:02d}__" if rank > 0 else ""
        case_dir = output_dir / group / f"{rank_prefix}{key[0]}_{key[1]}"
        pet_img = base.colorize(record["pet"], "magma")
        ct_img = base.gray_image(record["ct"])
        gt_img = base.mask_image(record["gt"])
        pred_img = train_mod._panel(record["pred"], "Pred", pred_fill=True)
        comparable_maps = []
        if "without_ucr" in record:
            comparable_maps.append(record["without_ucr"])
        comparable_maps.extend([record["without_a_physio"], record["full"]])
        rendered_maps = render_shared_jet_maps(*comparable_maps)
        if "without_ucr" in record:
            without_ucr_heatmap, no_physio_heatmap, ucr_a_physio_heatmap = rendered_maps
        else:
            no_physio_heatmap, ucr_a_physio_heatmap = rendered_maps
        effect_img = render_a_physio_effect_image(
            base,
            record["full"],
            record["without_a_physio"],
            record["gt"],
        )
        evidence_img = evidence_overlay_image(base, ucr_a_physio_heatmap.resize((256, 256)), record["components"])
        comparison_img = compose_a_physio_comparison(
            base,
            no_physio_heatmap,
            ucr_a_physio_heatmap,
            effect_img,
            record,
            tile=image_size,
        )

        base.save_image(case_dir / "PET_enhanced.png", pet_img, image_size)
        base.save_image(case_dir / "CT.png", ct_img, image_size)
        base.save_image(case_dir / "GT.png", gt_img, image_size)
        base.save_image(case_dir / "Pred.png", pred_img, image_size)
        if "without_ucr" in record:
            base.save_image(case_dir / "without_UCR_heatmap_jet.png", without_ucr_heatmap, image_size)
        base.save_image(case_dir / "without_A_PHYSIO_heatmap_jet.png", no_physio_heatmap, image_size)
        base.save_image(
            case_dir / "with_A_PHYSIO__also_without_MS_LAI_heatmap_jet.png",
            ucr_a_physio_heatmap,
            image_size,
        )
        base.save_image(case_dir / "A_PHYSIO_suppression_map.png", effect_img, image_size)
        comparison_img.save(case_dir / "A_PHYSIO_comparison.png")
        base.save_image(case_dir / "MS_LAI_evidence_overlay.png", evidence_img, image_size)

        mm = record["mask_metrics"]
        pm = dict(record["pixel_metrics"])
        pm.update(pixel_lookup.get(key, {}))
        hm = record["heatmap_metrics"]
        without_pm = record["without_a_physio_pixel_metrics"]
        without_hm = record["without_a_physio_heatmap_metrics"]
        effect_metrics = record["a_physio_effect_metrics"]
        full_aupr = record["pixel_metrics"].get("pixel_aupr", "")
        without_aupr = without_pm.get("pixel_aupr", "")
        aupr_gain = (
            float(full_aupr) - float(without_aupr)
            if full_aupr not in ("", None) and without_aupr not in ("", None)
            else ""
        )
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
            "without_ucr_raw_lesion_evidence": record.get("without_ucr_raw_score", ""),
            "without_ucr_image_score_z": record.get("without_ucr_image_score", ""),
            "without_ucr_pixel_auroc": record.get("without_ucr_pixel_metrics", {}).get("pixel_auroc", ""),
            "without_ucr_pixel_aupr": record.get("without_ucr_pixel_metrics", {}).get("pixel_aupr", ""),
            "pixel_auroc": pm.get("pixel_auroc", ""),
            "pixel_aupr": pm.get("pixel_aupr", ""),
            "without_a_physio_pixel_auroc": without_pm.get("pixel_auroc", ""),
            "without_a_physio_pixel_aupr": without_aupr,
            "a_physio_pixel_aupr_gain": aupr_gain,
            "hotspot_recall": hm["hotspot_recall"],
            "hotspot_precision": hm["hotspot_precision"],
            "without_a_physio_hotspot_recall": without_hm["hotspot_recall"],
            "without_a_physio_hotspot_precision": without_hm["hotspot_precision"],
            "a_physio_outside_suppression_ratio": effect_metrics["outside_suppression_ratio"],
            "a_physio_visible_suppression_fraction": effect_metrics["visible_suppression_fraction"],
            "a_physio_lesion_retention": effect_metrics["lesion_retention"],
            "a_physio_visual_effect_score": effect_metrics["visual_effect_score"],
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
    panel_names = panel_names_for_record(drawable[0][1])
    margin_x = 22
    margin_top = 18
    gap = 6
    header_h = 28
    col_h = 22
    row_h = header_h + tile + gap
    width = margin_x * 2 + len(panel_names) * tile + (len(panel_names) - 1) * gap
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
    for name in panel_names:
        draw.text((x, y), name, fill=(0, 0, 0), font=font_col)
        x += tile + gap
    y += col_h

    for row, record in drawable:
        key = base.row_key(row)
        mm = record["mask_metrics"]
        pm = dict(record["pixel_metrics"])
        pm.update(pixel_lookup.get(key, {}))
        hm = record["heatmap_metrics"]
        without_pm = record["without_a_physio_pixel_metrics"]
        without_hm = record["without_a_physio_heatmap_metrics"]
        effect_metrics = record["a_physio_effect_metrics"]
        full_aupr = record["pixel_metrics"].get("pixel_aupr", "")
        without_aupr = without_pm.get("pixel_aupr", "")
        aupr_gain = (
            float(full_aupr) - float(without_aupr)
            if full_aupr not in ("", None) and without_aupr not in ("", None)
            else None
        )
        lesion_pixels = int(float(row.get("lesion_pixels", mm["gt_pixels"]) or 0))
        area = 100.0 * float(row.get("lesion_area_ratio", 0.0) or 0.0)
        gain_text = "NA" if aupr_gain is None else f"{aupr_gain:+.3f}"
        meta = (
            f"{row.get('requested_group', row.get('lesion_size_group', '')).capitalize()}  |  "
            f"AUPR full/w/o={base.fmt_metric(full_aupr)}/{base.fmt_metric(without_aupr)}  delta={gain_text}  |  "
            f"HotRecall full/w/o={hm['hotspot_recall']:.3f}/{without_hm['hotspot_recall']:.3f}  |  "
            f"SuppOut={effect_metrics['outside_suppression_ratio']:.3f}  Retain={effect_metrics['lesion_retention']:.3f}  |  "
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
    parser.add_argument(
        "--without_ucr_experiment_name",
        type=str,
        default=None,
        help="Optional separately trained no-uncertainty experiment used for the w/o UCR heatmap.",
    )
    parser.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tile", type=int, default=160)
    parser.add_argument("--individual_top_k", type=int, default=5)
    parser.add_argument("--individual_size", type=int, default=256)
    parser.add_argument(
        "--select_a_physio_gain",
        action="store_true",
        help="Select one positive A-PHYSIO pixel-AUPR gain case per lesion-size group.",
    )
    parser.add_argument("--min_a_physio_aupr_gain", type=float, default=0.0)
    parser.add_argument("--max_a_physio_hotspot_drop", type=float, default=0.05)
    parser.add_argument("--selected_per_group", type=int, default=5)
    parser.add_argument("--min_visible_suppression_fraction", type=float, default=0.005)
    parser.add_argument(
        "--min_lesion_retention",
        type=float,
        default=0.0,
        help="Optional hard raw-response retention gate; 0 disables it.",
    )
    parser.add_argument(
        "--min_pred_dice",
        type=float,
        default=0.0,
        help="Optional hard fixed-threshold Dice gate; 0 keeps Dice as a ranking term.",
    )
    parser.add_argument(
        "--min_pred_precision",
        type=float,
        default=0.0,
        help="Optional hard fixed-threshold precision gate; 0 keeps precision as a ranking term.",
    )
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

    if args.select_a_physio_gain and args.group_root is None:
        raise ValueError("--select_a_physio_gain requires --group_root with small/medium/large slices.csv files.")

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

    selection_audit = []
    if args.select_a_physio_gain:
        selected_rows, selection_audit = select_a_physio_gain_rows(
            selected_rows,
            records,
            key_fn=base.row_key,
            groups=args.groups,
            min_aupr_gain=args.min_a_physio_aupr_gain,
            max_hotspot_drop=args.max_a_physio_hotspot_drop,
            top_per_group=args.selected_per_group,
            min_visible_suppression_fraction=args.min_visible_suppression_fraction,
            min_lesion_retention=args.min_lesion_retention,
            min_pred_dice=args.min_pred_dice,
            min_pred_precision=args.min_pred_precision,
        )
        individual_rows = list(selected_rows)
        if not selected_rows:
            status_counts = {}
            for row in selection_audit:
                status = row["selection_status"]
                status_counts[status] = status_counts.get(status, 0) + 1
            print(
                "WARNING: no A-PHYSIO candidates passed selection; "
                f"status_counts={status_counts}"
            )

    ckpt_without_ucr = None
    if args.without_ucr_experiment_name and selected_rows:
        import gc
        import torch

        del model, memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        no_ucr_train_args = without_ucr_args(
            train_mod,
            train_args,
            args.without_ucr_experiment_name,
        )
        (
            no_ucr_model,
            no_ucr_memory,
            no_ucr_prior,
            _no_ucr_dynamic_suppressor,
            no_ucr_calibration,
            no_ucr_test_loader,
            no_ucr_device,
            ckpt_without_ucr,
        ) = base.load_phytwin(train_mod, no_ucr_train_args)
        no_ucr_selected_by_key = {base.row_key(row): row for row in selected_rows}
        no_ucr_records = infer_records(
            train_mod,
            no_ucr_model,
            no_ucr_memory,
            no_ucr_prior,
            no_ucr_calibration,
            no_ucr_test_loader,
            no_ucr_device,
            no_ucr_train_args,
            no_ucr_selected_by_key,
        )
        attach_without_ucr_records(records, no_ucr_records)
        missing_ucr = [key for key in no_ucr_selected_by_key if key not in no_ucr_records]
        if missing_ucr:
            raise RuntimeError(
                f"w/o UCR checkpoint did not produce {len(missing_ucr)} selected cases; "
                f"first missing key={missing_ucr[0]}"
            )
        del no_ucr_model, no_ucr_memory, no_ucr_records
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.select_a_physio_gain:
        audit_fields = [
            "group",
            "case_id",
            "slice_id",
            "full_pixel_auroc",
            "without_a_physio_pixel_auroc",
            "full_pixel_aupr",
            "without_a_physio_pixel_aupr",
            "pixel_aupr_gain",
            "full_hotspot_recall",
            "without_a_physio_hotspot_recall",
            "hotspot_recall_gain",
            "outside_suppression_ratio",
            "visible_suppression_fraction",
            "lesion_retention",
            "visual_effect_score",
            "pred_dice",
            "pred_precision",
            "final_localization_quality",
            "balanced_selection_score",
            "selection_rank",
            "selection_status",
        ]
        write_csv(
            args.output_dir / "a_physio_candidate_metrics.csv",
            selection_audit,
            audit_fields,
        )
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
    best_rows = [
        row
        for group in args.groups
        for row in selected_rows
        if row.get("requested_group") == group and str(row.get("requested_rank")) == "1"
    ]
    n_best = draw_plate(
        train_mod,
        base,
        best_rows,
        records,
        pixel_lookup,
        args.output_dir / "ablation_best_by_size.png",
        "Best visible A-PHYSIO effect by lesion size",
        args.tile,
    )
    summary.append({"group": "best_by_size", "n_selected": len(best_rows), "n_drawn": n_best})

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
                "case_selection": (
                    "positive_a_physio_pixel_aupr_gain"
                    if args.select_a_physio_gain
                    else ("fixed_paper_cases" if args.group_root is None else "lesion_size_group_csv")
                ),
                "a_physio_selection": {
                    "enabled": bool(args.select_a_physio_gain),
                    "minimum_pixel_aupr_gain": float(args.min_a_physio_aupr_gain),
                    "maximum_hotspot_recall_drop": float(args.max_a_physio_hotspot_drop),
                    "selected_per_group": int(args.selected_per_group),
                    "minimum_visible_suppression_fraction": float(args.min_visible_suppression_fraction),
                    "minimum_lesion_retention": float(args.min_lesion_retention),
                    "minimum_pred_dice": float(args.min_pred_dice),
                    "minimum_pred_precision": float(args.min_pred_precision),
                    "display_scale": (
                        "joint 1st-to-99.5th percentile across w/o UCR, "
                        "w/o A-PHYSIO, and full maps per case"
                    ),
                },
                "checkpoint": ckpt_full,
                "without_ucr_checkpoint": ckpt_without_ucr,
                "panels": (
                    list(panel_names_for_record({"without_ucr": True}))
                    if ckpt_without_ucr
                    else list(PANEL_NAMES)
                ),
                "ms_lai_evidence_note": (
                    "The w/o MS-LAI panel shows the post-UCR and post-A-PHYSIO A* heatmap before lesion-aware "
                    "evidence aggregation. MS-LAI evidence overlays the selected multi-scale candidate components "
                    "on that same A* map. It is an interpretability visualization, not a new anomaly heatmap and "
                    "not a top-1% replacement score."
                ),
                "evidence_outline": "one white ellipse per merged multi-scale hotspot with external rank label",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved ablation and MS-LAI evidence figures to {args.output_dir}")


if __name__ == "__main__":
    main()
