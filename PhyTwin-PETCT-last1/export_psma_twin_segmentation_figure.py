#!/usr/bin/env python3
"""
Run on the server from the PhyTwin-PETCT repository root:

python export_psma_twin_segmentation_figure.py \
  --psma_data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA \
  --psma_pixel_metrics_csv saved_results_psma_eval/PhyTwin_PETCT_PSMA/figure_scores/pixel_slice_metrics.csv \
  --fdg_data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG \
  --fdg_pixel_metrics_csv saved_results_fdg_eval/PhyTwin_PETCT_FDG/figure_scores/pixel_slice_metrics.csv \
  --output_dir mixed_twin_segmentation_figure \
  --min_lesion_pixels 120 --min_visual_precision 0.30 --min_visual_dice 0.25 \
  --head_crop_fraction 0.12 --body_margin_fraction 0.06 --lower_crop_fraction 0.60 \
  --load_ckpt \
  --score_mode lesion_z --adaptive_physio --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 --image_size 256 --gpu 4

Automatically select four complementary examples from PSMA, FDG, or both and render:
CT | Normal Twin PET | PET | PhyTwin anomaly map | Threshold overlay.

The displayed CT/PET values are normalized PNG intensities, not HU/SUV. The
anomaly map is a calibrated PhyTwin response in arbitrary units, not delta SUV.
Ground-truth masks are used only to select representative evaluation examples
and to color the TP/FN/FP threshold overlay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


SCENES = (
    "upper_compact",
    "whole_body_disseminated",
    "lower_compact",
    "upper_multifocal",
)

SCENE_LABELS = {
    "upper_compact": "Upper compact",
    "whole_body_disseminated": "Whole-body disseminated",
    "lower_compact": "Lower compact",
    "upper_multifocal": "Upper multifocal",
}

TP_COLOR = np.array([0, 190, 0], dtype=np.uint8)
FN_COLOR = np.array([230, 35, 35], dtype=np.uint8)
FP_COLOR = np.array([30, 145, 235], dtype=np.uint8)


def _closeness(value: float, target: float, width: float) -> float:
    return max(0.0, 1.0 - abs(float(value) - float(target)) / max(float(width), 1e-8))


def extract_mask_features(mask, min_component_area: int = 3) -> dict[str, float | int]:
    """Return normalized location/distribution descriptors for one lesion mask."""
    from scipy.ndimage import label

    binary = np.squeeze(np.asarray(mask)) > 0.5
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2-D lesion mask, got shape={binary.shape}")
    h, w = binary.shape
    cc, n_raw = label(binary)
    components = []
    for component_id in range(1, n_raw + 1):
        region = cc == component_id
        area = int(region.sum())
        if area >= int(min_component_area):
            components.append(region)
    if not components:
        return {
            "lesion_pixels": 0,
            "area_fraction": 0.0,
            "n_components": 0,
            "dominant_component_ratio": 0.0,
            "centroid_y": 0.5,
            "centroid_x": 0.5,
            "y_min": 0.0,
            "y_max": 0.0,
            "vertical_span": 0.0,
            "occupied_bands": 0,
        }

    clean = np.logical_or.reduce(components)
    yy, xx = np.nonzero(clean)
    areas = [int(region.sum()) for region in components]
    lesion_pixels = int(clean.sum())
    band_edges = np.linspace(0, h, 6, dtype=int)
    occupied_bands = sum(
        bool(clean[band_edges[i]:band_edges[i + 1]].any())
        for i in range(len(band_edges) - 1)
    )
    y_min = float(yy.min() / max(1, h - 1))
    y_max = float(yy.max() / max(1, h - 1))
    lower_start = int(round(0.60 * h))
    lower_lesion_fraction = float(clean[lower_start:].sum() / lesion_pixels)
    return {
        "lesion_pixels": lesion_pixels,
        "area_fraction": float(lesion_pixels / clean.size),
        "n_components": len(components),
        "dominant_component_ratio": float(max(areas) / lesion_pixels),
        "centroid_y": float(yy.mean() / max(1, h - 1)),
        "centroid_x": float(xx.mean() / max(1, w - 1)),
        "y_min": y_min,
        "y_max": y_max,
        "vertical_span": float(y_max - y_min),
        "occupied_bands": int(occupied_bands),
        "lower_lesion_fraction": lower_lesion_fraction,
    }


def scene_fit_score(features: dict[str, float | int], scene: str) -> float:
    """Soft score in [0, 1] measuring how well a mask matches one target row."""
    if scene not in SCENES:
        raise ValueError(f"Unknown scene: {scene}")
    if int(features.get("lesion_pixels", 0)) <= 0:
        return 0.0

    cy = float(features["centroid_y"])
    span = float(features["vertical_span"])
    n_components = int(features["n_components"])
    dominant = float(features["dominant_component_ratio"])
    bands = int(features["occupied_bands"])

    if scene == "upper_compact":
        score = (
            0.38 * _closeness(cy, 0.25, 0.40)
            + 0.27 * _closeness(span, 0.14, 0.28)
            + 0.20 * min(1.0, dominant / 0.70)
            + 0.15 * _closeness(n_components, 2.0, 4.0)
        )
    elif scene == "whole_body_disseminated":
        score = (
            0.38 * min(1.0, span / 0.58)
            + 0.24 * min(1.0, bands / 4.0)
            + 0.23 * min(1.0, n_components / 5.0)
            + 0.15 * (1.0 - min(1.0, dominant))
        )
    elif scene == "lower_compact":
        score = (
            0.34 * _closeness(cy, 0.80, 0.30)
            + 0.22 * _closeness(span, 0.14, 0.26)
            + 0.18 * min(1.0, dominant / 0.65)
            + 0.10 * _closeness(n_components, 2.0, 4.0)
            + 0.16 * float(features.get("lower_lesion_fraction", 0.0))
        )
    else:
        score = (
            0.30 * _closeness(cy, 0.34, 0.40)
            + 0.25 * _closeness(span, 0.32, 0.35)
            + 0.25 * min(1.0, n_components / 4.0)
            + 0.20 * (1.0 - min(1.0, dominant))
        )
    return float(np.clip(score, 0.0, 1.0))


def strict_scene_match(features: dict[str, float | int], scene: str) -> bool:
    cy = float(features["centroid_y"])
    span = float(features["vertical_span"])
    y_min = float(features.get("y_min", max(0.0, cy - 0.5 * span)))
    n_components = int(features["n_components"])
    dominant = float(features["dominant_component_ratio"])
    bands = int(features["occupied_bands"])
    if scene == "upper_compact":
        return (
            0.12 <= cy <= 0.48
            and y_min >= 0.08
            and span <= 0.32
            and dominant >= 0.42
            and n_components <= 5
        )
    if scene == "whole_body_disseminated":
        return span >= 0.48 and bands >= 3 and n_components >= 3
    if scene == "lower_compact":
        return (
            cy >= 0.70
            and y_min >= 0.60
            and float(features.get("lower_lesion_fraction", 0.0)) >= 0.95
            and span <= 0.30
            and dominant >= 0.38
            and n_components <= 6
        )
    if scene == "upper_multifocal":
        return cy <= 0.56 and 0.16 <= span <= 0.58 and n_components >= 2 and dominant <= 0.82
    raise ValueError(f"Unknown scene: {scene}")


def lower_dominant_scene_match(features: dict[str, float | int]) -> bool:
    """Allow a small upper focus when the lesion burden is clearly lower-body dominant."""
    return (
        float(features["centroid_y"]) >= 0.62
        and float(features.get("y_max", 0.0)) >= 0.70
        and float(features.get("lower_lesion_fraction", 0.0)) >= 0.72
    )


def crop_box_for_scene(
    scene: str,
    shape: tuple[int, int],
    features: dict[str, float | int],
) -> tuple[int, int, int, int]:
    """Return y0, y1, x0, x1. The same box is applied to all row panels."""
    h, w = int(shape[0]), int(shape[1])
    if scene == "whole_body_disseminated":
        return 0, h, 0, w
    if scene == "upper_compact":
        y0 = 0
        y1 = min(h, max(int(round(0.46 * h)), int(round((float(features["y_max"]) + 0.10) * h))))
        y1 = min(y1, int(round(0.58 * h)))
    elif scene == "lower_compact":
        y0 = max(
            int(round(0.60 * h)),
            int(round((float(features["y_min"]) - 0.06) * h)),
        )
        y1 = h
    elif scene == "upper_multifocal":
        y0 = 0
        y1 = min(h, max(int(round(0.54 * h)), int(round((float(features["y_max"]) + 0.08) * h))))
        y1 = min(y1, int(round(0.68 * h)))
    else:
        raise ValueError(f"Unknown scene: {scene}")
    if y1 <= y0:
        y0, y1 = 0, h
    return int(y0), int(y1), 0, w


def _horizontal_body_bounds(ct_map, margin_fraction: float = 0.06) -> tuple[int, int]:
    """Find the CT body extent while preserving a small natural side margin."""
    arr = np.clip(np.squeeze(np.asarray(ct_map, dtype=np.float32)), 0.0, 1.0)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D CT map, got shape={arr.shape}")
    h, w = arr.shape
    border = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]))
    background = float(np.median(border))
    foreground = np.abs(arr - background) > 0.035
    min_column_pixels = max(2, int(round(0.02 * h)))
    columns = np.flatnonzero(foreground.sum(axis=0) >= min_column_pixels)
    if columns.size == 0:
        return 0, w
    body_x0 = int(columns.min())
    body_x1 = int(columns.max()) + 1
    margin = max(2, int(round((body_x1 - body_x0) * float(margin_fraction))))
    return max(0, body_x0 - margin), min(w, body_x1 + margin)


def display_crop_box(
    scene: str,
    record: dict[str, object],
    features: dict[str, float | int],
    head_crop_fraction: float = 0.12,
    body_margin_fraction: float = 0.06,
    lower_crop_fraction: float = 0.60,
) -> tuple[int, int, int, int]:
    """Create one body-aware crop shared by all columns in a figure row."""
    crop_reference = record.get("ct", record.get("pet", record.get("gt")))
    if crop_reference is None:
        raise KeyError("display_crop_box requires at least one of: ct, pet, gt")
    ct = np.squeeze(np.asarray(crop_reference))
    h, w = ct.shape
    y0, y1, _x0, _x1 = crop_box_for_scene(scene, (h, w), features)
    y0 = max(y0, int(round(float(head_crop_fraction) * h)))
    if scene == "lower_compact":
        y0 = max(y0, int(round(float(lower_crop_fraction) * h)))
    x0, x1 = _horizontal_body_bounds(ct, margin_fraction=body_margin_fraction)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Invalid display crop for {scene}: {(y0, y1, x0, x1)}")
    return y0, y1, x0, x1


def prediction_overlay(base, gt_mask, pred_mask, alpha: float = 0.78) -> np.ndarray:
    """Overlay TP/FN/FP categories on an inverted-grayscale PET image."""
    base_arr = np.squeeze(np.asarray(base, dtype=np.float32))
    if base_arr.ndim != 2:
        raise ValueError(f"Expected 2-D base image, got shape={base_arr.shape}")
    base_arr = np.clip(base_arr, 0.0, 1.0)
    rgb = np.repeat(np.rint(base_arr[..., None] * 255.0).astype(np.uint8), 3, axis=2)
    gt = np.squeeze(np.asarray(gt_mask)) > 0.5
    pred = np.squeeze(np.asarray(pred_mask)) > 0.5
    if gt.shape != base_arr.shape or pred.shape != base_arr.shape:
        raise ValueError("base, gt_mask, and pred_mask must have identical shapes")

    for region, color in (
        (gt & pred, TP_COLOR),
        (gt & ~pred, FN_COLOR),
        (~gt & pred, FP_COLOR),
    ):
        if not region.any():
            continue
        blended = (1.0 - alpha) * rgb[region].astype(np.float32) + alpha * color.astype(np.float32)
        rgb[region] = np.rint(blended).astype(np.uint8)
    return rgb


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_slice_id(value: object) -> str:
    try:
        return f"{int(float(value)):04d}"
    except (TypeError, ValueError):
        return Path(str(value)).stem


def case_slice_key(row: dict[str, object]) -> tuple[str, str]:
    path = Path(str(row.get("path", "")))
    case_id = str(row.get("case_id", "") or "")
    if not case_id and path.parent.name.lower() in {"pet", "ct", "label", "mask", "masks", "gt"}:
        case_id = path.parent.parent.name
    slice_id = row.get("slice_id", "") or path.stem
    return case_id, normalize_slice_id(slice_id)


def load_pixel_metric_lookup(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {case_slice_key(row): row for row in read_csv_rows(path)}


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def scan_candidates(
    data_root: Path,
    pixel_metric_lookup: dict[tuple[str, str], dict[str, str]],
    min_component_area: int = 3,
    dataset: str = "",
) -> list[dict[str, object]]:
    """Scan abnormal masks without loading the model."""
    from PIL import Image
    from phytwin_petct.data import resolve_data_root

    root = Path(resolve_data_root(str(data_root)))
    abnormal_root = root / "test" / "abnormal"
    if not abnormal_root.is_dir():
        raise FileNotFoundError(f"Missing abnormal test directory: {abnormal_root}")

    candidates: list[dict[str, object]] = []
    for patient_dir in sorted(path for path in abnormal_root.iterdir() if path.is_dir()):
        label_dir = patient_dir / "label"
        pet_dir = patient_dir / "pet"
        ct_dir = patient_dir / "ct"
        if not label_dir.is_dir() or not pet_dir.is_dir() or not ct_dir.is_dir():
            continue
        for mask_path in sorted(label_dir.glob("*.png")):
            pet_path = pet_dir / mask_path.name
            ct_path = ct_dir / mask_path.name
            if not pet_path.is_file() or not ct_path.is_file():
                continue
            mask = np.asarray(Image.open(mask_path).convert("L"))
            features = extract_mask_features(mask, min_component_area=min_component_area)
            if int(features["lesion_pixels"]) <= 0:
                continue
            key = (patient_dir.name, normalize_slice_id(mask_path.stem))
            metric_row = pixel_metric_lookup.get(key, {})
            row: dict[str, object] = {
                "dataset": str(dataset),
                "case_id": key[0],
                "slice_id": key[1],
                "path": str(pet_path),
                "pet_path": str(pet_path),
                "ct_path": str(ct_path),
                "mask_path": str(mask_path),
                "pixel_auroc_csv": _float_value(metric_row.get("pixel_auroc", 0.0)),
                "pixel_aupr_csv": _float_value(metric_row.get("pixel_aupr", 0.0)),
            }
            row.update(features)
            candidates.append(row)
    if not candidates:
        raise RuntimeError(f"No non-empty abnormal masks found under {abnormal_root}")
    return candidates


def preselect_candidates(
    candidates: list[dict[str, object]],
    per_scene: int = 12,
    max_slices_per_patient: int = 2,
    min_lesion_pixels: int = 120,
) -> dict[str, list[dict[str, object]]]:
    """Rank CPU-only candidates before the expensive model inference pass."""
    selected: dict[str, list[dict[str, object]]] = {}
    for scene in SCENES:
        ranked = []
        size_eligible = [
            row for row in candidates
            if int(row.get("lesion_pixels", 0)) >= int(min_lesion_pixels)
        ]
        size_filter_fallback = int(not bool(size_eligible))
        scene_candidates = size_eligible if size_eligible else candidates
        for base in scene_candidates:
            row = dict(base)
            fit = scene_fit_score(row, scene)
            strict = strict_scene_match(row, scene)
            quality = (
                0.78 * _float_value(row.get("pixel_aupr_csv"))
                + 0.22 * _float_value(row.get("pixel_auroc_csv"))
            )
            visibility = min(
                1.0,
                math.log1p(max(0, int(row.get("lesion_pixels", 0))))
                / math.log1p(max(1, int(min_lesion_pixels) * 5)),
            )
            row.update({
                "scene": scene,
                "scene_fit": fit,
                "strict_scene_match": int(strict),
                "lower_selection_tier": (
                    "strict"
                    if scene == "lower_compact" and strict
                    else (
                        "lower_dominant"
                        if scene == "lower_compact" and lower_dominant_scene_match(row)
                        else ""
                    )
                ),
                "lesion_visibility_score": visibility,
                "size_filter_fallback": size_filter_fallback,
                "preselection_score": 0.52 * fit + 0.30 * quality + 0.18 * visibility,
            })
            ranked.append(row)
        strict_rows = [row for row in ranked if row["strict_scene_match"]]
        if scene == "lower_compact":
            if not strict_rows:
                lower_dominant_rows = [
                    row for row in ranked
                    if row["lower_selection_tier"] == "lower_dominant"
                ]
                has_complete_morphology = all(
                    "y_min" in row and "lower_lesion_fraction" in row
                    for row in ranked
                )
                if lower_dominant_rows:
                    pool = lower_dominant_rows
                elif has_complete_morphology:
                    raise RuntimeError(
                        "No lower-body-dominant lesion candidate was found. "
                        "Required centroid_y>=0.62, y_max>=0.70, and "
                        "lower_lesion_fraction>=0.72."
                    )
                else:
                    pool = ranked
            else:
                pool = strict_rows
        else:
            pool = strict_rows if len(strict_rows) >= max(2, per_scene // 3) else ranked
        pool.sort(
            key=lambda row: (
                _float_value(row["preselection_score"]),
                _float_value(row["pixel_aupr_csv"]),
                int(row["lesion_pixels"]),
                str(row["case_id"]),
                str(row["slice_id"]),
            ),
            reverse=True,
        )
        kept = []
        patient_counts: dict[str, int] = {}
        for row in pool:
            patient = str(row["case_id"])
            if patient_counts.get(patient, 0) >= int(max_slices_per_patient):
                continue
            kept.append(row)
            patient_counts[patient] = patient_counts.get(patient, 0) + 1
            if len(kept) >= int(per_scene):
                break
        selected[scene] = kept
    return selected


def candidate_final_quality_score(row: dict[str, object]) -> float:
    """Rank representative examples, emphasizing clean threshold localization."""
    lesion_pixels = max(0, int(row.get("lesion_pixels", 0)))
    visibility = _float_value(
        row.get("lesion_visibility_score"),
        min(1.0, math.log1p(lesion_pixels) / math.log1p(600)),
    )
    return float(
        0.18 * _float_value(row.get("scene_fit"))
        + 0.25 * _float_value(row.get("pixel_aupr"))
        + 0.22 * _float_value(row.get("dice"))
        + 0.16 * _float_value(row.get("precision"))
        + 0.08 * _float_value(row.get("recall"))
        + 0.06 * _float_value(row.get("hotspot_recall"))
        + 0.05 * visibility
    )


def _inference_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["case_id"]), normalize_slice_id(row["slice_id"])


def _global_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row["case_id"]),
        normalize_slice_id(row["slice_id"]),
    )


def _patient_identity(row: dict[str, object]) -> str:
    return f"{row.get('dataset', '')}:{row.get('case_id', '')}"


def infer_candidates(
    train_mod,
    train_args,
    model,
    memory,
    prior,
    dynamic_suppressor,
    calibration,
    loader,
    device,
    candidate_rows: Iterable[dict[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    """Run the repository's exact final anomaly-map pipeline for selected slices."""
    import torch
    from export_selected_lesion_case_figure import (
        heatmap_overlap_metrics,
        mask_metrics,
        pixel_metrics_from_map,
    )

    selected = {_inference_key(row) for row in candidate_rows}
    records: dict[tuple[str, str], dict[str, object]] = {}
    model.eval()
    if memory is not None:
        memory = memory.to(device)
    adaptive_suppressor = train_mod.build_adaptive_physio(train_args)

    with torch.no_grad():
        for _idx, batch in train_mod.limited(loader, train_args.max_test_batches):
            path = batch["path"][0] if isinstance(batch["path"], (list, tuple)) else str(batch["path"])
            key = case_slice_key({"path": path})
            if key not in selected:
                continue

            pet = batch["pet"].to(device)
            ct = batch["ct"].to(device)
            pred, logvar = model(ct)
            residual = train_mod.residual_map(
                pet,
                pred,
                logvar if train_args.uncertainty else None,
            )
            residual_np = residual[0].detach().cpu().numpy().astype(np.float32)
            residual_z = train_mod.positive_zscore_map(
                residual_np,
                calibration["residual_mean"],
                calibration["residual_std"],
            )
            memory_z = train_mod.memory_z_map(memory, residual, calibration, train_args)
            final = train_mod.fuse_maps(
                residual_z,
                memory_z,
                residual_weight=train_args.residual_weight,
                memory_weight=train_args.memory_weight,
                physio_prior=prior,
                sigma=train_args.gaussian_sigma,
            )
            pet_gray = train_mod.pet_tensor_to_gray(batch["pet"])[0].numpy().astype(np.float32)
            ct_gray = train_mod.pet_tensor_to_gray(batch["ct"])[0].numpy().astype(np.float32)
            pred_gray = train_mod.pet_tensor_to_gray(pred.detach().cpu())[0].numpy().astype(np.float32)
            if dynamic_suppressor is not None:
                final, _ = dynamic_suppressor.suppress(
                    final,
                    pet_gray,
                    static_prior=None if prior is None else prior.prior,
                )
            final, _ = train_mod.apply_adaptive_physio(
                final,
                pet_gray,
                ct_gray,
                prior,
                adaptive_suppressor,
            )
            pred_mask = train_mod.predict_mask(
                final,
                threshold_quantile=train_args.pred_mask_quantile,
                min_area=train_args.pred_mask_min_area,
            )
            gt = batch["mask"][0].cpu().numpy()
            records[key] = {
                "case_id": key[0],
                "slice_id": key[1],
                "path": path,
                "ct": ct_gray,
                "normal_twin_pet": pred_gray,
                "pet": pet_gray,
                "gt": np.squeeze(gt).astype(np.float32),
                "pred": np.squeeze(pred_mask).astype(np.uint8),
                "anomaly": np.asarray(final, dtype=np.float32),
                "mask_metrics": mask_metrics(gt, pred_mask),
                "pixel_metrics": pixel_metrics_from_map(final, gt),
                "heatmap_metrics": heatmap_overlap_metrics(final, gt),
            }
            if len(records) == len(selected):
                break
    return records


def attach_inference_metrics(
    candidates_by_scene: dict[str, list[dict[str, object]]],
    records: dict[tuple[str, str], dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for scene, rows in candidates_by_scene.items():
        enriched = []
        for base in rows:
            key = _inference_key(base)
            record = records.get(key)
            if record is None:
                continue
            row = dict(base)
            row["record"] = record
            pixel = record["pixel_metrics"]
            mask = record["mask_metrics"]
            hotspot = record["heatmap_metrics"]
            pixel_aupr = _float_value(pixel.get("pixel_aupr"), _float_value(row.get("pixel_aupr_csv")))
            pixel_auroc = _float_value(pixel.get("pixel_auroc"), _float_value(row.get("pixel_auroc_csv")))
            row.update({
                "pixel_aupr": pixel_aupr,
                "pixel_auroc": pixel_auroc,
                "dice": _float_value(mask.get("dice")),
                "iou": _float_value(mask.get("iou")),
                "recall": _float_value(mask.get("recall")),
                "precision": _float_value(mask.get("precision")),
                "hotspot_recall": _float_value(hotspot.get("hotspot_recall")),
                "hotspot_precision": _float_value(hotspot.get("hotspot_precision")),
            })
            row["final_selection_score"] = candidate_final_quality_score(row)
            enriched.append(row)
        enriched.sort(
            key=lambda row: (
                _float_value(row["final_selection_score"]),
                _float_value(row["pixel_aupr"]),
                str(row["case_id"]),
                str(row["slice_id"]),
            ),
            reverse=True,
        )
        output[scene] = enriched
    return output


def choose_final_cases(
    candidates_by_scene: dict[str, list[dict[str, object]]],
    min_visual_precision: float = 0.30,
    min_visual_dice: float = 0.25,
) -> dict[str, dict[str, object]]:
    """Greedily select the best case per row while preferring unique patients."""
    selected: dict[str, dict[str, object]] = {}
    used_patients: set[str] = set()
    for scene in SCENES:
        ranked = sorted(
            candidates_by_scene.get(scene, []),
            key=lambda row: (
                _float_value(row.get("final_selection_score")),
                str(row.get("case_id", "")),
                str(row.get("slice_id", "")),
            ),
            reverse=True,
        )
        if not ranked:
            raise RuntimeError(f"No inferred candidate is available for scene={scene}")
        quality_rows = [
            row for row in ranked
            if _float_value(row.get("precision")) >= float(min_visual_precision)
            and _float_value(row.get("dice")) >= float(min_visual_dice)
        ]
        quality_gate_fallback = int(not bool(quality_rows))
        quality_pool = quality_rows if quality_rows else ranked
        unused = [row for row in quality_pool if _patient_identity(row) not in used_patients]
        choice = unused[0] if unused else quality_pool[0]
        choice = dict(choice)
        choice["unique_patient_fallback"] = int(not bool(unused))
        choice["quality_gate_fallback"] = quality_gate_fallback
        selected[scene] = choice
        used_patients.add(_patient_identity(choice))
    return selected


def _gt_contour_overlay(base, gt_mask) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion

    gray = np.clip(np.squeeze(np.asarray(base, dtype=np.float32)), 0.0, 1.0)
    rgb = np.repeat(np.rint(gray[..., None] * 255.0).astype(np.uint8), 3, axis=2)
    gt = np.squeeze(np.asarray(gt_mask)) > 0.5
    outer = binary_dilation(gt, iterations=1)
    inner = binary_erosion(gt, iterations=1)
    contour = outer & ~inner
    rgb[contour] = TP_COLOR
    return rgb


def _green_gt_mask(gt_mask) -> np.ndarray:
    gt = np.squeeze(np.asarray(gt_mask)) > 0.5
    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    rgb[gt] = TP_COLOR
    return rgb


def _crop(arr, box: tuple[int, int, int, int]):
    y0, y1, x0, x1 = box
    return np.asarray(arr)[y0:y1, x0:x1]


def _robust_anomaly_vmax(selected: dict[str, dict[str, object]]) -> float:
    values = []
    for scene in SCENES:
        anomaly = np.asarray(selected[scene]["record"]["anomaly"], dtype=np.float32)
        finite = anomaly[np.isfinite(anomaly)]
        if finite.size:
            values.append(finite)
    if not values:
        return 1.0
    merged = np.concatenate(values)
    vmax = float(np.percentile(merged, 99.5))
    return vmax if vmax > 1e-8 else max(1.0, float(np.max(merged)))


def render_publication_figure(
    selected: dict[str, dict[str, object]],
    output_prefix: Path,
    dpi: int = 600,
    head_crop_fraction: float = 0.12,
    body_margin_fraction: float = 0.06,
    lower_crop_fraction: float = 0.60,
) -> dict[str, str]:
    """Render the final five-column/four-row manuscript image plate."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    missing = [scene for scene in SCENES if scene not in selected]
    if missing:
        raise ValueError(f"Missing selected scenes: {missing}")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.titleweight": "normal",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    boxes = {}
    row_ratios = []
    for scene in SCENES:
        row = selected[scene]
        box = display_crop_box(
            scene,
            row["record"],
            row,
            head_crop_fraction=head_crop_fraction,
            body_margin_fraction=body_margin_fraction,
            lower_crop_fraction=lower_crop_fraction,
        )
        boxes[scene] = box
        row_ratios.append(max(0.28, (box[1] - box[0]) / max(1, box[3] - box[2])))

    fig = plt.figure(figsize=(165 / 25.4, 138 / 25.4), facecolor="white")
    gs = fig.add_gridspec(
        5,
        5,
        height_ratios=[*row_ratios, 0.22],
        hspace=0.035,
        wspace=0.005,
        left=0.018,
        right=0.992,
        top=0.945,
        bottom=0.08,
    )
    column_titles = ("CT", "Normal PET Ref.", "PET", "Anomaly map", "Threshold")
    anomaly_vmax = _robust_anomaly_vmax(selected)

    for row_index, scene in enumerate(SCENES):
        row = selected[scene]
        record = row["record"]
        box = boxes[scene]
        ct = np.clip(_crop(record["ct"], box), 0.0, 1.0)
        twin = np.clip(_crop(record["normal_twin_pet"], box), 0.0, 1.0)
        pet = np.clip(_crop(record["pet"], box), 0.0, 1.0)
        anomaly = np.clip(_crop(record["anomaly"], box), 0.0, anomaly_vmax)
        gt = _crop(record["gt"], box)
        pred = _crop(record["pred"], box)
        overlay = prediction_overlay(1.0 - pet, gt, pred)
        panel_data = (
            (ct, "gray", 0.0, 1.0, "bilinear"),
            (1.0 - twin, "gray", 0.0, 1.0, "bilinear"),
            (1.0 - pet, "gray", 0.0, 1.0, "bilinear"),
            (anomaly, "Reds", 0.0, anomaly_vmax, "bilinear"),
            (overlay, None, None, None, "nearest"),
        )
        for col_index, (image, cmap, vmin, vmax, interpolation) in enumerate(panel_data):
            ax = fig.add_subplot(gs[row_index, col_index])
            ax.imshow(
                image,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation=interpolation,
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(column_titles[col_index], pad=3)

    colorbar_specs = (
        ("gray", Normalize(0.0, 1.0), "Normalized CT intensity", [0.0, 0.5, 1.0]),
        ("gray_r", Normalize(0.0, 1.0), "Normalized PET uptake", [0.0, 0.5, 1.0]),
        ("gray_r", Normalize(0.0, 1.0), "Normalized PET uptake", [0.0, 0.5, 1.0]),
        ("Reds", Normalize(0.0, anomaly_vmax), "PhyTwin anomaly score (a.u.)", [0.0, anomaly_vmax]),
    )
    for col_index, (cmap, norm, label, ticks) in enumerate(colorbar_specs):
        cax = fig.add_subplot(gs[4, col_index])
        cax.set_axis_off()
        inset = cax.inset_axes([0.04, 0.43, 0.92, 0.24])
        scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(scalar, cax=inset, orientation="horizontal", ticks=ticks)
        cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
        tick_labels = cb.ax.get_xticklabels()
        if tick_labels:
            tick_labels[0].set_horizontalalignment("left")
            tick_labels[-1].set_horizontalalignment("right")
        cb.outline.set_linewidth(0.45)
        cb.set_label(label, fontsize=6, labelpad=1.5)

    legend_ax = fig.add_subplot(gs[4, 4])
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=[
            Patch(facecolor=TP_COLOR / 255.0, label="TP"),
            Patch(facecolor=FN_COLOR / 255.0, label="FN"),
            Patch(facecolor=FP_COLOR / 255.0, label="FP"),
        ],
        loc="center",
        ncol=3,
        frameon=True,
        edgecolor="#b5b5b5",
        fontsize=6,
        handlelength=1.5,
        handleheight=0.7,
        columnspacing=0.8,
        borderpad=0.35,
    )

    outputs = {
        "png": str(output_prefix.with_suffix(".png")),
        "tiff": str(output_prefix.with_suffix(".tiff")),
        "pdf": str(output_prefix.with_suffix(".pdf")),
        "svg": str(output_prefix.with_suffix(".svg")),
    }
    fig.savefig(outputs["png"], dpi=dpi, facecolor="white")
    fig.savefig(outputs["tiff"], dpi=dpi, facecolor="white")
    fig.savefig(outputs["pdf"], facecolor="white")
    fig.savefig(outputs["svg"], facecolor="white")
    plt.close(fig)
    return outputs


def render_ground_truth_figure(
    selected: dict[str, dict[str, object]],
    output_prefix: Path,
    dpi: int = 600,
    head_crop_fraction: float = 0.12,
    body_margin_fraction: float = 0.06,
    lower_crop_fraction: float = 0.60,
) -> dict[str, str]:
    """Export a separate GT reference without changing the five-column main plate."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.titleweight": "normal",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    boxes = {}
    row_ratios = []
    for scene in SCENES:
        row = selected[scene]
        box = display_crop_box(
            scene,
            row["record"],
            row,
            head_crop_fraction=head_crop_fraction,
            body_margin_fraction=body_margin_fraction,
            lower_crop_fraction=lower_crop_fraction,
        )
        boxes[scene] = box
        row_ratios.append(max(0.28, (box[1] - box[0]) / max(1, box[3] - box[2])))

    fig = plt.figure(figsize=(89 / 25.4, 142 / 25.4), facecolor="white")
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=row_ratios,
        hspace=0.035,
        wspace=0.015,
        left=0.025,
        right=0.99,
        top=0.945,
        bottom=0.025,
    )
    for row_index, scene in enumerate(SCENES):
        row = selected[scene]
        record = row["record"]
        box = boxes[scene]
        pet = np.clip(_crop(record["pet"], box), 0.0, 1.0)
        gt = _crop(record["gt"], box)
        images = (_gt_contour_overlay(1.0 - pet, gt), _green_gt_mask(gt))
        for col_index, image in enumerate(images):
            ax = fig.add_subplot(gs[row_index, col_index])
            ax.imshow(image, interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(("PET + GT contour", "GT mask")[col_index], pad=3)

    outputs = {
        "png": str(output_prefix.with_suffix(".png")),
        "tiff": str(output_prefix.with_suffix(".tiff")),
        "pdf": str(output_prefix.with_suffix(".pdf")),
        "svg": str(output_prefix.with_suffix(".svg")),
    }
    fig.savefig(outputs["png"], dpi=dpi, facecolor="white")
    fig.savefig(outputs["tiff"], dpi=dpi, facecolor="white")
    fig.savefig(outputs["pdf"], facecolor="white")
    fig.savefig(outputs["svg"], facecolor="white")
    plt.close(fig)
    return outputs


def render_candidate_contact_sheet(
    candidates_by_scene: dict[str, list[dict[str, object]]],
    output_path: Path,
    per_scene: int = 5,
    dpi: int = 220,
) -> None:
    """Render top alternatives so final case selection remains inspectable."""
    import matplotlib.pyplot as plt

    ncols = max(1, int(per_scene))
    fig, axes = plt.subplots(
        len(SCENES),
        ncols,
        figsize=(2.0 * ncols, 2.2 * len(SCENES)),
        squeeze=False,
        facecolor="white",
    )
    for row_index, scene in enumerate(SCENES):
        rows = candidates_by_scene.get(scene, [])[:ncols]
        for col_index in range(ncols):
            ax = axes[row_index, col_index]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_index >= len(rows):
                ax.set_axis_off()
                continue
            row = rows[col_index]
            record = row["record"]
            pet = np.asarray(record["pet"], dtype=np.float32)
            overlay = prediction_overlay(
                1.0 - np.clip(pet, 0.0, 1.0),
                record["gt"],
                record["pred"],
            )
            ax.imshow(overlay, interpolation="nearest")
            ax.set_title(
                f"{row.get('dataset', '')} | {row['case_id']} / {row['slice_id']}\n"
                f"AUPR={_float_value(row.get('pixel_aupr')):.3f}  "
                f"Dice={_float_value(row.get('dice')):.3f}",
                fontsize=6,
            )
            if col_index == 0:
                ax.set_ylabel(SCENE_LABELS[scene], fontsize=7, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.995, top=0.97, bottom=0.02, wspace=0.08, hspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def _serializable_candidate(row: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key != "record" and isinstance(value, (str, int, float, bool, type(None), np.integer, np.floating))
    }


def _parse_overrides(values: list[str]) -> dict[str, tuple[str, str, str]]:
    overrides = {}
    for value in values:
        if "=" not in value or ":" not in value:
            raise ValueError(
                f"Invalid --override {value!r}; expected SCENE=DATASET:CASE_ID:SLICE_ID"
            )
        scene, case_slice = value.split("=", 1)
        parts = case_slice.split(":")
        if len(parts) == 2:
            dataset = ""
            case_id, slice_id = parts
        elif len(parts) == 3:
            dataset, case_id, slice_id = parts
            dataset = dataset.upper()
        else:
            raise ValueError(
                f"Invalid --override {value!r}; expected SCENE=DATASET:CASE_ID:SLICE_ID"
            )
        if scene not in SCENES:
            raise ValueError(f"Unknown override scene={scene}; choices={SCENES}")
        overrides[scene] = (dataset, case_id, normalize_slice_id(slice_id))
    return overrides


def _cohort_specs(args, train_argv: list[str]) -> list[dict[str, object]]:
    dedicated = any(
        value is not None
        for value in (
            args.psma_data_root,
            args.psma_pixel_metrics_csv,
            args.fdg_data_root,
            args.fdg_pixel_metrics_csv,
        )
    )
    specs = []
    if dedicated:
        if "--experiment_name" in train_argv:
            raise ValueError(
                "Do not pass --experiment_name in dual-cohort mode; use "
                "--psma_experiment_name and --fdg_experiment_name."
            )
        for dataset, root, metrics, experiment in (
            ("PSMA", args.psma_data_root, args.psma_pixel_metrics_csv, args.psma_experiment_name),
            ("FDG", args.fdg_data_root, args.fdg_pixel_metrics_csv, args.fdg_experiment_name),
        ):
            if root is None and metrics is None:
                continue
            if root is None or metrics is None:
                raise ValueError(f"{dataset} requires both data root and pixel metrics CSV.")
            specs.append({
                "dataset": dataset,
                "data_root": Path(root),
                "pixel_metrics_csv": Path(metrics),
                "experiment_name": str(experiment),
            })
    else:
        if args.data_root is None or args.pixel_metrics_csv is None:
            raise ValueError(
                "Provide --data_root and --pixel_metrics_csv for single-cohort mode, "
                "or provide the PSMA/FDG-specific arguments."
            )
        dataset = str(args.dataset_name or Path(args.data_root).name).upper()
        specs.append({
            "dataset": dataset,
            "data_root": Path(args.data_root),
            "pixel_metrics_csv": Path(args.pixel_metrics_csv),
            "experiment_name": None,
        })
    if not specs:
        raise ValueError("No cohort configuration was provided.")
    for spec in specs:
        if not Path(spec["pixel_metrics_csv"]).is_file():
            raise FileNotFoundError(
                f"Missing {spec['dataset']} pixel metrics CSV: {spec['pixel_metrics_csv']}"
            )
    return specs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--pixel_metrics_csv", type=Path, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--psma_data_root", type=Path, default=None)
    parser.add_argument("--psma_pixel_metrics_csv", type=Path, default=None)
    parser.add_argument("--psma_experiment_name", type=str, default="PhyTwin_PETCT_PSMA")
    parser.add_argument("--fdg_data_root", type=Path, default=None)
    parser.add_argument("--fdg_pixel_metrics_csv", type=Path, default=None)
    parser.add_argument("--fdg_experiment_name", type=str, default="PhyTwin_PETCT_FDG")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--candidate_per_scene", type=int, default=12)
    parser.add_argument("--contact_per_scene", type=int, default=5)
    parser.add_argument("--max_slices_per_patient", type=int, default=2)
    parser.add_argument("--min_component_area", type=int, default=3)
    parser.add_argument("--min_lesion_pixels", type=int, default=120)
    parser.add_argument("--min_visual_precision", type=float, default=0.30)
    parser.add_argument("--min_visual_dice", type=float, default=0.25)
    parser.add_argument("--head_crop_fraction", type=float, default=0.12)
    parser.add_argument("--body_margin_fraction", type=float, default=0.06)
    parser.add_argument("--lower_crop_fraction", type=float, default=0.60)
    parser.add_argument("--figure_dpi", type=int, default=600)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="SCENE=DATASET:CASE_ID:SLICE_ID",
        help=f"Override one final row. Scene choices: {', '.join(SCENES)}",
    )
    return parser.parse_known_args(argv)


def main(argv=None):
    args, train_argv = parse_args(sys.argv[1:] if argv is None else argv)
    cohort_specs = _cohort_specs(args, train_argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overrides = _parse_overrides(args.override)
    from export_selected_lesion_case_figure import import_train_args, load_phytwin

    enriched = {scene: [] for scene in SCENES}
    cohort_manifest = []
    override_matches = {scene: 0 for scene in overrides}
    total_scanned = 0
    total_candidates = 0
    total_inferred = 0
    all_missing = []

    for spec in cohort_specs:
        dataset = str(spec["dataset"])
        pixel_lookup = load_pixel_metric_lookup(Path(spec["pixel_metrics_csv"]))
        scanned = scan_candidates(
            Path(spec["data_root"]),
            pixel_lookup,
            min_component_area=args.min_component_area,
            dataset=dataset,
        )
        candidates_by_scene = preselect_candidates(
            scanned,
            per_scene=args.candidate_per_scene,
            max_slices_per_patient=args.max_slices_per_patient,
            min_lesion_pixels=args.min_lesion_pixels,
        )
        scanned_by_key = {_global_key(row): row for row in scanned}
        for scene, override_key in overrides.items():
            override_dataset, case_id, slice_id = override_key
            if override_dataset and override_dataset != dataset:
                continue
            key = (dataset, case_id, slice_id)
            if key not in scanned_by_key:
                continue
            override_matches[scene] += 1
            override_row = dict(scanned_by_key[key])
            override_row.update({
                "scene": scene,
                "scene_fit": scene_fit_score(override_row, scene),
                "strict_scene_match": int(strict_scene_match(override_row, scene)),
                "preselection_score": 1.0,
                "manual_override": 1,
            })
            candidates_by_scene[scene] = [override_row] + [
                row for row in candidates_by_scene[scene] if _global_key(row) != key
            ]

        union = {}
        for rows in candidates_by_scene.values():
            for row in rows:
                union[_inference_key(row)] = row

        cohort_train_argv = ["--data_root", str(spec["data_root"])]
        if spec["experiment_name"]:
            cohort_train_argv += ["--experiment_name", str(spec["experiment_name"])]
        cohort_train_argv += train_argv
        train_mod, train_args = import_train_args(cohort_train_argv)
        model, memory, prior, dynamic_suppressor, calibration, test_loader, device, ckpt_path = load_phytwin(
            train_mod,
            train_args,
        )
        records = infer_candidates(
            train_mod,
            train_args,
            model,
            memory,
            prior,
            dynamic_suppressor,
            calibration,
            test_loader,
            device,
            union.values(),
        )
        cohort_enriched = attach_inference_metrics(candidates_by_scene, records)
        for scene in SCENES:
            enriched[scene].extend(cohort_enriched[scene])
        missing = sorted(set(union) - set(records))
        all_missing.extend(
            f"{dataset}:{case_id}:{slice_id}" for case_id, slice_id in missing
        )
        cohort_manifest.append({
            "dataset": dataset,
            "data_root": str(train_args.data_root),
            "pixel_metrics_csv": str(spec["pixel_metrics_csv"]),
            "checkpoint": ckpt_path,
            "n_scanned_nonempty_masks": len(scanned),
            "n_inference_candidates": len(union),
            "n_inferred": len(records),
        })
        total_scanned += len(scanned)
        total_candidates += len(union)
        total_inferred += len(records)

        del model, memory, prior, calibration, test_loader, records
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    for scene, count in override_matches.items():
        if count != 1:
            raise KeyError(
                f"Override for {scene} matched {count} cases. In mixed mode use "
                "SCENE=DATASET:CASE_ID:SLICE_ID."
            )
    for scene in SCENES:
        enriched[scene].sort(
            key=lambda row: (
                _float_value(row.get("final_selection_score")),
                str(row.get("dataset", "")),
                str(row.get("case_id", "")),
                str(row.get("slice_id", "")),
            ),
            reverse=True,
        )

    selected = choose_final_cases(
        enriched,
        min_visual_precision=args.min_visual_precision,
        min_visual_dice=args.min_visual_dice,
    )
    for scene, key in overrides.items():
        dataset, case_id, slice_id = key
        matches = [
            row for row in enriched[scene]
            if (not dataset or str(row.get("dataset", "")) == dataset)
            and str(row["case_id"]) == case_id
            and normalize_slice_id(row["slice_id"]) == slice_id
        ]
        if not matches:
            raise RuntimeError(f"Override case was not produced by inference: {scene}={key}")
        selected[scene] = matches[0]
        selected[scene]["manual_override"] = 1

    mixed_mode = len(cohort_specs) > 1
    figure_stem = "phytwin_mixed_twin_segmentation_figure" if mixed_mode else "psma_twin_segmentation_figure"
    gt_stem = "phytwin_mixed_twin_segmentation_ground_truth" if mixed_mode else "psma_twin_segmentation_ground_truth"
    figure_outputs = render_publication_figure(
        selected,
        args.output_dir / figure_stem,
        dpi=args.figure_dpi,
        head_crop_fraction=args.head_crop_fraction,
        body_margin_fraction=args.body_margin_fraction,
        lower_crop_fraction=args.lower_crop_fraction,
    )
    gt_figure_outputs = render_ground_truth_figure(
        selected,
        args.output_dir / gt_stem,
        dpi=args.figure_dpi,
        head_crop_fraction=args.head_crop_fraction,
        body_margin_fraction=args.body_margin_fraction,
        lower_crop_fraction=args.lower_crop_fraction,
    )
    render_candidate_contact_sheet(
        enriched,
        args.output_dir / "candidate_contact_sheet.png",
        per_scene=args.contact_per_scene,
    )

    candidate_rows = []
    for scene in SCENES:
        for rank, row in enumerate(enriched[scene], start=1):
            serial = _serializable_candidate(row)
            serial["candidate_rank"] = rank
            candidate_rows.append(serial)
    selected_rows = []
    for row_index, scene in enumerate(SCENES, start=1):
        serial = _serializable_candidate(selected[scene])
        serial["row_index"] = row_index
        serial["scene"] = scene
        serial["scene_label"] = SCENE_LABELS[scene]
        serial["crop_box_y0_y1_x0_x1"] = ",".join(
            str(value)
            for value in display_crop_box(
                scene,
                selected[scene]["record"],
                selected[scene],
                head_crop_fraction=args.head_crop_fraction,
                body_margin_fraction=args.body_margin_fraction,
                lower_crop_fraction=args.lower_crop_fraction,
            )
        )
        selected_rows.append(serial)

    candidate_fields = sorted({key for row in candidate_rows for key in row})
    selected_fields = sorted({key for row in selected_rows for key in row})
    write_csv(args.output_dir / "candidate_metrics.csv", candidate_rows, candidate_fields)
    write_csv(args.output_dir / "selected_cases.csv", selected_rows, selected_fields)

    manifest = {
        "mode": "mixed" if mixed_mode else "single",
        "cohorts": cohort_manifest,
        "output_dir": str(args.output_dir),
        "figure_outputs": figure_outputs,
        "ground_truth_figure_outputs": gt_figure_outputs,
        "candidate_contact_sheet": str(args.output_dir / "candidate_contact_sheet.png"),
        "n_scanned_nonempty_masks": total_scanned,
        "n_inference_candidates": total_candidates,
        "n_inferred": total_inferred,
        "selection_gates": {
            "min_lesion_pixels": args.min_lesion_pixels,
            "min_visual_precision": args.min_visual_precision,
            "min_visual_dice": args.min_visual_dice,
            "head_crop_fraction": args.head_crop_fraction,
            "body_margin_fraction": args.body_margin_fraction,
            "lower_crop_fraction": args.lower_crop_fraction,
        },
        "missing_inference_candidates": all_missing,
        "scenes": list(SCENES),
        "manual_overrides": {
            scene: {"dataset": key[0], "case_id": key[1], "slice_id": key[2]}
            for scene, key in overrides.items()
        },
        "labels": {
            "ct": "Normalized CT intensity",
            "pet": "Normalized PET uptake",
            "anomaly": "PhyTwin anomaly score (a.u.)",
            "overlay": {"TP": TP_COLOR.tolist(), "FN": FN_COLOR.tolist(), "FP": FP_COLOR.tolist()},
        },
        "selection_note": (
            "Representative evaluation cases were selected using ground-truth morphology "
            "and localization metrics; this is not an unbiased performance estimate."
        ),
    }
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("Selected cases:")
    for row in selected_rows:
        print(
            f"  row {row['row_index']} {row['scene']}: "
            f"{row.get('dataset', '')}:{row['case_id']}:{row['slice_id']} "
            f"pAUPR={_float_value(row.get('pixel_aupr')):.3f} "
            f"Dice={_float_value(row.get('dice')):.3f}"
        )
    print(f"Saved figure bundle to {args.output_dir}")


if __name__ == "__main__":
    main()
