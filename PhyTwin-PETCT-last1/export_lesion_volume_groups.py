#!/usr/bin/env python3
# Run examples:
#   python export_lesion_volume_groups.py --slice_csv saved_results_psma_eval/PhyTwin_PETCT_PSMA/figure_scores/slice_scores.csv --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA --dataset PSMA --skip_existing
#   python export_lesion_volume_groups.py --slice_csv saved_results_fdg_eval/PhyTwin_PETCT_FDG/figure_scores/slice_scores.csv --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG --dataset FDG --skip_existing
"""Export per-slice lesion size and small/medium/large case-slice lists.

This script uses GT mask pixels, not model predictions. By default, positive
slices are split into small/medium/large groups by 1/3 and 2/3 quantiles of
lesion_pixels. Inside each group, rows are sorted by the PhyTwin anomaly score
from best to worst by default, so the top rows are the first candidates for
visual examples.

python export_lesion_volume_groups.py \
  --slice_csv saved_results_fdg_eval/PhyTwin_PETCT_FDG/figure_scores/slice_scores.csv \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG \
  --dataset FDG \
  --skip_existing
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


MASK_DIR_NAMES = ("label", "labels", "mask", "masks", "gt", "ground_truth")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy")


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def normalize_case_id(path_value, fallback):
    if fallback:
        return str(fallback)
    p = Path(str(path_value))
    if p.parent.name.lower() in {"pet", "ct", "label", "labels", "mask", "masks", "gt"}:
        return p.parent.parent.name
    return p.parent.name


def normalize_slice_id(path_value, fallback):
    if fallback:
        return str(fallback)
    stem = Path(str(path_value)).stem
    return stem or "unknown"


def sibling_mask_candidates(image_path):
    p = Path(str(image_path))
    candidates = []
    if p.parent.name.lower() in {"pet", "ct"}:
        case_dir = p.parent.parent
        for mask_dir in MASK_DIR_NAMES:
            for ext in IMAGE_EXTS:
                candidates.append(case_dir / mask_dir / f"{p.stem}{ext}")
    for mask_dir in MASK_DIR_NAMES:
        for ext in IMAGE_EXTS:
            candidates.append(p.parent / mask_dir / f"{p.stem}{ext}")
            candidates.append(p.parent.parent / mask_dir / f"{p.stem}{ext}")
    return candidates


def data_root_mask_candidates(data_root, case_id, slice_id):
    if not data_root:
        return []
    root = Path(data_root)
    candidates = []
    for split in ("test", ""):
        split_root = root / split if split else root
        for label_group in ("abnormal", "normal", ""):
            group_root = split_root / label_group if label_group else split_root
            case_root = group_root / str(case_id)
            for mask_dir in MASK_DIR_NAMES:
                for ext in IMAGE_EXTS:
                    candidates.append(case_root / mask_dir / f"{slice_id}{ext}")
                    candidates.append(case_root / mask_dir / f"{Path(str(slice_id)).stem}{ext}")
    return candidates


def find_mask_path(image_path, data_root, case_id, slice_id):
    seen = set()
    for candidate in sibling_mask_candidates(image_path) + data_root_mask_candidates(data_root, case_id, slice_id):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def read_mask(path):
    if str(path).lower().endswith(".npy"):
        import numpy as np

        return np.load(path)
    try:
        import cv2
        import numpy as np

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"cv2 failed to read {path}")
        if arr.ndim == 3:
            arr = arr[..., 0]
        return np.asarray(arr)
    except Exception:
        from PIL import Image
        import numpy as np

        return np.asarray(Image.open(path))


def lesion_pixels_from_mask(mask_path):
    import numpy as np

    mask = read_mask(mask_path)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    positive = mask > 0
    return int(positive.sum()), int(positive.size)


def choose_group(lesion_pixels, small_thr, large_thr):
    if lesion_pixels <= 0:
        return "none"
    if lesion_pixels <= small_thr:
        return "small"
    if lesion_pixels >= large_thr:
        return "large"
    return "medium"


def quantile_thresholds(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    small_thr = float(np.quantile(arr, 1.0 / 3.0))
    large_thr = float(np.quantile(arr, 2.0 / 3.0))
    return small_thr, large_thr


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grouped_patient_rows(slice_rows, descending=True):
    grouped = defaultdict(list)
    for row in slice_rows:
        grouped[row["case_id"]].append(row)
    out = []
    for case_id in sorted(grouped):
        rows = sorted(
            grouped[case_id],
            key=lambda r: (float(r["ours_score"]), int(r["lesion_pixels"])),
            reverse=descending,
        )
        best = rows[0]
        out.append({
            "case_id": case_id,
            "n_slices": len(rows),
            "slice_ids": ";".join(str(r["slice_id"]) for r in rows),
            "total_lesion_pixels": sum(int(r["lesion_pixels"]) for r in rows),
            "max_slice_lesion_pixels": max(int(r["lesion_pixels"]) for r in rows),
            "best_slice_id": best["slice_id"],
            "best_ours_score": best["ours_score"],
            "paths": ";".join(str(r["path"]) for r in rows),
            "mask_paths": ";".join(str(r["mask_path"]) for r in rows),
        })
    out.sort(key=lambda r: (float(r["best_ours_score"]), int(r["max_slice_lesion_pixels"])), reverse=descending)
    for rank, row in enumerate(out, start=1):
        row["rank_in_size_group"] = rank
    return out


def main():
    parser = argparse.ArgumentParser(description="Export lesion volume groups from slice_scores.csv and GT masks.")
    parser.add_argument("--slice_csv", required=True, help="Existing slice_scores.csv from export_figure_scores.py.")
    parser.add_argument("--data_root", default=None, help="Dataset root, e.g. /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA.")
    parser.add_argument("--dataset", default=None, help="Dataset label written to the output CSV.")
    parser.add_argument("--output_dir", default=None, help="Defaults to the slice CSV directory.")
    parser.add_argument("--small_max_pixels", type=float, default=None, help="Fixed upper threshold for small lesions.")
    parser.add_argument("--large_min_pixels", type=float, default=None, help="Fixed lower threshold for large lesions.")
    parser.add_argument("--score_order", default="desc", choices=["desc", "asc"], help="Sort candidates by PhyTwin score. desc means best positive examples first.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if lesion_volumes.csv and group files already exist.")
    args = parser.parse_args()

    slice_csv = Path(args.slice_csv)
    output_dir = Path(args.output_dir) if args.output_dir else slice_csv.resolve().parent
    volume_csv = output_dir / "lesion_volumes.csv"
    group_root = output_dir / "lesion_size_groups"
    expected = [
        volume_csv,
        group_root / "small" / "slices.csv",
        group_root / "medium" / "slices.csv",
        group_root / "large" / "slices.csv",
        group_root / "small" / "patients_slices.csv",
        group_root / "medium" / "patients_slices.csv",
        group_root / "large" / "patients_slices.csv",
        group_root / "manifest.json",
    ]
    if args.skip_existing and all(path.exists() for path in expected):
        print(f"Skip existing lesion volume outputs: {output_dir}")
        for path in expected:
            print(path)
        return

    rows = read_csv_rows(slice_csv)
    volume_rows = []
    missing_masks = []
    for idx, row in enumerate(rows):
        path = row.get("path") or row.get("image_path") or row.get("img_path") or ""
        case_id = normalize_case_id(path, row.get("case_id") or row.get("patient_id"))
        slice_id = normalize_slice_id(path, row.get("slice_id"))
        true_label = int(float(row.get("true_label", row.get("label", 0)) or 0))
        ours_score = float(row.get("anomaly_score", row.get("score", 0.0)) or 0.0)
        mask_path = find_mask_path(path, args.data_root, case_id, slice_id)
        lesion_pixels = 0
        n_pixels = 0
        if mask_path is None:
            if true_label > 0:
                missing_masks.append({"case_id": case_id, "slice_id": slice_id, "path": path})
        else:
            lesion_pixels, n_pixels = lesion_pixels_from_mask(mask_path)
        volume_rows.append({
            "dataset": args.dataset or row.get("dataset", ""),
            "method": row.get("method", ""),
            "case_id": case_id,
            "slice_id": slice_id,
            "true_label": true_label,
            "lesion_pixels": lesion_pixels,
            "n_pixels": n_pixels,
            "lesion_area_ratio": (float(lesion_pixels) / float(n_pixels)) if n_pixels else 0.0,
            "lesion_size_group": "none",
            "ours_score": ours_score,
            "rank_in_size_group": "",
            "path": path,
            "mask_path": str(mask_path) if mask_path else "",
            "row_index": idx,
        })

    positive_sizes = [r["lesion_pixels"] for r in volume_rows if int(r["lesion_pixels"]) > 0]
    small_thr, large_thr = quantile_thresholds(positive_sizes)
    if args.small_max_pixels is not None:
        small_thr = float(args.small_max_pixels)
    if args.large_min_pixels is not None:
        large_thr = float(args.large_min_pixels)
    if large_thr < small_thr:
        raise ValueError("--large_min_pixels must be >= --small_max_pixels.")

    for row in volume_rows:
        row["lesion_size_group"] = choose_group(int(row["lesion_pixels"]), small_thr, large_thr)

    descending = args.score_order == "desc"
    for group in ("small", "medium", "large"):
        group_rows = [r for r in volume_rows if r["lesion_size_group"] == group]
        group_rows.sort(key=lambda r: (float(r["ours_score"]), int(r["lesion_pixels"])), reverse=descending)
        for rank, row in enumerate(group_rows, start=1):
            row["rank_in_size_group"] = rank

    volume_fields = [
        "dataset", "method", "case_id", "slice_id", "true_label", "lesion_pixels", "n_pixels",
        "lesion_area_ratio", "lesion_size_group", "ours_score", "rank_in_size_group", "path", "mask_path", "row_index",
    ]
    group_order = {"small": 0, "medium": 1, "large": 2, "none": 3}
    volume_rows.sort(key=lambda r: (group_order.get(r["lesion_size_group"], 9), int(r["rank_in_size_group"]) if r["rank_in_size_group"] else 10**12))
    write_csv(str(volume_csv), volume_rows, volume_fields)

    slice_fields = [
        "dataset", "method", "case_id", "slice_id", "true_label", "lesion_pixels", "n_pixels",
        "lesion_area_ratio", "lesion_size_group", "ours_score", "rank_in_size_group", "path", "mask_path", "row_index",
    ]
    patient_fields = [
        "rank_in_size_group", "case_id", "n_slices", "slice_ids", "total_lesion_pixels",
        "max_slice_lesion_pixels", "best_slice_id", "best_ours_score", "paths", "mask_paths",
    ]
    for group in ("small", "medium", "large"):
        group_rows = [r for r in volume_rows if r["lesion_size_group"] == group]
        group_dir = group_root / group
        write_csv(str(group_dir / "slices.csv"), group_rows, slice_fields)
        write_csv(str(group_dir / "patients_slices.csv"), grouped_patient_rows(group_rows, descending=descending), patient_fields)

    manifest = {
        "slice_csv": str(slice_csv),
        "data_root": args.data_root,
        "volume_csv": str(volume_csv),
        "group_root": str(group_root),
        "small_max_pixels": small_thr,
        "large_min_pixels": large_thr,
        "score_order": args.score_order,
        "ranking_rule": "positive slices are sorted by PhyTwin anomaly_score; descending means better positive examples first",
        "threshold_rule": "positive-slice tertiles unless fixed thresholds are provided",
        "n_rows": len(volume_rows),
        "n_positive_slices_with_mask": len(positive_sizes),
        "n_missing_positive_masks": len(missing_masks),
        "missing_positive_masks": missing_masks[:100],
    }
    os.makedirs(group_root, exist_ok=True)
    with open(group_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved lesion volumes: {volume_csv}")
    print(f"Saved lesion groups:  {group_root}")
    if missing_masks:
        print(f"Warning: missing masks for {len(missing_masks)} positive slices; see manifest.json")


if __name__ == "__main__":
    main()
