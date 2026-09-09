#!/usr/bin/env python3
# Run examples:
#   python export_figure_scores.py --input_mode dual --gpu 4 --num_epochs 30 --data_path /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA --save_name pet_psma --skip_existing --skip_require_pixel
#   python export_figure_scores.py --input_mode dual --gpu 4 --num_epochs 30 --data_path /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG --save_name pet_fdg --skip_existing --skip_require_pixel
# These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""Run Dinomaly PET/CT checkpoint inference and export figure score CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


def expected_output_files(output_dir, require_pixel=False):
    files = [
        os.path.join(output_dir, "slice_scores.csv"),
        os.path.join(output_dir, "patient_scores.csv"),
        os.path.join(output_dir, "manifest.json"),
    ]
    if require_pixel:
        files.append(os.path.join(output_dir, "pixel_slice_metrics.csv"))
    return files

def maybe_skip_existing(output_dir, require_pixel=False):
    if not output_dir:
        return False
    files = expected_output_files(output_dir, require_pixel=require_pixel)
    if all(os.path.exists(path) for path in files):
        print(f"Skip existing outputs: {output_dir}")
        for path in files:
            print(path)
        return True
    return False


def case_id_from_path(path):
    p = Path(str(path))
    if p.parent.name.lower() in {"pet", "ct", "label", "mask", "masks"}:
        return p.parent.parent.name
    return p.parent.name


def slice_id_from_path(path, idx):
    stem = Path(str(path)).stem
    return stem or f"{idx:06d}"


def top_ratio_score(anomaly_map, max_ratio):
    import numpy as np

    flat = anomaly_map.reshape(-1)
    if max_ratio <= 0:
        return float(flat.max())
    k = max(1, int(flat.shape[0] * max_ratio))
    return float(np.sort(flat)[-k:].mean())


def pixel_metric_row(method, dataset, modality, case_id, slice_id, path, anomaly_map, mask):
    import cv2
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    amap = np.asarray(anomaly_map, dtype=np.float32)
    gt = np.asarray(mask)
    gt = np.squeeze(gt)
    if gt.shape != amap.shape:
        gt = cv2.resize(gt.astype(np.float32), (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_NEAREST)
    y_true = (gt.reshape(-1) > 0.5).astype(np.uint8)
    y_score = amap.reshape(-1).astype(np.float32)
    n_pos = int(y_true.sum())
    n_pixels = int(y_true.size)
    if n_pos == 0 or n_pos == n_pixels:
        return None
    return {
        "method": method,
        "dataset": dataset,
        "modality": modality,
        "case_id": case_id,
        "slice_id": slice_id,
        "pixel_auroc": float(roc_auc_score(y_true, y_score)),
        "pixel_aupr": float(average_precision_score(y_true, y_score)),
        "n_pixels": n_pixels,
        "n_positive_pixels": n_pos,
        "path": str(path),
    }


def collect_scores(args):
    import numpy as np
    import torch
    from scipy.ndimage import gaussian_filter
    from tqdm import tqdm

    from dinomaly_petct import build_model, build_petct_loaders, load_checkpoint
    from utils import cal_anomaly_map

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    _train_loader, test_loader, _train_data, test_data, _image_size, _crop_size = build_petct_loaders(args, train=False)
    model, _trainable = build_model(device)
    load_checkpoint(model, args.ckpt_path, device)
    model.eval()

    rows = []
    pixel_rows = []
    sample_idx = 0
    with torch.no_grad():
        for _batch_idx, (img, gt, label, path) in enumerate(tqdm(test_loader, desc="Exporting Dinomaly scores")):
            img = img.to(device)
            en, de = model(img)
            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode="a")
            anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
            if anomaly_map.ndim == 2:
                anomaly_map = anomaly_map[None, ...]
            labels = label.detach().cpu().numpy().reshape(-1).tolist() if hasattr(label, "detach") else list(label)
            masks = gt.detach().cpu().numpy()
            if masks.ndim == 4:
                masks = masks[:, 0]
            elif masks.ndim == 2:
                masks = masks[None, ...]
            paths = list(path) if isinstance(path, (list, tuple)) else [path]
            for map_i, mask_i, label_i, path_i in zip(anomaly_map, masks, labels, paths):
                map_i = gaussian_filter(map_i, sigma=4).astype(np.float32)
                path_i = str(path_i)
                case_id = case_id_from_path(path_i)
                slice_id = slice_id_from_path(path_i, sample_idx)
                rows.append({
                    "method": args.method,
                    "dataset": Path(args.data_path).name,
                    "modality": args.input_mode,
                    "case_id": case_id,
                    "slice_id": slice_id,
                    "true_label": int(label_i),
                    "anomaly_score": top_ratio_score(map_i, args.max_ratio),
                    "path": path_i,
                })
                pix = pixel_metric_row(args.method, Path(args.data_path).name, args.input_mode, case_id, slice_id, path_i, map_i, mask_i)
                if pix is not None:
                    pixel_rows.append(pix)
                sample_idx += 1
    return rows, pixel_rows


def write_outputs(rows, pixel_rows, args):
    os.makedirs(args.output_dir, exist_ok=True)
    slice_csv = os.path.join(args.output_dir, "slice_scores.csv")
    patient_csv = os.path.join(args.output_dir, "patient_scores.csv")
    pixel_csv = os.path.join(args.output_dir, "pixel_slice_metrics.csv")
    manifest_json = os.path.join(args.output_dir, "manifest.json")

    slice_fields = ["method", "dataset", "modality", "case_id", "slice_id", "true_label", "anomaly_score", "path"]
    with open(slice_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=slice_fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(lambda: {"scores": [], "labels": []})
    for row in rows:
        grouped[row["case_id"]]["scores"].append(float(row["anomaly_score"]))
        grouped[row["case_id"]]["labels"].append(int(row["true_label"]))
    patient_rows = []
    for case_id in sorted(grouped):
        item = grouped[case_id]
        patient_rows.append({
            "method": args.method,
            "dataset": Path(args.data_path).name,
            "modality": args.input_mode,
            "case_id": case_id,
            "true_label": max(item["labels"]),
            "anomaly_score": max(item["scores"]),
            "n_slices": len(item["scores"]),
        })
    patient_fields = ["method", "dataset", "modality", "case_id", "true_label", "anomaly_score", "n_slices"]
    with open(patient_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=patient_fields)
        writer.writeheader()
        writer.writerows(patient_rows)

    pixel_fields = ["method", "dataset", "modality", "case_id", "slice_id", "pixel_auroc", "pixel_aupr", "n_pixels", "n_positive_pixels", "path"]
    with open(pixel_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pixel_fields)
        writer.writeheader()
        writer.writerows(pixel_rows)

    manifest = {
        "method": args.method,
        "data_path": args.data_path,
        "input_mode": args.input_mode,
        "checkpoint": args.ckpt_path,
        "slice_csv": slice_csv,
        "patient_csv": patient_csv,
        "pixel_slice_metrics_csv": pixel_csv,
        "patient_aggregation": "max slice anomaly_score",
        "n_slices": len(rows),
        "n_patients": len(patient_rows),
        "n_pixel_slices": len(pixel_rows),
        "n_abnormal_slices": int(sum(row["true_label"] for row in rows)),
        "n_abnormal_patients": int(sum(row["true_label"] for row in patient_rows)),
    }
    with open(manifest_json, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Export Dinomaly PET/CT figure score CSV files.")
    parser.add_argument("--data_path", default="/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG")
    parser.add_argument("--input_mode", default="dual", choices=["ct", "pet", "dual"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default="./saved_results")
    parser.add_argument("--save_name", default=None)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--max_ratio", type=float, default=0.01)
    parser.add_argument("--method", default="Dinomaly")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    args = parser.parse_args()
    if args.save_name is None:
        args.save_name = f"dinomaly_petct_{args.input_mode}_ep{args.num_epochs}_s{args.seed}"
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(args.save_dir, args.save_name, f"epoch_{args.num_epochs:02d}_model.pth")
    if args.output_dir is None:
        args.output_dir = os.path.join(args.save_dir, args.save_name, "figure_scores")
    return args


def main():
    args = parse_args()
    if getattr(args, "skip_existing", False) and maybe_skip_existing(getattr(args, "output_dir", None), getattr(args, "skip_require_pixel", False)):
        return
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    rows, pixel_rows = collect_scores(args)
    manifest = write_outputs(rows, pixel_rows, args)
    print(f"Saved slice scores:   {manifest['slice_csv']}")
    print(f"Saved patient scores: {manifest['patient_csv']}")
    print(f"Saved pixel metrics:  {manifest['pixel_slice_metrics_csv']}")
    print(f"Saved manifest:       {os.path.join(args.output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
