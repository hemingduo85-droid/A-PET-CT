#!/usr/bin/env python3
# Run examples:
#   python export_figure_scores.py --dataset psma --modalities ct,pet --input-mode ct_pet_pet --gpu 7 --skip_existing --skip_require_pixel
#   python export_figure_scores.py --dataset fdg --modalities ct,pet --input-mode ct_pet_pet --gpu 7 --skip_existing --skip_require_pixel
# These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""Run RD4AD PET/CT checkpoint inference and export figure score CSV files."""

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

    from config import checkpoint_path, get_dataset_config, parse_modalities, resolve_input_mode
    from dataset import MVTecDataset, get_data_transforms
    from model_utils import select_device
    from test_modified import cal_anomaly_map, load_checkpoint, make_model

    dataset_config = get_dataset_config(args.dataset)
    data_root = args.data_root or dataset_config["data_root"]
    modalities = parse_modalities(args.modality, args.modalities)
    input_mode = resolve_input_mode(modalities, args.input_mode)
    ckp_path = args.checkpoint or checkpoint_path(dataset_config, modalities, input_mode)
    if not os.path.exists(ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")

    device = select_device(physical_gpu=args.gpu)
    data_transform, gt_transform = get_data_transforms(256, 256)
    test_data = MVTecDataset(
        root=os.path.join(data_root, "test"),
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
        modalities=modalities,
        input_mode=input_mode,
        image_size=256,
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    encoder, bn, decoder = make_model(device, modalities, input_mode)
    load_checkpoint(bn, decoder, ckp_path, device)
    bn.eval()
    decoder.eval()

    rows = []
    pixel_rows = []
    with torch.no_grad():
        for idx, (img, gt, label, path) in enumerate(test_loader):
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode="a")
            anomaly_map = gaussian_filter(anomaly_map, sigma=4).astype(np.float32)
            path = str(path[0] if isinstance(path, (list, tuple)) else path)
            case_id = case_id_from_path(path)
            slice_id = slice_id_from_path(path, idx)
            rows.append({
                "method": args.method,
                "dataset": dataset_config["name"],
                "modality": "+".join(modalities) if len(modalities) > 1 else modalities[0],
                "case_id": case_id,
                "slice_id": slice_id,
                "true_label": int(label.item() if hasattr(label, "item") else label),
                "anomaly_score": float(np.max(anomaly_map)),
                "path": path,
            })
            mask = gt.detach().cpu().numpy()
            pix = pixel_metric_row(args.method, dataset_config["name"], "+".join(modalities) if len(modalities) > 1 else modalities[0], case_id, slice_id, path, anomaly_map, mask)
            if pix is not None:
                pixel_rows.append(pix)
    args._resolved_checkpoint = ckp_path
    args._resolved_data_root = data_root
    args._resolved_modality = "+".join(modalities) if len(modalities) > 1 else modalities[0]
    args._resolved_input_mode = input_mode
    return rows, pixel_rows


def write_outputs(rows, pixel_rows, args):
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "figure_scores",
            str(args.dataset).lower(),
            f"{args._resolved_modality}_{args._resolved_input_mode}",
        )
    os.makedirs(args.output_dir, exist_ok=True)
    slice_csv = os.path.join(args.output_dir, "slice_scores.csv")
    patient_csv = os.path.join(args.output_dir, "patient_scores.csv")
    pixel_csv = os.path.join(args.output_dir, "pixel_slice_metrics.csv")
    manifest_json = os.path.join(args.output_dir, "manifest.json")

    fields = ["method", "dataset", "modality", "case_id", "slice_id", "true_label", "anomaly_score", "path"]
    with open(slice_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(lambda: {"scores": [], "labels": []})
    for row in rows:
        grouped[row["case_id"]]["scores"].append(float(row["anomaly_score"]))
        grouped[row["case_id"]]["labels"].append(int(row["true_label"]))
    patient_rows = []
    for case_id in sorted(grouped):
        item = grouped[case_id]
        first = next(row for row in rows if row["case_id"] == case_id)
        patient_rows.append({
            "method": args.method,
            "dataset": first["dataset"],
            "modality": first["modality"],
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
        "dataset": args.dataset,
        "data_root": args._resolved_data_root,
        "modality": args._resolved_modality,
        "input_mode": args._resolved_input_mode,
        "checkpoint": args._resolved_checkpoint,
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
    parser = argparse.ArgumentParser(description="Export RD4AD figure score CSV files.")
    parser.add_argument("--dataset", default="psma", choices=["psma", "fdg"])
    parser.add_argument("--data-root", dest="data_root", default="")
    parser.add_argument("--modality", default="pet", choices=["pet", "ct"])
    parser.add_argument("--modalities", default="", help="Comma-separated, e.g. pet,ct")
    parser.add_argument("--input-mode", dest="input_mode", default="auto", choices=["auto", "rgb", "ct_pet_pet", "concat"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--method", default="RD4AD")
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    from config import get_dataset_config, parse_modalities, resolve_input_mode
    dataset_config = get_dataset_config(args.dataset)
    modalities = parse_modalities(args.modality, args.modalities)
    input_mode = resolve_input_mode(modalities, args.input_mode)
    resolved_modality = "+".join(modalities) if len(modalities) > 1 else modalities[0]
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "figure_scores",
            str(args.dataset).lower(),
            f"{resolved_modality}_{input_mode}",
        )
    if getattr(args, "skip_existing", False) and maybe_skip_existing(getattr(args, "output_dir", None), getattr(args, "skip_require_pixel", False)):
        return
    rows, pixel_rows = collect_scores(args)
    manifest = write_outputs(rows, pixel_rows, args)
    print(f"Saved slice scores:   {manifest['slice_csv']}")
    print(f"Saved patient scores: {manifest['patient_csv']}")
    print(f"Saved pixel metrics:  {manifest['pixel_slice_metrics_csv']}")
    print(f"Saved manifest:       {os.path.join(args.output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
