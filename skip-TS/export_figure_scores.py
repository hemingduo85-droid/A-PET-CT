#!/usr/bin/env python3
# Run examples:
#   python export_figure_scores.py --dataset psma --modalities ct pet --input_mode pseudo_rgb --cuda 4 --epochs 30 --skip_existing --skip_require_pixel
#   python export_figure_scores.py --dataset fdg --modalities ct pet --input_mode pseudo_rgb --cuda 7 --epochs 30 --skip_existing --skip_require_pixel
# These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""Run skip-TS test inference and export figure score CSV files."""

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


DATASET_PATHS = {
    "fdg": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG",
    "psma": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA",
}


def case_id_from_path(path):
    p = Path(str(path))
    if p.parent.name.lower() in {"pet", "ct", "label", "mask", "masks"}:
        return p.parent.parent.name
    return p.parent.name


def slice_id_from_path(path, idx):
    stem = Path(str(path)).stem
    return stem or f"{idx:06d}"


def maybe_first_path(value):
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def resolve_data_path(dataset, data_path):
    return data_path or DATASET_PATHS[dataset]


def resolve_save_path(checkpoint_root, dataset, modalities, input_mode):
    modality_tag = "+".join(modalities)
    return os.path.join(checkpoint_root, dataset, f"{modality_tag}_{input_mode}")


def default_checkpoint_path(save_path, net, dataset, modalities, input_mode, epochs, seed):
    modality_tag = "+".join(modalities)
    filename = f"{net}_{dataset}_{modality_tag}_{input_mode}_epoch{epochs}_seed{seed}.pth"
    return os.path.join(save_path, filename)


def resolve_device_name(device, cuda):
    if device == "cpu" or device.startswith("cuda:"):
        return device
    return f"cuda:{cuda}"


def pixel_metric_row(method, dataset, modality, case_id, slice_id, path, anomaly_map, mask):
    import cv2
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    amap = np.asarray(anomaly_map, dtype=np.float32)
    gt = np.squeeze(np.asarray(mask))
    if gt.shape != amap.shape:
        gt = cv2.resize(gt.astype(np.float32), (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_NEAREST)
    y_true = (gt.reshape(-1) > 0.5).astype(np.uint8)
    y_score = np.nan_to_num(amap.reshape(-1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
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

    from eval_func import cal_anomaly_map, top1pct_score
    from train_and_test import _build_models, _build_test_dataloader, _device

    device = _device(resolve_device_name(args.device, args.cuda))
    test_loader = _build_test_dataloader(
        args.data_path,
        modalities=args.modalities,
        replicate_channels=args.replicate_channels,
        input_mode=args.input_mode,
        num_workers=args.num_workers,
    )
    encoder, decoder = _build_models(args.modalities, args.replicate_channels, args.input_mode, device)
    state = torch.load(args.checkpoint, map_location=device)
    decoder.load_state_dict(state, strict=True)
    encoder.eval()
    decoder.eval()

    rows = []
    pixel_rows = []
    with torch.no_grad():
        for idx, (img, mask_tensor, label, sample_path) in enumerate(tqdm(test_loader, desc="Exporting skip-TS scores")):
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(inputs[3], inputs[0:3], args.res)
            anomaly_map, _ = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode="a")
            anomaly_map = gaussian_filter(anomaly_map, sigma=4).astype(np.float32)
            path = maybe_first_path(sample_path)
            case_id = case_id_from_path(path)
            slice_id = slice_id_from_path(path, idx)
            modality = "+".join(args.modalities) if len(args.modalities) > 1 else args.modalities[0]
            rows.append({
                "method": args.method,
                "dataset": args.dataset.upper(),
                "modality": modality,
                "case_id": case_id,
                "slice_id": slice_id,
                "true_label": int(label.numpy()[0] if hasattr(label, "numpy") else label),
                "anomaly_score": float(top1pct_score(anomaly_map)),
                "path": path,
            })
            mask_np = mask_tensor.detach().cpu().numpy() if hasattr(mask_tensor, "detach") else mask_tensor
            pix = pixel_metric_row(args.method, args.dataset.upper(), modality, case_id, slice_id, path, anomaly_map, mask_np)
            if pix is not None:
                pixel_rows.append(pix)
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
            "dataset": args.dataset.upper(),
            "modality": "+".join(args.modalities) if len(args.modalities) > 1 else args.modalities[0],
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
        "dataset": args.dataset.upper(),
        "modality": "+".join(args.modalities) if len(args.modalities) > 1 else args.modalities[0],
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
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
    parser = argparse.ArgumentParser(description="Export skip-TS figure score CSV files by running test inference.")
    parser.add_argument("--dataset", choices=sorted(DATASET_PATHS), default="fdg")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--checkpoint_root", default="./checkpoints")
    parser.add_argument("--results_root", default="./results")
    parser.add_argument("--modalities", nargs="+", default=["ct", "pet"], choices=["pet", "ct"])
    parser.add_argument("--input_mode", choices=["pseudo_rgb", "dual", "single"], default="pseudo_rgb")
    parser.add_argument("--replicate_channels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--net", default="wide_res50")
    parser.add_argument("--res", type=int, default=3)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--method", default="skip-TS")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    args = parser.parse_args()
    args.data_path = resolve_data_path(args.dataset, args.data_path)
    effective_input_mode = args.input_mode if len(args.modalities) > 1 else "single"
    save_path = resolve_save_path(args.checkpoint_root, args.dataset, args.modalities, effective_input_mode)
    args.checkpoint = args.checkpoint or default_checkpoint_path(
        save_path, args.net, args.dataset, args.modalities, effective_input_mode, args.epochs, args.seed
    )
    args.output_dir = args.output_dir or os.path.join(
        args.results_root, args.dataset, f"{'+'.join(args.modalities)}_{effective_input_mode}", "figure_scores"
    )
    args.input_mode = effective_input_mode
    return args


def main():
    args = parse_args()
    if getattr(args, "skip_existing", False) and maybe_skip_existing(getattr(args, "output_dir", None), getattr(args, "skip_require_pixel", False)):
        return
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    rows, pixel_rows = collect_scores(args)
    manifest = write_outputs(rows, pixel_rows, args)
    print(f"Saved slice scores:   {manifest['slice_csv']}")
    print(f"Saved patient scores: {manifest['patient_csv']}")
    print(f"Saved pixel metrics:  {manifest['pixel_slice_metrics_csv']}")
    print(f"Saved manifest:       {os.path.join(args.output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
