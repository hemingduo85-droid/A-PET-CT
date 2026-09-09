#!/usr/bin/env python3
# Run examples:
#   python export_figure_scores.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer PSMA --modalities ct,pet --device cuda:4 --skip_existing --skip_require_pixel
#   python export_figure_scores.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer FDG --modalities ct,pet --device cuda:4 --skip_existing --skip_require_pixel
# These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""Export INPFormer PET/CT slice/patient scores for figure plotting."""

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


def parse_id(value):
    parts = str(value).split("__")
    if len(parts) >= 3:
        return parts[1], parts[2]
    p = Path(str(value))
    if p.parent.name.lower() in {"ct", "pet", "label"}:
        return p.parent.parent.name, p.stem
    return p.parent.name, p.stem


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


def write_outputs(rows, pixel_rows, output_dir, manifest):
    os.makedirs(output_dir, exist_ok=True)
    slice_csv = os.path.join(output_dir, "slice_scores.csv")
    patient_csv = os.path.join(output_dir, "patient_scores.csv")
    pixel_csv = os.path.join(output_dir, "pixel_slice_metrics.csv")
    manifest_json = os.path.join(output_dir, "manifest.json")
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
        patient_rows.append({
            "method": manifest["method"],
            "dataset": manifest["dataset"],
            "modality": manifest["modality"],
            "case_id": case_id,
            "true_label": max(grouped[case_id]["labels"]),
            "anomaly_score": max(grouped[case_id]["scores"]),
            "n_slices": len(grouped[case_id]["scores"]),
        })
    with open(patient_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "dataset", "modality", "case_id", "true_label", "anomaly_score", "n_slices"])
        writer.writeheader()
        writer.writerows(patient_rows)
    pixel_fields = ["method", "dataset", "modality", "case_id", "slice_id", "pixel_auroc", "pixel_aupr", "n_pixels", "n_positive_pixels", "path"]
    with open(pixel_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pixel_fields)
        writer.writeheader()
        writer.writerows(pixel_rows)
    manifest.update({
        "slice_csv": slice_csv,
        "patient_csv": patient_csv,
        "pixel_slice_metrics_csv": pixel_csv,
        "n_slices": len(rows),
        "n_patients": len(patient_rows),
        "n_pixel_slices": len(pixel_rows),
        "n_abnormal_slices": int(sum(r["true_label"] for r in rows)),
        "n_abnormal_patients": int(sum(r["true_label"] for r in patient_rows)),
        "patient_aggregation": "max slice anomaly_score",
    })
    with open(manifest_json, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Export INPFormer figure score CSV files.")
    parser.add_argument("--data_root", default="../A_data/2d_equal_mask50")
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct,pet")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=392)
    parser.add_argument("--INP_num", type=int, default=6)
    parser.add_argument("--anomaly_source", default="reconstruction", choices=["prototype", "reconstruction", "fused"])
    parser.add_argument("--label_mode", default="folder", choices=["folder", "mask"])
    parser.add_argument("--channel_fill", default="mean_modalities", choices=["zero", "imagenet_mean", "mean_modalities"])
    parser.add_argument("--max_ratio", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--method", default="INPFormer")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    return parser


def main():
    args = build_parser().parse_args()
    args.tracer = args.tracer.upper()
    from paper_eval_utils import parse_modalities
    args.modalities = parse_modalities(args.modalities)
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint_dir) / args.tracer / "_".join(args.modalities) / "figure_scores")
    if getattr(args, "skip_existing", False) and maybe_skip_existing(getattr(args, "output_dir", None), getattr(args, "skip_require_pixel", False)):
        return

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from dataset import MVTecDataset
    from paper_eval_utils import top_percent_mean
    from train import build_model, build_transforms, find_checkpoint, resolve_device, resolve_tracer_root
    from utils import build_anomaly_map

    device = resolve_device(args.device)
    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    data_transform, gt_transform = build_transforms(args.input_size, args.crop_size)
    eval_size = args.input_size if args.crop_size <= 0 else args.crop_size
    dataset = MVTecDataset(
        root=tracer_root,
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
        modalities=args.modalities,
        label_mode=args.label_mode,
        channel_fill=args.channel_fill,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model, _ = build_model(args, device)
    ckpt = find_checkpoint(args, args.modalities)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()

    rows = []
    pixel_rows = []
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(loader, desc="Exporting INPFormer scores"):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map = build_anomaly_map(model, en, de, eval_size, args.anomaly_source)
            if anomaly_map.ndim == 4:
                maps = anomaly_map[:, 0].detach().cpu().numpy().astype(np.float32)
            else:
                maps = anomaly_map.detach().cpu().numpy().astype(np.float32)
            scores = top_percent_mean(maps, args.max_ratio * 100.0)
            labels = label.detach().cpu().view(-1).numpy().astype(int).tolist()
            masks = gt.detach().cpu().numpy()
            for path, lab, score, amap, mask_i in zip(img_path, labels, scores, maps, masks):
                case_id, slice_id = parse_id(path)
                rows.append({
                    "method": args.method,
                    "dataset": args.tracer,
                    "modality": "_".join(args.modalities),
                    "case_id": case_id,
                    "slice_id": slice_id,
                    "true_label": int(lab),
                    "anomaly_score": float(score),
                    "path": str(path),
                })
                pix = pixel_metric_row(args.method, args.tracer, "_".join(args.modalities), case_id, slice_id, path, amap, mask_i)
                if pix is not None:
                    pixel_rows.append(pix)

    output_dir = args.output_dir
    manifest = write_outputs(rows, pixel_rows, output_dir, {
        "method": args.method,
        "dataset": args.tracer,
        "modality": "_".join(args.modalities),
        "data_root": str(tracer_root),
        "checkpoint": str(ckpt),
        "score_rule": f"top {args.max_ratio * 100.0:g}% mean from INPFormer anomaly map",
    })
    print(f"Saved slice scores:   {manifest['slice_csv']}")
    print(f"Saved patient scores: {manifest['patient_csv']}")
    print(f"Saved pixel metrics:  {manifest['pixel_slice_metrics_csv']}")
    print(f"Saved manifest:       {os.path.join(output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
