#!/usr/bin/env python3
# Run examples:
#   python export_figure_scores.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer PSMA --modalities ct,pet --device cuda:4 --skip_existing --skip_require_pixel
#   python export_figure_scores.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer FDG --modalities ct,pet --device cuda:4 --skip_existing --skip_require_pixel
# These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""Export MMRAD PET/CT slice/patient scores for figure plotting."""

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


def split_id(value):
    parts = str(value).split("__")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return "unknown", str(value)


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
    fields = ["method", "dataset", "modality", "case_id", "slice_id", "true_label", "anomaly_score", "path"]
    with open(slice_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(lambda: {"scores": [], "labels": []})
    for row in rows:
        grouped[row["case_id"]]["scores"].append(float(row["anomaly_score"]))
        grouped[row["case_id"]]["labels"].append(int(row["true_label"]))
    patient_rows = [{
        "method": manifest["method"], "dataset": manifest["dataset"], "modality": manifest["modality"],
        "case_id": pid, "true_label": max(grouped[pid]["labels"]),
        "anomaly_score": max(grouped[pid]["scores"]), "n_slices": len(grouped[pid]["scores"]),
    } for pid in sorted(grouped)]
    with open(patient_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "dataset", "modality", "case_id", "true_label", "anomaly_score", "n_slices"])
        writer.writeheader()
        writer.writerows(patient_rows)
    pixel_fields = ["method", "dataset", "modality", "case_id", "slice_id", "pixel_auroc", "pixel_aupr", "n_pixels", "n_positive_pixels", "path"]
    with open(pixel_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pixel_fields)
        writer.writeheader()
        writer.writerows(pixel_rows)
    manifest.update({"slice_csv": slice_csv, "patient_csv": patient_csv, "pixel_slice_metrics_csv": pixel_csv, "n_slices": len(rows), "n_patients": len(patient_rows), "n_pixel_slices": len(pixel_rows), "patient_aggregation": "max slice anomaly_score"})
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Export MMRAD figure score CSV files.")
    parser.add_argument("--data_root", default="../A_data/2d_equal_mask50")
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct,pet")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--topk_percent", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--method", default="MMRAD")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    return parser


def main():
    args = build_parser().parse_args()
    from infer import resolve_path
    from paper_eval_utils import parse_modalities
    args.tracer = args.tracer.upper()
    modalities = parse_modalities(args.modalities)
    if args.output_dir is None:
        args.output_dir = str(resolve_path(args.checkpoint_root) / args.tracer / "_".join(modalities) / "figure_scores")
    if getattr(args, "skip_existing", False) and maybe_skip_existing(getattr(args, "output_dir", None), getattr(args, "skip_require_pixel", False)):
        return

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from dataloader import MultimodalGrayDataset
    from infer import checkpoint_path, load_model
    from paper_eval_utils import top_percent_mean

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = resolve_path(args.data_root)
    tracer_root = data_root if (data_root / "test").exists() else data_root / args.tracer
    dataset = MultimodalGrayDataset(tracer_root, modalities=modalities, mode="test", image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = load_model(args, modalities, device)
    rows = []
    pixel_rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Exporting MMRAD scores"):
            images = batch["image"].to(device)
            recon = model(images)
            residual = F.mse_loss(recon, images, reduction="none").mean(dim=1)
            maps = residual.detach().cpu().numpy().astype(np.float32)
            scores = top_percent_mean(maps, args.topk_percent)
            labels = batch["label"].detach().cpu().view(-1).numpy().astype(int).tolist()
            masks = batch["mask"].detach().cpu().numpy()
            for sample_id, label, score, amap, mask_i in zip(batch["id"], labels, scores, maps, masks):
                case_id, slice_id = split_id(sample_id)
                rows.append({
                    "method": args.method,
                    "dataset": args.tracer,
                    "modality": "_".join(modalities),
                    "case_id": case_id,
                    "slice_id": slice_id,
                    "true_label": int(label),
                    "anomaly_score": float(score),
                    "path": str(sample_id),
                })
                pix = pixel_metric_row(args.method, args.tracer, "_".join(modalities), case_id, slice_id, sample_id, amap, mask_i)
                if pix is not None:
                    pixel_rows.append(pix)
    ckpt = checkpoint_path(args, modalities)
    output_dir = args.output_dir
    manifest = write_outputs(rows, pixel_rows, output_dir, {
        "method": args.method,
        "dataset": args.tracer,
        "modality": "_".join(modalities),
        "data_root": str(tracer_root),
        "checkpoint": str(ckpt),
        "score_rule": f"top {args.topk_percent:g}% mean from MMRAD residual map",
    })
    print(f"Saved slice scores:   {manifest['slice_csv']}")
    print(f"Saved patient scores: {manifest['patient_csv']}")
    print(f"Saved pixel metrics:  {manifest['pixel_slice_metrics_csv']}")
    print(f"Saved manifest:       {os.path.join(output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
