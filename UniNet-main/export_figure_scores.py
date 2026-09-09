#!/usr/bin/env python3
# Run examples:
#   python export_figure_scores.py --dataset_name PSMA --modality petct --gpu 4 --epochs 30 --skip_existing --skip_require_pixel
#   python export_figure_scores.py --dataset_name FDG --modality petct --gpu 6 --epochs 30 --skip_existing --skip_require_pixel
# These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""Run UniNet PET/CT checkpoint inference and export figure score CSV files."""

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


DATASET_ROOTS = {
    "psma": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA",
    "fdg": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG",
}


def case_id_from_path(path):
    p = Path(str(path))
    if p.parent.name.lower() in {"pet", "ct", "label", "mask", "masks"}:
        return p.parent.parent.name
    return p.parent.name


def slice_id_from_path(path, idx):
    stem = Path(str(path)).stem
    return stem or f"{idx:06d}"


def final_ckpt_suffix(args):
    return args.ckpt_suffix or f"EPOCH_{args.epochs:03d}"


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


def prepare_config(args):
    dataset_tag = args.dataset_name.strip().replace(" ", "_").upper()
    args.dataset_name = dataset_tag
    if args.data_root is None:
        if dataset_tag.lower() not in DATASET_ROOTS:
            raise ValueError(f"unknown dataset_name={args.dataset_name}; pass --data_root explicitly")
        args.data_root = DATASET_ROOTS[dataset_tag.lower()]
    args.dataset = f"PETCT_{dataset_tag}_{args.modality.upper()}"
    args._class_ = args.dataset
    args.domain = "medical"
    args.setting = "oc"
    return args


def collect_scores(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import gaussian_filter

    import main_petct
    from datasets_petct import get_petct_dataloaders
    from eval_protocol import image_score_from_map
    from UniNet_lib.mechanism import weighted_decision_mechanism
    from utils import load_weights

    args = prepare_config(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _train_loader, test_loader = get_petct_dataloaders(
        data_root=args.data_root,
        modality=args.modality,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model, _bn, _student, dfs, _target_teacher = main_petct.build_model(args, device)
    ckpt_path = os.path.join(args.ckpt_dir, args.dataset)
    suffix = final_ckpt_suffix(args)
    modules = [model.t.t_t, model.bn.bn, model.s.s1, dfs]
    state = load_weights(modules, ckpt_path, suffix, device=device)
    model.t.t_t = state["tt"]
    model.bn.bn = state["bn"]
    model.s.s1 = state["st"]
    model.dfs = state["dfs"]
    model.train_or_eval(type="eval")

    n = model.n
    output_list = [[] for _ in range(n * 3)]
    labels = []
    paths = []
    masks = []
    weights_cnt = 0
    with torch.no_grad():
        for img, label, mask, path in test_loader:
            img = img.to(device)
            t_tf, de_features = model(img)
            labels.extend(label.numpy().tolist())
            paths.extend(path if isinstance(path, (list, tuple)) else [path])
            masks.extend(mask.detach().cpu().numpy())
            weights_cnt += 1
            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                output_list[l].append(1 - F.cosine_similarity(t, s))

    _unused, anomaly_map = weighted_decision_mechanism(
        weights_cnt, output_list, args.alpha, args.beta, out_size=args.image_size
    )
    anomaly_map = np.stack([gaussian_filter(anomaly_map[i], sigma=4) for i in range(len(anomaly_map))])
    scores = np.array([image_score_from_map(anomaly_map[i], agg=args.image_agg) for i in range(len(anomaly_map))])
    rows = []
    pixel_rows = []
    for idx, (label, score, path, mask) in enumerate(zip(labels, scores, paths, masks)):
        path = str(path)
        case_id = case_id_from_path(path)
        slice_id = slice_id_from_path(path, idx)
        rows.append({
            "method": args.method,
            "dataset": args.dataset_name,
            "modality": args.modality,
            "case_id": case_id,
            "slice_id": slice_id,
            "true_label": int(label),
            "anomaly_score": float(score),
            "path": path,
        })
        pix = pixel_metric_row(args.method, args.dataset_name, args.modality, case_id, slice_id, path, anomaly_map[idx], mask)
        if pix is not None:
            pixel_rows.append(pix)
    args._resolved_checkpoint = os.path.join(ckpt_path, f"{suffix}.pth")
    return rows, pixel_rows


def write_outputs(rows, pixel_rows, args):
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
            "modality": args.modality,
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
        "dataset": args.dataset_name,
        "data_root": args.data_root,
        "modality": args.modality,
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
    parser = argparse.ArgumentParser(description="Export UniNet figure score CSV files.")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--dataset_name", default="PSMA")
    parser.add_argument("--modality", default="petct", choices=["pet", "ct", "petct"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr_s", type=float, default=5e-3)
    parser.add_argument("--lr_t", type=float, default=1e-6)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--weighted_decision_mechanism", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.00003)
    parser.add_argument("--default", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", default="./saved_results")
    parser.add_argument("--ckpt_dir", default="./ckpts")
    parser.add_argument("--ckpt_suffix", default=None)
    parser.add_argument("--image_agg", default="top1pct", choices=["top1pct", "max"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--method", default="UniNet")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.output_dir is None:
        dataset_tag = args.dataset_name.strip().replace(" ", "_").upper()
        args.output_dir = os.path.join("./saved_results", f"PETCT_{dataset_tag}_{args.modality.upper()}", "figure_scores")
    return args


def main():
    args = parse_args()
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
