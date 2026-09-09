#!/usr/bin/env python3
"""
Run examples:
  python export_figure_scores.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer PSMA \
    --modalities ct,pet --device cuda:4 --skip_existing --skip_require_pixel
  python export_figure_scores.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer FDG \
    --modalities ct,pet --device cuda:5 --skip_existing --skip_require_pixel
These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from functools import partial
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

PROJECT_DIR = Path(__file__).resolve().parent


def parse_modalities(value):
    modalities = [m.strip().lower() for m in value.split(",") if m.strip()]
    invalid = sorted(set(modalities) - {"ct", "pet"})
    if invalid:
        raise ValueError(f"Invalid modalities: {invalid}. Use ct, pet, or ct,pet.")
    if not modalities:
        raise ValueError("At least one modality is required.")
    return modalities


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    for candidate in (PROJECT_DIR / path, PROJECT_DIR.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_DIR / path).resolve()


def find_checkpoint(args, modalities):
    if args.checkpoint:
        checkpoint = resolve_path(args.checkpoint)
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    checkpoint_dir = resolve_path(args.checkpoint_dir) / args.tracer.upper() / "_".join(modalities)
    for name in ("best_model.pt", "model.pth", "best_model.pth"):
        checkpoint = checkpoint_dir / name
        if checkpoint.exists():
            return checkpoint
    candidates = sorted(checkpoint_dir.glob("*.pt")) + sorted(checkpoint_dir.glob("*.pth"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}; provide --checkpoint")


def top_percent_mean(values, percent):
    import numpy as np

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    flat = arr.reshape(arr.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * float(percent) / 100.0)))
    return np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)


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


def split_id(value):
    parts = str(value).split("__")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return "unknown", str(value)


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
    parser = argparse.ArgumentParser(description="Export ESC-DRKD figure score CSV files.")
    parser.add_argument("--data_root", default="../A_data/2d_equal_mask50")
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct,pet")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--topk_percent", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--method", default="ESC-DRKD")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    return parser


def main():
    args = build_parser().parse_args()
    args.tracer = args.tracer.upper()
    modalities = parse_modalities(args.modalities)
    if args.output_dir is None:
        args.output_dir = str(resolve_path(args.checkpoint_dir) / args.tracer / "_".join(modalities) / "figure_scores")
    if getattr(args, "skip_existing", False) and maybe_skip_existing(getattr(args, "output_dir", None), getattr(args, "skip_require_pixel", False)):
        return

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from checkpoint_schema import require_complete_model_state
    from dataloader import MultimodalDataset, collate_fn, resolve_tracer_root
    from new_model import ESC_DRKD_Multimodal

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    ckpt = find_checkpoint(args, modalities)

    model = ESC_DRKD_Multimodal(modalities=modalities).to(device).eval()
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(require_complete_model_state(checkpoint), strict=True)
    print(f"Loaded checkpoint: {ckpt}")

    dataset = MultimodalDataset(tracer_root, modalities=modalities, mode="test", img_size=args.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_fn, modalities=modalities))

    rows = []
    pixel_rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Exporting ESC-DRKD scores"):
            x_dict = {m: batch["modalities"][m].to(device) for m in modalities}
            x = torch.stack([x_dict[mod] for mod in modalities], dim=0).mean(dim=0)
            teacher_features = model.teacher.backbone(x)
            reconstruction = model.student(teacher_features[-1], teacher_features[:-1])
            recon_features = model.teacher.backbone(reconstruction)
            anomaly_maps = []
            for teacher_feature, recon_feature in zip(teacher_features[:3], recon_features[:3]):
                if recon_feature.shape[-2:] != teacher_feature.shape[-2:]:
                    recon_feature = F.interpolate(recon_feature, size=teacher_feature.shape[-2:], mode="bilinear")
                cosine = F.cosine_similarity(teacher_feature, recon_feature, dim=1)
                anomaly = 1 - cosine.unsqueeze(1)
                anomaly = F.interpolate(anomaly, size=(args.image_size, args.image_size), mode="bilinear")
                anomaly_maps.append(anomaly)
            amap = torch.mean(torch.cat(anomaly_maps, dim=1), dim=1).squeeze().detach().cpu().numpy().astype(np.float32)
            score = float(top_percent_mean(amap, args.topk_percent)[0])
            sample_id = str(batch["id"][0])
            case_id, slice_id = split_id(sample_id)
            mask = batch["mask"].detach().cpu().numpy()
            rows.append({
                "method": args.method,
                "dataset": args.tracer,
                "modality": "_".join(modalities),
                "case_id": case_id,
                "slice_id": slice_id,
                "true_label": int(batch["label"].view(-1)[0].item()),
                "anomaly_score": score,
                "path": sample_id,
            })
            pix = pixel_metric_row(args.method, args.tracer, "_".join(modalities), case_id, slice_id, sample_id, amap, mask)
            if pix is not None:
                pixel_rows.append(pix)
    output_dir = args.output_dir
    manifest = write_outputs(rows, pixel_rows, output_dir, {
        "method": args.method,
        "dataset": args.tracer,
        "modality": "_".join(modalities),
        "data_root": str(tracer_root),
        "checkpoint": str(ckpt),
        "score_rule": f"top {args.topk_percent:g}% mean from ESC-DRKD anomaly map",
    })
    print(f"Saved slice scores:   {manifest['slice_csv']}")
    print(f"Saved patient scores: {manifest['patient_csv']}")
    print(f"Saved pixel metrics:  {manifest['pixel_slice_metrics_csv']}")
    print(f"Saved manifest:       {os.path.join(output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
