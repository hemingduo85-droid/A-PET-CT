#!/usr/bin/env python3
# Run examples:
#   python export_figure_scores.py --dataset psma --modality petct --gpu 4 --skip_existing --skip_require_pixel
#   python export_figure_scores.py --dataset fdg --modality petct --gpu 4 --skip_existing --skip_require_pixel
# Add --skip_require_pixel if existing runs should only be skipped when pixel_slice_metrics.csv also exists.
"""Export GLASS PET-CT slice/patient scores for manuscript figures."""

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GLASS PET-CT inference and export slice/patient score tables."
    )
    parser.add_argument("--dataset", choices=["psma", "fdg"], default="psma")
    parser.add_argument("--data_path", default=None, help="Override PET-CT dataset root.")
    parser.add_argument("--modality", choices=["pet", "ct", "petct"], default="petct")
    parser.add_argument("--save_dir", default=None, help="Directory containing GLASS models/.")
    parser.add_argument("--output_dir", default=None, help="Defaults to <save_dir>/figure_scores.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--backbone", default="wideresnet50")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1536)
    parser.add_argument("--target_embed_dimension", type=int, default=1536)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--meta_epochs", type=int, default=30)
    parser.add_argument("--eval_epochs", type=int, default=30)
    parser.add_argument("--dsc_layers", type=int, default=2)
    parser.add_argument("--dsc_hidden", type=int, default=1024)
    parser.add_argument("--dsc_margin", type=float, default=0.5)
    parser.add_argument("--pre_proj", type=int, default=1)
    parser.add_argument("--mining", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--radius", type=float, default=0.75)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--svd", type=int, default=0)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--limit", type=int, default=1840)
    return parser.parse_args()


def patient_id_from_path(path: str) -> str:
    p = Path(path)
    if p.parent.name in {"pet", "ct", "label"}:
        return p.parent.parent.name
    return p.parent.name


def pixel_metric_row(method, dataset, modality, slice_id, patient_id, label, image_path, anomaly_map, mask):
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
        "case_id": patient_id,
        "slice_id": slice_id,
        "pixel_auroc": float(roc_auc_score(y_true, y_score)),
        "pixel_aupr": float(average_precision_score(y_true, y_score)),
        "n_pixels": n_pixels,
        "n_positive_pixels": n_pos,
        "path": str(image_path),
    }


def write_exports(output_dir, slice_rows, pixel_rows, manifest):
    os.makedirs(output_dir, exist_ok=True)
    slice_csv = os.path.join(output_dir, "slice_scores.csv")
    patient_csv = os.path.join(output_dir, "patient_scores.csv")
    pixel_csv = os.path.join(output_dir, "pixel_slice_metrics.csv")
    manifest_json = os.path.join(output_dir, "manifest.json")

    with open(slice_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slice_id", "patient_id", "label", "score", "image_path"])
        writer.writeheader()
        writer.writerows(slice_rows)

    grouped = defaultdict(lambda: {"scores": [], "labels": []})
    for row in slice_rows:
        grouped[row["patient_id"]]["scores"].append(float(row["score"]))
        grouped[row["patient_id"]]["labels"].append(int(row["label"]))

    patient_rows = []
    for pid in sorted(grouped):
        patient_rows.append(
            {
                "patient_id": pid,
                "label": max(grouped[pid]["labels"]),
                "score": max(grouped[pid]["scores"]),
                "n_slices": len(grouped[pid]["scores"]),
            }
        )
    with open(patient_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "label", "score", "n_slices"])
        writer.writeheader()
        writer.writerows(patient_rows)

    pixel_fields = ["method", "dataset", "modality", "case_id", "slice_id", "pixel_auroc", "pixel_aupr", "n_pixels", "n_positive_pixels", "path"]
    with open(pixel_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pixel_fields)
        writer.writeheader()
        writer.writerows(pixel_rows)

    manifest.update(
        {
            "slice_scores_csv": slice_csv,
            "patient_scores_csv": patient_csv,
            "pixel_slice_metrics_csv": pixel_csv,
            "n_slices": len(slice_rows),
            "n_patients": len(patient_rows),
            "n_pixel_slices": len(pixel_rows),
        }
    )
    with open(manifest_json, "w") as f:
        json.dump(manifest, f, indent=2)
    return slice_csv, patient_csv, pixel_csv, manifest_json


def main():
    args = parse_args()
    import petct_config
    data_path = args.data_path or petct_config.resolve_dataset(args.dataset)["data_path"]
    save_dir = args.save_dir or petct_config.default_save_dir("results", args.dataset, args.modality)
    output_dir = args.output_dir or os.path.join(save_dir, "figure_scores")
    if getattr(args, "skip_existing", False) and maybe_skip_existing(output_dir, getattr(args, "skip_require_pixel", False)):
        return

    import torch
    import backbones
    import glass
    import utils
    from datasets.petct import DatasetSplit, PETCTDataset

    device = utils.set_torch_device([args.gpu])
    utils.fix_seeds(args.seed, device)

    dataset = PETCTDataset(
        data_path,
        "",
        dataset_name="petct",
        classname=args.modality,
        resize=args.resize,
        imagesize=args.imagesize,
        split=DatasetSplit.TEST,
        seed=args.seed,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    test_loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    backbone = backbones.load(args.backbone)
    backbone.name, backbone.seed = args.backbone, None
    model = glass.GLASS(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=args.layers,
        device=device,
        input_shape=dataset.imagesize,
        pretrain_embed_dimension=args.pretrain_embed_dimension,
        target_embed_dimension=args.target_embed_dimension,
        patchsize=args.patchsize,
        meta_epochs=args.meta_epochs,
        eval_epochs=args.eval_epochs,
        dsc_layers=args.dsc_layers,
        dsc_hidden=args.dsc_hidden,
        dsc_margin=args.dsc_margin,
        pre_proj=args.pre_proj,
        mining=args.mining,
        noise=args.noise,
        radius=args.radius,
        p=args.p,
        lr=args.lr,
        svd=args.svd,
        step=args.step,
        limit=args.limit,
    )
    model = model.to(device)
    dataset_name = f"petct_{args.modality}"
    model.set_model_dir(os.path.join(save_dir, "models", "backbone_0"), dataset_name)

    ckpts = model._find_checkpoint()
    if not ckpts:
        raise FileNotFoundError(f"No GLASS checkpoint found under {model.ckpt_dir}")
    state_dict = torch.load(ckpts[0], map_location=device)
    if "discriminator" in state_dict:
        model.discriminator.load_state_dict(state_dict["discriminator"])
        if "pre_projection" in state_dict:
            model.pre_projection.load_state_dict(state_dict["pre_projection"])
    else:
        model.load_state_dict(state_dict, strict=False)

    _, scores, segmentations, labels, masks_gt, paths = model.predict(test_loader)
    slice_rows = []
    pixel_rows = []
    for idx, (score, label, path, segmentation, mask_gt) in enumerate(zip(scores, labels, paths, segmentations, masks_gt)):
        pid = patient_id_from_path(path)
        slice_rows.append(
            {
                "slice_id": idx,
                "patient_id": pid,
                "label": int(label),
                "score": float(score),
                "image_path": path,
            }
        )
        pix = pixel_metric_row("GLASS", args.dataset, args.modality, idx, pid, label, path, segmentation, mask_gt)
        if pix is not None:
            pixel_rows.append(pix)

    outputs = write_exports(
        output_dir,
        slice_rows,
        pixel_rows,
        {
            "repo": "GLASS-main",
            "dataset": args.dataset,
            "modality": args.modality,
            "data_path": data_path,
            "save_dir": save_dir,
            "checkpoint": ckpts[0],
            "score_rule": "GLASS anomaly map top-1% mean, matching glass.GLASS._predict",
            "patient_rule": "max slice score per patient",
        },
    )
    print("Exported:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
