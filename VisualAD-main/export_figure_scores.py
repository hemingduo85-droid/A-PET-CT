#!/usr/bin/env python3
# Run examples:
#   python export_figure_scores.py --dataset psma --modality petct --device cuda:7 --epoch 30 --skip_existing --skip_require_pixel
#   python export_figure_scores.py --dataset fdg --modality petct --device cuda:7 --epoch 30 --skip_existing --skip_require_pixel
# These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""Run VisualAD PET/CT checkpoint inference and export figure score CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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


def setup_seed(seed):
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def collect_scores(args):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.ndimage import gaussian_filter
    from tqdm import tqdm

    import VisualAD_lib
    from dataset_petct import PETCTDataset
    from utils.anomaly_detection import generate_anomaly_map_from_tokens
    from utils.feature_transform import create_feature_transform
    from utils.petct_config import resolve_checkpoint_path, resolve_petct_paths
    from utils.scoring import DEFAULT_TOPK_RATIO, reduce_anomaly_map
    from utils.transforms import get_transform

    setup_seed(args.seed)
    _train_path, args.test_data_path = resolve_petct_paths(args.dataset, test_data_path=args.test_data_path)
    args.checkpoint_path = resolve_checkpoint_path(
        args.checkpoint_path, args.checkpoint_root, args.dataset, args.modality, args.epoch
    )
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    args.backbone = checkpoint.get("backbone", "ViT-L/14@336px")
    args.image_size = checkpoint.get("image_size", 256)
    args.features_list = checkpoint.get("features_list", [6, 12, 18, 24])

    preprocess, target_transform = get_transform(args)
    model, _ = VisualAD_lib.load(args.backbone, device=device)
    model.eval()
    model.to(device)
    feature_dim = model.visual.embed_dim
    model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
    model.visual.normal_token.data = checkpoint["normal_token"].to(device)
    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None and checkpoint.get("ln_post_weight") is not None:
        ln_post.weight.data = checkpoint["ln_post_weight"].to(device)
        ln_post.bias.data = checkpoint["ln_post_bias"].to(device)

    layer_transforms = nn.ModuleDict()
    if "layer_transforms" in checkpoint:
        for layer_name, state_dict in checkpoint["layer_transforms"].items():
            hidden_dim = state_dict["mlp.0.weight"].shape[0]
            layer_transforms[layer_name] = create_feature_transform(
                transform_type="mlp",
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
                dropout=0.0,
            ).to(device)
            layer_transforms[layer_name].load_state_dict(state_dict)
            layer_transforms[layer_name].eval()

    cross_attn = None
    if "cross_attn" in checkpoint:
        from utils.spatial_cross_attention import build_layer_adaptive_cross_attention

        config = checkpoint.get("cross_attn_config", {})
        cross_attn = build_layer_adaptive_cross_attention(
            layers=args.features_list,
            embed_dim=feature_dim,
            num_anchors=config.get("num_anchors", 4),
            dropout=config.get("dropout", 0.1),
            res_scale_init=config.get("res_scale_init", 0.01),
        ).to(device)
        cross_attn.load_state_dict(checkpoint["cross_attn"])
        cross_attn.eval()

    test_data = PETCTDataset(
        root=args.test_data_path,
        transform=preprocess,
        target_transform=target_transform,
        modality=args.modality,
        split="test",
    )
    loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    rows = []
    pixel_rows = []
    for idx, items in enumerate(tqdm(loader, desc="Exporting VisualAD scores")):
        image = items["img"].to(device)
        with torch.no_grad():
            vision_output = model.encode_image(image, args.features_list)
            anomaly_features = vision_output["anomaly_features"]
            normal_features = vision_output["normal_features"]
            patch_tokens = vision_output["patch_tokens"]
            patch_start_idx = vision_output["patch_start_idx"]
            patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
            if cross_attn is not None:
                adapted_list = cross_attn(anomaly_features, normal_features, patch_features_list, args.features_list)
                anomaly_features_list = [a["anomaly"] for a in adapted_list]
                normal_features_list = [a["normal"] for a in adapted_list]
            else:
                anomaly_features_list = [anomaly_features] * len(patch_tokens)
                normal_features_list = [normal_features] * len(patch_tokens)

            anomaly_map_list = []
            for layer_idx, patch_feature in enumerate(patch_tokens):
                af_norm = F.normalize(anomaly_features_list[layer_idx], dim=1, eps=1e-8)
                nf_norm = F.normalize(normal_features_list[layer_idx], dim=1, eps=1e-8)
                layer_key = f"layer_{args.features_list[layer_idx]}"
                if layer_key in layer_transforms:
                    bsz, num_tokens, dim = patch_feature.shape
                    patch_feature = layer_transforms[layer_key](
                        patch_feature.view(-1, dim)
                    ).view(bsz, num_tokens, dim)
                anomaly_map_list.append(
                    generate_anomaly_map_from_tokens(
                        af_norm, nf_norm, patch_feature[:, patch_start_idx:, :], args.image_size
                    )
                )
            final_map = torch.stack(anomaly_map_list).sum(dim=0).cpu()
            filtered = gaussian_filter(final_map[0].numpy(), sigma=args.sigma)
            final_map = torch.from_numpy(filtered).unsqueeze(0)
            score = reduce_anomaly_map(final_map, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO).item()

        path = str(items["img_path"][0])
        case_id = case_id_from_path(path)
        slice_id = slice_id_from_path(path, idx)
        rows.append({
            "method": args.method,
            "dataset": args.dataset.upper(),
            "modality": args.modality,
            "case_id": case_id,
            "slice_id": slice_id,
            "true_label": int(items["anomaly"].item()),
            "anomaly_score": float(score),
            "path": path,
        })
        mask = items["img_mask"].detach().cpu().numpy() if hasattr(items["img_mask"], "detach") else items["img_mask"]
        pix = pixel_metric_row(args.method, args.dataset.upper(), args.modality, case_id, slice_id, path, final_map.squeeze(0).numpy(), mask)
        if pix is not None:
            pixel_rows.append(pix)
    return rows, pixel_rows


def write_outputs(rows, pixel_rows, args):
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "figure_scores",
            str(args.dataset).lower(),
            str(args.modality).lower(),
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
        "dataset": args.dataset,
        "test_data_path": args.test_data_path,
        "modality": args.modality,
        "checkpoint": args.checkpoint_path,
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
    parser = argparse.ArgumentParser(description="Export VisualAD PET/CT figure score CSV files.")
    parser.add_argument("--dataset", default="psma", choices=["fdg", "psma"])
    parser.add_argument("--test_data_path", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--checkpoint_root", default="./experiments")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--modality", default="petct", choices=["pet", "ct", "petct"])
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", default="VisualAD")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "figure_scores",
            str(args.dataset).lower(),
            str(args.modality).lower(),
        )
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
