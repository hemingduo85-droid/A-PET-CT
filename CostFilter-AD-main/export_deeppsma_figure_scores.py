#!/usr/bin/env python3
"""DeepPSMA generalization commands (CostFilter-AD):
python export_deeppsma_figure_scores.py --source_dataset PSMA --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --modality petct --gpu 4 --test_stage costfilter --output_dir generalization_outputs/PSMA_weights/figure_scores --skip_existing --skip_require_pixel

python export_deeppsma_figure_scores.py --source_dataset FDG --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --modality petct --gpu 4 --test_stage costfilter --output_dir generalization_outputs/FDG_weights/figure_scores --skip_existing --skip_require_pixel
"""
# Run examples:
#   python export_deeppsma_figure_scores.py --dataset psma --modality petct --gpu 4 --test_stage costfilter --skip_existing --skip_require_pixel
#   python export_deeppsma_figure_scores.py --dataset fdg --modality petct --gpu 4 --test_stage costfilter --skip_existing --skip_require_pixel
# Add --skip_require_pixel if existing runs should only be skipped when pixel_slice_metrics.csv also exists.
"""Export CostFilter-AD PET-CT slice/patient scores for figure plotting."""


import argparse
import csv
import json
import os
import sys
from collections import defaultdict


DEFAULT_DEEPPSMA_ROOT = "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1"
GENERALIZATION_OUTPUT_TEMPLATE = "generalization_outputs/{source_dataset}_weights/figure_scores"


def prepare_deeppsma_view(data_root, view_root):
    """Create an idempotent patient-style symlink view of flat DeepPSMA slices."""
    data_root = os.path.abspath(os.path.expanduser(data_root))
    view_root = os.path.abspath(os.path.expanduser(view_root))
    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    total = 0
    for group, needs_label in (("normal", False), ("abnormal", True)):
        source_root = os.path.join(data_root, "test", group)
        modality_dirs = {name: os.path.join(source_root, name) for name in ("pet", "ct")}
        if needs_label:
            modality_dirs["label"] = os.path.join(source_root, "label")
        missing_dirs = [path for path in modality_dirs.values() if not os.path.isdir(path)]
        if missing_dirs:
            raise FileNotFoundError(f"Incomplete DeepPSMA {group} layout: {missing_dirs}")
        names = sorted(
            name for name in os.listdir(modality_dirs["pet"])
            if os.path.splitext(name)[1].lower() in image_exts
        )
        for name in names:
            sources = {key: os.path.join(folder, name) for key, folder in modality_dirs.items()}
            missing_files = [path for path in sources.values() if not os.path.isfile(path)]
            if missing_files:
                raise FileNotFoundError(f"Unpaired DeepPSMA slice {name}: {missing_files}")
            case_id = os.path.splitext(name)[0]
            for modality, source in sources.items():
                target_dir = os.path.join(view_root, "test", group, case_id, modality)
                os.makedirs(target_dir, exist_ok=True)
                target = os.path.join(target_dir, name)
                if os.path.lexists(target):
                    if os.path.islink(target) and os.path.realpath(target) == os.path.realpath(source):
                        continue
                    raise FileExistsError(f"Refusing to replace existing DeepPSMA view entry: {target}")
                os.symlink(os.path.abspath(source), target)
            total += 1
    if total == 0:
        raise RuntimeError(f"No DeepPSMA slices found under {data_root}")
    os.makedirs(os.path.join(view_root, "train"), exist_ok=True)
    return view_root



TARGET_DATASET_KEY = "target_dataset"


def rewrite_generalization_argv(argv, source_flag, data_flag, output_flag, source_case="lower"):
    """Map the common generalization CLI onto one method's original parser."""
    argv = list(argv)
    if "-h" in argv or "--help" in argv:
        return argv, "PSMA", DEFAULT_DEEPPSMA_ROOT, GENERALIZATION_OUTPUT_TEMPLATE.format(source_dataset="PSMA")
    extracted = {}
    cleaned = []
    index = 0
    aliases = {"--source_dataset": "source_dataset", "--data_root": "data_root", "--output_dir": "output_dir"}
    while index < len(argv):
        token = argv[index]
        matched = False
        for flag, key in aliases.items():
            if token == flag:
                if index + 1 >= len(argv):
                    raise ValueError(f"Missing value for {flag}")
                extracted[key] = argv[index + 1]
                index += 2
                matched = True
                break
            if token.startswith(flag + "="):
                extracted[key] = token.split("=", 1)[1]
                index += 1
                matched = True
                break
        if not matched:
            cleaned.append(token)
            index += 1
    source_dataset = extracted.get("source_dataset", "PSMA").upper()
    if source_dataset not in {"PSMA", "FDG"}:
        raise ValueError(f"source_dataset must be PSMA or FDG, got {source_dataset}")
    data_root = extracted.get("data_root", DEFAULT_DEEPPSMA_ROOT)
    output_dir = extracted.get(
        "output_dir",
        GENERALIZATION_OUTPUT_TEMPLATE.format(source_dataset=source_dataset),
    )
    os.environ["DEEPPSMA_SOURCE_DATASET"] = source_dataset
    os.environ["DEEPPSMA_TARGET_DATA_ROOT"] = os.path.abspath(data_root)
    view_root = prepare_deeppsma_view(data_root, os.path.join(output_dir, "_deeppsma_view"))
    source_value = source_dataset.lower() if source_case == "lower" else source_dataset
    cleaned.extend([source_flag, source_value, data_flag, view_root, output_flag, output_dir])
    return cleaned, source_dataset, data_root, output_dir



def expected_output_files(output_dir, require_pixel=False):
    files = [
        os.path.join(output_dir, "slice_scores.csv"),
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
        description="Run CostFilter-AD PET-CT inference and export slice/patient score tables."
    )
    parser.add_argument("--dataset", type=str, default="psma", choices=["psma", "fdg"])
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--modality", type=str, default="petct", choices=["ct", "pet", "petct"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=252)
    parser.add_argument("--costfilter_epochs", type=int, default=30)
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--dinomaly_ckpt_path", type=str, default=None)
    parser.add_argument("--costfilter_ckpt_path", type=str, default=None)
    parser.add_argument("--heatmap_dir", type=str, default=None)
    parser.add_argument("--heatmap_limit", type=int, default=128)
    parser.add_argument("--eval_cache_path", type=str, default=None)
    parser.add_argument("--test_stage", type=str, default="costfilter", choices=["dinomaly", "costfilter"])
    parser.add_argument("--ci_iters", type=int, default=500)
    parser.add_argument("--ci_seed", type=int, default=42)
    parser.add_argument("--ci_pixel_max_samples", type=int, default=200000)
    parser.add_argument("--ci_hist_bins", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    return parser.parse_args()


def compute_pixel_metric_row(amap, gt, slice_id, pid, label, image_path):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_score = np.asarray(amap, dtype=np.float32).reshape(-1)
    y_true = (np.asarray(gt).reshape(-1) > 0.5).astype(np.uint8)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"Pixel label/score size mismatch for {image_path}: "
            f"{y_true.shape[0]} labels vs {y_score.shape[0]} scores"
        )
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=float(np.max(y_score[np.isfinite(y_score)])) if np.isfinite(y_score).any() else 0.0)
    n_pixels = int(y_true.size)
    n_positive = int(y_true.sum())
    row = {
        "slice_id": slice_id,
        "patient_id": pid,
        "label": int(label),
        "pixel_auroc": "",
        "pixel_aupr": "",
        "n_pixels": n_pixels,
        "n_positive_pixels": n_positive,
        "image_path": image_path,
    }
    if 0 < n_positive < n_pixels:
        row["pixel_auroc"] = float(roc_auc_score(y_true, y_score))
        row["pixel_aupr"] = float(average_precision_score(y_true, y_score))
    return row


def write_exports(output_dir, amaps, gts, labels, pids, image_paths, manifest):
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    flat = amaps.reshape(amaps.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * 0.01)))
    scores = np.sort(flat, axis=1)[:, -k:].mean(axis=1)

    slice_rows = []
    for idx, (score, label, pid, path) in enumerate(zip(scores, labels, pids, image_paths)):
        slice_rows.append(
            {
                "slice_id": idx,
                "patient_id": pid,
                "label": int(label),
                "score": float(score),
                "image_path": path,
            }
        )

    slice_csv = os.path.join(output_dir, "slice_scores.csv")
    with open(slice_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slice_id", "patient_id", "label", "score", "image_path"])
        writer.writeheader()
        writer.writerows(slice_rows)

    grouped = defaultdict(lambda: {"scores": [], "labels": []})
    for row in slice_rows:
        grouped[row["patient_id"]]["scores"].append(float(row["score"]))
        grouped[row["patient_id"]]["labels"].append(int(row["label"]))
    patient_rows = [
        {
            "patient_id": pid,
            "label": max(grouped[pid]["labels"]),
            "score": max(grouped[pid]["scores"]),
            "n_slices": len(grouped[pid]["scores"]),
        }
        for pid in sorted(grouped)
    ]
    patient_csv = os.path.join(output_dir, "patient_scores.csv")
    with open(patient_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "label", "score", "n_slices"])
        writer.writeheader()
        writer.writerows(patient_rows)

    pixel_rows = [
        compute_pixel_metric_row(amap, gt, idx, pid, label, path)
        for idx, (amap, gt, label, pid, path) in enumerate(zip(amaps, gts, labels, pids, image_paths))
    ]
    pixel_csv = os.path.join(output_dir, "pixel_slice_metrics.csv")
    with open(pixel_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slice_id",
                "patient_id",
                "label",
                "pixel_auroc",
                "pixel_aupr",
                "n_pixels",
                "n_positive_pixels",
                "image_path",
            ],
        )
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
            "n_valid_pixel_metric_slices": int(sum(row["pixel_auroc"] != "" for row in pixel_rows)),
        }
    )
    manifest_json = os.path.join(output_dir, "manifest.json")
    with open(manifest_json, "w") as f:
        json.dump(manifest, f, indent=2)
    return slice_csv, patient_csv, pixel_csv, manifest_json


def main():
    args = parse_args()

    import logging
    import sys

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    import run_petct as rp

    rp.resolve_dataset_profile(args)
    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.save_dir,
            "figure_scores",
            args.dataset_tag,
            args.modality,
            args.test_stage,
        )
    if getattr(args, "skip_existing", False) and maybe_skip_existing(getattr(args, "output_dir", None), getattr(args, "skip_require_pixel", False)):
        return

    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    rp.setup_seed(args.seed)
    logger = rp.get_logger("export_figure_scores", args.save_dir)

    data_transform, gt_transform = rp.get_data_transforms(args.image_size, args.crop_size)
    test_dataset = rp.PETCTTestDataset(args.data_path, args.modality, data_transform, gt_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    captured = {}
    original_compute = rp._compute_three_level

    def capture_compute(amaps_np, gts_np, labels, pids, **kwargs):
        amaps_np = np.asarray(amaps_np)
        gts_np = np.asarray(gts_np)
        labels = np.asarray(labels, dtype=int)
        captured["amaps"] = amaps_np
        captured["gts"] = gts_np
        captured["labels"] = labels
        captured["pids"] = list(pids)
        flat = amaps_np.reshape(amaps_np.shape[0], -1)
        k = max(1, int(np.ceil(flat.shape[1] * 0.01)))
        image_scores = np.sort(flat, axis=1)[:, -k:].mean(axis=1)
        patient_ids = list(dict.fromkeys(captured["pids"]))
        patient_labels = {
            pid: max(int(label) for cur_pid, label in zip(captured["pids"], labels) if cur_pid == pid)
            for pid in patient_ids
        }
        return {
            "image_scores": image_scores,
            "n_patients": len(patient_ids),
            "n_abn_patients": int(sum(patient_labels.values())),
            "nonfinite_count": int((~np.isfinite(amaps_np)).sum()),
        }

    rp._compute_three_level = capture_compute
    try:
        model, embed_dim = rp.build_dinomaly(args.device)
        if not os.path.exists(args.dinomaly_ckpt):
            raise FileNotFoundError(f"Dinomaly checkpoint not found: {args.dinomaly_ckpt}")
        rp.load_dinomaly_checkpoint(model, args.dinomaly_ckpt, args.device)

        if args.test_stage == "dinomaly":
            rp.evaluate_dinomaly(
                model,
                test_loader,
                args.device,
                args.crop_size,
                sigma=4,
                logger=logger,
                prefix="[Export Dinomaly] ",
                ci_iters=args.ci_iters,
                ci_seed=args.ci_seed,
                ci_pixel_max_samples=args.ci_pixel_max_samples,
            )
            checkpoint = args.dinomaly_ckpt
        else:
            if not os.path.exists(args.costfilter_ckpt):
                raise FileNotFoundError(f"CostFilter checkpoint not found: {args.costfilter_ckpt}")
            feat_size = args.crop_size // 14
            feat_total = feat_size * feat_size
            pre_min_dim = min(768, feat_total)
            model_unet = rp.DiscriminativeSubNetwork_3d_att_dino_channel(
                in_channels=pre_min_dim,
                out_channels=2,
                base_channels=args.base_channels,
            ).to(args.device).float()
            rp.load_costfilter_checkpoint(model_unet, args.costfilter_ckpt, args.device)
            rp.evaluate_costfilter(
                model,
                model_unet,
                test_loader,
                args.device,
                args.crop_size,
                feat_size,
                pre_min_dim,
                embed_dim,
                sigma=4,
                lamda=args.lamda,
                heatmap_dir=None,
                heatmap_limit=0,
                modality=args.modality,
                logger=logger,
                prefix="[Export CostFilter] ",
                ci_iters=args.ci_iters,
                ci_seed=args.ci_seed,
                ci_pixel_max_samples=args.ci_pixel_max_samples,
            )
            checkpoint = args.costfilter_ckpt
    finally:
        rp._compute_three_level = original_compute

    if "amaps" not in captured:
        raise RuntimeError("No anomaly maps were captured from CostFilter evaluation.")

    paths = []
    for ct_path, pet_path, _, _, _ in test_dataset.samples:
        paths.append(pet_path if args.modality in {"pet", "petct"} else ct_path)

    output_dir = args.output_dir
    outputs = write_exports(
        output_dir,
        captured["amaps"],
        captured["gts"],
        captured["labels"],
        captured["pids"],
        paths,
        {
            "repo": "CostFilter-AD-main/Costfilter_Dinomaly",
            "dataset": "DeepPSMA",
            "source_dataset": os.environ["DEEPPSMA_SOURCE_DATASET"],
            "target_dataset": "DeepPSMA",
            "original_data_root": os.environ["DEEPPSMA_TARGET_DATA_ROOT"],
            "modality": args.modality,
            "test_stage": args.test_stage,
            "data_path": args.data_path,
            "save_dir": args.save_dir,
            "checkpoint": checkpoint,
            "score_rule": "mean of top-1% pixels from the same anomaly map used by run_petct._compute_three_level",
            "patient_rule": "max slice score per patient",
        },
    )
    print("Exported:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    sys.argv[1:], _SOURCE_DATASET, _TARGET_DATA_ROOT, _GENERALIZATION_OUTPUT_DIR = rewrite_generalization_argv(
        sys.argv[1:],
        source_flag="--dataset",
        data_flag="--data_path",
        output_flag="--output_dir",
        source_case="lower",
    )
    main()
