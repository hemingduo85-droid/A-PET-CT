#!/usr/bin/env python3
"""DeepPSMA generalization commands (GatingAno):
python export_deeppsma_figure_scores.py --source_dataset PSMA --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --modality petct --ckpt /data/cyf/codes/A-PET-CT/GatingAno/checkpoints/PSMA_petct_epoch30.pth --gpu 4 --output_dir generalization_outputs/PSMA_weights/figure_scores --skip_existing --skip_require_pixel

python export_deeppsma_figure_scores.py --source_dataset FDG --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --modality petct --ckpt /data/cyf/codes/A-PET-CT/GatingAno/checkpoints/FDG_petct_epoch30.pth --gpu 5 --output_dir generalization_outputs/FDG_weights/figure_scores --skip_existing --skip_require_pixel
"""
"""
Run examples:
  python export_deeppsma_figure_scores.py --dataset PSMA  \
    --modality petct  --ckpt /data/cyf/codes/A-PET-CT/GatingAno/checkpoints/ctctpet/PSMA_petct_epoch30.pth --gpu 4 \
    --skip_existing --skip_require_pixel
  python export_deeppsma_figure_scores.py --dataset FDG \
    --modality petct  --ckpt /data/cyf/codes/A-PET-CT/GatingAno/checkpoints/ctctpetfdg30/FDG_petct_epoch30.pth  --gpu 4 \
    --skip_existing --skip_require_pixel
These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""


import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


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


def build_transform(image_size):
    from torchvision import transforms as T

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_generator(config, ckpt_path):
    import torch

    from models import GatingAno

    generator = GatingAno(
        n_channels=config.input_channels,
        n_classes=config.output_channels,
    ).to(config.device)
    try:
        checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=config.device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    generator.load_state_dict(state_dict)
    if isinstance(checkpoint, dict) and "alpha" in checkpoint:
        config.alpha = checkpoint["alpha"]
    generator.eval()
    return generator


def patient_id_from_path_local(path):
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


def slice_id_from_path(path, index):
    stem = Path(str(path)).stem
    return stem or f"{index:06d}"


def pixel_metric_row(method, dataset, modality, case_id, slice_id, path, anomaly_map, mask):
    import cv2
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    amap = np.asarray(anomaly_map, dtype=np.float32)
    if amap.ndim == 3:
        amap = amap.mean(axis=0)
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


def default_output_dir(config):
    return os.path.join(config.save_dir, "figure_scores")


def collect_scores(config, generator, test_loader, method):
    import torch
    from tqdm import tqdm

    rows = []
    pixel_rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Exporting figure scores")):
            images, labels, masks, paths = batch
            images = images.to(config.device)
            labels = labels.cpu().numpy()
            masks = masks.cpu().numpy()
            if isinstance(paths, (str, bytes)):
                paths = [paths]

            _, reconstructed = generator(images, config.alpha)
            anomaly_map = torch.abs(images - reconstructed)
            scores = anomaly_map.mean(dim=[1, 2, 3]).cpu().numpy()

            maps = anomaly_map.detach().cpu().numpy()
            for i, (label, score, path, amap, mask) in enumerate(zip(labels, scores, paths, maps, masks)):
                path = str(path)
                case_id = patient_id_from_path_local(path)
                slice_id = slice_id_from_path(path, batch_idx * test_loader.batch_size + i)
                rows.append({
                    "method": method,
                    "dataset": "DeepPSMA",
                    "modality": config.modality,
                    "case_id": case_id,
                    "slice_id": slice_id,
                    "true_label": int(label),
                    "anomaly_score": float(score),
                    "path": path,
                })
                pix = pixel_metric_row(method, "DeepPSMA", config.modality, case_id, slice_id, path, amap, mask)
                if pix is not None:
                    pixel_rows.append(pix)
    return rows, pixel_rows


def write_slice_scores(rows, path):
    fields = ["method", "dataset", "modality", "case_id", "slice_id", "true_label", "anomaly_score", "path"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_patient_scores(rows, path):
    grouped = defaultdict(lambda: {"scores": [], "labels": []})
    for row in rows:
        grouped[row["case_id"]]["scores"].append(float(row["anomaly_score"]))
        grouped[row["case_id"]]["labels"].append(int(row["true_label"]))

    patient_rows = []
    for case_id in sorted(grouped):
        item = grouped[case_id]
        first = next(row for row in rows if row["case_id"] == case_id)
        patient_rows.append({
            "method": first["method"],
            "dataset": first["dataset"],
            "modality": first["modality"],
            "case_id": case_id,
            "true_label": max(item["labels"]),
            "anomaly_score": max(item["scores"]),
            "n_slices": len(item["scores"]),
        })

    fields = ["method", "dataset", "modality", "case_id", "true_label", "anomaly_score", "n_slices"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(patient_rows)
    return patient_rows


def write_pixel_scores(pixel_rows, path):
    fields = ["method", "dataset", "modality", "case_id", "slice_id", "pixel_auroc", "pixel_aupr", "n_pixels", "n_positive_pixels", "path"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pixel_rows)


def write_outputs(rows, pixel_rows, output_dir, config, ckpt_path, method):
    os.makedirs(output_dir, exist_ok=True)
    slice_csv = os.path.join(output_dir, "slice_scores.csv")
    patient_csv = os.path.join(output_dir, "patient_scores.csv")
    pixel_csv = os.path.join(output_dir, "pixel_slice_metrics.csv")
    manifest_json = os.path.join(output_dir, "manifest.json")

    write_slice_scores(rows, slice_csv)
    patient_rows = write_patient_scores(rows, patient_csv)
    write_pixel_scores(pixel_rows, pixel_csv)

    manifest = {
        "method": method,
        "dataset": "DeepPSMA",
        "source_dataset": os.environ["DEEPPSMA_SOURCE_DATASET"],
        "target_dataset": "DeepPSMA",
        "original_data_root": os.environ["DEEPPSMA_TARGET_DATA_ROOT"],
        "modality": config.modality,
        "checkpoint": ckpt_path,
        "test_root": config.test_root,
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
    parser = argparse.ArgumentParser(description="Export GatingAno figure score CSV files.")
    parser.add_argument("--modality", type=str, default="pet", choices=["pet", "ct", "petct"])
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dataset", type=str.upper, default="PSMA")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Defaults to checkpoints/{dataset}_{modality}_epoch30.pth")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Defaults to results/{dataset}_{modality}/figure_scores")
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    parser.add_argument("--method", type=str, default="GatingAno")
    parser.add_argument("--dual_input_mode", type=str, default="pseudo_rgb", choices=["pseudo_rgb"])
    return parser.parse_args()


def main():
    args = parse_args()
    from torch.utils.data import DataLoader

    from dataloader import PETCTAnomalyDataset
    from train import Config

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = Config(
        modality=args.modality,
        gpu=args.gpu,
        dataset=args.dataset,
        data_root=args.data_root,
        dual_input_mode=args.dual_input_mode,
    )
    output_dir = args.output_dir or default_output_dir(config)
    if getattr(args, "skip_existing", False) and maybe_skip_existing(output_dir, getattr(args, "skip_require_pixel", False)):
        return

    ckpt_path = args.ckpt or config.final_ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    test_dataset = PETCTAnomalyDataset(
        root=config.test_root,
        mode="test",
        modality=config.modality,
        dual_input_mode=config.dual_input_mode,
        return_path=True,
        transform=build_transform(config.image_size),
        image_size=config.image_size,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)
    generator = load_generator(config, ckpt_path)
    rows, pixel_rows = collect_scores(config, generator, test_loader, args.method)
    manifest = write_outputs(rows, pixel_rows, output_dir, config, ckpt_path, args.method)

    print(f"Saved slice scores:   {manifest['slice_csv']}")
    print(f"Saved patient scores: {manifest['patient_csv']}")
    print(f"Saved pixel metrics:  {manifest['pixel_slice_metrics_csv']}")
    print(f"Saved manifest:       {os.path.join(os.path.dirname(manifest['slice_csv']), 'manifest.json')}")


if __name__ == "__main__":
    sys.argv[1:], _SOURCE_DATASET, _TARGET_DATA_ROOT, _GENERALIZATION_OUTPUT_DIR = rewrite_generalization_argv(
        sys.argv[1:],
        source_flag="--dataset",
        data_flag="--data_root",
        output_flag="--output_dir",
        source_case="upper",
    )
    main()
