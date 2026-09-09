#!/usr/bin/env python3
"""DeepPSMA generalization commands (Recontrast):
python export_deeppsma_figure_scores.py --source_dataset PSMA --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --modality petct --cuda 4 --epochs 30 --output_dir generalization_outputs/PSMA_weights/figure_scores --skip_existing --skip_require_pixel

python export_deeppsma_figure_scores.py --source_dataset FDG --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --modality petct --cuda 5 --epochs 30 --output_dir generalization_outputs/FDG_weights/figure_scores --skip_existing --skip_require_pixel
"""
# Run examples:
#   python export_deeppsma_figure_scores.py --dataset psma --modality petct --cuda 6 --epochs 30 --skip_existing --skip_require_pixel
#   python export_deeppsma_figure_scores.py --dataset fdg --modality petct --cuda 6 --epochs 30 --skip_existing --skip_require_pixel
# These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""Run ReContrast PET/CT checkpoint inference and export figure score CSV files."""


import argparse
import copy
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


def resolve_run_name(args):
    return args.save_name or f"recontrast_{args.dataset.lower()}_{args.modality.lower()}"


def model_filename(args):
    return f"epoch{args.epochs}_{args.dataset.lower()}_{args.modality.lower()}.pth"


def build_model(args, device):
    import torch
    from models.de_resnet import de_wide_resnet50_2
    from models.recontrast import ReContrast
    from models.resnet import wide_resnet50_2
    from utils import replace_layers

    encoder, bn = wide_resnet50_2(pretrained=True, in_channels=args.replicate_channels)
    decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)
    replace_layers(decoder, torch.nn.ReLU, torch.nn.GELU())
    encoder, bn, decoder = encoder.to(device), bn.to(device), decoder.to(device)
    encoder_freeze = copy.deepcopy(encoder)
    return ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder)


def top_score(anomaly_map, max_ratio):
    import torch

    flat = anomaly_map.flatten(1)
    if max_ratio == 0:
        return torch.max(flat, dim=1)[0]
    k = max(1, int(flat.shape[1] * max_ratio))
    return torch.topk(flat, k=k, dim=1)[0].mean(dim=1)


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
    import torch
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import DataLoader

    from dataset import get_detection_dataset
    from ReContrast_petct import cal_anomaly_maps, get_gaussian_kernel

    data_dir = args.data_dir or DATASET_ROOTS[args.dataset]
    args.output_dir = args.output_dir or os.path.join(args.save_dir, resolve_run_name(args))
    checkpoint = args.checkpoint or os.path.join(args.output_dir, model_filename(args))
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cuda_config = args.cuda.lower()
    if cuda_config in ("cpu", "-1", "none") or not torch.cuda.is_available():
        device = "cpu"
    elif cuda_config.startswith("cuda:"):
        device = cuda_config
    else:
        device = "cuda:" + args.cuda

    dataset = get_detection_dataset(
        data_dir=data_dir,
        file_name="test",
        modality=args.modality.lower(),
        replicate_channels=args.replicate_channels,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(args, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    rows = []
    pixel_rows = []
    with torch.no_grad():
        sample_idx = 0
        for img, gt, label, paths in loader:
            img = img.to(device)
            en2, de2 = model(img)
            anomaly_map, _ = cal_anomaly_maps(en2, de2, img.shape[-1])
            if args.resize_mask is not None:
                anomaly_map = F.interpolate(
                    anomaly_map, size=(args.resize_mask, args.resize_mask), mode="bilinear", align_corners=False
                )
            anomaly_map = gaussian_kernel(anomaly_map)
            anomaly_map = torch.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
            scores = top_score(anomaly_map, args.max_ratio).cpu().numpy()
            maps = anomaly_map.squeeze(1).detach().cpu().numpy().astype(np.float32)
            masks = gt.detach().cpu().numpy() if hasattr(gt, "detach") else np.asarray(gt)
            labels = label.cpu().numpy()
            for score, label_value, path, amap, mask_i in zip(scores, labels, paths, maps, masks):
                path = str(path)
                case_id = case_id_from_path(path)
                slice_id = slice_id_from_path(path, sample_idx)
                rows.append({
                    "method": args.method,
                    "dataset": "DeepPSMA",
                    "modality": args.modality,
                    "case_id": case_id,
                    "slice_id": slice_id,
                    "true_label": int(label_value),
                    "anomaly_score": float(score),
                    "path": path,
                })
                pix = pixel_metric_row(args.method, "DeepPSMA", args.modality, case_id, slice_id, path, amap, mask_i)
                if pix is not None:
                    pixel_rows.append(pix)
                sample_idx += 1
    args._resolved_checkpoint = checkpoint
    args._resolved_data_dir = data_dir
    return rows, pixel_rows


def write_outputs(rows, pixel_rows, args):
    out_dir = args.figure_output_dir or os.path.join(args.output_dir, "figure_scores")
    os.makedirs(out_dir, exist_ok=True)
    slice_csv = os.path.join(out_dir, "slice_scores.csv")
    patient_csv = os.path.join(out_dir, "patient_scores.csv")
    pixel_csv = os.path.join(out_dir, "pixel_slice_metrics.csv")
    manifest_json = os.path.join(out_dir, "manifest.json")

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
        "dataset": "DeepPSMA",
        "source_dataset": os.environ["DEEPPSMA_SOURCE_DATASET"],
        "target_dataset": "DeepPSMA",
        "original_data_root": os.environ["DEEPPSMA_TARGET_DATA_ROOT"],
        "data_dir": args._resolved_data_dir,
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
    return manifest, out_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Export ReContrast figure score CSV files.")
    parser.add_argument("--dataset", default="psma", choices=["fdg", "psma"])
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--save_dir", default="./results")
    parser.add_argument("--save_name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--modality", default="petct", choices=["pet", "ct", "petct"])
    parser.add_argument("--replicate_channels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_ratio", type=float, default=0.01)
    parser.add_argument("--resize_mask", type=int, default=256)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--method", default="ReContrast")
    parser.add_argument("--figure_output_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir = args.output_dir or os.path.join(args.save_dir, resolve_run_name(args))
    skip_dir = args.figure_output_dir or os.path.join(args.output_dir, "figure_scores")
    if getattr(args, "skip_existing", False) and maybe_skip_existing(skip_dir, getattr(args, "skip_require_pixel", False)):
        return
    if args.modality == "petct" and args.replicate_channels != 3:
        raise ValueError("petct modality uses [CT,PET,PET], so --replicate_channels must be 3")
    rows, pixel_rows = collect_scores(args)
    manifest, out_dir = write_outputs(rows, pixel_rows, args)
    print(f"Saved slice scores:   {manifest['slice_csv']}")
    print(f"Saved patient scores: {manifest['patient_csv']}")
    print(f"Saved pixel metrics:  {manifest['pixel_slice_metrics_csv']}")
    print(f"Saved manifest:       {os.path.join(out_dir, 'manifest.json')}")


if __name__ == "__main__":
    sys.argv[1:], _SOURCE_DATASET, _TARGET_DATA_ROOT, _GENERALIZATION_OUTPUT_DIR = rewrite_generalization_argv(
        sys.argv[1:],
        source_flag="--dataset",
        data_flag="--data_dir",
        output_flag="--figure_output_dir",
        source_case="lower",
    )
    main()
