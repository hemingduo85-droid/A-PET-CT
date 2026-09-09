#!/usr/bin/env python3
"""DeepPSMA generalization commands (STFPM):
python export_deeppsma_figure_scores.py --source_dataset PSMA --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --modalities ct pet --input-mode pseudo_rgb --cuda 7 --epochs 30 --output_dir generalization_outputs/PSMA_weights/figure_scores --skip_existing --skip_require_pixel

python export_deeppsma_figure_scores.py --source_dataset FDG --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --modalities ct pet --input-mode pseudo_rgb --cuda 6 --epochs 30 --output_dir generalization_outputs/FDG_weights/figure_scores --skip_existing --skip_require_pixel
"""
# Run examples:
#   python export_deeppsma_figure_scores.py --dataset psma --modalities ct pet --input-mode pseudo_rgb --cuda 7 --epochs 30 --skip_existing --skip_require_pixel
#   python export_deeppsma_figure_scores.py --dataset fdg --modalities ct pet --input-mode pseudo_rgb --cuda 7 --epochs 30 --skip_existing --skip_require_pixel
# These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""Run STFPM PET/CT checkpoint inference and export figure score CSV files."""


import argparse
import csv
import json
import os
import sys
import re
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


def case_id_from_path(path):
    p = Path(str(path))
    if p.parent.name.lower() in {"pet", "ct", "label", "mask", "masks"}:
        return p.parent.parent.name
    return p.parent.name


def slice_id_from_path(path, idx):
    stem = Path(str(path)).stem
    return stem or f"{idx:06d}"


def parse_modalities(modality, modalities):
    raw = modalities or ([modality] if modality else [])
    parsed = []
    for token in raw:
        for part in re.split(r"[+,]", token):
            part = part.strip()
            if part:
                parsed.append(part)
    out = []
    for item in parsed:
        if item not in out:
            out.append(item)
    if not out:
        raise ValueError("Pass --modalities or --modality")
    return out


def mask_path_from_image_path(path):
    p = Path(str(path))
    candidates = []
    if p.parent.name.lower() in {"pet", "ct"}:
        case_dir = p.parent.parent
        for mask_dir in ("label", "labels", "mask", "masks", "gt", "ground_truth"):
            candidates.append(case_dir / mask_dir / p.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def pixel_metric_row(method, dataset, modality, case_id, slice_id, path, anomaly_map):
    import cv2
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    mask_path = mask_path_from_image_path(path)
    if mask_path is None:
        return None
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    amap = np.asarray(anomaly_map, dtype=np.float32)
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
    import cv2
    import numpy as np
    import torch
    from torchvision import transforms

    import main as stfpm

    args.dataset_path, args.dataset_key = stfpm.resolve_dataset_path(args)
    args.modalities = parse_modalities(args.modality, args.modalities)
    modality_set = set(m.lower() for m in args.modalities)
    if args.input_mode == "auto":
        args.input_mode = "pseudo_rgb" if modality_set == {"ct", "pet"} and len(args.modalities) == 2 else "concat"
    modalities_str = "+".join(args.modalities)
    args.checkpoint = args.checkpoint or stfpm.resolve_checkpoint(args)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    device = stfpm.select_device(args)

    transform_single = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_pos_loader, test_neg_loader, test_pos_dataset, test_neg_dataset = stfpm.build_test_loaders(args, transform_single)
    in_channels = 3 if (len(args.modalities) == 1 or args.input_mode == "pseudo_rgb") else 3 * len(args.modalities)
    teacher = stfpm.ResNet18_MS3(
        pretrained=not args.no_pretrained_teacher,
        in_channels=in_channels,
        weights_path=args.teacher_weights,
    ).to(device)
    student = stfpm.ResNet18_MS3(pretrained=False, in_channels=in_channels).to(device)
    saved_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    student.load_state_dict(saved_dict["state_dict"])

    pos = stfpm.test(teacher, student, test_pos_loader, device, args, name="abnormal")
    neg = stfpm.test(teacher, student, test_neg_loader, device, args, name="normal")
    maps = []
    for item in pos:
        maps.append(cv2.resize(item, (256, 256)))
    for item in neg:
        maps.append(cv2.resize(item, (256, 256)))
    maps = np.stack(maps)
    paths = stfpm.get_dataset_paths(test_pos_dataset) + stfpm.get_dataset_paths(test_neg_dataset)
    labels = np.concatenate((np.ones(pos.shape[0], dtype=int), np.zeros(neg.shape[0], dtype=int)), 0)
    scores = maps.max(-1).max(-1)

    rows = []
    pixel_rows = []
    for idx, (label, score, path, amap) in enumerate(zip(labels, scores, paths, maps)):
        case_id = case_id_from_path(path)
        slice_id = slice_id_from_path(path, idx)
        rows.append({
            "method": args.method,
            "dataset": "DeepPSMA",
            "modality": modalities_str,
            "case_id": case_id,
            "slice_id": slice_id,
            "true_label": int(label),
            "anomaly_score": float(score),
            "path": path,
        })
        pix = pixel_metric_row(args.method, "DeepPSMA", modalities_str, case_id, slice_id, path, amap)
        if pix is not None:
            pixel_rows.append(pix)
    args._resolved_modalities = modalities_str
    return rows, pixel_rows


def write_outputs(rows, pixel_rows, args):
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "figure_scores",
            str(args.dataset_key).lower(),
            f"{args._resolved_modalities}_{args.input_mode}",
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
            "modality": first["modality"],
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
        "dataset_path": args.dataset_path,
        "modality": args._resolved_modalities,
        "input_mode": args.input_mode,
        "checkpoint": args.checkpoint,
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
    parser = argparse.ArgumentParser(description="Export STFPM figure score CSV files.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--test-batch-size", dest="test_batch_size", type=int, default=32)
    parser.add_argument("--dataset", default="fdg", choices=["fdg", "psma"])
    parser.add_argument("--dataset-path", dest="dataset_path", default=None)
    parser.add_argument("--model-save-path", dest="model_save_path", default="snapshots")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--input-mode", dest="input_mode", choices=["auto", "pseudo_rgb", "concat"], default="auto")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--test-log-interval", dest="test_log_interval", type=int, default=50)
    parser.add_argument("--teacher-weights", dest="teacher_weights", default=None)
    parser.add_argument("--no-pretrained-teacher", dest="no_pretrained_teacher", action="store_true")
    parser.add_argument("--modality", default=None)
    parser.add_argument("--modalities", nargs="+", default=None)
    parser.add_argument("--strict-modal-match", dest="strict_modal_match", action="store_true")
    parser.add_argument("--method", default="STFPM")
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    import main as stfpm
    args.dataset_path, args.dataset_key = stfpm.resolve_dataset_path(args)
    args.modalities = parse_modalities(args.modality, args.modalities)
    modality_set = set(m.lower() for m in args.modalities)
    if args.input_mode == "auto":
        args.input_mode = "pseudo_rgb" if modality_set == {"ct", "pet"} and len(args.modalities) == 2 else "concat"
    args._resolved_modalities = "+".join(args.modalities)
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "figure_scores",
            str(args.dataset_key).lower(),
            f"{args._resolved_modalities}_{args.input_mode}",
        )
    if getattr(args, "skip_existing", False) and maybe_skip_existing(getattr(args, "output_dir", None), getattr(args, "skip_require_pixel", False)):
        return
    rows, pixel_rows = collect_scores(args)
    manifest = write_outputs(rows, pixel_rows, args)
    print(f"Saved slice scores:   {manifest['slice_csv']}")
    print(f"Saved patient scores: {manifest['patient_csv']}")
    print(f"Saved pixel metrics:  {manifest['pixel_slice_metrics_csv']}")
    print(f"Saved manifest:       {os.path.join(args.output_dir, 'manifest.json')}")


if __name__ == "__main__":
    sys.argv[1:], _SOURCE_DATASET, _TARGET_DATA_ROOT, _GENERALIZATION_OUTPUT_DIR = rewrite_generalization_argv(
        sys.argv[1:],
        source_flag="--dataset",
        data_flag="--dataset-path",
        output_flag="--output-dir",
        source_case="lower",
    )
    main()
