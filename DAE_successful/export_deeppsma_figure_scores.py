#!/usr/bin/env python3
"""DeepPSMA generalization commands (DAE):
python export_deeppsma_figure_scores.py --source_dataset PSMA --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --modalities ct,pet --device cuda:4 --output_dir generalization_outputs/PSMA_weights/figure_scores --skip_existing --skip_require_pixel

python export_deeppsma_figure_scores.py --source_dataset FDG --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --modalities ct,pet --device cuda:4 --output_dir generalization_outputs/FDG_weights/figure_scores --skip_existing --skip_require_pixel
"""
# Run examples:
#   python export_deeppsma_figure_scores.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer PSMA --modalities ct,pet --device cuda:4 --skip_existing --skip_require_pixel
#   python export_deeppsma_figure_scores.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer FDG --modalities ct,pet --device cuda:4 --skip_existing --skip_require_pixel
# These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""Export DAE PET/CT slice/patient scores for figure plotting."""


import argparse
import csv
import json
import os
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

PROJECT_DIR = Path(__file__).resolve().parent


def parse_modalities(value):
    return [m.strip().lower() for m in value.split(",") if m.strip()]


def resolve_project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def find_checkpoint(args, mod, modalities):
    if args.checkpoint:
        return Path(args.checkpoint).expanduser()
    return resolve_project_path(args.checkpoint_dir) / args.tracer.upper() / "_".join(modalities) / f"best_{mod}_model.pth"


def top_percent_mean(values, percent):
    import numpy as np

    arr = np.asarray(values)
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim >= 3 else arr.reshape(1, -1)
    k = max(1, int(np.ceil(flat.shape[1] * float(percent) / 100.0)))
    part = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


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
    with open(slice_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "dataset", "modality", "case_id", "slice_id", "true_label", "anomaly_score", "path"])
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(lambda: {"scores": [], "labels": []})
    for row in rows:
        grouped[row["case_id"]]["scores"].append(float(row["anomaly_score"]))
        grouped[row["case_id"]]["labels"].append(int(row["true_label"]))
    patient_rows = [{
        "method": manifest["method"],
        "dataset": manifest["dataset"],
        "modality": manifest["modality"],
        "case_id": pid,
        "true_label": max(grouped[pid]["labels"]),
        "anomaly_score": max(grouped[pid]["scores"]),
        "n_slices": len(grouped[pid]["scores"]),
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
    parser = argparse.ArgumentParser(description="Export DAE figure score CSV files.")
    parser.add_argument("--data_root", default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--source_dataset", default="PSMA", choices=["PSMA", "FDG", "psma", "fdg"])
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct,pet")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--noise_res", type=int, default=16)
    parser.add_argument("--topk_percent", type=float, default=1.0)
    parser.add_argument("--debug_ratio", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--method", default="DAE")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    return parser


def main():
    args = build_parser().parse_args()
    args.tracer = args.source_dataset.upper()
    modalities = parse_modalities(args.modalities)
    if args.output_dir is None:
        args.output_dir = GENERALIZATION_OUTPUT_TEMPLATE.format(source_dataset=args.tracer)
    if getattr(args, "skip_existing", False) and maybe_skip_existing(getattr(args, "output_dir", None), getattr(args, "skip_require_pixel", False)):
        return

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from dataloader import PETCTSliceDataset
    from denoising import denoising

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tracer_root = prepare_deeppsma_view(args.data_root, os.path.join(args.output_dir, "_deeppsma_view"))

    models = {}
    for mod in modalities:
        ckpt_path = find_checkpoint(args, mod, modalities)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for {mod}: {ckpt_path}")
        wrapper = denoising(
            identifier=f"dae_{mod}",
            n_input=1,
            noise_std=args.noise_std,
            noise_res=args.noise_res,
            device=device,
        )
        wrapper.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        wrapper.model.to(device).eval()
        models[mod] = wrapper.model
        print(f"Loaded {mod}: {ckpt_path}")

    dataset = PETCTSliceDataset(tracer_root, modalities=modalities, mode="test", target_size=(args.image_size, args.image_size), debug_ratio=args.debug_ratio)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    rows = []
    pixel_rows = []
    for batch in tqdm(loader, desc="Exporting DAE scores"):
        residuals = []
        with torch.no_grad():
            for mod in modalities:
                x = batch["modalities"][mod].to(device)
                recon = models[mod](x)
                if isinstance(recon, (tuple, list)):
                    recon = recon[0]
                residuals.append(torch.abs(x - recon))
            amap = torch.mean(torch.stack(residuals), dim=0).squeeze().cpu().numpy().astype(np.float32)
        score = float(top_percent_mean(amap, args.topk_percent)[0])
        sample_id = str(batch["id"][0])
        case_id, slice_id = split_id(sample_id)
        mask = batch["mask"].detach().cpu().numpy()
        rows.append({
            "method": args.method,
            "dataset": "DeepPSMA",
            "modality": "_".join(modalities),
            "case_id": case_id,
            "slice_id": slice_id,
            "true_label": int(batch["label"].view(-1)[0].item()),
            "anomaly_score": score,
            "path": sample_id,
        })
        pix = pixel_metric_row(args.method, "DeepPSMA", "_".join(modalities), case_id, slice_id, sample_id, amap, mask)
        if pix is not None:
            pixel_rows.append(pix)

    output_dir = args.output_dir
    manifest = write_outputs(rows, pixel_rows, output_dir, {
        "method": args.method,
        "dataset": "DeepPSMA",
        "source_dataset": args.tracer,
        "target_dataset": "DeepPSMA",
        "modality": "_".join(modalities),
        "data_root": os.path.abspath(os.path.expanduser(args.data_root)),
        "data_view": str(tracer_root),
        "patient_level_valid": False,
        "score_rule": f"top {args.topk_percent:g}% mean from DAE residual map",
    })
    print(f"Saved slice scores:   {manifest['slice_csv']}")
    print(f"Saved patient scores: {manifest['patient_csv']}")
    print(f"Saved pixel metrics:  {manifest['pixel_slice_metrics_csv']}")
    print(f"Saved manifest:       {os.path.join(output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
