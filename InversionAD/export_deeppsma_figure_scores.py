#!/usr/bin/env python3
"""DeepPSMA generalization commands (InvAD):
python export_deeppsma_figure_scores.py --source_dataset PSMA --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --input_mode dual --devices 1 --output_dir generalization_outputs/PSMA_weights/figure_scores --skip_existing --skip_require_pixel

python export_deeppsma_figure_scores.py --source_dataset FDG --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --input_mode dual --devices 1 --output_dir generalization_outputs/FDG_weights/figure_scores --skip_existing --skip_require_pixel
"""
# Run examples:
#   python export_deeppsma_figure_scores.py --dataset psma --input_mode dual --devices 6 --skip_existing --skip_require_pixel
#   python export_deeppsma_figure_scores.py --dataset fdg --input_mode dual --devices 6 --skip_existing --skip_require_pixel
# These examples skip only when slice, patient, manifest, and pixel_slice_metrics.csv all exist.
"""Export InversionAD PET-CT slice/patient scores for figure plotting."""


import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


DEFAULT_DEEPPSMA_ROOT = "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1"
GENERALIZATION_OUTPUT_TEMPLATE = "generalization_outputs/{source_dataset}_weights/figure_scores"
TARGET_DATASET_KEY = "target_dataset"


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run InversionAD PET-CT inference and export slice/patient score tables."
    )
    parser.add_argument("--fname", default="configs/exp_dit_petct/petct_dual.yml")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--source_dataset", type=str.upper, choices=["PSMA", "FDG"], default="PSMA")
    parser.add_argument("--data_root", default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--dataset", choices=["psma", "fdg"], default=None)
    parser.add_argument("--input_mode", choices=["ct", "pet", "dual"], default=None)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--eval_step", type=int, default=3)
    parser.add_argument("--noise_step", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema_model", action="store_true")
    parser.add_argument("--use_best_model", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
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
    patient_rows = [
        {
            "patient_id": pid,
            "label": max(grouped[pid]["labels"]),
            "score": max(grouped[pid]["scores"]),
            "n_slices": len(grouped[pid]["scores"]),
        }
        for pid in sorted(grouped)
    ]
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


def load_state_dict(path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    args = parse_args()
    args.dataset = args.source_dataset.lower()
    if args.use_ema_model and args.use_best_model:
        raise ValueError("Use at most one of --use_ema_model and --use_best_model.")

    import numpy as np
    import torch
    import yaml
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.backbones import get_backbone, get_backbone_feature_shape
    from src.config import resolve_petct_config
    from src.datasets import build_dataset
    from src.denoiser import get_denoiser
    from src.evaluate import init_denoiser, top1pct_score

    with open(args.fname, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.dataset is not None:
        config["data"]["petct_dataset"] = args.dataset
    if args.input_mode is not None:
        config["data"]["input_mode"] = args.input_mode
    config = resolve_petct_config(config)
    original_data_root = os.path.abspath(os.path.expanduser(args.data_root))
    output_dir = args.output_dir or GENERALIZATION_OUTPUT_TEMPLATE.format(
        source_dataset=args.source_dataset
    )
    config["data"]["data_root"] = prepare_deeppsma_view(
        original_data_root, os.path.join(output_dir, "_deeppsma_view")
    )

    selected_device = None
    if args.devices:
        selected_device = int(str(args.devices[0]).split(":")[-1])
    if torch.cuda.is_available():
        if selected_device is None:
            selected_device = torch.cuda.current_device()
        torch.cuda.set_device(selected_device)
        device = f"cuda:{selected_device}"
    else:
        device = "cpu"
    config.setdefault("meta", {})["device"] = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.save_dir is None:
        args.save_dir = config.get("logging", {}).get("save_dir")
    if args.save_dir is None:
        raise ValueError("Provide --save_dir or set logging.save_dir in the config.")
    if getattr(args, "skip_existing", False) and maybe_skip_existing(output_dir, getattr(args, "skip_require_pixel", False)):
        return

    dataset_config = dict(config["data"])
    dataset_config["train"] = False
    dataset_config["anom_only"] = True
    dataset_config["normal_only"] = False
    anom_dataset = build_dataset(**dataset_config)
    dataset_config["anom_only"] = False
    dataset_config["normal_only"] = True
    normal_dataset = build_dataset(**dataset_config)

    batch_size = args.batch_size or config["data"].get("batch_size", 8)
    num_workers = args.num_workers if args.num_workers is not None else config["data"].get("num_workers", 4)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "drop_last": False,
    }
    normal_loader = DataLoader(normal_dataset, **loader_kwargs)
    anom_loader = DataLoader(anom_dataset, **loader_kwargs)

    in_shape = get_backbone_feature_shape(model_type=config["backbone"]["model_type"])
    denoiser = get_denoiser(**config["diffusion"], input_shape=in_shape).to(device).eval()
    feature_extractor = get_backbone(**config["backbone"]).to(device).eval()

    if args.use_ema_model:
        checkpoint_path = os.path.join(args.save_dir, "model_ema_latest.pth")
    elif args.use_best_model:
        checkpoint_path = os.path.join(args.save_dir, "model_best.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(args.save_dir, "model_latest.pth")
    else:
        checkpoint_path = os.path.join(args.save_dir, "model_latest.pth")
    state = load_state_dict(checkpoint_path, device)
    if state and "module." in next(iter(state.keys())):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    denoiser.load_state_dict(state, strict=True)

    eval_denoiser = init_denoiser(args.eval_step, device, config, in_shape, inherit_model=denoiser)
    all_rows = []
    pixel_rows = []
    slice_id = 0
    with torch.no_grad():
        for loader, label_value, desc in [(normal_loader, 0, "normal"), (anom_loader, 1, "anomaly")]:
            for batch in tqdm(loader, desc=f"Export {desc}", leave=False):
                images = batch["samples"].to(device)
                cls_labels = batch["clslabels"].to(device)
                features, _ = feature_extractor(images)
                start_t = torch.zeros(images.shape[0], device=device, dtype=torch.long)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                    latents_last = eval_denoiser.ddim_reverse_sample(
                        features,
                        start_t,
                        cls_labels,
                        eta=0.0,
                    )
                latents_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
                maps = F.interpolate(
                    latents_l2.unsqueeze(0),
                    size=(images.shape[2], images.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).cpu().numpy()
                masks = batch["masks"].detach().cpu().numpy() if "masks" in batch else np.zeros_like(maps)

                for i, path in enumerate(batch["filenames"]):
                    score = top1pct_score(maps[i])
                    pid = patient_id_from_path(path)
                    all_rows.append(
                        {
                            "slice_id": slice_id,
                            "patient_id": pid,
                            "label": int(label_value),
                            "score": float(score),
                            "image_path": path,
                        }
                    )
                    pix = pixel_metric_row(
                        "InversionAD",
                        "DeepPSMA",
                        config["data"].get("input_mode"),
                        slice_id,
                        pid,
                        label_value,
                        path,
                        maps[i],
                        masks[i],
                    )
                    if pix is not None:
                        pixel_rows.append(pix)
                    slice_id += 1

    outputs = write_exports(
        output_dir,
        all_rows,
        pixel_rows,
        {
            "repo": "InversionAD1",
            "config": args.fname,
            "dataset": "DeepPSMA",
            "source_dataset": args.source_dataset,
            "target_dataset": "DeepPSMA",
            "original_data_root": original_data_root,
            "input_mode": config["data"].get("input_mode"),
            "data_path": config["data"].get("data_root"),
            "save_dir": args.save_dir,
            "checkpoint": checkpoint_path,
            "score_rule": "mean of top-1% pixels from the inversion anomaly map, matching src.evaluate",
            "patient_rule": "max slice score per patient",
        },
    )
    print("Exported:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
