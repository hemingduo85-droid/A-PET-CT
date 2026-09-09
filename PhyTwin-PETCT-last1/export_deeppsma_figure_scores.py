#!/usr/bin/env python3
"""DeepPSMA generalization commands (ours):
python export_deeppsma_figure_scores.py --source_dataset PSMA --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --load_ckpt --experiment_name PhyTwin_PETCT_PSMA --score_mode lesion_z --adaptive_physio --adaptive_alpha 0.50 --adaptive_lesion_protect 0.70 --batch_size 1 --image_size 256 --gpu 4 --output_dir generalization_outputs/PSMA_weights/figure_scores --skip_existing --skip_require_pixel

python export_deeppsma_figure_scores.py --source_dataset FDG --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --load_ckpt --experiment_name PhyTwin_PETCT_FDG --score_mode lesion_z --adaptive_physio --adaptive_alpha 0.50 --adaptive_lesion_protect 0.70 --batch_size 1 --image_size 256 --gpu 6 --output_dir generalization_outputs/FDG_weights/figure_scores --skip_existing --skip_require_pixel
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


def build_deeppsma_test_loader(data_root, image_size, batch_size, num_workers):
    """Build only the flat DeepPSMA test loader; source calibration stays in the checkpoint."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    from phytwin_petct.data import IMAGENET_MEAN, IMAGENET_STD

    class DeepPSMASliceDataset(Dataset):
        def __init__(self):
            self.samples = []
            for group, label, needs_mask in (("normal", 0, False), ("abnormal", 1, True)):
                base = os.path.join(data_root, "test", group)
                pet_dir, ct_dir = os.path.join(base, "pet"), os.path.join(base, "ct")
                mask_dir = os.path.join(base, "label")
                if not os.path.isdir(pet_dir) or not os.path.isdir(ct_dir):
                    raise FileNotFoundError(f"Missing DeepPSMA {group} PET/CT directories under {base}")
                for name in sorted(os.listdir(pet_dir)):
                    if Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                        continue
                    pet_path, ct_path = os.path.join(pet_dir, name), os.path.join(ct_dir, name)
                    mask_path = os.path.join(mask_dir, name) if needs_mask else None
                    if not os.path.isfile(ct_path) or (needs_mask and not os.path.isfile(mask_path)):
                        raise FileNotFoundError(f"Unpaired DeepPSMA slice: {name}")
                    self.samples.append((pet_path, ct_path, mask_path, label))
            if not self.samples:
                raise RuntimeError(f"No DeepPSMA test slices found under {data_root}")
            self.image_transform = T.Compose([
                T.Resize((image_size, image_size), InterpolationMode.LANCZOS),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            self.mask_transform = T.Compose([
                T.Resize((image_size, image_size), InterpolationMode.NEAREST),
                T.ToTensor(),
            ])

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            pet_path, ct_path, mask_path, label = self.samples[index]
            pet_gray = Image.open(pet_path).convert("L")
            ct_gray = Image.open(ct_path).convert("L")
            mask = Image.new("L", pet_gray.size, 0) if mask_path is None else Image.open(mask_path).convert("L")
            to_rgb = lambda image: Image.merge("RGB", [image, image, image])
            return {
                "pet": self.image_transform(to_rgb(pet_gray)),
                "ct": self.image_transform(to_rgb(ct_gray)),
                "mask": (self.mask_transform(mask) > 0.5).float(),
                "path": pet_path,
                "label": int(label),
            }

    return DataLoader(
        DeepPSMASliceDataset(),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )



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


def check_lesion_calibration(calibration, current_refinement_config):
    """Validate source-domain calibration using the evaluator's legacy fallback contract."""
    has_lesion_calibration = any(
        key in calibration
        for key in ("normal_lesion_ms_mean", "normal_lesion_mean", "normal_image_mean")
    )
    if not has_lesion_calibration:
        raise RuntimeError(
            "Checkpoint lacks lesion/image normal calibration statistics. "
            "Run train.py once on the source dataset with train/normal, then rerun this exporter."
        )
    return calibration.get("image_refinement_config") != current_refinement_config


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


def write_outputs(rows, pixel_rows, output_dir, method, data_root, checkpoint, source_dataset):
    os.makedirs(output_dir, exist_ok=True)
    slice_csv = os.path.join(output_dir, "slice_scores.csv")
    patient_csv = os.path.join(output_dir, "patient_scores.csv")
    pixel_csv = os.path.join(output_dir, "pixel_slice_metrics.csv")
    manifest_json = os.path.join(output_dir, "manifest.json")

    slice_fields = ["method", "dataset", "modality", "case_id", "slice_id", "true_label", "anomaly_score", "path"]
    with open(slice_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=slice_fields)
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
            "method": first["method"],
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
        "method": method,
        "dataset": "DeepPSMA",
        "source_dataset": source_dataset,
        "target_dataset": "DeepPSMA",
        "data_root": data_root,
        "checkpoint": checkpoint,
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


def collect_phytwin_rows(train_mod, model, memory, prior, dynamic_suppressor, calibration, loader, device, args, method):
    import numpy as np
    import torch
    from tqdm import tqdm

    model.eval()
    if memory is not None:
        memory = memory.to(device)

    adaptive_suppressor = train_mod.build_adaptive_physio(args)
    prior_arr = None if prior is None else prior.prior
    rows = []
    pixel_rows = []
    slice_counter = 0

    with torch.no_grad():
        for _batch_idx, batch in tqdm(train_mod.limited(loader, args.max_test_batches), desc="Exporting PhyTwin scores"):
            pet = batch["pet"].to(device)
            ct = batch["ct"].to(device)
            pred, logvar = model(ct)
            residual = train_mod.residual_map(pet, pred, logvar if args.uncertainty else None)
            pet_grays = train_mod.pet_tensor_to_gray(batch["pet"]).numpy().astype(np.float32)
            ct_grays = train_mod.pet_tensor_to_gray(batch["ct"]).numpy().astype(np.float32)
            labels = batch["label"]
            if hasattr(labels, "detach"):
                labels = labels.detach().cpu().reshape(-1).tolist()
            else:
                labels = list(labels)
            paths = list(batch["path"]) if isinstance(batch["path"], (list, tuple)) else [str(batch["path"])]
            masks = batch["mask"]
            if hasattr(masks, "detach"):
                masks = masks.detach().cpu().numpy()
            else:
                masks = np.asarray(masks)
            batch_size = int(pet.shape[0])
            if not (len(labels) == len(paths) == batch_size):
                raise RuntimeError(
                    f"DeepPSMA batch metadata mismatch: images={batch_size}, "
                    f"labels={len(labels)}, paths={len(paths)}"
                )

            for sample_index in range(batch_size):
                residual_one = residual[sample_index : sample_index + 1]
                residual_np = residual_one[0].detach().cpu().numpy().astype(np.float32)
                residual_z = train_mod.positive_zscore_map(
                    residual_np, calibration["residual_mean"], calibration["residual_std"]
                )
                memory_z = train_mod.memory_z_map(memory, residual_one, calibration, args)
                final = train_mod.fuse_maps(
                    residual_z,
                    memory_z,
                    residual_weight=args.residual_weight,
                    memory_weight=args.memory_weight,
                    physio_prior=prior,
                    sigma=args.gaussian_sigma,
                )
                pet_gray = pet_grays[sample_index]
                ct_gray = ct_grays[sample_index]
                if dynamic_suppressor is not None:
                    final, _dynamic_mask = dynamic_suppressor.suppress(
                        final,
                        pet_gray,
                        static_prior=None if prior is None else prior.prior,
                    )
                final, _adaptive_mask = train_mod.apply_adaptive_physio(
                    final, pet_gray, ct_gray, prior, adaptive_suppressor
                )

                if args.score_mode == "component":
                    image_score = train_mod.lesion_component_score(
                        final,
                        physio_prior=prior_arr,
                        threshold_quantile=args.component_quantile,
                        min_area=args.component_min_area,
                        prior_penalty=args.prior_penalty,
                    )
                elif args.score_mode == "normal_z":
                    image_score = train_mod.normal_calibrated_score(
                        final,
                        calibration["normal_image_mean"],
                        calibration["normal_image_std"],
                        physio_prior=prior_arr,
                        prior_penalty=args.prior_penalty,
                    )
                elif args.score_mode == "lesion_z":
                    raw_score = train_mod.compute_lesion_score(
                        final,
                        pet_gray,
                        ct_gray,
                        prior,
                        args,
                        normal_hotspot_memory=calibration.get("normal_hotspot_memory") if args.use_hotspot_memory else None,
                    )
                    normal_mean = calibration.get(
                        "normal_lesion_ms_mean",
                        calibration.get("normal_lesion_mean", calibration["normal_image_mean"]),
                    )
                    normal_std = calibration.get(
                        "normal_lesion_ms_std",
                        calibration.get("normal_lesion_std", calibration["normal_image_std"]),
                    )
                    image_score = (raw_score - normal_mean) / (normal_std + 1e-8)
                else:
                    image_score = train_mod.topk_score(final, fraction=0.01)

                label = int(labels[sample_index])
                path = str(paths[sample_index])
                dataset_name = "DeepPSMA"
                case_id = case_id_from_path(path)
                slice_id = slice_id_from_path(path, slice_counter)
                rows.append({
                    "method": method,
                    "dataset": dataset_name,
                    "modality": "petct",
                    "case_id": case_id,
                    "slice_id": slice_id,
                    "true_label": label,
                    "anomaly_score": float(image_score),
                    "path": path,
                })
                pix = pixel_metric_row(
                    method,
                    dataset_name,
                    "petct",
                    case_id,
                    slice_id,
                    path,
                    final,
                    masks[sample_index],
                )
                if pix is not None:
                    pixel_rows.append(pix)
                slice_counter += 1

    return rows, pixel_rows


def parse_export_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Export PhyTwin figure score CSVs. All unknown arguments are forwarded "
            "to train.py's parser, so use the same checkpoint/data/scoring flags."
        )
    )
    parser.add_argument("--output_dir", default=None,
                        help="Defaults to <save_dir>/<method_name>/figure_scores")
    parser.add_argument("--source_dataset", type=str.upper, choices=["PSMA", "FDG"], default="PSMA")
    parser.add_argument("--data_root", default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--skip_existing", action="store_true", help="Skip inference if output CSV files already exist.")
    parser.add_argument("--skip_require_pixel", action="store_true", help="When skipping, also require pixel_slice_metrics.csv to exist.")
    parser.add_argument("--method", default=None,
                        help="Defaults to train.py method_name/experiment_name")
    args, train_argv = parser.parse_known_args(argv)
    train_argv.extend(["--data_root", args.data_root])
    return args, train_argv


def main(argv=None):
    export_args, train_argv = parse_export_args(sys.argv[1:] if argv is None else argv)

    import torch
    import train as train_mod

    old_argv = sys.argv[:]
    try:
        sys.argv = ["train.py"] + train_argv
        args = train_mod.parse_args()
    finally:
        sys.argv = old_argv

    if args.experiment_name is None:
        args.method_name = f"PhyTwin_PETCT_{export_args.source_dataset}"
    output_dir = export_args.output_dir or GENERALIZATION_OUTPUT_TEMPLATE.format(
        source_dataset=export_args.source_dataset
    )
    if getattr(export_args, "skip_existing", False) and maybe_skip_existing(output_dir, getattr(export_args, "skip_require_pixel", False)):
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = build_deeppsma_test_loader(
        args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = train_mod.NormalTwinUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        uncertainty=args.uncertainty,
    ).to(device)

    ckpt_full = os.path.join(args.ckpt_dir, args.method_name, "BEST_PHYTWIN.pth")
    if not os.path.exists(ckpt_full):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_full}")

    model, memory, prior, calibration = train_mod.load_full_checkpoint(ckpt_full, model)
    model = model.to(device)
    if args.no_physio:
        prior = None
    if args.score_mode == "lesion_z":
        refinement_mismatch = check_lesion_calibration(
            calibration, train_mod.image_refinement_config(args)
        )
        if refinement_mismatch:
            print(
                "Warning: checkpoint lesion_z refinement config differs from current args; "
                "reusing the checkpoint's available source-domain calibration statistics."
            )

    dynamic_suppressor = None
    if args.use_dynamic_physio:
        dynamic_suppressor = train_mod.DynamicPhysioSuppressor(
            alpha=args.dynamic_alpha,
            pet_quantile=args.dynamic_pet_quantile,
            min_area=args.dynamic_min_area,
        )

    method = export_args.method or args.method_name
    rows, pixel_rows = collect_phytwin_rows(
        train_mod, model, memory, prior, dynamic_suppressor, calibration, test_loader, device, args, method
    )
    manifest = write_outputs(
        rows, pixel_rows, output_dir, method, args.data_root, ckpt_full, export_args.source_dataset
    )
    print(f"Saved slice scores:   {manifest['slice_csv']}")
    print(f"Saved patient scores: {manifest['patient_csv']}")
    print(f"Saved pixel metrics:  {manifest['pixel_slice_metrics_csv']}")
    print(f"Saved manifest:       {os.path.join(output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
