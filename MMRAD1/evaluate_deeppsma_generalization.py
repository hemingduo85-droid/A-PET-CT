"""
Retrain corrected dual-modality models:
  python train.py --mode train --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer PSMA --modalities ct,pet --device cuda:0 --save_dir checkpoints_v2
  python train.py --mode train --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer FDG --modalities ct,pet --device cuda:1 --save_dir checkpoints_v2

Evaluate DeepPSMA1 with the corrected weights:
  python evaluate_deeppsma_generalization.py --source_tracer PSMA --modality ct,pet --device cuda:0 --save_dir checkpoints_v2 --batch_size 8
  python evaluate_deeppsma_generalization.py --source_tracer FDG --modality ct,pet --device cuda:1 --save_dir checkpoints_v2 --batch_size 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from deeppsma_slice_metrics import compute_slice_only_metrics, top_percent_mean
from train import (
    load_model_from_checkpoint,
    parse_modalities,
    resolve_path,
    resolve_device,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DeepPSMAFlatDataset(Dataset):
    def __init__(self, data_root, modalities=("ct", "pet"), image_size=256):
        self.data_root = Path(data_root).expanduser().resolve()
        self.modalities = [m.lower() for m in modalities]
        self.image_size = int(image_size)
        self.samples = []
        self.skipped_unpaired_or_unmasked = 0
        self._load_samples()
        if not self.samples:
            raise FileNotFoundError(f"No valid DeepPSMA samples found under {self.data_root}")

    def _list_images(self, folder):
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)

    def _load_samples(self):
        test_root = self.data_root / "test"
        for class_name, label in (("normal", 0), ("abnormal", 1)):
            class_dir = test_root / class_name
            base_dir = class_dir / self.modalities[0]
            for base_path in self._list_images(base_dir):
                modality_paths = {mod: class_dir / mod / base_path.name for mod in self.modalities}
                if not all(path.is_file() for path in modality_paths.values()):
                    self.skipped_unpaired_or_unmasked += 1
                    continue
                mask_path = class_dir / "label" / base_path.name if label else None
                if label and not (mask_path and mask_path.is_file()):
                    self.skipped_unpaired_or_unmasked += 1
                    continue
                self.samples.append({
                    "id": f"{class_name}__deeppsma1__{base_path.stem}",
                    "label": label,
                    "modalities": modality_paths,
                    "mask": mask_path,
                })

    def class_distribution(self):
        return {
            "normal": sum(sample["label"] == 0 for sample in self.samples),
            "abnormal": sum(sample["label"] == 1 for sample in self.samples),
            "skipped_unpaired_or_unmasked": self.skipped_unpaired_or_unmasked,
        }

    def _read_gray(self, path, is_mask=False):
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        with Image.open(path) as img:
            img = img.convert("L").resize((self.image_size, self.image_size), resample=resample)
            arr = np.asarray(img, dtype=np.float32)
        if is_mask:
            return (arr > 127.5).astype(np.float32)
        return arr / 255.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        channels = [self._read_gray(sample["modalities"][mod], is_mask=False) for mod in self.modalities]
        image = np.stack(channels, axis=0)
        if sample["label"]:
            mask = self._read_gray(sample["mask"], is_mask=True)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.tensor(float(sample["label"]), dtype=torch.float32),
            "mask": torch.from_numpy(mask).float(),
            "id": sample["id"],
        }


@torch.no_grad()
def evaluate_deeppsma(model, loader, criterion, device, args):
    model.eval()
    test_loss = 0.0
    labels, masks, maps, filenames = [], [], [], []

    for batch in tqdm(loader, desc="DeepPSMA eval", leave=True):
        images = batch["image"].to(device)
        batch_masks = batch["mask"].cpu().numpy()
        reconstructions = model(images)
        test_loss += criterion(reconstructions, images).item()
        residual = F.mse_loss(reconstructions, images, reduction="none")
        pixel_score = residual.mean(dim=1, keepdim=True)
        pixel_score = F.interpolate(
            pixel_score,
            size=batch_masks.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).cpu().numpy()

        maps.extend(pixel_score.astype(np.float32))
        masks.extend(batch_masks.astype(np.uint8))
        labels.extend(batch["label"].cpu().numpy().astype(np.int32).tolist())
        filenames.extend(list(batch["id"]))

    labels_np = np.asarray(labels, dtype=np.int32)
    masks_np = np.asarray(masks, dtype=np.uint8)
    maps_np = np.asarray(maps, dtype=np.float32)
    scores = top_percent_mean(maps_np, args.topk_percent)
    formatted = compute_slice_only_metrics(
        labels_np,
        masks_np,
        maps_np,
        scores,
        bootstrap_iters=args.bootstrap_iters,
        hist_bins=args.ci_hist_bins,
        seed=args.ci_seed,
    )
    return {
        "test_loss": test_loss / max(1, len(loader)),
        "image_auroc": np.nan,
        "image_ap": np.nan,
        "image_f1": np.nan,
        "pixel_auroc": np.nan,
        "pixel_aupr": np.nan,
        "_formatted_metrics": formatted,
        "_labels": labels_np,
        "_masks": masks_np,
        "_maps": maps_np,
        "_image_scores": scores,
        "_filenames": np.asarray(filenames, dtype=object),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct DeepPSMA1 generalization evaluator for MMRAD.")
    parser.add_argument("--data_root", default="/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1")
    parser.add_argument("--source_tracer", choices=["PSMA", "FDG", "psma", "fdg"], default="PSMA")
    parser.add_argument("--modality", default="ct,pet", help="ct, pet, or ct,pet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--topk_percent", type=float, default=1.0)
    parser.add_argument("--bootstrap_iters", type=int, default=500)
    parser.add_argument("--ci_seed", type=int, default=42)
    parser.add_argument("--ci_pixel_max_samples", type=int, default=200000)
    parser.add_argument("--ci_hist_bins", type=int, default=16384)
    parser.add_argument("--save_npz", action="store_true")
    args = parser.parse_args()

    args.modalities = parse_modalities(args.modality)
    args.tracer = args.source_tracer.upper()
    device = resolve_device(args.device)
    dataset = DeepPSMAFlatDataset(args.data_root, args.modalities, image_size=args.img_size)
    distribution = dataset.class_distribution()
    if distribution["normal"] == 0 or distribution["abnormal"] == 0:
        raise RuntimeError(f"DeepPSMA must contain both classes, got: {distribution}")
    print(f"DeepPSMA root: {dataset.data_root}")
    print(f"Source weights: {args.tracer}")
    print(f"Modalities: {args.modalities}")
    print(f"DeepPSMA distribution: {distribution}")
    print(f"Device: {device}")
    model = load_model_from_checkpoint(args, device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    metrics = evaluate_deeppsma(model, loader, nn.MSELoss(), device, args)
    print(metrics["_formatted_metrics"])
    save_dir = resolve_path(args.save_dir) / args.tracer / "_".join(args.modalities)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_txt = save_dir / "deeppsma1_eval_metrics.txt"
    metrics_txt.write_text(metrics["_formatted_metrics"] + "\n", encoding="utf-8")
    print(f"Metrics file saved to: {metrics_txt}")
    if args.save_npz:
        cache_path = save_dir / "deeppsma1_eval_cache.npz"
        np.savez_compressed(
            cache_path,
            labels=metrics["_labels"].astype(np.int32),
            masks=metrics["_masks"].astype(np.uint8),
            maps=metrics["_maps"].astype(np.float32),
            image_scores=metrics["_image_scores"].astype(np.float64),
            paths=metrics["_filenames"],
        )
        print(f"Slice-only eval cache saved to: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
