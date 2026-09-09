#!/usr/bin/env python3
"""
DeepPSMA1 direct generalization test for DAE.

Data root:
  /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1

Run with existing PSMA weights:
  cd /data/cyf/codes/A-PET-CT/DAE_successful
  python evaluate_deeppsma_generalization.py --source_tracer PSMA --modality ct,pet --device cuda:0

Run with existing FDG weights:
  cd /data/cyf/codes/A-PET-CT/DAE_successful
  python evaluate_deeppsma_generalization.py --source_tracer FDG --modality ct,pet --device cuda:1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dae_eval_protocol import eval_protocol_compute_metrics, eval_protocol_format_metrics, top_percent_mean
from denoising import denoising


DEFAULT_DEEPPSMA_ROOT = "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
REPO_DIR = Path(__file__).resolve().parent


def parse_modalities(value: str) -> list[str]:
    modalities = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(modalities) - {"ct", "pet"})
    if invalid or not modalities:
        raise ValueError(f"--modality must be ct, pet, or ct,pet; got {value!r}")
    return modalities


class DeepPSMAFlatDataset(Dataset):
    """Direct reader for flat DeepPSMA test/{normal,abnormal}/{pet,ct,label}."""

    def __init__(self, root: str, modalities: list[str], image_size: int = 256):
        self.root = Path(root).expanduser()
        self.modalities = modalities
        self.image_size = int(image_size)
        self.samples: list[dict] = []
        self.skipped = 0
        self._collect()
        if not self.samples:
            raise RuntimeError(
                f"No paired DeepPSMA samples found under {self.root}. "
                "Expected test/{normal,abnormal}/{pet,ct,label}/*.png"
            )

    @staticmethod
    def _is_image(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in IMG_EXTS and path.stat().st_size > 0

    def _collect_group(self, group: str, label_value: int) -> None:
        group_root = self.root / "test" / group
        primary_dir = group_root / self.modalities[0]
        if not primary_dir.is_dir():
            return
        label_dir = group_root / "label"
        for primary_path in sorted(p for p in primary_dir.iterdir() if self._is_image(p)):
            paths = {self.modalities[0]: primary_path}
            valid = True
            for mod in self.modalities[1:]:
                mod_path = group_root / mod / primary_path.name
                if not self._is_image(mod_path):
                    valid = False
                    self.skipped += 1
                    break
                paths[mod] = mod_path
            if not valid:
                continue
            mask_path = None
            if label_value == 1:
                candidate = label_dir / primary_path.name
                if not self._is_image(candidate):
                    self.skipped += 1
                    continue
                mask_path = candidate
            self.samples.append({
                "id": f"{group}__slice_{primary_path.stem}__{primary_path.stem}",
                "label": int(label_value),
                "modalities": paths,
                "mask": mask_path,
            })

    def _collect(self) -> None:
        self._collect_group("normal", 0)
        self._collect_group("abnormal", 1)

    def _read_gray(self, path: Path, is_mask: bool = False) -> torch.Tensor:
        try:
            with Image.open(path) as img:
                img = img.convert("L")
                resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
                img = img.resize((self.image_size, self.image_size), resample=resample)
                arr = np.asarray(img, dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Failed to read image: {path}") from exc
        if is_mask:
            arr = (arr > 0.5).astype(np.float32)
        return torch.from_numpy(arr).float().unsqueeze(0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        modalities = {mod: self._read_gray(sample["modalities"][mod]) for mod in self.modalities}
        if sample["mask"] is None:
            mask = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        else:
            mask = self._read_gray(sample["mask"], is_mask=True)
        return {
            "modalities": modalities,
            "label": torch.tensor(float(sample["label"]), dtype=torch.float32),
            "mask": mask,
            "id": sample["id"],
        }

    def distribution(self) -> dict[str, int]:
        normal = sum(1 for sample in self.samples if sample["label"] == 0)
        abnormal = sum(1 for sample in self.samples if sample["label"] == 1)
        return {"normal": normal, "abnormal": abnormal, "skipped_unpaired_or_unmasked": self.skipped}


@torch.no_grad()
def evaluate(models, loader, modalities, device, bootstrap_iters: int, hist_bins: int, max_batches: int = 0):
    for wrapper in models.values():
        wrapper.model.eval()

    labels, maps, masks, paths = [], [], [], []
    for batch_idx, batch in enumerate(tqdm(loader, desc="DeepPSMA eval", leave=True), start=1):
        if max_batches and batch_idx > max_batches:
            break
        residuals = []
        for mod in modalities:
            x = batch["modalities"][mod].to(device)
            recon, _ = models[mod].forward(x)
            residuals.append(torch.abs(x - recon))
        anomaly_map = torch.mean(torch.stack(residuals), dim=0).squeeze().cpu().numpy()
        maps.append(anomaly_map.astype(np.float32))
        masks.append(batch["mask"].squeeze().cpu().numpy().astype(np.uint8))
        labels.append(int(batch["label"].view(-1)[0].item()))
        paths.append(batch["id"][0])

    labels_np = np.asarray(labels, dtype=np.int64)
    maps_np = np.asarray(maps, dtype=np.float32)
    masks_np = np.asarray(masks, dtype=np.uint8)
    scores = top_percent_mean(maps_np, 1.0)
    slice_metrics, patient_metrics = eval_protocol_compute_metrics(
        labels_np,
        masks_np,
        maps_np,
        scores,
        paths,
        bootstrap_iters=bootstrap_iters,
        hist_bins=hist_bins,
        progress_callback=print,
    )
    return eval_protocol_format_metrics(slice_metrics, patient_metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DAE direct DeepPSMA1 generalization evaluator")
    parser.add_argument("--data_root", default=DEFAULT_DEEPPSMA_ROOT)
    parser.add_argument("--source_tracer", default="PSMA", choices=["PSMA", "FDG", "psma", "fdg"])
    parser.add_argument("--modality", default="ct,pet", help="ct, pet, or ct,pet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--noise_std", default=0.2, type=float)
    parser.add_argument("--noise_res", default=16, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--bootstrap_iters", default=500, type=int)
    parser.add_argument("--hist_bins", default=16384, type=int)
    parser.add_argument("--max_batches", default=0, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    modalities = parse_modalities(args.modality)
    source = args.source_tracer.upper()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset = DeepPSMAFlatDataset(args.data_root, modalities, image_size=args.image_size)
    print(f"DeepPSMA root: {Path(args.data_root).expanduser().resolve()}")
    print(f"Source weights: {source}")
    print(f"Modalities: {modalities}")
    print(f"Distribution: {dataset.distribution()}")
    print(f"Device: {device}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    ckpt_dir = (Path(args.save_dir).expanduser() / source / "_".join(modalities)).resolve()
    models = {}
    for mod in modalities:
        ckpt_path = ckpt_dir / f"best_{mod}_model.pth"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        wrapper = denoising(
            identifier=f"dae_{mod}",
            n_input=1,
            lr=1e-3,
            noise_std=args.noise_std,
            noise_res=args.noise_res,
            device=device,
        )
        state_dict = torch.load(ckpt_path, map_location=device)
        wrapper.model.load_state_dict(state_dict)
        wrapper.model.to(device).eval()
        models[mod] = wrapper
        print(f"Loaded {mod} checkpoint: {ckpt_path}")

    formatted = evaluate(
        models,
        loader,
        modalities,
        device,
        bootstrap_iters=args.bootstrap_iters,
        hist_bins=args.hist_bins,
        max_batches=args.max_batches,
    )
    print(formatted)


if __name__ == "__main__":
    main()
