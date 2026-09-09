"""
DeepPSMA generalization commands for the retrained complete checkpoints.

  python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer PSMA --modality ct,pet --device cuda:0 --save_dir checkpoints_fixed --seed 42

  python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer FDG --modality ct,pet --device cuda:1 --save_dir checkpoints_fixed --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from deeppsma_slice_metrics import compute_slice_only_metrics, top_percent_mean
from checkpoint_schema import require_complete_model_state
from new_model import ESC_DRKD_Multimodal
from train import (
    collate_fn,
    find_checkpoint,
    fuse_modalities,
    get_anomaly_map,
    parse_modalities,
    print_log,
    resolve_path,
    seed_everything,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DeepPSMAFlatDataset(Dataset):
    def __init__(self, data_root, modalities=("ct", "pet"), img_size=256):
        self.data_root = Path(data_root).expanduser().resolve()
        self.modalities = [m.lower() for m in modalities]
        self.img_size = int(img_size)
        self.samples = []
        self._load_samples()
        if not self.samples:
            raise FileNotFoundError(f"No valid DeepPSMA samples found under {self.data_root}")

    def _list_images(self, folder):
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)

    def _load_samples(self):
        for class_name, label in (("normal", 0), ("abnormal", 1)):
            class_dir = self.data_root / "test" / class_name
            base_dir = class_dir / self.modalities[0]
            for base_path in self._list_images(base_dir):
                paths = {mod: class_dir / mod / base_path.name for mod in self.modalities}
                if not all(path.is_file() for path in paths.values()):
                    continue
                mask_path = class_dir / "label" / base_path.name if label else None
                if label and not (mask_path and mask_path.is_file()):
                    continue
                self.samples.append({
                    "id": f"{class_name}__deeppsma1__{base_path.stem}",
                    "label": label,
                    "modalities": paths,
                    "mask": mask_path,
                })

    def _read_gray(self, path, is_mask=False):
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        with Image.open(path) as image:
            image = image.convert("L").resize((self.img_size, self.img_size), resample=resample)
            arr = np.asarray(image, dtype=np.float32)
        if is_mask:
            return (arr > 127.5).astype(np.float32)
        return arr / 255.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        modalities = {
            mod: torch.from_numpy(self._read_gray(sample["modalities"][mod], is_mask=False)).float().unsqueeze(0)
            for mod in self.modalities
        }
        if sample["label"]:
            mask = self._read_gray(sample["mask"], is_mask=True)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        return {
            "modalities": modalities,
            "label": int(sample["label"]),
            "mask": torch.from_numpy(mask).float(),
            "id": sample["id"],
        }


def load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def evaluate_deeppsma(model, loader, args, log=None):
    model.eval()
    labels, masks, maps = [], [], []

    for batch in tqdm(loader, desc="DeepPSMA eval", leave=True):
        x_dict = {mod: batch["modalities"][mod].to(args.device) for mod in args.modalities}
        x = fuse_modalities(x_dict, args.modalities)
        t_features = model.teacher.backbone(x)
        reconstruction = model.student(t_features[-1], t_features[:-1])
        anomaly_map = get_anomaly_map(model, t_features, reconstruction, args.img_size)

        maps.extend(anomaly_map.squeeze(1).cpu().numpy().astype(np.float32))
        masks.extend(batch["mask"].cpu().numpy().astype(np.uint8))
        labels.extend(batch["label"].cpu().numpy().astype(np.int32).tolist())

    labels_np = np.asarray(labels, dtype=np.int32)
    masks_np = np.asarray(masks, dtype=np.uint8)
    maps_np = np.asarray(maps, dtype=np.float32)
    image_scores = top_percent_mean(maps_np, 1.0)
    print_log(
        f"Eval arrays ready: labels={labels_np.shape}, masks={masks_np.shape}, "
        f"maps={maps_np.shape}, bootstrap_iters={args.bootstrap_iters}",
        log,
    )
    return compute_slice_only_metrics(
        labels_np,
        masks_np,
        maps_np,
        image_scores,
        bootstrap_iters=args.bootstrap_iters,
        hist_bins=args.ci_hist_bins,
        seed=args.ci_seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct DeepPSMA1 generalization evaluator for ESC-DRKD.")
    parser.add_argument("--data_root", default="/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1")
    parser.add_argument("--source_tracer", choices=["PSMA", "FDG", "psma", "fdg"], default="PSMA")
    parser.add_argument("--modality", default="ct,pet", help="ct, pet, or ct,pet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--bootstrap_iters", type=int, default=500)
    parser.add_argument("--ci_seed", type=int, default=0)
    parser.add_argument("--ci_pixel_max_samples", type=int, default=200000)
    parser.add_argument("--ci_hist_bins", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.modalities = parse_modalities(args.modality)
    args.tracer = args.source_tracer.upper()
    seed_everything(args.seed)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = resolve_path(args.save_dir) / args.tracer / "_".join(args.modalities)
    save_dir.mkdir(parents=True, exist_ok=True)
    log = open(save_dir / "deeppsma1_eval.log", "w", encoding="utf-8")

    dataset = DeepPSMAFlatDataset(args.data_root, args.modalities, img_size=args.img_size)
    print_log(f"DeepPSMA1 data root: {Path(args.data_root).expanduser().resolve()}", log)
    print_log(f"Samples: {len(dataset)}", log)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = ESC_DRKD_Multimodal(modalities=args.modalities).to(args.device)
    checkpoint = find_checkpoint(args, save_dir)
    state = load_state(checkpoint, args.device)
    model.load_state_dict(require_complete_model_state(state), strict=True)
    print_log(f"Loaded checkpoint: {checkpoint}", log)
    formatted_metrics = evaluate_deeppsma(model, loader, args, log)
    print_log(formatted_metrics, log)
    with open(save_dir / "deeppsma1_eval_metrics.txt", "w", encoding="utf-8") as handle:
        handle.write(formatted_metrics + "\n")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
