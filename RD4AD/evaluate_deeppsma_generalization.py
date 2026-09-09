"""
Run examples:
  python evaluate_deeppsma_generalization.py --source_tracer PSMA --modality petct --gpu 0
  python evaluate_deeppsma_generalization.py --source_tracer FDG --modality petct --gpu 1
  python evaluate_deeppsma_generalization.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 --source_tracer PSMA --modality ct --gpu 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import checkpoint_path, get_dataset_config
from dataset import get_data_transforms, gray_to_rgb_tensor
from eval_protocol import (
    BOOTSTRAP_ITERS,
    BOOTSTRAP_SEED,
    HIST_BINS,
    _bootstrap_ci,
    _pixel_slice_bootstrap,
    _safe_aupr,
    _safe_auroc,
    f1_score_max,
    format_image_metrics,
    format_pixel_metrics,
)
from test_modified import cal_anomaly_map, load_checkpoint, make_model, parse_heatmap_limit, save_heatmap
from model_utils import select_device

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DeepPSMAFlatMVTecDataset(Dataset):
    def __init__(self, data_root, data_transform, gt_transform, modalities, input_mode, image_size=256):
        self.data_root = Path(data_root).expanduser().resolve()
        self.data_transform = data_transform
        self.gt_transform = gt_transform
        self.modalities = modalities
        self.input_mode = input_mode
        self.image_size = int(image_size)
        self.gray_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])
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
                mask = class_dir / "label" / base_path.name if label else None
                if label and not (mask and mask.is_file()):
                    continue
                self.samples.append((paths, mask, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, mask, label = self.samples[idx]
        if self.input_mode == "ct_pet_pet":
            ct = self.gray_transform(Image.open(paths["ct"]).convert("L"))
            pet = self.gray_transform(Image.open(paths["pet"]).convert("L"))
            img = torch.cat([ct, pet, pet], dim=0)
            sample_path = str(paths["ct"])
        else:
            img = gray_to_rgb_tensor(Image.open(paths[self.modalities[0]]).convert("L"), self.data_transform)
            sample_path = str(paths[self.modalities[0]])
        if label:
            gt = self.gt_transform(Image.open(mask).convert("L"))
        else:
            gt = torch.zeros([1, img.size(-2), img.size(-1)])
        return img, gt, label, sample_path


def parse_modalities(modality):
    if modality == "petct":
        return ["pet", "ct"], "ct_pet_pet"
    return [modality], "rgb"


def evaluate_slice_only(encoder, bn, decoder, dataloader, device, save_heatmaps="0",
                        heatmap_dir="./heatmaps", run_name="run"):
    bn.eval()
    decoder.eval()
    gt_labels, gt_masks, anomaly_maps, image_scores, paths = [], [], [], [], []
    heatmap_limit = parse_heatmap_limit(save_heatmaps)
    saved = 0
    output_dir = os.path.join(heatmap_dir, run_name)

    with torch.no_grad():
        for idx, (img, gt, label, path) in enumerate(dataloader):
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode="a")
            anomaly_map = gaussian_filter(anomaly_map, sigma=4).astype(np.float32)

            gt_binary = gt.cpu().numpy().astype(np.float32)
            gt_binary = (gt_binary > 0.5).astype(np.float32)
            label_value = int(label.item())
            path_value = path[0] if isinstance(path, (list, tuple)) else str(path)

            gt_labels.append(label_value)
            gt_masks.append(gt_binary[0])
            anomaly_maps.append(anomaly_map)
            image_scores.append(float(np.max(anomaly_map)))
            paths.append(path_value)

            if heatmap_limit is None or saved < heatmap_limit:
                if heatmap_limit != 0:
                    save_heatmap(path_value, anomaly_map, output_dir, idx, "abnormal" if label_value else "normal")
                    saved += 1

    labels_np = np.asarray(gt_labels)
    masks_np = np.asarray(gt_masks)
    maps_np = np.asarray(anomaly_maps)
    image_scores_np = np.asarray(image_scores)
    print(
        f"Cache loaded: labels={labels_np.shape}, masks={masks_np.shape}, "
        f"maps={maps_np.shape}",
        flush=True,
    )
    print("Computing image-level metrics and 95CI...", flush=True)
    slice_metrics = {
        "img_auroc": _bootstrap_ci(labels_np, image_scores_np, _safe_auroc, BOOTSTRAP_ITERS, BOOTSTRAP_SEED),
        "img_aupr": _bootstrap_ci(labels_np, image_scores_np, _safe_aupr, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 1),
        "img_f1": _bootstrap_ci(labels_np, image_scores_np, f1_score_max, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 2),
        "_pr_sp": image_scores_np,
        "_gt_sp": labels_np,
    }
    print(format_image_metrics(slice_metrics), flush=True)
    print("Computing pixel-level Slice-Px(abn) metrics and 95CI...", flush=True)
    print("Computing exact full-pixel point estimates with sklearn...", flush=True)
    px_auroc, px_aupr = _pixel_slice_bootstrap(
        labels_np,
        masks_np,
        maps_np,
        BOOTSTRAP_ITERS,
        BOOTSTRAP_SEED + 10,
        HIST_BINS,
        progress=True,
    )
    slice_metrics["px_auroc"] = px_auroc
    slice_metrics["px_aupr"] = px_aupr
    print(format_pixel_metrics(slice_metrics), flush=True)
    if heatmap_limit != 0:
        print(f"saved heatmaps: {saved} -> {output_dir}")
    print("Results:")
    print("\n".join([format_image_metrics(slice_metrics), format_pixel_metrics(slice_metrics)]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct DeepPSMA1 generalization evaluator for RD4AD.")
    parser.add_argument("--data_root", default="/data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1")
    parser.add_argument("--source_tracer", choices=["PSMA", "FDG", "psma", "fdg"], default="PSMA")
    parser.add_argument("--modality", choices=["ct", "pet", "petct"], default="petct")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--save_heatmaps", default="0")
    parser.add_argument("--heatmap_dir", default="./heatmaps_deeppsma1")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    modalities, input_mode = parse_modalities(args.modality)
    dataset_config = get_dataset_config(args.source_tracer.lower())
    ckp_path = args.checkpoint or checkpoint_path(dataset_config, modalities, input_mode)
    device = select_device(physical_gpu=args.gpu)
    data_transform, gt_transform = get_data_transforms(256, 256)
    dataset = DeepPSMAFlatMVTecDataset(args.data_root, data_transform, gt_transform, modalities, input_mode, image_size=256)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    encoder, bn, decoder = make_model(device, modalities, input_mode)
    load_checkpoint(bn, decoder, ckp_path, device)
    evaluate_slice_only(
        encoder,
        bn,
        decoder,
        loader,
        device,
        save_heatmaps=args.save_heatmaps,
        heatmap_dir=args.heatmap_dir,
        run_name=f"deeppsma1_from_{args.source_tracer.lower()}_{args.modality}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
