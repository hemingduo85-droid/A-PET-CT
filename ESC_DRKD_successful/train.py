"""
Reproducible ESC-DRKD training and evaluation commands.

Train new complete PSMA and FDG checkpoints without overwriting legacy weights:

  python train.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer PSMA --modalities ct,pet --epochs 30 --device cuda:0 --save_dir checkpoints_fixed --seed 42

  python train.py --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer FDG --modalities ct,pet --epochs 30 --device cuda:1 --save_dir checkpoints_fixed --seed 42

Evaluate the retrained checkpoints on their original test sets:

  python train.py --mode eval --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer PSMA --modalities ct,pet --device cuda:0 --save_dir checkpoints_fixed --seed 42

  python train.py --mode eval --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 --tracer FDG --modalities ct,pet --device cuda:1 --save_dir checkpoints_fixed --seed 42
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from new_model import ESC_DRKD_Multimodal
from checkpoint_schema import build_complete_checkpoint, require_complete_model_state
from utils import AverageMeter, print_log


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from paper_eval_utils import (
    eval_protocol_compute_metrics,
    eval_protocol_format_image_metrics,
    eval_protocol_format_metrics,
    top_percent_mean,
)
RESAMPLE_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")
RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def parse_modalities(value):
    modalities = [m.strip().lower() for m in value.split(",") if m.strip()]
    invalid = sorted(set(modalities) - {"ct", "pet"})
    if invalid:
        raise ValueError(f"Invalid modalities: {invalid}. Use ct, pet, or ct,pet.")
    if not modalities:
        raise ValueError("At least one modality is required.")
    return modalities


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    for candidate in (PROJECT_DIR / path, PROJECT_DIR.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_DIR / path).resolve()


def resolve_tracer_root(data_root, tracer):
    root = resolve_path(data_root)
    if (root / "train").exists() and (root / "test").exists():
        return root
    return root / tracer.upper()


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_checkpoint_file(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class PETCTSliceDataset(torch.utils.data.Dataset):
    """PET/CT slice dataset.

    Expected layout:
        root/train/normal/<patient>/<ct|pet>/<slice>.png
        root/test/normal/<patient>/<ct|pet>/<slice>.png
        root/test/abnormal/<patient>/<ct|pet>/<slice>.png
        root/test/abnormal/<patient>/label/<slice>.png
    """

    def __init__(self, root, modalities, mode="train", img_size=256):
        self.root = Path(root)
        self.modalities = modalities
        self.mode = mode
        self.img_size = img_size
        self.samples = []

        if mode not in {"train", "test"}:
            raise ValueError(f"Invalid mode: {mode}")

        self._load_samples()
        if not self.samples:
            raise FileNotFoundError(
                f"No {mode} samples found under {self.root} with modalities={self.modalities}"
            )

    def _load_samples(self):
        split_dir = self.root / self.mode
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        class_names = ["normal"] if self.mode == "train" else ["normal", "abnormal"]
        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                if class_name == "normal":
                    raise FileNotFoundError(f"Required class directory not found: {class_dir}")
                continue

            for patient_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                base_dir = patient_dir / self.modalities[0]
                if not base_dir.exists():
                    continue

                for base_path in sorted(p for p in base_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
                    modality_paths = {self.modalities[0]: base_path}
                    valid = True
                    for mod in self.modalities[1:]:
                        mod_path = patient_dir / mod / base_path.name
                        if not mod_path.is_file():
                            valid = False
                            break
                        modality_paths[mod] = mod_path
                    if not valid:
                        continue

                    mask_path = None
                    if self.mode == "test" and class_name == "abnormal":
                        candidate = patient_dir / "label" / base_path.name
                        if candidate.is_file() and candidate.stat().st_size > 0:
                            mask_path = candidate

                    self.samples.append(
                        {
                            "id": f"{class_name}__{patient_dir.name}__{base_path.stem}",
                            "label": int(class_name == "abnormal"),
                            "modalities": modality_paths,
                            "mask": mask_path,
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def _read_gray(self, path, is_mask=False):
        try:
            with Image.open(path) as image:
                image = image.convert("L")
                resample = RESAMPLE_NEAREST if is_mask else RESAMPLE_BILINEAR
                image = image.resize((self.img_size, self.img_size), resample=resample)
                img = np.asarray(image, dtype=np.float32) / 255.0
        except OSError as exc:
            raise RuntimeError(f"Failed to read image: {path}") from exc
        if is_mask:
            img = (img > 0.5).astype(np.float32)
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = {}
        for mod in self.modalities:
            img = self._read_gray(sample["modalities"][mod], is_mask=False)
            data[mod] = torch.from_numpy(img).float().unsqueeze(0)

        if self.mode == "test" and sample["mask"] is not None:
            mask = self._read_gray(sample["mask"], is_mask=True)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        item = {"modalities": data}
        if self.mode == "test":
            item.update(
                {
                    "label": sample["label"],
                    "mask": torch.from_numpy(mask).float(),
                    "id": sample["id"],
                }
            )
        return item

    def class_distribution(self):
        if self.mode == "train":
            return {"normal": len(self.samples)}
        normal = sum(1 for sample in self.samples if sample["label"] == 0)
        abnormal = sum(1 for sample in self.samples if sample["label"] == 1)
        return {"normal": normal, "abnormal": abnormal}


def collate_fn(batch):
    modalities = batch[0]["modalities"].keys()
    out = {
        "modalities": {
            mod: torch.stack([item["modalities"][mod] for item in batch], dim=0)
            for mod in modalities
        }
    }
    if "label" in batch[0]:
        out.update(
            {
                "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
                "mask": torch.stack([item["mask"] for item in batch], dim=0),
                "id": [item["id"] for item in batch],
            }
        )
    return out


def fuse_modalities(x_dict, modalities):
    # Original reproduction path: average selected modalities into one grayscale input.
    return torch.stack([x_dict[mod] for mod in modalities], dim=0).mean(dim=0)


def calculate_aupro(anomaly_maps, ground_truth_masks):
    all_scores = np.concatenate([am.reshape(-1) for am in anomaly_maps])
    if all_scores.size == 0 or np.max(all_scores) == np.min(all_scores):
        return 0.0

    thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 100)
    curve = []
    for threshold in thresholds:
        tp = fn = 0.0
        fpr_sum = valid = 0
        for anomaly_map, gt_mask in zip(anomaly_maps, ground_truth_masks):
            pred = (anomaly_map > threshold).astype(np.float32)
            gt = (gt_mask > 0.5).astype(np.float32)
            tp += np.sum((pred == 1) & (gt == 1))
            fn += np.sum((pred == 0) & (gt == 1))
            normal_pixels = np.sum(gt == 0)
            if normal_pixels > 0:
                fpr_sum += np.sum((pred == 1) & (gt == 0)) / normal_pixels
                valid += 1
        recall = tp / (tp + fn + 1e-8)
        fpr = fpr_sum / (valid + 1e-8)
        curve.append((fpr, recall))

    curve = np.array(sorted(curve))
    fpr, idx = np.unique(curve[:, 0], return_index=True)
    recall = curve[idx, 1]
    keep = fpr <= 0.6
    if not np.any(keep):
        return 0.0
    fpr = fpr[keep]
    recall = recall[keep]
    if fpr[0] > 0:
        fpr = np.insert(fpr, 0, 0.0)
        recall = np.insert(recall, 0, recall[0])
    if fpr[-1] < 0.6:
        fpr = np.append(fpr, 0.6)
        recall = np.append(recall, recall[-1])
    return float(np.trapz(recall, fpr) / 0.6)


def find_checkpoint(args, save_dir):
    if args.checkpoint:
        checkpoint = resolve_path(args.checkpoint)
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    for name in ("best_model.pt", "model.pth"):
        checkpoint = save_dir / name
        if checkpoint.exists():
            return checkpoint

    pt_files = sorted(save_dir.glob("*.pt")) + sorted(save_dir.glob("*.pth"))
    if pt_files:
        return pt_files[-1]
    raise FileNotFoundError(f"No checkpoint found in {save_dir}; provide --checkpoint")


def get_anomaly_map(model, t_features, reconstruction, img_size):
    anomaly_maps = []
    with torch.no_grad():
        recon_features = model.teacher.backbone(reconstruction)
        for t_feat, recon_feat in zip(t_features[:3], recon_features[:3]):
            if recon_feat.shape[-2:] != t_feat.shape[-2:]:
                recon_feat = F.interpolate(recon_feat, size=t_feat.shape[-2:], mode="bilinear")
            cos_sim = F.cosine_similarity(t_feat, recon_feat, dim=1)
            anomaly_map = 1 - cos_sim.unsqueeze(1)
            anomaly_map = F.interpolate(anomaly_map, size=(img_size, img_size), mode="bilinear")
            anomaly_maps.append(anomaly_map)
    return torch.mean(torch.cat(anomaly_maps, dim=1), dim=1, keepdim=True)


def train_one_epoch(model, loader, optimizer, args, log):
    model.train()
    losses = AverageMeter()
    pbar = tqdm(loader, desc="Train")

    for batch in pbar:
        x_dict = {mod: batch["modalities"][mod].to(args.device) for mod in args.modalities}
        x = fuse_modalities(x_dict, args.modalities)

        with torch.no_grad():
            t_features = model.teacher.backbone(x)

        reconstruction = model.student(t_features[-1], t_features[:-1])
        loss = model.compute_loss(
            {
                "reconstruction": reconstruction,
                "t_features": t_features,
                "t_recon_features": model.teacher.backbone(reconstruction),
            },
            x,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), x.size(0))
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})

    print_log(f"Train Loss: {losses.avg:.4f}", log)
    return losses.avg


@torch.no_grad()
def evaluate(model, loader, args, log=None):
    model.eval()
    pixel_maps, pixel_masks = [], []
    labels, filenames = [], []

    for batch in tqdm(loader, desc="Eval"):
        x_dict = {mod: batch["modalities"][mod].to(args.device) for mod in args.modalities}
        x = fuse_modalities(x_dict, args.modalities)
        t_features = model.teacher.backbone(x)
        reconstruction = model.student(t_features[-1], t_features[:-1])
        anomaly_map = get_anomaly_map(model, t_features, reconstruction, args.img_size)

        pixel_maps.append(anomaly_map.squeeze().cpu().numpy())
        pixel_masks.append(batch["mask"].squeeze().cpu().numpy())
        labels.append(int(batch["label"].view(-1)[0].item()))
        filenames.append(batch["id"][0])

    labels = np.asarray(labels, dtype=np.int64)
    pixel_maps = np.asarray(pixel_maps, dtype=np.float32)
    pixel_masks = np.asarray(pixel_masks, dtype=np.uint8)
    image_scores = top_percent_mean(pixel_maps, 1.0)
    print_log(
        f"Eval arrays ready: labels={labels.shape}, masks={pixel_masks.shape}, "
        f"maps={pixel_maps.shape}, bootstrap_iters={args.bootstrap_iters}",
        log,
    )

    def progress_callback(message):
        print_log(message, log)

    slice_metrics, pat_metrics = eval_protocol_compute_metrics(
        labels,
        pixel_masks,
        pixel_maps,
        image_scores,
        filenames,
        bootstrap_iters=args.bootstrap_iters,
        ci_seed=args.ci_seed,
        ci_pixel_max_samples=args.ci_pixel_max_samples,
        ci_hist_bins=args.ci_hist_bins,
        progress_callback=progress_callback,
    )
    return {
        "image_auroc": slice_metrics["img_auroc"],
        "image_ap": slice_metrics["img_ap"],
        "image_f1": slice_metrics["img_f1"],
        "pixel_auroc": slice_metrics["px_auroc_abn"],
        "pixel_aupr": slice_metrics["px_aupr_abn"],
        # "aupro": calculate_aupro(pixel_maps, pixel_masks),
        "_formatted_image_metrics": eval_protocol_format_image_metrics(slice_metrics),
        "_formatted_metrics": eval_protocol_format_metrics(slice_metrics, pat_metrics),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Train ESC-DRKD on A_data PET/CT slices.")
    parser.add_argument("--data_root", default="../A_data/2d_equal_mask50", type=str)
    parser.add_argument("--tracer", default="FDG", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct", type=str, help="ct, pet, or ct,pet")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--device", default="cuda:2", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--save_dir", default="checkpoints", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--mode", default="train", choices=["train", "eval"], type=str)
    parser.add_argument("--early_stop_patience", default=20, type=int)
    parser.add_argument("--eval_interval", default=1, type=int)
    parser.add_argument("--bootstrap_iters", default=500, type=int)
    parser.add_argument("--ci_seed", default=0, type=int)
    parser.add_argument("--ci_pixel_max_samples", default=200000, type=int)
    parser.add_argument("--ci_hist_bins", default=16384, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def main():
    args = build_parser().parse_args()
    args.modalities = parse_modalities(args.modalities)
    args.tracer = args.tracer.upper()
    seed_everything(args.seed)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    # 自动根据 lr 和 batch_size 创建不同文件夹
    save_dir = resolve_path(args.save_dir) / args.tracer / "_".join(args.modalities)
    #save_dir = save_dir  / f"lr{args.lr}_bs{args.batch_size}"  # 关键在这里
    save_dir.mkdir(parents=True, exist_ok=True)

    log = open(save_dir / "training.log", "w", encoding="utf-8")
    print_log(f"Data root: {tracer_root}", log)
    print_log(f"Modalities: {args.modalities}", log)
    print_log(f"Device: {args.device}", log)
    print_log("Fusion: original mean fusion into one channel", log)

    test_dataset = PETCTSliceDataset(tracer_root, args.modalities, mode="test", img_size=args.img_size)
    print_log(f"Test distribution: {test_dataset.class_distribution()}", log)

    train_loader = None
    if args.mode == "train":
        train_dataset = PETCTSliceDataset(tracer_root, args.modalities, mode="train", img_size=args.img_size)
        print_log(f"Train distribution: {train_dataset.class_distribution()}", log)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = ESC_DRKD_Multimodal(modalities=args.modalities).to(args.device)
    if args.mode == "eval":
        checkpoint = find_checkpoint(args, save_dir)
        state = load_checkpoint_file(checkpoint, args.device)
        model.load_state_dict(require_complete_model_state(state), strict=True)
        print_log(f"Loaded checkpoint: {checkpoint}", log)
        metrics = evaluate(model, test_loader, args, log)
        formatted_metrics = metrics["_formatted_metrics"]
        print_log(formatted_metrics, log)
        with open(save_dir / "eval_metrics.txt", "w", encoding="utf-8") as f:
            f.write(formatted_metrics + "\n")
        log.close()
        return

    optimizer = optim.Adam(model.student.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print_log(f"Epoch {epoch}/{args.epochs}", log)
        train_one_epoch(model, train_loader, optimizer, args, log)

    checkpoint = build_complete_checkpoint(
        model,
        optimizer,
        epoch=args.epochs,
        modalities=args.modalities,
        tracer=args.tracer,
        fusion="mean",
        seed=args.seed,
    )
    torch.save(checkpoint, save_dir / "best_model.pt")
    metrics = evaluate(model, test_loader, args, log)
    formatted_metrics = metrics["_formatted_metrics"]
    print_log(formatted_metrics, log)
    with open(save_dir / "best_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Best Epoch: {args.epochs}\n")
        f.write(formatted_metrics + "\n")

    log.close()


if __name__ == "__main__":
    main()
