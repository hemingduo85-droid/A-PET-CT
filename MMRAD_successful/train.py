import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import MultimodalGrayDataset
from model import MultiModalAnomalyDetector
"""
cd /data/cyf/codes/lyh/MMRAD_successful

python train.py \
  --mode eval \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA \
  --tracer PSMA \
  --modalities ct,pet \
  --device cuda:3 \
  --bootstrap_iters 500 > psma_ci.log 2>&1 &

python train.py \
  --mode eval \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG \
  --tracer FDG \
  --modalities ct,pet \
  --device cuda:6 \
  --bootstrap_iters 500 > fdg_ci.log 2>&1 &

nohup python3 -u train.py \
  --mode eval \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG \
  --tracer FDG \
  --modalities ct,pet \
  --save_dir ./checkpoints \
  --device cuda:6 \
  --batch_size 8 \
  --num_workers 4 \
  --bootstrap_iters 500 \
  --ci_pixel_max_samples 0 \
  > fdg_ci.log 2>&1 &

95ci
nohup python3 -u train.py \
  --mode eval \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA \
  --tracer PSMA \
  --modalities ct,pet \
  --save_dir ./checkpoints \
  --device cuda:0 \
  --batch_size 8 \
  --num_workers 4 \
  --bootstrap_iters 500 \
  > psma_npz.log 2>&1 &

nohup python3 -u train.py \
  --mode eval \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG \
  --tracer FDG \
  --modalities ct,pet \
  --save_dir ./checkpoints \
  --device cuda:2 \
  --batch_size 8 \
  --num_workers 4 \
  --bootstrap_iters 500 \
  > fdg_npz.log 2>&1 &
"""

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent

from paper_eval_utils import (
    cache_output_paths,
    eval_protocol_compute_metrics,
    eval_protocol_format_image_metrics,
    eval_protocol_format_metrics,
    save_cache,
    top_percent_mean,
    write_metrics,
)


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


def resolve_device(device_arg):
    if device_arg == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is not available.")
        if device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        torch.cuda.set_device(device)
    return device


def checkpoint_path(args, modalities):
    if args.checkpoint:
        return Path(args.checkpoint).expanduser()
    ckpt_dir = resolve_path(args.save_dir) / args.tracer.upper() / "_".join(modalities)
    for name in ("model.pth", "best_model.pth"):
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate
    return ckpt_dir / "model.pth"


def load_model_from_checkpoint(args, device):
    model = MultiModalAnomalyDetector(
        num_modalities=len(args.modalities),
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
    ).to(device)
    model.image_score_mode = args.image_score_mode
    model.image_score_topk_ratio = args.image_score_topk_ratio
    ckpt_path = checkpoint_path(args, args.modalities)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


def safe_auroc(labels, scores):
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def safe_ap(labels, scores):
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def best_f1(labels, scores):
    if len(np.unique(labels)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(np.max(f1))


def compute_image_scores(residual, mode="mean", topk_ratio=0.01):
    pixel_score = residual.mean(dim=1)
    if mode == "mean":
        return pixel_score.mean(dim=(1, 2))
    if mode != "topk":
        raise ValueError(f"Unsupported image score mode: {mode}")

    flat_score = pixel_score.flatten(1)
    k = max(1, int(flat_score.shape[1] * topk_ratio))
    topk_score = torch.topk(flat_score, k=k, dim=1).values
    return topk_score.mean(dim=1)


def calculate_aupro(anomaly_maps, ground_truth_masks):
    all_scores = np.concatenate([m.reshape(-1) for m in anomaly_maps])
    if all_scores.size == 0 or np.max(all_scores) == np.min(all_scores):
        return 0.0

    thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 100)
    curve = []
    for threshold in thresholds:
        tp = fn = 0.0
        fpr_sum = valid = 0
        for anomaly_map, gt in zip(anomaly_maps, ground_truth_masks):
            pred = (anomaly_map >= threshold).astype(np.uint8)
            gt = (gt > 0.5).astype(np.uint8)
            tp += np.sum((pred == 1) & (gt == 1))
            fn += np.sum((pred == 0) & (gt == 1))
            normal_pixels = np.sum(gt == 0)
            if normal_pixels > 0:
                fpr_sum += np.sum((pred == 1) & (gt == 0)) / normal_pixels
                valid += 1
        curve.append((fpr_sum / (valid + 1e-8), tp / (tp + fn + 1e-8)))

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


@torch.no_grad()
def evaluate(
    model,
    test_loader,
    criterion,
    device,
    topk_percent=1.0,
    bootstrap_iters=500,
    ci_seed=42,
    ci_pixel_max_samples=200000,
    ci_hist_bins=16384,
    progress_callback=None,
):
    model.eval()
    test_loss = 0.0
    image_labels = []
    pixel_maps = []
    pixel_masks = []
    filenames = []

    for batch in tqdm(test_loader, desc="Eval"):
        images = batch["image"].to(device)
        labels = batch["label"].cpu().numpy()
        masks = batch["mask"].cpu().numpy()

        reconstructions = model(images)
        test_loss += criterion(reconstructions, images).item()

        residual = F.mse_loss(reconstructions, images, reduction="none")
        image_score = compute_image_scores(
            residual,
            mode=getattr(model, "image_score_mode", "mean"),
            topk_ratio=getattr(model, "image_score_topk_ratio", 0.01),
        ).cpu().numpy()
        pixel_score = residual.mean(dim=1, keepdim=True)
        pixel_score = F.interpolate(
            pixel_score,
            size=masks.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).cpu().numpy()

        image_labels.extend(labels.tolist())
        pixel_maps.extend([pixel_score[i] for i in range(pixel_score.shape[0])])
        pixel_masks.extend([masks[i] for i in range(masks.shape[0])])
        filenames.extend(list(batch["id"]))

    image_labels = np.asarray(image_labels, dtype=np.int64)
    pixel_maps = np.asarray(pixel_maps, dtype=np.float32)
    pixel_masks = np.asarray(pixel_masks, dtype=np.uint8)
    image_scores = top_percent_mean(pixel_maps, topk_percent)
    slice_metrics, pat_metrics = eval_protocol_compute_metrics(
        image_labels,
        pixel_masks,
        pixel_maps,
        image_scores,
        filenames,
        bootstrap_iters=bootstrap_iters,
        ci_seed=ci_seed,
        ci_pixel_max_samples=ci_pixel_max_samples,
        ci_hist_bins=ci_hist_bins,
        progress_callback=progress_callback,
    )

    return {
        "test_loss": test_loss / max(1, len(test_loader)),
        "image_auroc": slice_metrics["img_auroc"],
        "image_ap": slice_metrics["img_ap"],
        "image_f1": slice_metrics["img_f1"],
        "pixel_auroc": slice_metrics["px_auroc_abn"],
        "pixel_aupr": slice_metrics["px_aupr_abn"],
        "_formatted_metrics": eval_protocol_format_metrics(slice_metrics, pat_metrics),
        "_slice_metrics": slice_metrics,
        "_pat_metrics": pat_metrics,
        "_filenames": filenames,
        "_labels": image_labels,
        "_scores": image_scores,
        "_pixel_maps": pixel_maps,
        "_pixel_masks": pixel_masks,
    }


def save_evaluation_cache(args, metrics, save_dir, metrics_filename="metrics.json"):
    metrics_txt_path = save_dir / ("eval_metrics.txt" if metrics_filename != "metrics.json" else "best_metrics.txt")
    with metrics_txt_path.open("w", encoding="utf-8") as handle:
        if metrics_txt_path.name == "best_metrics.txt":
            handle.write(f"Best Epoch: {args.epochs}\n")
        handle.write(metrics["_formatted_metrics"] + "\n")
    print(f"Unified metrics file saved to: {metrics_txt_path}")

    if not args.save_npz:
        print("NPZ cache disabled; use --save_npz to save test_cache.npz and metrics JSON.")
        return

    cache_file, metrics_file, _ = cache_output_paths("MMRAD", args.tracer, args.modalities)
    if metrics_filename != "metrics.json":
        metrics_file = metrics_file.with_name(metrics_filename)
    save_cache(
        cache_file,
        "MMRAD",
        args.tracer,
        args.modalities,
        metrics["_filenames"],
        metrics["_labels"],
        metrics["_scores"],
        metrics["_pixel_maps"],
        metrics["_pixel_masks"],
        args.topk_percent,
    )
    payload = {
        "method": "MMRAD",
        "tracer": args.tracer,
        "modalities": args.modalities,
        "score_source": "topk_from_map",
        "topk_percent": float(args.topk_percent),
        "slice_metrics": metrics["_slice_metrics"],
        "pat_metrics": metrics["_pat_metrics"],
        "formatted_metrics": metrics["_formatted_metrics"],
    }
    write_metrics(metrics_file, payload)
    print(f"Cache saved: {cache_file}")
    print(f"Metrics saved: {metrics_file}")


def print_log(msg, log_file):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def train_model(args):
    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    # 鎸塴r鍜宐atchsize鍒涘缓鐙珛鏂囦欢澶?
    # save_dir = resolve_path(args.save_dir) / args.tracer.upper() / "_".join(args.modalities) / f"lr{args.lr}_bs{args.batch_size}"
    save_dir = resolve_path(args.save_dir) / args.tracer.upper() / "_".join(args.modalities)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 璁粌鏃ュ織
    log = open(save_dir / "training.log", "w", encoding="utf-8")
    device = resolve_device(args.device)
    
    model = MultiModalAnomalyDetector(
        num_modalities=len(args.modalities),
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
    ).to(device)
    model.image_score_mode = args.image_score_mode
    model.image_score_topk_ratio = args.image_score_topk_ratio

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_dataset = MultimodalGrayDataset(
        tracer_root, modalities=args.modalities, mode="train", image_size=args.img_size
    )
    test_dataset = MultimodalGrayDataset(
        tracer_root, modalities=args.modalities, mode="test", image_size=args.img_size
    )

    # 缁熶竴鏃ュ織鎵撳嵃
    print_log(f"Data root: {tracer_root}", log)
    print_log(f"Modalities: {args.modalities}", log)
    print_log(f"Train distribution: {train_dataset.class_distribution()}", log)
    print_log(f"Test distribution: {test_dataset.class_distribution()}", log)
    print_log(f"Checkpoints: {save_dir}", log)
    print_log(f"Device: {device}", log)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    writer = SummaryWriter(log_dir=str(save_dir / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"):
            images = batch["image"].to(device)
            reconstructions = model(images)
            loss = criterion(reconstructions, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))
        scheduler.step()
        writer.add_scalar("loss/train", avg_train_loss, epoch)
        print_log(f"Epoch {epoch}/{args.epochs}, Train Loss={avg_train_loss:.4f}", log)
    torch.save(model.state_dict(), save_dir / "model.pth")
    print_log(f"\nTraining completed! Saved final epoch weights to: {save_dir / 'model.pth'}", log)

    def print_progress(message):
        print_log(message, log)

    metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        topk_percent=args.topk_percent,
        bootstrap_iters=args.bootstrap_iters,
        ci_seed=args.ci_seed,
        ci_pixel_max_samples=args.ci_pixel_max_samples,
        ci_hist_bins=args.ci_hist_bins,
        progress_callback=print_progress,
    )
    writer.add_scalar("loss/test", metrics["test_loss"], args.epochs)
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            writer.add_scalar(f"metrics/{key}", value, args.epochs)

    formatted_metrics = metrics["_formatted_metrics"]
    print_log(formatted_metrics, log)

    save_evaluation_cache(args, metrics, save_dir)

    writer.close()
    log.close()


def eval_model(args):
    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    save_dir = resolve_path(args.save_dir) / args.tracer.upper() / "_".join(args.modalities)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model = load_model_from_checkpoint(args, device).eval()
    criterion = nn.MSELoss()
    test_dataset = MultimodalGrayDataset(
        tracer_root, modalities=args.modalities, mode="test", image_size=args.img_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    def print_progress(message):
        print(message, flush=True)

    metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        topk_percent=args.topk_percent,
        bootstrap_iters=args.bootstrap_iters,
        ci_seed=args.ci_seed,
        ci_pixel_max_samples=args.ci_pixel_max_samples,
        ci_hist_bins=args.ci_hist_bins,
        progress_callback=print_progress,
    )
    print(metrics["_formatted_metrics"])
    save_evaluation_cache(args, metrics, save_dir, metrics_filename="eval_metrics.json")


def build_parser():
    parser = argparse.ArgumentParser(description="Train MMRAD on A_data PET/CT slices.")
    parser.add_argument("--data_root", default="../A_data/2d_equal_mask50", type=str)
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct,pet", type=str, help="ct, pet, or ct,pet")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--patch_size", default=8, type=int)
    parser.add_argument("--embed_dim", default=32, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--device", default="cuda:3", type=str)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--save_dir", default="checkpoints", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--topk_percent", default=1.0, type=float)
    parser.add_argument("--mode", default="eval", choices=["train", "eval"], type=str)
    parser.add_argument("--save_npz", action="store_true", help="Save test_cache.npz and metrics JSON for plotting.")
    parser.add_argument(
        "--image_score_mode",
        default="topk",
        choices=["mean", "topk"],
        help="Image-level anomaly score aggregation from pixel residuals.",
    )
    parser.add_argument(
        "--image_score_topk_ratio",
        default=0.01,
        type=float,
        help="Top-k ratio used when image_score_mode=topk.",
    )
    parser.add_argument("--bootstrap_iters", default=500, type=int, help="Bootstrap iterations for 95% CI.")
    parser.add_argument("--ci_seed", default=42, type=int, help="Random seed for bootstrap confidence intervals.")
    parser.add_argument(
        "--ci_pixel_max_samples",
        default=200000,
        type=int,
        help="Deprecated; pixel CI uses abnormal-slice histogram bootstrap with full-pixel point estimates.",
    )
    parser.add_argument("--ci_hist_bins", default=16384, type=int, help="Histogram bins for pixel-level bootstrap CI.")
    # 鏂板鍙傛暟锛氭棭鍋?+ 淇濆瓨鎸囨爣
    parser.add_argument("--early_stop_patience", default=10, type=int, help="Deprecated; no early stopping is used.")
    parser.add_argument(
        "--save_metric",
        default="image_ap",
        choices=["image_auroc", "image_ap", "image_f1", "pixel_auroc", "pixel_aupr"],
        help="Deprecated; final epoch is always saved.",
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.modalities = parse_modalities(parsed.modalities)
    parsed.tracer = parsed.tracer.upper()
    if parsed.mode == "eval":
        eval_model(parsed)
    else:
        train_model(parsed)
