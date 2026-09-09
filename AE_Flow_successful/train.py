import argparse
import os
import sys
from pathlib import Path

from env_sanitize import disable_user_site_packages
"""
95ci
nohup python train.py --mode eval --tracer PSMA --modalities ct,pet --device cuda:6 --bootstrap_iters 500  > psma_npz.log 2>&1 &

nohup python train.py --mode eval --tracer FDG --modalities ct,pet --device cuda:5 --bootstrap_iters 500  > fdg_npz.log 2>&1 &

不保存npz
nohup python train.py --mode eval --tracer FDG --modalities ct,pet --device cuda:2 --bootstrap_iters 500 --hist_bins 16384 > fdg_ct_pet_eval_ci.log 2>&1 &
tail -f fdg_ct_pet_eval_ci.log

"""

disable_user_site_packages()

import numpy as np
import torch
import torch.nn.functional as F
from dataloader import BrainTumorAnomalyDataset
from HYPERPARAMETER import alpha, beta
from model import ae_flow
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=FutureWarning, message="Importing.*from.*torchmetrics.*")

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TRACER_ROOTS = {
    "FDG": [
        Path("/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA/FDG"),
        Path("/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG"),
    ],
    "PSMA": [
        Path("/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA"),
    ],
}
LEGACY_DATA_ROOT = Path("A_data/2d_equal_mask50")

from paper_eval_utils import (
    top_percent_mean,
    eval_protocol_compute_metrics,
    eval_protocol_format_metrics,
    cache_output_paths,
    save_cache,
    resolve_path,
)

def find_checkpoint(args, modalities):
    if args.checkpoint:
        return Path(args.checkpoint)
    script_dir = Path(__file__).resolve().parent
    base = script_dir / args.checkpoint_root
    cand = base / args.tracer / "_".join(modalities)

    if cand.exists():
        pth_files = sorted(cand.glob("*.pth"))
        if pth_files:
            return pth_files[-1]
    raise FileNotFoundError(f"No checkpoint found in {cand}, provide --checkpoint")

def parse_modalities(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]

def modality_tag(modalities):
    return "_".join(modalities)

def best_checkpoint_name(tracer, modalities):
    return f"best_{tracer.upper()}_{modality_tag(modalities)}_model.pth"

def _has_split_dirs(path):
    return (path / "train").exists() and (path / "test").exists()


def _configured_roots(tracer):
    return [path.expanduser() for path in DEFAULT_TRACER_ROOTS[tracer.upper()]]


def resolve_tracer_root(data_root, tracer):
    tracer = tracer.upper()
    tried = []
    if data_root is None:
        for configured in _configured_roots(tracer):
            tried.append(configured)
            if _has_split_dirs(configured):
                return str(configured)
        data_root = LEGACY_DATA_ROOT

    data_root = Path(data_root).expanduser()
    if not data_root.is_absolute():
        for candidate in (data_root, PROJECT_DIR / data_root, PROJECT_DIR.parent / data_root):
            if candidate.exists():
                data_root = candidate
                break
    data_root = data_root.resolve()
    tried.append(data_root)
    if _has_split_dirs(data_root):
        return str(data_root)
    tracer_root = data_root / tracer
    tried.append(tracer_root)
    if _has_split_dirs(tracer_root):
        return str(tracer_root)
    for configured in _configured_roots(tracer):
        tried.append(configured)
        if _has_split_dirs(configured):
            return str(configured)
    tried_text = "\n  - ".join(str(path) for path in tried)
    raise FileNotFoundError(
        f"No dataset split root found for tracer={tracer}. Tried:\n  - {tried_text}\n"
        "Expected each root to contain train/ and test/ directories."
    )


def resolve_project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()



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


def make_dataloader(dataset, batch_size, num_workers, shuffle):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = not shuffle
    return DataLoader(dataset, **kwargs)


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    print(f"Loaded checkpoint from {checkpoint_path} with keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}")    
    return model


def generate_cache(args, model, device):
    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    dataset = BrainTumorAnomalyDataset(
        root=tracer_root,
        mode="test",
        modalities=args.modalities,
        image_size=args.image_size,
        debug_ratio=args.debug_ratio,
    )
    loader = make_dataloader(dataset, batch_size=1, num_workers=0, shuffle=False)

    all_filenames, all_labels, all_maps, all_masks = [], [], [], []
    for batch in tqdm(loader, desc="AE cache", ncols=80):
        if len(batch) == 4:
            img, label, mask, filename = batch
        elif len(batch) == 3:
            img, label, mask = batch
            filename = [f"unknown__unknown__{len(all_filenames)}"]
        else:
            continue

        img = img.to(device)
        rec_img, _, _ = model(img)
        anomap = torch.abs(img - rec_img).mean(dim=1).detach().cpu().numpy().astype(np.float32)

        mask_np = mask.detach().cpu().numpy()
        if mask_np.ndim == 4:
            mask_np = mask_np.max(axis=1)
        mask_np = (mask_np > 0.5).astype(np.uint8)

        all_maps.extend(list(anomap))
        all_masks.extend(list(mask_np))
        all_labels.extend(label.detach().cpu().view(-1).numpy().astype(np.int64).tolist())
        all_filenames.extend(list(filename))

    maps = np.stack(all_maps).astype(np.float32)
    masks = np.stack(all_masks).astype(np.uint8)
    labels = np.asarray(all_labels, dtype=np.int64)
    scores = top_percent_mean(maps, args.topk_percent)

    cache_file, metrics_file, _ = cache_output_paths("AE", args.tracer, args.modalities)
    save_cache(cache_file, "AE", args.tracer, args.modalities, all_filenames, labels, scores, maps, masks, args.topk_percent)
    print(f"Cache saved: {cache_file}")
    return all_filenames, labels, maps, masks, scores


def print_cache_metric_header(labels, masks, maps, bootstrap_iters):
    print(
        f"Cache loaded: labels={labels.shape}, masks={masks.shape}, "
        f"maps={maps.shape}, bootstrap_iters={bootstrap_iters}"
    )


def eval_mode(args):
    device = resolve_device(args.device)
    model = ae_flow.AE_FLOW(subnet=args.subnet).to(device)
    checkpoint = find_checkpoint(args, args.modalities)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    filenames, labels, maps, masks, scores = generate_cache(args, model, device)
    print_cache_metric_header(labels, masks, maps, args.bootstrap_iters)
    slice_metrics, pat_metrics = eval_protocol_compute_metrics(
        labels,
        masks,
        maps,
        scores,
        filenames,
        bootstrap_iters=args.bootstrap_iters,
        ci_seed=args.ci_seed,
        ci_pixel_max_samples=args.ci_pixel_max_samples,
        hist_bins=args.hist_bins,
        progress_callback=print,
    )
    formatted_metrics = eval_protocol_format_metrics(slice_metrics, pat_metrics)
    print(formatted_metrics)

    save_dir = os.path.join(resolve_path(args.save_dir), args.tracer.upper(), modality_tag(args.modalities))
    os.makedirs(save_dir, exist_ok=True)
    metrics_txt_path = os.path.join(save_dir, "eval_metrics.txt")
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write(formatted_metrics + "\n")
    print(f"Eval metrics file saved to: {metrics_txt_path}")



def train(args):
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = ae_flow.AE_FLOW(subnet=args.subnet)
    model.to(device)

    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    train_set = BrainTumorAnomalyDataset(
        root=tracer_root,
        mode="train",
        modalities=args.modalities,
        image_size=args.image_size,
        debug_ratio=args.debug_ratio,
    )
    test_set = BrainTumorAnomalyDataset(
        root=tracer_root,
        mode="test",
        modalities=args.modalities,
        image_size=args.image_size,
        debug_ratio=args.debug_ratio,
    )

    print(f"Data root: {args.data_root}")
    print(f"Tracer root: {tracer_root}")
    print(f"Modalities: {args.modalities}")
    print(f"Train batch size: {args.train_batch_size}")
    print(f"Eval batch size: {args.test_batch_size}")

    train_loader = make_dataloader(train_set, args.train_batch_size, args.train_num_workers, shuffle=True)
    test_loader = make_dataloader(test_set, args.test_batch_size, args.test_num_workers, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = os.path.join(resolve_path(args.save_dir), args.tracer.upper(), modality_tag(args.modalities))
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoint dir: {save_dir}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=True, ncols=60, mininterval=5, ascii=True)
        for batch in train_bar:
            img, label = batch
            img = img.to(device)

            rec_img, z_hat, jac = model(img)
            recon_loss = F.mse_loss(rec_img, img)
            flow_loss, log_z = model.flow_loss()
            loss = (1 - alpha) * recon_loss + alpha * flow_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch}, Avg Loss: {avg_loss:.4f}")

    # Save training checkpoint at the final epoch as the optimal model weights
    ckpt_path = os.path.join(save_dir, best_checkpoint_name(args.tracer, args.modalities))
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "data_root": tracer_root,
            "tracer": args.tracer.upper(),
            "modalities": args.modalities,
        },
        ckpt_path,
    )
    print(f"Training completed! Saved final epoch weights to: {ckpt_path}")

    # Generate .npz cache and compute unified final metrics
    filenames, labels, maps, masks, scores = generate_cache(args, model, device)
    print_cache_metric_header(labels, masks, maps, args.bootstrap_iters)
    slice_metrics, pat_metrics = eval_protocol_compute_metrics(
        labels,
        masks,
        maps,
        scores,
        filenames,
        bootstrap_iters=args.bootstrap_iters,
        ci_seed=args.ci_seed,
        ci_pixel_max_samples=args.ci_pixel_max_samples,
        hist_bins=args.hist_bins,
        progress_callback=print,
    )
    formatted_metrics = eval_protocol_format_metrics(slice_metrics, pat_metrics)

    print(formatted_metrics)

    metrics_txt_path = os.path.join(save_dir, "best_metrics.txt")
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Best Epoch: {args.epochs}\n")
        f.write(formatted_metrics + "\n")
    print(f"Unified metrics file saved to: {metrics_txt_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train/evaluate AE_FLOW.")
    parser.add_argument("--data_root", default=None, type=str)
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA"])
    parser.add_argument("--modalities", default="ct,pet", type=parse_modalities)
    parser.add_argument("--subnet", default="conv_type", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--checkpoint_root", default="checkpoints", type=str)
    parser.add_argument("--topk_percent", default=1.0, type=float)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--eval_interval", default=5, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--train_num_workers", default=2, type=int)
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--test_num_workers", default=1, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--save_dir", default="checkpoints", type=str)
    parser.add_argument("--debug_ratio", default=1.0, type=float)
    parser.add_argument("--skip_pixel_eval", action="store_true")
    parser.add_argument("--bootstrap_iters", default=500, type=int)
    parser.add_argument("--ci_seed", default=42, type=int)
    parser.add_argument("--ci_pixel_max_samples", default=200000, type=int)
    parser.add_argument("--hist_bins", default=16384, type=int)
    parser.add_argument("--device", default="cuda:4", type=str)
    parser.add_argument("--mode", default="eval", choices=["train", "eval"], type=str)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "eval":
        eval_mode(args)
    else:
        train(args)
