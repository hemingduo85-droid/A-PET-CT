import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import PETCTSliceDataset, resolve_tracer_root
from denoising import denoising
"""
python3 train.py --mode eval --tracer PSMA --modalities ct,pet --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA --device cuda:4 > psma_ci.log 2>&1 &

python3 train.py --mode eval --tracer FDG --modalities ct,pet --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG --device cuda:4 > fdg_ci.log 2>&1 &

95ci
nohup python3 train.py \
  --mode eval \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA \
  --tracer PSMA \
  --modalities ct,pet \
  --device cuda:4 \
  --num_workers 0 \
  > psma_npz.log 2>&1 &

nohup python3 train.py \
  --mode eval \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG \
  --tracer FDG \
  --modalities ct,pet \
  --device cuda:5 \
  --num_workers 0 \
  > fdg_npz.log 2>&1 &

泛化性
nohup python3 train.py \
  --mode eval \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/deeppsma1 \
  --tracer PSMA \
  --modalities ct,pet \
  --device cuda:4 \
  --num_workers 0 \
  > psma_deeppsma1_generalization.log 2>&1 &
"""


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dae_eval_protocol import top_percent_mean, eval_protocol_compute_metrics, eval_protocol_format_metrics


def parse_modalities(value):
    return [m.strip().lower() for m in value.split(",") if m.strip()]


def resolve_project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def make_loaders(args, tracer_root, modalities):
    target_size = (args.image_size, args.image_size)
    train_set = PETCTSliceDataset(
        tracer_root,
        modalities=modalities,
        mode="train",
        target_size=target_size,
        debug_ratio=args.debug_ratio,
    )
    test_set = PETCTSliceDataset(
        tracer_root,
        modalities=modalities,
        mode="test",
        target_size=target_size,
        debug_ratio=args.debug_ratio,
    )
    kwargs = {"num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)
    return train_set, test_set, train_loader, test_loader


@torch.no_grad()
def evaluate(models, test_loader, modalities, device, save_dir=None, max_batches=0):
    for mod in modalities:
        ckpt_path = save_dir / f"best_{mod}_model.pth"
        state_dict = torch.load(ckpt_path, map_location=device)
        models[mod].model.load_state_dict(state_dict)  # 加载磁盘权重到模型
        models[mod].model.eval()

    image_labels = []
    pixel_maps = []
    pixel_masks = []
    filenames = []

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Eval", leave=True), start=1):
        if max_batches and batch_idx > max_batches:
            break

        residuals = []
        for mod in modalities:
            x = batch["modalities"][mod].to(device)
            recon, _ = models[mod].forward(x)
            residuals.append(torch.abs(x - recon))

        anomaly_map = torch.mean(torch.stack(residuals), dim=0).squeeze().cpu().numpy()

        image_labels.append(float(batch["label"].view(-1)[0].item()))
        pixel_maps.append(anomaly_map)
        pixel_masks.append(batch["mask"].squeeze().cpu().numpy())
        filenames.append(batch["id"][0])

    labels = np.asarray(image_labels, dtype=np.int64)
    maps = np.asarray(pixel_maps, dtype=np.float32)
    masks = np.asarray(pixel_masks, dtype=np.uint8)
    scores = top_percent_mean(maps, 1.0)

    slice_metrics, pat_metrics = eval_protocol_compute_metrics(
        labels,
        masks,
        maps,
        scores,
        filenames,
        bootstrap_iters=500,
        progress_callback=print,
    )
    return {
        "image_auroc": slice_metrics["img_auroc"]["value"],
        "image_aupr": slice_metrics["img_aupr"]["value"],
        "image_f1": slice_metrics["img_f1"]["value"],
        "pixel_auroc": slice_metrics["px_auroc_abn"]["value"],
        "pixel_aupr": slice_metrics["px_aupr_abn"]["value"],
        "patient_auroc": pat_metrics["pat_auroc"]["value"],
        "patient_aupr": pat_metrics["pat_aupr"]["value"],
        "patient_f1": pat_metrics["pat_f1"]["value"],
        "_formatted_metrics": eval_protocol_format_metrics(slice_metrics, pat_metrics),
    }


def train_model(args):
    modalities = parse_modalities(args.modalities)
    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    save_dir = resolve_project_path(args.save_dir) / args.tracer.upper() / "_".join(modalities)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ===================== 修复 1：自动创建 runs 目录 =====================
    runs_dir = save_dir / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_set, test_set, train_loader, test_loader = make_loaders(args, tracer_root, modalities)
    print("Train distribution:", train_set.get_class_distribution())
    print("Test distribution:", test_set.get_class_distribution())
    print("Device:", device)
    print("Checkpoints:", save_dir)

    models = {}
    optimizers = {}
    for mod in modalities:
        wrapper = denoising(
            identifier=f"dae_{mod}",
            n_input=1,
            lr=args.lr,
            noise_std=args.noise_std,
            noise_res=args.noise_res,
            device=device,
        )
        wrapper.model.to(device)
        models[mod] = wrapper
        optimizers[mod] = wrapper.optimiser

    writer = SummaryWriter(log_dir=str(runs_dir))

    # for epoch in range(1, args.epochs + 1):
    #     train_loss = {mod: 0.0 for mod in modalities}
    #     for mod in modalities:
    #         models[mod].model.train()

    #     for batch in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"):
    #         for mod in modalities:
    #             x = batch["modalities"][mod].to(device)
    #             recon, _ = models[mod].forward(x)
    #             loss = torch.nn.functional.mse_loss(recon, x)
    #             optimizers[mod].zero_grad()
    #             loss.backward()
    #             optimizers[mod].step()
    #             train_loss[mod] += loss.item()

    #     for mod in modalities:
    #         avg_loss = train_loss[mod] / max(1, len(train_loader))
    #         writer.add_scalar(f"loss/train_{mod}", avg_loss, epoch)

    #     loss_text = ", ".join(
    #         f"{mod}={train_loss[mod] / max(1, len(train_loader)):.4f}" for mod in modalities
    #     )
    #     print(f"Epoch {epoch}/{args.epochs}, Avg Loss: {loss_text}")

    # for mod in modalities:
    #     ckpt_path = save_dir / f"best_{mod}_model.pth"
    #     torch.save(models[mod].model.state_dict(), ckpt_path)
    # print(f"\nTraining completed! Saved final epoch weights to: {save_dir}")

    metrics = evaluate(models, test_loader, modalities, device, save_dir=save_dir, max_batches=args.eval_max_batches)
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            writer.add_scalar(f"metrics/{key}", value, args.epochs)

    formatted_metrics = metrics["_formatted_metrics"]
    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS (Unified Protocol)")
    print("=" * 50)
    print(formatted_metrics)
    print("=" * 50 + "\n")

    metrics_txt_path = save_dir / "best_metrics.txt"
    with metrics_txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Best Epoch: {args.epochs}\n")
        f.write(formatted_metrics + "\n")
    print(f"Unified metrics file saved to: {metrics_txt_path}")

    writer.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Train DAE on A_data PET/CT slices.")
    parser.add_argument("--data_root", default="A_data/2d_equal_mask50", type=str)
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct,pet", help="Comma separated: ct, pet, or ct,pet")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--noise_std", default=0.2, type=float)
    parser.add_argument("--noise_res", default=16, type=int)
    parser.add_argument("--eval_interval", default=1, type=int, help="Deprecated; evaluation runs once after training.")
    parser.add_argument("--eval_max_batches", default=0, type=int, help="Limit eval batches for quick debugging. 0 means full test set.")
    parser.add_argument("--device", default="cuda:3", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--debug_ratio", default=1.0, type=float)
    parser.add_argument("--save_dir", default="checkpoints", type=str)
    parser.add_argument("--mode", default="eval", choices=["eval"], help="Compatibility flag; this script currently runs evaluation only.")
    return parser


if __name__ == "__main__":
    train_model(build_parser().parse_args())
