import argparse
import json
import os
import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import MVTecDataset
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Aggregation_Block, Mlp, Prototype_Block
from optimizers import StableAdamW
from utils import WarmCosineScheduler, build_anomaly_map, get_logger, global_cosine_hm_adaptive, setup_seed
"""
cd /Users/zongjinying/Desktop/lyh/INPFormer

nohup python3 -u train.py \
  --mode eval \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 \
  --tracer PSMA \
  --modalities ct,pet \
  --checkpoint_dir ./checkpoints \
  --device cuda:4 \
  --bootstrap_iters 500 \
  > psma_npz.log 2>&1 &

nohup python3 -u train.py \
  --mode eval \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 \
  --tracer FDG \
  --modalities ct,pet \
  --checkpoint_dir ./checkpoints \
  --device cuda:4 \
  --bootstrap_iters 500 \
  > fdg_npz.log 2>&1 &
"""

warnings.filterwarnings("ignore")
PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_eval_utils import (
    cache_output_paths,
    eval_protocol_compute_metrics,
    eval_protocol_format_metrics,
    json_ready_metrics,
    resolve_path,
    save_cache,
    top_percent_mean,
)


def configure_cpu_threads(num_threads):
    if num_threads is None or num_threads <= 0:
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, num_threads)))
    except RuntimeError:
        pass


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


def build_transforms(input_size, crop_size):
    image_steps = [transforms.Resize((input_size, input_size))]
    mask_steps = [transforms.Resize((input_size, input_size))]
    if crop_size > 0 and crop_size != input_size:
        image_steps.append(transforms.CenterCrop(crop_size))
        mask_steps.append(transforms.CenterCrop(crop_size))
    image_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mask_steps.append(transforms.ToTensor())
    data_transform = transforms.Compose(image_steps)
    gt_transform = transforms.Compose(mask_steps)
    return data_transform, gt_transform


def build_model(args, device):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(args.encoder)
    if "small" in args.encoder:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 4, 6, 8]
        fuse_layer_encoder = [[0, 1, 2, 3]]
        fuse_layer_decoder = [[0, 1, 2, 3]]
    elif "base" in args.encoder:
        embed_dim, num_heads = 768, 12
        target_layers = [4, 6, 8, 10]
        fuse_layer_encoder = [[0, 1, 2, 3]]
        fuse_layer_decoder = [[0, 1, 2, 3]]
    elif "large" in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]

    bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)])
    inp_tokens = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim))])
    aggregation = nn.ModuleList(
        [
            Aggregation_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
        ]
    )
    decoder = nn.ModuleList(
        [
            Prototype_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
            for _ in range(8)
        ]
    )

    model = INP_Former(
        encoder=encoder,
        bottleneck=bottleneck,
        aggregation=aggregation,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        prototype_token=inp_tokens,
    ).to(device)
    trainable = nn.ModuleList([bottleneck, decoder, aggregation, inp_tokens])
    return model, trainable


def init_trainable(trainable):
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


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
        return Path(args.checkpoint)
    return resolve_path(args.checkpoint_dir) / args.tracer / "_".join(modalities) / "model.pth"


def find_checkpoint(args, modalities):
    if args.checkpoint:
        return Path(args.checkpoint)
    base = resolve_path(args.checkpoint_dir) / args.tracer / "_".join(modalities)
    if not base.exists():
        raise FileNotFoundError(f"No checkpoint directory found: {base}")
    pth_files = sorted(base.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No checkpoint found in {base}")
    return pth_files[-1]


@torch.no_grad()
def evaluate_unified(model, dataloader, device, args, eval_size, save_root=None):
    model.eval()
    labels = []
    maps = []
    masks = []
    filenames = []
    for img, gt, label, img_path in tqdm(dataloader, ncols=80, desc="Eval"):
        img = img.to(device)
        output = model(img)
        en, de = output[0], output[1]
        anomaly_map = build_anomaly_map(model, en, de, eval_size, args.anomaly_source)
        if gt.shape[-2:] != anomaly_map.shape[-2:]:
            gt = nn.functional.interpolate(gt, size=anomaly_map.shape[-2:], mode="nearest")
        if gt.shape[1] > 1:
            gt = torch.max(gt, dim=1, keepdim=True)[0]
        maps.extend(anomaly_map[:, 0].detach().cpu().numpy().astype(np.float32))
        masks.extend((gt[:, 0].detach().cpu().numpy() > 0.5).astype(np.uint8))
        labels.extend(label.detach().cpu().view(-1).numpy().astype(np.int64).tolist())
        for path in img_path:
            p = Path(path)
            filenames.append(f"{p.parent.parent.parent.name}__{p.parent.parent.name}__{p.stem}")

    maps = np.stack(maps).astype(np.float32)
    masks = np.stack(masks).astype(np.uint8)
    labels = np.asarray(labels, dtype=np.int64)
    scores = top_percent_mean(maps, args.max_ratio * 100.0)
    progress_callback = getattr(args, "_metric_progress_callback", None)
    slice_metrics, pat_metrics = eval_protocol_compute_metrics(
        labels,
        masks,
        maps,
        scores,
        filenames,
        bootstrap_iters=args.bootstrap_iters,
        ci_pixel_max_samples=args.ci_pixel_max_samples,
        progress_callback=progress_callback,
    )
    formatted_metrics = eval_protocol_format_metrics(slice_metrics, pat_metrics)
    if save_root is not None:
        cache_file, metrics_file, _ = cache_output_paths("INP", args.tracer, args.modalities)
        save_cache(cache_file, "INP", args.tracer, args.modalities, filenames, labels, scores, maps, masks, args.max_ratio * 100.0)
        metrics_payload = {
            "method": "INP",
            "tracer": args.tracer,
            "modalities": args.modalities,
            "score_source": "topk_from_map",
            "topk_percent": float(args.max_ratio * 100.0),
            "slice_metrics": json_ready_metrics(slice_metrics),
            "pat_metrics": json_ready_metrics(pat_metrics),
        }
        with open(metrics_file, "w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, indent=2)
        metrics_txt_path = save_root / "best_metrics.txt"
        with metrics_txt_path.open("w", encoding="utf-8") as handle:
            handle.write(formatted_metrics + "\n")
        print(f"Metrics saved to: {metrics_file}")
        print(f"Unified metrics file saved to: {metrics_txt_path}")
    return [
        slice_metrics["img_auroc"],
        slice_metrics["img_ap"],
        slice_metrics["img_f1"],
        slice_metrics["px_auroc_abn"],
        slice_metrics["px_aupr_abn"],
        float("nan"),
        float("nan"),
    ], formatted_metrics


def eval_mode(args):
    configure_cpu_threads(args.cpu_threads)
    setup_seed(args.seed)
    args.tracer = args.tracer.upper()
    args.modalities = parse_modalities(args.modalities)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    save_root = resolve_path(args.save_dir) / args.tracer / "_".join(args.modalities)
    save_root.mkdir(parents=True, exist_ok=True)

    logger = get_logger("INPFormer", save_root)
    print_fn = logger.info
    print_fn(f"Data root: {tracer_root}")
    print_fn(f"Modalities: {args.modalities}")
    print_fn(f"Device: {device}")

    data_transform, gt_transform = build_transforms(args.input_size, args.crop_size)
    eval_size = args.input_size if args.crop_size <= 0 else args.crop_size
    test_data = MVTecDataset(
        root=tracer_root,
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
        modalities=args.modalities,
        label_mode=args.label_mode,
        channel_fill=args.channel_fill,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model, _ = build_model(args, device)
    ckpt = find_checkpoint(args, args.modalities)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()
    args._metric_progress_callback = print_fn
    results, formatted_metrics = evaluate_unified(model, test_loader, device, args, eval_size, save_root=save_root)
    print_fn("\n" + "=" * 50)
    print_fn("FINAL EVALUATION METRICS (Unified Protocol)")
    print_fn("=" * 50)
    print_fn(formatted_metrics)
    print_fn("=" * 50)
    return results


def train(args):
    configure_cpu_threads(args.cpu_threads)
    setup_seed(args.seed)
    args.tracer = args.tracer.upper()
    args.modalities = parse_modalities(args.modalities)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    save_root = resolve_path(args.save_dir) / args.tracer / "_".join(args.modalities)
    save_root.mkdir(parents=True, exist_ok=True)

    logger = get_logger("INPFormer", save_root)
    print_fn = logger.info
    print_fn(f"Data root: {tracer_root}")
    print_fn(f"Modalities: {args.modalities}")
    print_fn(f"Device: {device}")

    data_transform, gt_transform = build_transforms(args.input_size, args.crop_size)
    eval_size = args.input_size if args.crop_size <= 0 else args.crop_size
    train_data = MVTecDataset(
        root=tracer_root,
        transform=data_transform,
        gt_transform=gt_transform,
        phase="train",
        modalities=args.modalities,
        label_mode=args.label_mode,
        channel_fill=args.channel_fill,
    )
    test_data = MVTecDataset(
        root=tracer_root,
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
        modalities=args.modalities,
        label_mode=args.label_mode,
        channel_fill=args.channel_fill,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model, trainable = build_model(args, device)
    init_trainable(trainable)
    optimizer = StableAdamW(
        [{"params": trainable.parameters()}],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        amsgrad=True,
        eps=1e-10,
    )
    scheduler = WarmCosineScheduler(
        optimizer,
        base_value=args.lr,
        final_value=args.min_lr,
        total_iters=args.total_epochs * max(1, len(train_loader)),
        warmup_iters=args.warmup_iters,
    )

    for epoch in range(1, args.total_epochs + 1):
        model.train()
        loss_list = []
        for img, _, _, _ in tqdm(train_loader, ncols=80, desc=f"Train {epoch}/{args.total_epochs}"):
            img = img.to(device)
            en, de, g_loss = model(img)
            loss = global_cosine_hm_adaptive(en, de, y=3) + 0.2 * g_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            loss_list.append(loss.item())

        print_fn(f"Epoch [{epoch}/{args.total_epochs}], Loss: {np.mean(loss_list):.4f}")

    ckpt_path = save_root / "model.pth"
    torch.save(model.state_dict(), ckpt_path)
    print_fn(f"Training completed! Saved final epoch weights to {ckpt_path}")

    args._metric_progress_callback = print_fn
    results, formatted_metrics = evaluate_unified(model, test_loader, device, args, eval_size, save_root=save_root)
    print_fn("\n" + "=" * 50)
    print_fn("FINAL EVALUATION METRICS (Unified Protocol)")
    print_fn("=" * 50)
    print_fn(formatted_metrics)
    print_fn("=" * 50)

    with open(save_root / "best_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Best Epoch: {args.total_epochs}\n")
        f.write(formatted_metrics + "\n")

    return results


def main(args):
    if args.mode == "eval":
        return eval_mode(args)
    return train(args)


def print_results(print_fn, prefix, results):
    auroc_sp, ap_sp, f1_sp, auroc_px, aupr_px, _, _ = results
    print_fn(
        f"{prefix}: "
        f"I-AUROC={auroc_sp:.4f}, I-AUPR={ap_sp:.4f}, I-F1={f1_sp:.4f}, "
        f"P-AUROC={auroc_px:.4f}, P-AUPR={aupr_px:.4f}"
    )


def describe_channel_adapter(channel_fill):
    if channel_fill == "mean_modalities":
        return "[ct,pet,(ct+pet)/2]"
    if channel_fill == "imagenet_mean":
        return "[ct,pet,imagenet_mean]"
    return "[ct,pet,0]"


def build_parser():
    parser = argparse.ArgumentParser(description="Train INPFormer on A_data PET/CT slices.")
    parser.add_argument("--data_root", default="../A_data/2d_equal_mask50", type=str)
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct,pet", type=str, help="ct, pet, or ct,pet")
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14", type=str)
    parser.add_argument("--input_size", default=448, type=int)
    parser.add_argument("--crop_size", default=392, type=int, help="Use <=0 for no CenterCrop after resize.")
    parser.add_argument("--INP_num", default=6, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--total_epochs", default=30, type=int)
    parser.add_argument("--mode", default="eval", choices=["train", "eval"], type=str)
    parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--lr", default=1e-4, type=float) #1e-3
    parser.add_argument("--min_lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--warmup_iters", default=100, type=int)
    parser.add_argument("--max_ratio", default=0.01, type=float)
    parser.add_argument(
        "--image_score_mode",
        default="topk_mean",
        choices=[ "topk_mean", "max"],
        help="Image-level anomaly score aggregation from the anomaly map.",
    )

    parser.add_argument(
        "--label_mode",
        default="folder",
        choices=["folder", "mask"],
        help="folder uses test/abnormal as image label. mask is only for optional lesion-positive slice analysis.",
    )
    parser.add_argument(
        "--channel_fill",
        default="mean_modalities",
        choices=["zero", "imagenet_mean", "mean_modalities"],
        help="Third channel for ct,pet input. mean_modalities uses (ct+pet)/2 and is the PET/CT adapted default.",
    )
    parser.add_argument(
        "--save_metric",
        default="image_ap",
        choices=["image_auroc", "image_ap", "image_f1", "pixel_auroc", "pixel_aupr"],
        help="Metric used to select model.pth.",
    )
    parser.add_argument(
        "--anomaly_source",
        default="reconstruction",
        choices=["prototype", "reconstruction", "fused"],
        help="Map used for metrics: INP prototype distance, decoder reconstruction cosine map, or their sum.",
    )
    parser.add_argument(
        "--score_csv",
        default="",
        type=str,
        help="Optional CSV filename under save_dir/tracer/modalities for per-image score diagnostics.",
    )
    parser.add_argument("--device", default="cuda:5", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--cpu_threads", default=4, type=int, help="Limit PyTorch/BLAS CPU threads. Use <=0 to keep library defaults.")
    parser.add_argument("--eval_interval", default=5, type=int, help="Run full test evaluation every N epochs; final epoch is always evaluated.")
    parser.add_argument("--compute_aupro", action="store_true", help="Compute AUPRO during evaluation. Off by default to reduce CPU/GPU load while training.")
    parser.add_argument("--aupro_nstrips", default=200, type=int)
    parser.add_argument("--bootstrap_iters", default=500, type=int, help="Bootstrap iterations for all 95% CI estimates.")
    parser.add_argument("--ci_pixel_max_samples", default=200000, type=int, help="Deprecated compatibility option; pixel CI now uses abnormal-slice histogram bootstrap.")
    parser.add_argument("--save_dir", default="./checkpoints", type=str)
    parser.add_argument("--seed", default=2, type=int)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
