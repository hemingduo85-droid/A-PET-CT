import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecDataset
from train import build_model, build_transforms, resolve_tracer_root
from utils import build_anomaly_map, setup_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_eval_utils import (
    best_f1_threshold,
    display_score_map,
    hot_output_root,
    load_selected_samples,
    parse_modalities,
    save_case_outputs,
    selected_csv_path,
    top_percent_mean,
)


METHOD = "INP"
PROJECT_DIR = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    for candidate in (PROJECT_DIR / path, PROJECT_DIR.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
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


def checkpoint_path(args, modalities):
    if args.checkpoint:
        return Path(args.checkpoint)
    return resolve_path(args.checkpoint_dir) / args.tracer.upper() / "_".join(modalities) / "model.pth"


def load_model(args, modalities, device):
    args.modalities = modalities
    model, _ = build_model(args, device)
    model.load_state_dict(torch.load(checkpoint_path(args, modalities), map_location=device), strict=True)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path(args, modalities)}")
    return model


def make_input_tensor(images, modalities, data_transform):
    from PIL import Image

    channels = [images[m] for m in modalities]
    if len(channels) == 1:
        channels = channels * 3
    elif len(channels) == 2:
        channels.append(np.mean(np.stack(channels, axis=0), axis=0).astype(np.float32))
    fused = np.stack(channels[:3], axis=-1)
    image = Image.fromarray((np.clip(fused, 0, 1) * 255).astype(np.uint8))
    return data_transform(image).unsqueeze(0)


@torch.no_grad()
def infer_map(model, images, modalities, device, args, data_transform, out_size):
    x = make_input_tensor(images, modalities, data_transform).to(device)
    en, de, _ = model(x)
    amap = build_anomaly_map(model, en, de, out_size, args.anomaly_source)
    return amap.squeeze().detach().cpu().numpy().astype(np.float32)


def run_selected(args, model, modalities, device):
    data_transform, _ = build_transforms(args.input_size, args.crop_size)
    eval_size = args.input_size if args.crop_size <= 0 else args.crop_size
    csv_path = Path(args.selected_csv) if args.selected_csv else selected_csv_path(args.tracer)
    samples = load_selected_samples(csv_path, modalities, image_size=eval_size)
    out_root = hot_output_root(METHOD, args.tracer, modalities)
    out_root.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(samples, 1):
        raw_map = infer_map(model, sample["images"], modalities, device, args, data_transform, eval_size)
        roi = np.zeros_like(raw_map, dtype=np.uint8)
        for image in sample["images"].values():
            roi |= (image > args.roi_min).astype(np.uint8)
        disp = display_score_map(raw_map, roi=roi, mode=args.heatmap_norm, q_low=args.score_q_low, q_high=args.score_q_high)
        threshold = args.thr
        best_f1 = None
        if args.threshold_mode == "gt_best_f1":
            threshold, best_f1 = best_f1_threshold(sample["mask"], raw_map)
        score = float(top_percent_mean(raw_map, args.topk_percent)[0])
        save_case_outputs(
            out_root / sample["case_name"],
            sample["images"],
            sample["mask"],
            raw_map,
            disp,
            threshold,
            score=score,
            case_name=sample["case_name"],
            roi=roi,
            min_component_pixels=args.min_component_pixels,
        )
        suffix = f" best_f1={best_f1:.4f}" if best_f1 is not None else ""
        print(f"[{idx}/{len(samples)}] {sample['case_name']} score={score:.4f} thr={threshold:.4f}{suffix}")
    print(f"Done, saved to {out_root}")


def build_parser():
    parser = argparse.ArgumentParser(description="INPFormer compact inference for PET/CT comparisons.")
    parser.add_argument("--data_root", default="../A_data/2d_equal_mask50")
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct,pet")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--selected_csv", default=None)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=392)
    parser.add_argument("--INP_num", type=int, default=6)
    parser.add_argument("--anomaly_source", default="reconstruction", choices=["prototype", "reconstruction", "fused"])
    parser.add_argument("--label_mode", default="folder", choices=["folder", "mask"])
    parser.add_argument("--channel_fill", default="mean_modalities", choices=["zero", "imagenet_mean", "mean_modalities"])
    parser.add_argument("--roi_min", type=float, default=0.03)
    parser.add_argument("--score_q_low", type=float, default=0.01)
    parser.add_argument("--score_q_high", type=float, default=0.995)
    parser.add_argument("--heatmap_norm", default="percentile", choices=["percentile", "minmax", "none"])
    parser.add_argument("--threshold_mode", default="gt_best_f1", choices=["fixed", "gt_best_f1"])
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--min_component_pixels", type=int, default=0)
    parser.add_argument("--topk_percent", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main():
    args = build_parser().parse_args()
    setup_seed(args.seed)
    args.tracer = args.tracer.upper()
    modalities = parse_modalities(args.modalities)
    device = resolve_device(args.device)
    model = load_model(args, modalities, device)
    run_selected(args, model, modalities, device)


if __name__ == "__main__":
    main()
