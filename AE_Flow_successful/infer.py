import argparse
import os
import sys
from pathlib import Path

from env_sanitize import disable_user_site_packages

disable_user_site_packages()

import numpy as np
import torch
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import ae_flow
from paper_eval_utils import (
    best_f1_threshold,
    display_score_map,
    hot_output_root,
    load_selected_samples,
    save_case_outputs,
    selected_csv_path,
    top_percent_mean,
    parse_modalities
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



def load_model(checkpoint, device, subnet):
    model = ae_flow.AE_FLOW(subnet=subnet).to(device).eval()
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model


def build_input_tensor(images, modalities):
    channels = [images[m] for m in modalities]
    if len(channels) == 1:
        channels = channels * 3
    elif len(channels) == 2:
        channels.append((channels[0] + channels[1]) / 2.0)
    else:
        channels = channels[:3]
    fused = np.stack(channels, axis=0).astype(np.float32)
    fused = (fused - 0.5) / 0.5
    return torch.from_numpy(fused).unsqueeze(0)


@torch.no_grad()
def infer_map(model, images, modalities, device):
    x = build_input_tensor(images, modalities).to(device)
    rec, _, _ = model(x)
    raw = torch.abs(x - rec).mean(dim=1).squeeze(0)
    return raw.detach().cpu().numpy().astype(np.float32)


def run_selected(args, model, modalities, device):
    csv_path = Path(args.selected_csv) if args.selected_csv else selected_csv_path(args.tracer)
    samples = load_selected_samples(csv_path, modalities, image_size=args.image_size)
    out_root = hot_output_root("AE", args.tracer, modalities)
    out_root.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(samples, 1):
        raw_map = infer_map(model, sample["images"], modalities, device)
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
    parser = argparse.ArgumentParser(description="AE-Flow compact inference for PET/CT comparisons.")
    parser.add_argument("--data_root", default="A_data/2d_equal_mask50")
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA", "fdg", "psma"]) 
    parser.add_argument("--modalities", default="ct,pet")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--selected_csv", default=None)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--roi_min", type=float, default=0.03)
    parser.add_argument("--score_q_low", type=float, default=0.01)
    parser.add_argument("--score_q_high", type=float, default=0.995)
    parser.add_argument("--heatmap_norm", default="percentile", choices=["percentile", "minmax", "none"])
    parser.add_argument("--threshold_mode", default="gt_best_f1", choices=["fixed", "gt_best_f1"])
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--min_component_pixels", type=int, default=0)
    parser.add_argument("--topk_percent", type=float, default=1.0)
    parser.add_argument("--debug_ratio", type=float, default=1.0)
    parser.add_argument("--subnet", default="conv_type")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.tracer = args.tracer.upper()
    modalities = parse_modalities(args.modalities)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint = find_checkpoint(args, modalities)
    model = load_model(checkpoint, device, args.subnet)
    run_selected(args, model, modalities, device)


if __name__ == "__main__":
    main()
