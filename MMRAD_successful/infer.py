import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataloader import MultimodalGrayDataset
from model import MultiModalAnomalyDetector

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


METHOD = "MMRAD"
PROJECT_DIR = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    for candidate in (PROJECT_DIR / path, PROJECT_DIR.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_DIR / path).resolve()


def checkpoint_path(args, modalities):
    if args.checkpoint:
        return Path(args.checkpoint)
    ckpt_dir = resolve_path(args.checkpoint_root) / args.tracer.upper() / "_".join(modalities)
    for name in ("model.pth", "best_model.pth"):
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate
    return ckpt_dir / "model.pth"


def load_model(args, modalities, device):
    model = MultiModalAnomalyDetector(
        num_modalities=len(modalities),
        img_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
    ).to(device).eval()
    ckpt = torch.load(checkpoint_path(args, modalities), map_location=device)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path(args, modalities)}")
    return model


@torch.no_grad()
def infer_map(model, images, modalities, device):
    x = np.stack([images[m] for m in modalities], axis=0)
    x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
    recon = model(x_t)
    residual = F.mse_loss(recon, x_t, reduction="none").mean(dim=1)
    return residual.squeeze().detach().cpu().numpy().astype(np.float32)


def run_selected(args, model, modalities, device):
    csv_path = Path(args.selected_csv) if args.selected_csv else selected_csv_path(args.tracer)
    samples = load_selected_samples(csv_path, modalities, image_size=args.image_size)
    out_root = hot_output_root(METHOD, args.tracer, modalities)
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
    parser = argparse.ArgumentParser(description="MMRAD paper inference/cache for PET/CT comparisons.")
    parser.add_argument("--data_root", default="../A_data/2d_equal_mask50")
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--selected_csv", default=None)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--roi_min", type=float, default=0.03)
    parser.add_argument("--score_q_low", type=float, default=0.01)
    parser.add_argument("--score_q_high", type=float, default=0.995)
    parser.add_argument("--heatmap_norm", default="percentile", choices=["percentile", "minmax", "none"])
    parser.add_argument("--threshold_mode", default="gt_best_f1", choices=["fixed", "gt_best_f1"])
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--min_component_pixels", type=int, default=0)
    parser.add_argument("--topk_percent", type=float, default=1.0)
    return parser


def main():
    args = build_parser().parse_args()
    args.tracer = args.tracer.upper()
    modalities = parse_modalities(args.modalities)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args, modalities, device)
    run_selected(args, model, modalities, device)


if __name__ == "__main__":
    main()
