import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import PETCTSliceDataset, resolve_tracer_root
from denoising import denoising

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_eval_utils import (
    best_f1_threshold,
    cache_output_paths,
    display_score_map,
    hot_output_root,
    load_selected_samples,
    parse_modalities,
    save_cache,
    save_case_outputs,
    selected_csv_path,
    top_percent_mean,
)
from dae_eval_protocol import eval_protocol_compute_metrics, eval_protocol_format_metrics


METHOD = "DAE"
PROJECT_DIR = Path(__file__).resolve().parent


def resolve_project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    for candidate in (PROJECT_DIR / path, PROJECT_DIR.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_DIR / path).resolve()


def find_checkpoint(args, mod, modalities):
    if args.checkpoint:
        return Path(args.checkpoint)
    return resolve_project_path(args.checkpoint_dir) / args.tracer.upper() / "_".join(modalities) / f"best_{mod}_model.pth"


def load_models(args, modalities, device):
    models = {}
    for mod in modalities:
        ckpt_path = find_checkpoint(args, mod, modalities)
        wrapper = denoising(
            identifier=f"dae_{mod}",
            n_input=1,
            noise_std=args.noise_std,
            noise_res=args.noise_res,
            device=device,
        )
        wrapper.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        wrapper.model.to(device).eval()
        models[mod] = wrapper.model
        print(f"Loaded {mod}: {ckpt_path}")
    return models


@torch.no_grad()
def infer_map(models, images, modalities, device):
    residuals = []
    for mod in modalities:
        arr = torch.from_numpy(images[mod]).float().unsqueeze(0).unsqueeze(0).to(device)
        recon = models[mod](arr)
        if isinstance(recon, (tuple, list)):
            recon = recon[0]
        residuals.append(torch.abs(arr - recon))
    return torch.mean(torch.stack(residuals), dim=0).squeeze().cpu().numpy().astype(np.float32)


@torch.no_grad()
def generate_cache(args, models, modalities, device):
    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    dataset = PETCTSliceDataset(
        tracer_root,
        modalities=modalities,
        mode="test",
        target_size=(args.image_size, args.image_size),
        debug_ratio=args.debug_ratio,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    filenames, labels, maps, masks = [], [], [], []
    for batch in tqdm(loader, desc=f"{METHOD} cache", ncols=80):
        images = {mod: batch["modalities"][mod].squeeze().numpy() for mod in modalities}
        maps.append(infer_map(models, images, modalities, device))
        masks.append((batch["mask"].squeeze().numpy() > 0.5).astype(np.uint8))
        labels.append(int(batch["label"].view(-1)[0].item()))
        filenames.append(str(batch["id"][0]))
    maps = np.stack(maps).astype(np.float32)
    masks = np.stack(masks).astype(np.uint8)
    labels = np.asarray(labels, dtype=np.int64)
    scores = top_percent_mean(maps, args.topk_percent)
    cache_file, metrics_file, _ = cache_output_paths(METHOD, args.tracer, modalities)
    save_cache(cache_file, METHOD, args.tracer, modalities, filenames, labels, scores, maps, masks, args.topk_percent)

    slice_metrics, pat_metrics = eval_protocol_compute_metrics(
        labels,
        masks,
        maps,
        scores,
        filenames,
        bootstrap_iters=500,
        progress_callback=print,
    )
    formatted_metrics = eval_protocol_format_metrics(slice_metrics, pat_metrics)
    metrics_payload = {
        "method": METHOD,
        "tracer": args.tracer.upper(),
        "modalities": modalities,
        "score_source": "topk_from_map",
        "topk_percent": float(args.topk_percent),
        "slice_metrics": slice_metrics,
        "pat_metrics": pat_metrics,
    }
    with open(metrics_file, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    print(f"Cache saved: {cache_file}")
    print(f"Metrics saved: {metrics_file}")
    print(formatted_metrics)


def run_selected(args, models, modalities, device):
    csv_path = Path(args.selected_csv) if args.selected_csv else selected_csv_path(args.tracer)
    samples = load_selected_samples(csv_path, modalities, image_size=args.image_size)
    out_root = hot_output_root(METHOD, args.tracer, modalities)
    out_root.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(samples, 1):
        raw_map = infer_map(models, sample["images"], modalities, device)
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
    parser = argparse.ArgumentParser(description="DAE paper inference/cache for PET/CT comparisons.")
    parser.add_argument("--data_root", default="A_data/2d_equal_mask50")
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct,pet")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--selected_csv", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--noise_res", type=int, default=16)
    parser.add_argument("--roi_min", type=float, default=0.03)
    parser.add_argument("--score_q_low", type=float, default=0.01)
    parser.add_argument("--score_q_high", type=float, default=0.995)
    parser.add_argument("--heatmap_norm", default="percentile", choices=["percentile", "minmax", "none"])
    parser.add_argument("--threshold_mode", default="gt_best_f1", choices=["fixed", "gt_best_f1"])
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--min_component_pixels", type=int, default=0)
    parser.add_argument("--topk_percent", type=float, default=1.0)
    parser.add_argument("--debug_ratio", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--make_cache", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.tracer = args.tracer.upper()
    modalities = parse_modalities(args.modalities)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = load_models(args, modalities, device)
    if args.make_cache:
        generate_cache(args, models, modalities, device)
    else:
        run_selected(args, models, modalities, device)


if __name__ == "__main__":
    main()
