import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from new_model import ESC_DRKD_Multimodal
from train import PETCTSliceDataset, collate_fn, fuse_modalities, get_anomaly_map, resolve_tracer_root

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_eval_utils import (
    best_f1_threshold,
    cache_output_paths,
    compute_metrics,
    display_score_map,
    find_checkpoint,
    hot_output_root,
    load_selected_samples,
    parse_modalities,
    save_cache,
    save_case_outputs,
    selected_csv_path,
    top_percent_mean,
    write_metrics,
    resolve_path
)


METHOD = "ESC"
PROJECT_DIR = Path(__file__).resolve().parent


def load_model(args, modalities, device):
    model = ESC_DRKD_Multimodal(modalities=modalities).to(device).eval()
    ckpt = torch.load(find_checkpoint(args, modalities), map_location=device)
    state = ckpt.get("student", ckpt.get("model", ckpt.get("state_dict", ckpt))) if isinstance(ckpt, dict) else ckpt
    model.student.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {find_checkpoint(args, modalities)}")
    return model


@torch.no_grad()
def infer_map(model, images, modalities, device, image_size):
    x_dict = {m: torch.from_numpy(images[m]).float().unsqueeze(0).unsqueeze(0).to(device) for m in modalities}
    x = fuse_modalities(x_dict, modalities)
    t_features = model.teacher.backbone(x)
    reconstruction = model.student(t_features[-1], t_features[:-1])
    amap = get_anomaly_map(model, t_features, reconstruction, image_size)
    return amap.squeeze().detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def generate_cache(args, model, modalities, device):
    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    dataset = PETCTSliceDataset(tracer_root, modalities, mode="test", img_size=args.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    filenames, labels, maps, masks = [], [], [], []
    for batch in tqdm(loader, desc=f"{METHOD} cache", ncols=80):
        images = {m: batch["modalities"][m].squeeze().numpy() for m in modalities}
        maps.append(infer_map(model, images, modalities, device, args.image_size))
        masks.append((batch["mask"].squeeze().numpy() > 0.5).astype(np.uint8))
        labels.append(int(batch["label"].view(-1)[0].item()))
        filenames.append(str(batch["id"][0]))
    maps = np.stack(maps).astype(np.float32)
    masks = np.stack(masks).astype(np.uint8)
    scores = top_percent_mean(maps, args.topk_percent)
    cache_file, metrics_file, _ = cache_output_paths(METHOD, args.tracer, modalities)
    save_cache(cache_file, METHOD, args.tracer, modalities, filenames, labels, scores, maps, masks, args.topk_percent)
    metrics = compute_metrics(labels, scores, masks, maps)
    metrics.update({"method": METHOD, "score_source": "topk_from_map", "topk_percent": args.topk_percent})
    write_metrics(metrics_file, metrics)
    print(f"Cache saved: {cache_file}")


def run_selected(args, model, modalities, device):
    csv_path = Path(args.selected_csv) if args.selected_csv else selected_csv_path(args.tracer)
    samples = load_selected_samples(csv_path, modalities, image_size=args.image_size)
    out_root = hot_output_root(METHOD, args.tracer, modalities)
    out_root.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(samples, 1):
        raw_map = infer_map(model, sample["images"], modalities, device, args.image_size)
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
    parser = argparse.ArgumentParser(description="ESC-DRKD paper inference/cache for PET/CT comparisons.")
    parser.add_argument("--data_root", default="../A_data/2d_equal_mask50")
    parser.add_argument("--tracer", default="PSMA", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--selected_csv", default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--roi_min", type=float, default=0.03)
    parser.add_argument("--score_q_low", type=float, default=0.01)
    parser.add_argument("--score_q_high", type=float, default=0.995)
    parser.add_argument("--heatmap_norm", default="percentile", choices=["percentile", "minmax", "none"])
    parser.add_argument("--threshold_mode", default="gt_best_f1", choices=["fixed", "gt_best_f1"])
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--min_component_pixels", type=int, default=0)
    parser.add_argument("--topk_percent", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--make_cache", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.tracer = args.tracer.upper()
    modalities = parse_modalities(args.modalities)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args, modalities, device)
    if args.make_cache:
        generate_cache(args, model, modalities, device)
    else:
        run_selected(args, model, modalities, device)


if __name__ == "__main__":
    main()
