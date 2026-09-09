"""Evaluate + optional visualization for ESC-DRKD on 2d_equal.

Default: metrics on full test set + matplotlib panels for abnormal slices only.
--visualize all  : save every test slice
--visualize none : metrics only
--sample_ids     : restrict to listed ids (like legacy dyx_ESC_view target list)
"""

import argparse
import csv
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import MultimodalDataset, collate_fn, modality_tag, resolve_tracer_root
from infer_utils import build_model, forward_batch, load_student_weights
from metrics import ap_safe, auc_safe, best_f1_and_thr, calculate_aupro
from utils import print_log
from visualize import load_display_image, visualize_sample


def parse_modalities(value):
    return [m.strip().lower() for m in value.split(",") if m.strip()]


def resolve_device(device_str):
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="ESC-DRKD eval + visualization")
    parser.add_argument("--data_root", default="A_data/2d_equal", type=str)
    parser.add_argument("--tracer", default="FDG", choices=["FDG", "PSMA"])
    parser.add_argument("--modalities", default="ct,pet", type=parse_modalities)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--save_dir", default="checkpoints", type=str)
    parser.add_argument("--vis_root", default="visualization_results", type=str)
    parser.add_argument(
        "--visualize",
        choices=["abnormal", "all", "none"],
        default="abnormal",
        help="abnormal=only test abnormal; all=all test slices; none=skip figures",
    )
    parser.add_argument(
        "--sample_ids",
        default=None,
        type=str,
        help="Comma-separated ids: normal__patient__slice or abnormal__...",
    )
    parser.add_argument("--high_dice_threshold", default=0.9, type=float)
    args = parser.parse_args()

    device = resolve_device(args.device)
    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    tag = modality_tag(args.modalities)
    run_dir = Path(args.save_dir) / args.tracer.upper() / tag
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"

    target_ids = [x.strip() for x in args.sample_ids.split(",") if x.strip()] if args.sample_ids else None

    vis_dir = Path(args.vis_root) / args.tracer.upper() / tag
    vis_all_dir = vis_dir / "all_samples"
    vis_high_dir = vis_dir / "high_dice"
    if args.visualize != "none":
        vis_all_dir.mkdir(parents=True, exist_ok=True)
        vis_high_dir.mkdir(parents=True, exist_ok=True)

    log = (run_dir / "view.log").open("w", encoding="utf-8")
    test_set = MultimodalDataset(
        tracer_root,
        modalities=args.modalities,
        mode="test",
        img_size=args.img_size,
        target_sample_ids=target_ids,
    )
    loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=partial(collate_fn, modalities=args.modalities),
    )

    model = build_model(args.modalities, device)
    print_log(f"Loading {ckpt}", log)
    load_student_weights(model, ckpt, device)
    model.eval()

    gt_labels, pred_scores, pred_maps, gt_masks = [], [], [], []
    for batch in tqdm(loader, desc="Forward"):
        anomaly_map, _ = forward_batch(model, batch, args.modalities, device, args.img_size)
        gt_labels.append(int(batch["label"].item()))
        pred_scores.append(float(anomaly_map.max().item()))
        pred_maps.append(anomaly_map.squeeze().cpu().numpy())
        gt_masks.append(batch["mask"].squeeze().cpu().numpy())

    pixel_thr, _ = best_f1_and_thr(
        np.concatenate([m.reshape(-1) for m in pred_maps]),
        np.concatenate([(m > 0.5).astype(np.uint8).reshape(-1) for m in gt_masks]),
    )

    vis_count = 0
    per_sample_rows = []
    if args.visualize != "none":
        for batch in tqdm(loader, desc="Visualize"):
            gt_label = int(batch["label"].item())
            if args.visualize == "abnormal" and gt_label != 1:
                continue

            anomaly_map, _ = forward_batch(model, batch, args.modalities, device, args.img_size)
            pred_map = anomaly_map.squeeze().cpu().numpy()
            gt_mask = batch["mask"].squeeze().cpu().numpy()
            sample_id = batch["id"][0]
            img_path = batch.get("img_path", [None])[0]
            display = load_display_image(img_path, args.img_size)

            folder, dice = visualize_sample(
                sample_id,
                display,
                pred_map,
                gt_mask,
                pixel_thr,
                str(vis_all_dir),
                save_single_images=True,
                high_dice_dir=str(vis_high_dir) if gt_label == 1 else None,
                high_dice_threshold=args.high_dice_threshold,
            )
            vis_count += 1
            per_sample_rows.append({"id": sample_id, "label": gt_label, "dice": dice, "folder": folder})

    image_labels = (np.asarray(gt_labels) > 0).astype(int)
    pred_scores = np.asarray(pred_scores)
    summary = {
        "image_auroc": auc_safe(image_labels, pred_scores),
        "image_ap": ap_safe(image_labels, pred_scores),
        "aupro": calculate_aupro(pred_maps, gt_masks),
        "pixel_threshold": pixel_thr,
        "num_test": len(gt_labels),
        "visualized_samples": vis_count,
    }
    for key, value in summary.items():
        print_log(f"{key}: {value}", log)

    if per_sample_rows:
        csv_path = vis_dir / "visualized_samples.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=per_sample_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_sample_rows)

    log.close()
    print(f"Done. Visualized {vis_count} / {len(gt_labels)} test samples -> {vis_all_dir}")


if __name__ == "__main__":
    main()
