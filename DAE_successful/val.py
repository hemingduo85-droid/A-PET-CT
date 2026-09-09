import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import PETCTSliceDataset, resolve_tracer_root
from denoising import denoising


PROJECT_DIR = Path(__file__).resolve().parent


def parse_modalities(value):
    return [m.strip().lower() for m in value.split(",") if m.strip()]


def resolve_project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def auc_safe(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def ap_safe(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def best_f1_and_thr(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.0, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    idx = int(np.argmax(f1))
    return float(f1[idx]), float(thresholds[idx])


def confusion(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp, fp, fn


def dsc_score(pred, gt):
    tp, fp, fn = confusion(pred, gt)
    return float(2 * tp / (2 * tp + fp + fn + 1e-8))


def ppv_score(pred, gt):
    tp, fp, _ = confusion(pred, gt)
    return float(tp / (tp + fp + 1e-8))


def hd95_assd(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 999.0, 999.0

    pred_surface = pred ^ binary_erosion(pred)
    gt_surface = gt ^ binary_erosion(gt)
    pred_pts = np.argwhere(pred_surface)
    gt_pts = np.argwhere(gt_surface)
    if pred_pts.size == 0 or gt_pts.size == 0:
        return 999.0, 999.0

    dist = cdist(pred_pts, gt_pts)
    d1 = dist.min(axis=1)
    d2 = dist.min(axis=0)
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95))), float((d1.mean() + d2.mean()) / 2)


def calculate_aupro(anomaly_maps, ground_truth_masks):
    all_scores = np.concatenate([m.reshape(-1) for m in anomaly_maps])
    if all_scores.size == 0 or np.max(all_scores) == np.min(all_scores):
        return 0.0

    thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 200)
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


def find_checkpoint(args, mod, modalities):
    if args.checkpoint:
        return Path(args.checkpoint)
    checkpoint_dir = resolve_project_path(args.checkpoint_dir) / args.tracer.upper() / "_".join(modalities)
    return checkpoint_dir / f"best_{mod}_model.pth"


@torch.no_grad()
def evaluate_dae(args):
    modalities = parse_modalities(args.modalities)
    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = PETCTSliceDataset(
        tracer_root,
        modalities=modalities,
        mode="test",
        target_size=(args.image_size, args.image_size),
        debug_ratio=args.debug_ratio,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print("Test distribution:", dataset.get_class_distribution())

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

    image_scores, image_labels = [], []
    pixel_maps, pixel_masks, filenames = [], [], []

    for batch in tqdm(loader, desc="Validate"):
        residuals = []
        scores = []
        for mod in modalities:
            x = batch["modalities"][mod].to(device)
            recon = models[mod](x)
            if isinstance(recon, (tuple, list)):
                recon = recon[0]
            residuals.append(torch.abs(x - recon))
            scores.append(torch.mean((x - recon) ** 2, dim=(1, 2, 3)))

        anomaly_map = torch.mean(torch.stack(residuals), dim=0).squeeze().cpu().numpy()
        image_scores.append(torch.mean(torch.stack(scores), dim=0).item())
        image_labels.append(float(batch["label"].view(-1)[0].item()))
        pixel_maps.append(anomaly_map)
        pixel_masks.append(batch["mask"].squeeze().cpu().numpy())
        filenames.append(batch["id"][0])

    image_labels = np.asarray(image_labels)
    image_scores = np.asarray(image_scores)
    pixel_scores = np.concatenate([m.reshape(-1) for m in pixel_maps])
    pixel_labels = np.concatenate([m.reshape(-1) for m in pixel_masks])
    pixel_f1, pixel_thr = best_f1_and_thr(pixel_labels, pixel_scores)

    summary = {
        "dataset": args.tracer.upper(),
        "method": "DAE",
        "image_auroc": auc_safe(image_labels, image_scores),
        "image_ap": ap_safe(image_labels, image_scores),
        "image_f1": best_f1_and_thr(image_labels, image_scores)[0],
        "pixel_auroc": auc_safe(pixel_labels, pixel_scores),
        "pixel_ap": ap_safe(pixel_labels, pixel_scores),
        "pixel_f1": pixel_f1,
        "pixel_thr": pixel_thr,
        "aupro": calculate_aupro(pixel_maps, pixel_masks),
    }

    rows = []
    for filename, anomaly_map, gt in zip(filenames, pixel_maps, pixel_masks):
        pred = (anomaly_map >= pixel_thr).astype(np.uint8)
        gt = (gt > 0.5).astype(np.uint8)
        hd95, assd = hd95_assd(pred, gt)
        rows.append(
            {
                "filename": filename,
                "dataset": args.tracer.upper(),
                "method": "DAE",
                "PPV": ppv_score(pred, gt),
                "DSC": dsc_score(pred, gt),
                "HD95": hd95,
                "ASSD": assd,
            }
        )

    output_dir = resolve_project_path(args.output_dir) / args.tracer.upper() / "_".join(modalities)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path) if args.csv_path else output_dir / "dae_slice_metrics.csv"
    summary_path = Path(args.summary_path) if args.summary_path else output_dir / "dae_summary.txt"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("CSV saved:", csv_path)
    print("Summary saved:", summary_path)


def build_parser():
    parser = argparse.ArgumentParser(description="Validate DAE on A_data PET/CT slices.")
    parser.add_argument("--data_root", default="A_data/2d_equal", type=str)
    parser.add_argument("--tracer", default="FDG", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct,pet", help="Comma separated: ct, pet, or ct,pet")
    parser.add_argument("--checkpoint", default=None, type=str, help="Single checkpoint path, mainly for one modality.")
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
    parser.add_argument("--output_dir", default="results", type=str)
    parser.add_argument("--csv_path", default=None, type=str)
    parser.add_argument("--summary_path", default=None, type=str)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--noise_std", default=0.2, type=float)
    parser.add_argument("--noise_res", default=16, type=int)
    parser.add_argument("--device", default="cuda:4", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--debug_ratio", default=1.0, type=float)
    return parser


if __name__ == "__main__":
    evaluate_dae(build_parser().parse_args())
