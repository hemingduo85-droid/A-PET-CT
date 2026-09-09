import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    from thop import clever_format, profile
except ImportError:
    clever_format = None
    profile = None

from dataset import MVTecDataset
from models.uad import INP_Former
from train import build_model, build_transforms, parse_modalities, resolve_path, resolve_tracer_root
from utils import build_anomaly_map, compute_pro, get_gaussian_kernel, get_logger, setup_seed


def f1_score_max(y_true, y_score):
    precs, recs, _ = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    return float(f1s[:-1].max()) if len(f1s) > 1 else 0.0


def best_threshold(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    if len(thrs) == 0:
        return float(np.percentile(y_score, 99.5))
    f1s = 2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1] + 1e-7)
    return float(thrs[int(np.argmax(f1s))])


def dice_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 0.0
    return float((2 * np.logical_and(pred, gt).sum()) / (denom + eps))


def hd95_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return 0.0
    pred_pts = np.column_stack(np.where(pred))
    gt_pts = np.column_stack(np.where(gt))
    return float(max(directed_hausdorff(pred_pts, gt_pts)[0], directed_hausdorff(gt_pts, pred_pts)[0]))


def assd_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return 0.0

    pred_distance = distance_transform_edt(~pred)
    gt_distance = distance_transform_edt(~gt)
    pred_to_gt = gt_distance[pred]
    gt_to_pred = pred_distance[gt]
    if pred_to_gt.size == 0 or gt_to_pred.size == 0:
        return 0.0
    return float((pred_to_gt.mean() + gt_to_pred.mean()) / 2.0)


def ppv_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp + eps))


def sensitivity_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn + eps))


def load_weights(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=True)


def safe_aupro(gt_px, pr_px):
    masks = (gt_px > 0.5).astype(np.uint8)
    if masks.max() == 0:
        return 0.0
    try:
        return float(compute_pro(masks, pr_px))
    except Exception:
        return 0.0


def calculate_model_complexity(model, device, input_size=(3, 448, 448)):
    try:
        if profile is None or clever_format is None:
            raise ImportError("thop is not installed")
        dummy_input = torch.randn(1, *input_size).to(device)
        flops_list, params_list = [], []
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                flops, params = profile(
                    model,
                    inputs=(dummy_input,),
                    verbose=False,
                    custom_ops={INP_Former: None},
                )
                flops_list.append(flops)
                params_list.append(params)

        flops_mean = float(np.mean(flops_list))
        flops_std = float(np.std(flops_list))
        params_mean = float(np.mean(params_list))
        params_std = float(np.std(params_list))
        flops_mean_str, params_mean_str = clever_format([flops_mean, params_mean], "%.2f")
        flops_std_str, params_std_str = clever_format([flops_std, params_std], "%.2f")
        return {
            "flops_mean": flops_mean_str,
            "flops_std": flops_std_str,
            "params_mean": params_mean_str,
            "params_std": params_std_str,
            "flops_mean_value": flops_mean,
            "params_mean_value": params_mean,
        }
    except Exception as e:
        return {
            "flops_mean": "failed",
            "flops_std": "0",
            "params_mean": "failed",
            "params_std": "0",
            "flops_mean_value": 0.0,
            "params_mean_value": 0.0,
            "error": str(e),
        }


def evaluate_to_csv(model, loader, device, args):
    model.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    gt_px, pr_px, gt_sp, pr_sp = [], [], [], []
    per_image_rows = []
    abnormal_rows = []

    with torch.no_grad():
        for img, gt, label, img_path in tqdm(loader, ncols=80, desc="Evaluating"):
            img = img.to(device)
            en, de = model(img)[:2]
            amap = build_anomaly_map(model, en, de, img.shape[-1], args.anomaly_source)
            if args.resize_mask:
                amap = F.interpolate(amap, size=args.resize_mask, mode="bilinear", align_corners=False)
                gt = F.interpolate(gt, size=args.resize_mask, mode="nearest")
            amap = gaussian_kernel(amap)
            gt = (gt > 0.5).float()

            image_score = amap.flatten(1).max(dim=1)[0]
            gt_px.append(gt)
            pr_px.append(amap.cpu())
            gt_sp.append(label)
            pr_sp.append(image_score.cpu())

            flat_map = amap.flatten(1)
            flat_gt = gt.flatten(1)
            for i in range(img.shape[0]):
                base = Path(img_path[i]).stem
                label_i = int(label[i].item())
                gt_i = gt[i, 0].cpu().numpy().astype(np.uint8)
                amap_i = amap[i, 0].cpu().numpy()
                threshold = best_threshold(gt_i.ravel(), amap_i.ravel()) if gt_i.any() else np.percentile(amap_i, 99.5)
                pred_i = binary_fill_holes(amap_i >= threshold).astype(np.uint8)

                per_image_rows.append({
                    "path": img_path[i],
                    "filename": base,
                    "label": label_i,
                    "image_score": float(image_score[i].cpu().item()),
                    "mask_pixels": int(flat_gt[i].sum().cpu().item()),
                    "map_min": float(flat_map[i].min().cpu().item()),
                    "map_mean": float(flat_map[i].mean().cpu().item()),
                    "map_max": float(flat_map[i].max().cpu().item()),
                })

                if label_i == 1:
                    abnormal_rows.append({
                        "filename": base,
                        "dataset": args.tracer,
                        "method": "INPFormer",
                        "image_score": float(image_score[i].cpu().item()),
                        "PPV": ppv_score(pred_i, gt_i),
                        "DSC": dice_score(pred_i, gt_i),
                        "HD95": hd95_score(pred_i, gt_i),
                        "ASSD": assd_score(pred_i, gt_i),
                        "Sensitive": sensitivity_score(pred_i, gt_i),
                    })

    gt_px = torch.cat(gt_px, dim=0)[:, 0].numpy()
    pr_px = torch.cat(pr_px, dim=0)[:, 0].numpy()
    gt_sp = torch.cat(gt_sp).flatten().numpy()
    pr_sp = torch.cat(pr_sp).flatten().numpy()

    metrics = {
        "I-AUROC": float(roc_auc_score(gt_sp, pr_sp)),
        "I-AP": float(average_precision_score(gt_sp, pr_sp)),
        "I-F1": f1_score_max(gt_sp, pr_sp),
        "P-AUROC": float(roc_auc_score(gt_px.ravel(), pr_px.ravel())),
        "P-AP": float(average_precision_score(gt_px.ravel(), pr_px.ravel())),
        "P-F1": f1_score_max(gt_px.ravel(), pr_px.ravel()),
        "P-AUPRO": safe_aupro(gt_px, pr_px),
    }
    metrics.update(summarize_abnormal_metrics(abnormal_rows))
    return metrics, per_image_rows, abnormal_rows


def summarize_abnormal_metrics(rows):
    summary = {}
    key_map = {
        "DSC": "dice",
        "HD95": "hd95",
        "ASSD": "assd",
        "PPV": "ppv",
        "Sensitive": "sensitive",
    }
    for key, out_name in key_map.items():
        values = np.array([row[key] for row in rows], dtype=np.float32)
        if values.size == 0:
            summary[f"{out_name}_mean"] = 0.0
            summary[f"{out_name}_std"] = 0.0
        else:
            summary[f"{out_name}_mean"] = float(values.mean())
            summary[f"{out_name}_std"] = float(values.std())
    summary["baseline_dsc_mean"] = summary["dice_mean"]
    summary["baseline_dsc_std"] = summary["dice_std"]
    summary["baseline_hd95_mean"] = summary["hd95_mean"]
    summary["baseline_hd95_std"] = summary["hd95_std"]
    summary["abnormal_samples_count"] = int(len(rows))
    return summary


def build_results_dict(metrics, medical_metrics, complexity, score_csv, abnormal_csv):
    return {
        "sample_level": {
            "auroc": float(metrics["I-AUROC"]),
            "ap": float(metrics["I-AP"]),
            "f1": float(metrics["I-F1"]),
        },
        "pixel_level": {
            "auroc": float(metrics["P-AUROC"]),
            "ap": float(metrics["P-AP"]),
            "f1_dsc": float(metrics["P-F1"]),
            "aupro": float(metrics["P-AUPRO"]),
        },
        "medical_metrics": medical_metrics,
        "model_complexity": complexity,
        "score_csv_path": str(score_csv),
        "csv_metrics_path": str(abnormal_csv),
        "abnormal_samples_count": int(medical_metrics["abnormal_samples_count"]),
    }


def main(args):
    setup_seed(args.seed)
    args.tracer = args.tracer.upper()
    args.modalities = parse_modalities(args.modalities)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tracer_root = resolve_tracer_root(args.data_root, args.tracer)
    save_root = resolve_path(args.save_dir) / args.tracer / "_".join(args.modalities)
    save_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path).expanduser() if args.model_path else save_root / "model.pth"

    logger = get_logger("INPFormerCSV", save_root)
    print_fn = logger.info
    print_fn(f"Data root: {tracer_root}")
    print_fn(f"Model path: {model_path}")
    print_fn(f"Modalities: {args.modalities}")
    print_fn(f"Device: {device}")
    print_fn(f"Anomaly source: {args.anomaly_source}")
    print_fn("Image score: max")

    data_transform, gt_transform = build_transforms(args.input_size, args.crop_size)
    args.resize_mask = args.input_size if args.crop_size <= 0 else args.crop_size
    dataset = MVTecDataset(
        root=tracer_root,
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
        modalities=args.modalities,
        label_mode=args.label_mode,
        channel_fill=args.channel_fill,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, _ = build_model(args, device)
    load_weights(model, model_path, device)
    complexity = calculate_model_complexity(model, device, input_size=(3, args.input_size, args.input_size))
    metrics, per_image_rows, abnormal_rows = evaluate_to_csv(model, loader, device, args)
    medical_metrics = summarize_abnormal_metrics(abnormal_rows)

    score_csv = save_root / args.score_csv
    abnormal_csv = save_root / args.abnormal_csv
    json_path = save_root / args.json_name
    pd.DataFrame(per_image_rows).to_csv(score_csv, index=False)
    pd.DataFrame(abnormal_rows).to_csv(abnormal_csv, index=False)
    results_dict = build_results_dict(metrics, medical_metrics, complexity, score_csv, abnormal_csv)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    print_fn(
        "\n--- Final Test Results ---\n"
        f"Sample-level:\n"
        f"  AUROC: {metrics['I-AUROC']:.4f}, AP: {metrics['I-AP']:.4f}, F1: {metrics['I-F1']:.4f}\n"
        f"Pixel-level:\n"
        f"  AUROC: {metrics['P-AUROC']:.4f}, AP: {metrics['P-AP']:.4f}, "
        f"F1(DSC): {metrics['P-F1']:.4f}, AUPRO: {metrics['P-AUPRO']:.4f}\n"
        f"Medical Metrics (Abnormal Samples Only, Mean +/- Std):\n"
        f"  Total abnormal samples: {medical_metrics['abnormal_samples_count']}\n"
        f"  Dice:     {medical_metrics['dice_mean']:.4f} +/- {medical_metrics['dice_std']:.4f}\n"
        f"  HD95:     {medical_metrics['hd95_mean']:.4f} +/- {medical_metrics['hd95_std']:.4f}\n"
        f"  ASSD:     {medical_metrics['assd_mean']:.4f} +/- {medical_metrics['assd_std']:.4f}\n"
        f"  PPV:      {medical_metrics['ppv_mean']:.4f} +/- {medical_metrics['ppv_std']:.4f}\n"
        f"  Sensitive:{medical_metrics['sensitive_mean']:.4f} +/- {medical_metrics['sensitive_std']:.4f}\n"
        f"Baseline Medical Metrics (Mean +/- Std):\n"
        f"  DSC (Best Threshold): {medical_metrics['baseline_dsc_mean']:.4f} +/- "
        f"{medical_metrics['baseline_dsc_std']:.4f}\n"
        f"  HD95: {medical_metrics['baseline_hd95_mean']:.4f} +/- {medical_metrics['baseline_hd95_std']:.4f}\n"
        f"Model Complexity:\n"
        f"  FLOPs: {complexity['flops_mean']} (std: {complexity['flops_std']})\n"
        f"  Params: {complexity['params_mean']} (std: {complexity['params_std']})"
    )
    print_fn(f"Saved scores to: {score_csv}")
    print_fn(f"Saved abnormal metrics to: {abnormal_csv}")
    print_fn(f"Saved JSON to: {json_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="INPFormer PET/CT CSV evaluator.")
    parser.add_argument("--data_root", default="../A_data/2d_equal", type=str)
    parser.add_argument("--tracer", default="FDG", choices=["FDG", "PSMA", "fdg", "psma"])
    parser.add_argument("--modalities", default="ct,pet", type=str)
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14", type=str)
    parser.add_argument("--input_size", default=448, type=int)
    parser.add_argument("--crop_size", default=0, type=int)
    parser.add_argument("--INP_num", default=6, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--label_mode", default="folder", choices=["folder", "mask"])
    parser.add_argument("--channel_fill", default="mean_modalities", choices=["zero", "imagenet_mean", "mean_modalities"])
    parser.add_argument("--anomaly_source", default="prototype", choices=["prototype", "reconstruction", "fused"])
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--score_csv", default="scores.csv", type=str)
    parser.add_argument("--abnormal_csv", default="abnormal_metrics.csv", type=str)
    parser.add_argument("--json_name", default="evaluation_results.json", type=str)
    parser.add_argument("--device", default="cuda:5", type=str)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--save_dir", default="./saved_results", type=str)
    parser.add_argument("--seed", default=1, type=int)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
