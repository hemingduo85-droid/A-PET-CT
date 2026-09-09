import csv
import json
import math
import os
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
VALID_MODALITIES = {"pet", "ct"}

def parse_modalities(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]

def top_percent_mean(score_maps, percent=1.0):
    score_maps = np.asarray(score_maps, dtype=np.float32)
    if score_maps.ndim == 2:
        score_maps = score_maps[None]
    flat = score_maps.reshape(score_maps.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * percent / 100.0)))
    return np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)


def hot_output_root(method, tracer, modalities):
    return PROJECT_ROOT / "paper_figures" / method / "sample_hot_figures" / tracer.upper() / modality_output_dir(modalities)


def modality_output_dir(modalities):
    modalities = sorted(m.lower() for m in modalities)
    if modalities == ["ct"]:
        return "CT"
    if modalities == ["pet"]:
        return "PET"
    if modalities == ["ct", "pet"]:
        return "PETCT"
    raise ValueError(f"Unsupported modality combination: {modalities}")


def resolve_path(path, base=PROJECT_ROOT):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    for candidate in (base / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (base / path).resolve()


def selected_csv_path(tracer):
    return PROJECT_ROOT / "paper_figures" /"samples_data"/ tracer.upper() / "selected_samples.csv"

def read_gray(path, image_size=256, resample=Image.BILINEAR):
    image = Image.open(path).convert("L")
    if image_size:
        image = image.resize((image_size, image_size), resample=resample)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path, image_size=256):
    return (read_gray(path, image_size, Image.NEAREST) > 0).astype(np.uint8)


def normalize01(image):
    image = image.astype(np.float32)
    lo = float(np.nanmin(image))
    hi = float(np.nanmax(image))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return (image - lo) / (hi - lo)


def percentile_normalize(image, roi=None, q_low=0.01, q_high=0.995):
    image = image.astype(np.float32)
    values = image[roi.astype(bool)] if roi is not None and np.count_nonzero(roi) > 10 else image.ravel()
    lo = float(np.quantile(values, q_low))
    hi = float(np.quantile(values, q_high))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def display_score_map(score_map, roi=None, mode="percentile", q_low=0.01, q_high=0.995):
    if mode == "none":
        out = score_map.astype(np.float32)
    elif mode == "minmax":
        out = normalize01(score_map)
    elif mode == "percentile":
        out = percentile_normalize(score_map, roi=roi, q_low=q_low, q_high=q_high)
    else:
        raise ValueError(f"Unknown display normalization: {mode}")
    if roi is not None:
        out = out.copy()
        out[roi == 0] = 0
    return out


def best_f1_threshold(gt_mask, score_map):
    from sklearn.metrics import precision_recall_curve
    gt = gt_mask.astype(np.uint8).ravel()
    score = score_map.astype(np.float32).ravel()
    precision, recall, thresholds = precision_recall_curve(gt, score)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    f1 = f1[:-1]
    if len(f1) == 0:
        return 0.5, 0.0
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def patient_id_from_path(path):
    path_str = str(path)
    if "__" in path_str:
        parts = os.path.basename(path_str).split("__")
        if len(parts) >= 2:
            return parts[1]
    return os.path.basename(os.path.dirname(os.path.dirname(path_str)))


def load_selected_samples(csv_path, modalities, image_size=256):
    csv_path = Path(csv_path)
    samples = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case", "patient", "slice", "mask_path"} | {f"{m}_path" for m in modalities}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            images = {m: read_gray(resolve_path(row[f"{m}_path"], csv_path.parent), image_size) for m in modalities}
            mask = load_mask(resolve_path(row["mask_path"], csv_path.parent), image_size)
            samples.append({
                "case": row["case"],
                "patient": row["patient"],
                "slice": row["slice"],
                "case_name": f"{row['case']}_{row['patient']}__{row['slice']}",
                "label": int(str(row["case"]).lower() == "abnormal"),
                "images": images,
                "mask": mask,
            })
    return samples


def connected_components(mask):
    mask = (mask > 0).astype(np.uint8)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    components = []
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
            components.append(coords)
    return components


def remove_small_components(mask, min_pixels):
    if min_pixels <= 0:
        return mask.astype(np.uint8)
    output = np.zeros_like(mask, dtype=np.uint8)
    for component in connected_components(mask):
        if len(component) >= min_pixels:
            for y, x in component:
                output[y, x] = 1
    return output


def save_image(data, output, cmap="gray", vmin=None, vmax=None):
    fig = plt.figure(figsize=(4, 4))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_annotated_image(data, output, modality):
    # -------------------------- 可调布局区域 --------------------------
    # 整张画布大小，单位英寸
    fig = plt.figure(figsize=(4, 4))

    # 主图像区域 [left, bottom, width, height] 范围0~1（画布相对占比）
    # left：图像距离画布左边界
    # bottom：图像距离画布底部（数值变大，图像整体往上移）
    # width / height：图像宽高占画布比例
    ax = plt.Axes(fig, [0.05, 0.15, 0.9, 0.9])
    ax.set_axis_off()
    fig.add_axes(ax)
    # -------------------------------------------------------------------

    if modality == "pet":
        cmap = "gray_r"
        vmin, vmax = 0, 1
    elif modality == "ct":
        cmap = "gray"
        vmin, vmax = 0, 1
    else:
        cmap = "gray"
        vmin, vmax = None, None
        
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    # -------------------------- 色条位置可调 --------------------------
    # 色条区域 [left, bottom, width, height]
    cax = fig.add_axes([0.05, -0.1, 0.9, 0.1])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    # 刻度文字大小
    cbar.ax.tick_params(labelsize=22)
    # -----------------------------------------------------------------
    
    if modality == "pet":
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['0', '1', '2', '3', '4', '5'])
        cax.set_title('SUV', fontsize=22, pad=10)
    elif modality == "ct":
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(['-250', '50', '350'])
        cax.set_title('HU', fontsize=22, pad=10)
        
    # 整张图四周留白大小
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def save_heatmap(data, output, cmap="jet", label="Anomaly score"):
    # 画布尺寸 和 save_annotated_image 完全一致
    fig = plt.figure(figsize=(4, 4))

    # 主图区域 [left, bottom, width, height] 完全照搬PET/CT代码
    ax = plt.Axes(fig, [0.05, 0.15, 0.9, 0.9])
    ax.set_axis_off()
    fig.add_axes(ax)

    # ----------------值域容错（防空图/全NaN/全同值崩溃）----------------
    try:
        data_min = float(np.nanmin(data))
        data_max = float(np.nanmax(data))
    except Exception:
        data_min, data_max = 0.0, 1.0

    if np.isclose(data_min, data_max):
        vmin = np.floor(data_min) - 1
        vmax = np.ceil(data_max) + 1
    else:
        vmin = np.floor(data_min)
        vmax = np.ceil(data_max)
    # ---------------------------------------------------------

    # 修正：Reds改为双向蓝白红bwr色板，适配正负ΔSUV
    if cmap == "Reds":
        # 蓝(-vmin) → 白(0) → 红(vmax)，完美匹配论文ΔSUV图
        cmap_obj = plt.get_cmap("bwr")
        cax = fig.add_axes([0.05, -0.1, 0.9, 0.1])
    else:
        cmap_obj = plt.get_cmap(cmap)
        cax = fig.add_axes([-0.1, -0.1, 1.2, 0.05])
    im = ax.imshow(data, cmap=cmap_obj, vmin=vmin, vmax=vmax)

    
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    # 刻度字号统一22，和PET/CT代码匹配
    cbar.ax.tick_params(labelsize=22)

    # 按色板区分两套刻度规则
    if cmap == "Reds":
        # ΔSUV双向图：最小值、0、最大值 三档整数刻度
        ticks = [vmin, 0, vmax]
        tick_labels = [str(int(t)) for t in ticks]
    else:
        # 普通jet热力图：固定0~1六等分刻度
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        tick_labels = ["0", "0.2", "0.4", "0.6", "0.8", "1"]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)

    # 标题字号、间距和原图统一
    cax.set_title(label, fontsize=20, pad=10)

    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_overlay(base, pred, gt, alpha=0.72):
    rgb = np.repeat((base)[..., None], 3, axis=2)
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = pred & gt
    fn = (~pred) & gt
    fp = pred & (~gt)

    rgb[tp] = rgb[tp] * (1 - alpha) + np.array([0.0, 0.9, 0.0]) * alpha
    rgb[fn] = rgb[fn] * (1 - alpha) + np.array([1.0, 0.05, 0.05]) * alpha
    rgb[fp] = rgb[fp] * (1 - alpha) + np.array([0.0, 0.45, 1.0]) * alpha
    return rgb

def save_threshold_panel(base, pred, gt, output):
    overlay = make_overlay(base, pred, gt)
    fig = plt.figure(figsize=(4, 4))
    # 主图像区域 和 save_annotated_image 完全一致 [left, bottom, w, h]
    ax = plt.Axes(fig, [0.05, 0.15, 0.9, 0.9])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(overlay)
    
    # 标题放大，和其他子图标题视觉匹配
    ax.set_title("Threshold", fontsize=22, pad=10)
    
    legend = [
        Patch(facecolor=[0.0, 0.9, 0.0], label="TP"),
        Patch(facecolor=[1.0, 0.05, 0.05], label="FN"),
        Patch(facecolor=[0.0, 0.45, 1.0], label="FP"),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        framealpha=0.9,
        fontsize=20,        # 图例文字和标题同字号22
        handlelength=1.8,   # 色块拉长匹配大字体
        handleheight=1.3,
        handletextpad=0.5,
        columnspacing=0.75,
        borderpad=0.4
    )
    # 保存参数统一
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_overview(output_dir, name, images, gt, pred, score_map, threshold_overlay, dice, score):
    image_panels = []
    if "pet" in images:
        image_panels.append(("PET", images["pet"], "magma", None, None))
    if "ct" in images:
        image_panels.append(("CT", images["ct"], "gray", None, None))
    if not image_panels:
        base_name, base_image = next(iter(images.items()))
        image_panels.append((base_name.upper(), base_image, "gray", None, None))

    panels = [
        *image_panels,
        ("GT", gt, "gray", 0, 1),
        ("Pred", pred, "gray", 0, 1),
        ("Heatmap", score_map, "jet", 0, 1),
        ("Anomaly map", score_map, "Reds", 0, 1),
        ("Threshold", threshold_overlay, None, None, None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 2.6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, data, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    fig.suptitle(f"{name} | Dice={dice:.4f} | score={score:.4f}", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def dice_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def save_case_outputs(
    case_out,
    images,
    gt,
    raw_map,
    display_map,
    threshold,
    score=None,
    case_name=None,
    roi=None,
    min_component_pixels=0,
):
    case_out.mkdir(parents=True, exist_ok=True)
    pred = (raw_map >= threshold).astype(np.uint8)
    if roi is not None:
        pred = (pred & (roi > 0)).astype(np.uint8)
    pred = remove_small_components(pred, min_component_pixels)
    if "pet" in images:
        save_annotated_image(images["pet"], case_out / "pet_original.png", modality="pet")
    if "ct" in images:
        save_annotated_image(images["ct"], case_out / "ct_original.png", modality="ct")
    save_image(gt, case_out / "gt_mask.png", "gray", 0, 1)
    legacy_raw_pred = case_out / "raw_pred_mask.png"
    if legacy_raw_pred.exists():
        legacy_raw_pred.unlink()
    save_image(pred, case_out / "pred_mask.png", "gray", 0, 1)
    save_heatmap(display_map, case_out / "pred_heatmap.png", cmap="jet", label="Anomaly score")
    save_heatmap(display_map, case_out / "anomaly_map.png", cmap="Reds", label="Anomaly score")
    base = images.get("pet", images.get("ct", next(iter(images.values()))))
    save_threshold_panel(base, pred, gt, case_out / "threshold_map.png")
    threshold_overlay = make_overlay(base, pred, gt)
    if score is None:
        score = float(top_percent_mean(raw_map, 1.0)[0])
    save_overview(
        case_out,
        case_name or case_out.name,
        images,
        gt,
        pred,
        display_map,
        threshold_overlay,
        dice_score(pred, gt),
        float(score),
    )
    return pred


def save_cache(cache_file, method, tracer, modalities, filenames, labels, scores, maps, masks, topk_percent):
    np.savez_compressed(
        cache_file,
        schema_version=2,
        method=method,
        tracer=tracer.upper(),
        modalities=",".join(modalities),
        score_source="topk_from_map",
        topk_percent=float(topk_percent),
        filenames=np.asarray(filenames, dtype=object),
        slice_labels=np.asarray(labels, dtype=np.int64),
        slice_scores=np.asarray(scores, dtype=np.float32),
        pixel_maps=np.asarray(maps, dtype=np.float32),
        pixel_masks=np.asarray(masks, dtype=np.uint8),
    )

def compute_pro(masks, amaps, num_th=200):
    import pandas as pd
    from skimage import measure
    from sklearn.metrics import auc

    masks = np.asarray(masks, dtype=np.uint8)
    amaps = np.asarray(amaps, dtype=np.float32)
    if masks.ndim == 2:
        masks = masks[None]
        amaps = amaps[None]

    abnormal_indices = np.where(masks.max(axis=(1, 2)) > 0)[0]
    if len(abnormal_indices) == 0:
        return 0.0

    masks = masks[abnormal_indices]
    amaps = amaps[abnormal_indices]

    min_th = amaps.min()
    max_th = amaps.max()
    if max_th <= min_th:
        return 0.0

    delta = (max_th - min_th) / num_th
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])

    labeled_masks = [measure.label(mask) for mask in masks]
    regions_list = [measure.regionprops(labeled) for labeled in labeled_masks]

    inverse_masks = 1 - masks
    inverse_masks_sum = inverse_masks.sum() + 1e-8

    for th in np.arange(min_th, max_th, delta):
        binary_amaps = (amaps > th)
        pros = []

        for binary_amap, regions in zip(binary_amaps, regions_list):
            for region in regions:
                coords = region.coords
                tp_pixels = binary_amap[coords[:, 0], coords[:, 1]].sum()
                pros.append(tp_pixels / region.area)

        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks_sum

        if pros:
            df = pd.concat([df, pd.DataFrame([{
                'pro': float(np.mean(pros)),
                'fpr': float(fpr),
                'threshold': float(th)
            }])], ignore_index=True)

    if len(df) < 2:
        return 0.0

    df = df[df["fpr"] < 0.3]
    if df.empty:
        return 0.0

    df["fpr"] = df["fpr"] / df["fpr"].max()
    try:
        return float(auc(df["fpr"], df["pro"]))
    except Exception:
        return 0.0



def eval_protocol_compute_metrics(gt_labels, gt_masks, anomaly_maps, image_scores, paths):
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

    def f1_score_max(y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if len(np.unique(y_true)) < 2:
            return 0.0
        p, r, _ = precision_recall_curve(y_true, y_score)
        f1 = 2 * p * r / (p + r + 1e-7)
        f1 = f1[:-1]
        return float(f1.max()) if len(f1) else 0.0

    def safe_auroc(y_true, y_score):
        y_true = np.asarray(y_true)
        if len(np.unique(y_true)) < 2:
            return 0.0
        return float(roc_auc_score(y_true, y_score))

    def safe_ap(y_true, y_score):
        y_true = np.asarray(y_true)
        if len(np.unique(y_true)) < 2:
            return 0.0
        return float(average_precision_score(y_true, y_score))

    def _sens_at_spec(y_true, y_score, spec):
        y_true = np.asarray(y_true)
        if len(np.unique(y_true)) < 2:
            return 0.0
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ok = fpr <= (1.0 - spec)
        return float(tpr[ok].max()) if ok.any() else 0.0

    def pixel_f1_abnormal_minmax(gt_labels, gt_masks, anomaly_maps):
        gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
        gt_masks = np.asarray(gt_masks, dtype=np.float32)
        maps = np.asarray(anomaly_maps, dtype=np.float32)
        if gt_masks.ndim == 4:
            gt_masks = gt_masks.squeeze(1)
        if maps.ndim == 4:
            maps = maps.squeeze(1)

        keep = gt_labels == 1
        if not np.any(keep):
            return 0.0

        y_true = gt_masks[keep].reshape(-1).astype(bool)
        y_score = maps[keep].reshape(-1).astype(np.float64)
        if y_score.size == 0 or len(np.unique(y_true)) < 2:
            return 0.0

        score_min = float(np.nanmin(y_score))
        score_max = float(np.nanmax(y_score))
        y_score = (y_score - score_min) / (score_max - score_min + 1e-8)
        return f1_score_max(y_true, y_score)

    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    image_scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    gt_px_abn = gt_masks[gt_labels == 1].reshape(-1).astype(bool)
    pr_px_abn = maps[gt_labels == 1].reshape(-1).astype(np.float64)
    px_auroc_abn = safe_auroc(gt_px_abn, pr_px_abn)
    px_ap_abn = safe_ap(gt_px_abn, pr_px_abn)
    
    # Compute AUPRO
    masks_abn = gt_masks[gt_labels == 1]
    maps_abn = maps[gt_labels == 1]
    if masks_abn.size > 0:
        px_aupro_abn = compute_pro(masks_abn, maps_abn)
    else:
        px_aupro_abn = 0.0

    px_f1_abn = pixel_f1_abnormal_minmax(gt_labels, gt_masks, maps)

    slice_metrics = {
        "img_auroc": safe_auroc(gt_labels, image_scores),
        "img_ap": safe_ap(gt_labels, image_scores),
        "img_f1": f1_score_max(gt_labels, image_scores),
        "px_auroc_abn": px_auroc_abn,
        "px_ap_abn": px_ap_abn,
        "px_aupro_abn": px_aupro_abn,
        "px_f1_abn": px_f1_abn,
        "_pr_sp": image_scores,
        "_gt_sp": gt_labels,
    }

    patient_scores = defaultdict(list)
    patient_labels = defaultdict(list)
    for path, score, label_value in zip(paths, image_scores, gt_labels):
        pid = patient_id_from_path(path)
        patient_scores[pid].append(float(score))
        patient_labels[pid].append(int(label_value))
    pat_score, pat_label = [], []
    for pid in patient_scores:
        pat_score.append(float(np.max(patient_scores[pid])))
        pat_label.append(int(np.max(patient_labels[pid])))
    pat_score = np.asarray(pat_score, dtype=np.float64)
    pat_label = np.asarray(pat_label, dtype=np.int32)
    pat_metrics = {
        "pat_auroc": safe_auroc(pat_label, pat_score),
        "pat_ap": safe_ap(pat_label, pat_score),
        "pat_f1": f1_score_max(pat_label, pat_score),
        "pat_sens90": _sens_at_spec(pat_label, pat_score, 0.90),
        "pat_sens95": _sens_at_spec(pat_label, pat_score, 0.95),
        "n_patients": len(pat_label),
        "n_abn_patients": int(pat_label.sum()),
    }
    return slice_metrics, pat_metrics


def eval_protocol_format_metrics(slice_metrics, pat_metrics, as_percent=True):
    s = 100.0 if as_percent else 1.0
    m = slice_metrics
    lines = [
        f"[Slice-Img]  AUROC={m['img_auroc']*s:.2f}%  AP={m['img_ap']*s:.2f}%  F1={m['img_f1']*s:.2f}%",
        f"[Slice-Px(abn)]  AUROC={m['px_auroc_abn']*s:.2f}%  AP={m['px_ap_abn']*s:.2f}%  F1={m['px_f1_abn']*s:.2f}% AUPRO={m['px_aupro_abn']*s:.2f}%",
    ]
    p = pat_metrics
    lines.append(
        f"[Patient({p['n_abn_patients']}/{p['n_patients']}abn)]  AUROC={p['pat_auroc']*s:.2f}%  "
        f"AP={p['pat_ap']*s:.2f}%  F1={p['pat_f1']*s:.2f}%  "
        f"Sens@90Spec={p['pat_sens90']*s:.2f}%  Sens@95Spec={p['pat_sens95']*s:.2f}%"
    )
    return "\n".join(lines)

def cache_output_paths(method, tracer, modalities):
    run_tag = f"{tracer.upper()}_{'_'.join(modalities)}"
    cache_dir = PROJECT_ROOT / "paper_figures" / method / "plot_cache" / run_tag
    out_dir = PROJECT_ROOT / "paper_figures" / method / "full_evaluation" / run_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "test_cache.npz", cache_dir / "metrics.json", out_dir
