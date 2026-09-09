from collections import defaultdict

from sklearn import metrics
from skimage import measure

import cv2
import numpy as np
import os
import pandas as pd

BOOTSTRAP_ITERATIONS = 500
BOOTSTRAP_SEED = 20260717
HISTOGRAM_BINS = 16384


def _bootstrap_ci(labels, scores, metric_fn, n_bootstraps=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        value = float(metric_fn(labels, scores))
        return value, value

    rng = np.random.default_rng(seed)
    values = []
    n = len(labels)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        sample_labels = labels[idx]
        if len(np.unique(sample_labels)) < 2:
            continue
        values.append(float(metric_fn(sample_labels, scores[idx])))
    if not values:
        value = float(metric_fn(labels, scores))
        return value, value
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _metrics_from_hist(pos_hist, neg_hist):
    tp = pos_hist[::-1].astype(np.float64, copy=False)
    fp = neg_hist[::-1].astype(np.float64, copy=False)
    total_pos = float(tp.sum())
    total_neg = float(fp.sum())
    if total_pos <= 0 or total_neg <= 0:
        return 0.0, 0.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    tpr = cum_tp / total_pos
    fpr = cum_fp / total_neg
    auroc = float(np.trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))

    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)
    aupr = float(np.sum(precision * (tp / total_pos)))
    return auroc, aupr


def _pixel_slice_histogram_bootstrap(
        labels,
        masks,
        maps,
        n_bootstraps=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
        bins=HISTOGRAM_BINS,
        exact_auroc=None,
        exact_aupr=None,
        progress=False,
):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    masks = np.asarray(masks)
    maps = np.asarray(maps)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    keep = labels == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0 or n_bootstraps <= 0:
        return (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0))

    score_min = float(np.min(maps[keep]))
    score_max = float(np.max(maps[keep]))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    scale = (bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        slice_scores = maps[idx].reshape(-1)
        slice_mask = masks[idx].reshape(-1).astype(bool)
        bin_idx = np.floor((slice_scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[slice_mask], minlength=bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~slice_mask], minlength=bins).astype(np.uint32)

    rng = np.random.default_rng(seed)
    aurocs = []
    auprs = []
    n_slices = len(abn_idx)
    for _ in range(n_bootstraps):
        sample = rng.integers(0, n_slices, size=n_slices)
        weights = np.bincount(sample, minlength=n_slices).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if progress and (len(aurocs) % 50 == 0 or len(aurocs) == n_bootstraps):
            print(f"Pixel histogram slice bootstrap: {len(aurocs)}/{n_bootstraps}", flush=True)

    if not aurocs:
        auroc_value = 0.0 if exact_auroc is None else float(exact_auroc)
        aupr_value = 0.0 if exact_aupr is None else float(exact_aupr)
        return (auroc_value, (auroc_value, auroc_value)), (aupr_value, (aupr_value, aupr_value))

    auroc_value = float(np.mean(aurocs)) if exact_auroc is None else float(exact_auroc)
    aupr_value = float(np.mean(auprs)) if exact_aupr is None else float(exact_aupr)
    auroc_ci = tuple(float(v) for v in np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = tuple(float(v) for v in np.percentile(auprs, [2.5, 97.5]))
    return (auroc_value, auroc_ci), (aupr_value, aupr_ci)


def _average_precision(labels, scores):
    return _safe_ap(labels, scores)


def _add_metric_with_ci(result, key, value, ci):
    result[key] = float(value)
    result[f"{key}_ci_low"] = float(ci[0])
    result[f"{key}_ci_high"] = float(ci[1])


def format_metric_with_ci(row, key, label=None):
    label = label or key
    value = row[key] * 100
    low = row.get(f"{key}_ci_low")
    high = row.get(f"{key}_ci_high")
    if low is None or high is None:
        return f"{label}={value:.2f}%"
    return f"{label}={value:.2f}% (95% CI {low * 100:.2f}-{high * 100:.2f}%)"


def format_image_metrics(row):
    return "[Slice-Img]  " + "  ".join([
        format_metric_with_ci(row, "image_auroc", "AUROC"),
        format_metric_with_ci(row, "image_ap", "AUPR"),
        format_metric_with_ci(row, "image_f1", "F1"),
    ])


def format_pixel_metrics(row):
    return "[Slice-Px(abn)]  " + "  ".join([
        format_metric_with_ci(row, "pixel_auroc_abn", "AUROC"),
        format_metric_with_ci(row, "pixel_aupr_abn", "AUPR"),
    ])


def format_patient_metrics(row):
    return f"[Patient({int(row.get('n_abn_patients', 0))}/{int(row.get('n_patients', 0))}abn)]  " + "  ".join([
        format_metric_with_ci(row, "patient_auroc", "AUROC"),
        format_metric_with_ci(row, "patient_ap", "AUPR"),
        format_metric_with_ci(row, "patient_f1", "F1"),
    ])


def print_metric_group(title, row, keys):
    labels = {
        "image_auroc": "AUROC",
        "image_ap": "AUPR",
        "image_f1": "F1",
        "pixel_auroc_abn": "AUROC",
        "pixel_aupr_abn": "AUPR",
        "patient_auroc": "AUROC",
        "patient_ap": "AUPR",
        "patient_f1": "F1",
    }
    parts = [format_metric_with_ci(row, key, labels.get(key, key)) for key in keys if key in row]
    if parts:
        print(f"{title}  " + "  ".join(parts), flush=True)


def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights):
    """
    Computes the best precision, recall and threshold for a given set of
    anomaly ground truth labels and anomaly prediction weights.
    """
    precision, recall, thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_precision = precision[np.argmax(f1_scores)]
    best_recall = recall[np.argmax(f1_scores)]
    print(best_threshold, best_precision, best_recall)

    return best_threshold, best_precision, best_recall


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).
    """
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    ap = 0. if path == 'training' else metrics.average_precision_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    return {"auroc": auroc, "ap": ap}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, path='train'):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    auroc = metrics.roc_auc_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)
    ap = 0. if path == 'training' else metrics.average_precision_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)

    return {"auroc": auroc, "ap": ap}


def compute_f1_max(labels, scores):
    """PR 曲线上所有阈值中的最大 F1。"""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return 0.0
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    f1 = f1[:-1]
    return float(np.max(f1)) if len(f1) else 0.0


def compute_sens_at_spec(labels, scores, spec_target):
    """特异性 ≥ spec_target 时可达到的最高灵敏度。"""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return 0.0
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    valid = np.where(fpr <= 1.0 - spec_target)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def _minmax_norm(arr):
    """测试集全局 min-max 归一化到 [0, 1]。"""
    arr = np.asarray(arr, dtype=np.float64)
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        return (arr - vmin) / (vmax - vmin)
    return np.zeros_like(arr, dtype=np.float64)


def compute_image_metrics(labels_gt, scores):
    """切片图像级：AUROC / AP / F1-Max。"""
    labels = np.asarray(labels_gt)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return {"auroc": 0.0, "ap": 0.0, "f1": 0.0}
    return {
        "auroc": metrics.roc_auc_score(labels, scores),
        "ap": metrics.average_precision_score(labels, scores),
        "f1": compute_f1_max(labels, scores),
    }


def compute_pixel_metrics(segmentations, masks_gt, labels_gt=None, abn_only=False):
    """
    切片像素级：测试集全局 min-max 归一化后报告 AUROC / AP / F1-Max。
    abn_only=True 时仅使用异常切片（缓解类不平衡）。
    """
    segs = np.asarray(segmentations)
    masks = np.asarray(masks_gt)
    if masks.ndim == 4:
        masks = masks[:, 0]

    if abn_only and labels_gt is not None:
        abn_idx = np.where(np.asarray(labels_gt) == 1)[0]
        if len(abn_idx) == 0:
            return {"auroc": 0.0, "ap": 0.0, "f1": 0.0}
        segs = segs[abn_idx]
        masks = masks[abn_idx]

    flat_scores = _minmax_norm(segs.ravel())
    flat_labels = masks.ravel().astype(int)

    if len(np.unique(flat_labels)) < 2:
        return {"auroc": 0.0, "ap": 0.0, "f1": 0.0}

    return {
        "auroc": metrics.roc_auc_score(flat_labels, flat_scores),
        "ap": metrics.average_precision_score(flat_labels, flat_scores),
        "f1": compute_f1_max(flat_labels, flat_scores),
    }


def compute_patient_metrics(scores, labels_gt, img_paths):
    """
    患者级：同一患者所有切片分数取 max；患者标签取切片标签 max。
    Legacy helper: reports AUROC / AP / F1-Max.
    """
    patient_scores = {}
    patient_labels = {}

    for score, label, path in zip(scores, labels_gt, img_paths):
        try:
            pid = os.path.basename(os.path.dirname(os.path.dirname(path)))
        except Exception:
            pid = path
        if pid not in patient_scores:
            patient_scores[pid] = []
            patient_labels[pid] = []
        patient_scores[pid].append(float(score))
        patient_labels[pid].append(int(label))

    p_scores = np.array([max(patient_scores[k]) for k in sorted(patient_scores)])
    p_labels = np.array([max(patient_labels[k]) for k in sorted(patient_labels)])

    if len(np.unique(p_labels)) < 2:
        return {"auroc": 0.0, "ap": 0.0, "f1": 0.0, "sens90": 0.0, "sens95": 0.0}

    return {
        "auroc": metrics.roc_auc_score(p_labels, p_scores),
        "ap": metrics.average_precision_score(p_labels, p_scores),
        "f1": compute_f1_max(p_labels, p_scores),
        "sens90": compute_sens_at_spec(p_labels, p_scores, 0.90),
        "sens95": compute_sens_at_spec(p_labels, p_scores, 0.95),
    }


def _safe_auroc(labels, scores):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(metrics.roc_auc_score(labels, scores))


def _safe_ap(labels, scores):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(metrics.average_precision_score(labels, scores))


def patient_id_from_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


def compute_protocol_metrics(labels_gt, masks_gt, segmentations, scores, img_paths, print_image_first=False):
    """
    Metrics aligned with /Users/zongjinying/Desktop/eval_protocol.py.

    Reports slice image metrics, abnormal-slice pixel metrics, and patient metrics.
    Patient score is max slice score; patient label is max slice label.
    """
    labels = np.asarray(labels_gt, dtype=np.int32).reshape(-1)
    masks = np.asarray(masks_gt, dtype=np.float32)
    maps = np.asarray(segmentations, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    result = {}

    if print_image_first:
        print("Computing image-level metrics and 95CI...", flush=True)
    image_auroc = _safe_auroc(labels, scores)
    _add_metric_with_ci(result, "image_auroc", image_auroc, _bootstrap_ci(labels, scores, _safe_auroc))
    image_ap = _safe_ap(labels, scores)
    _add_metric_with_ci(result, "image_ap", image_ap, _bootstrap_ci(labels, scores, _average_precision))
    image_f1 = compute_f1_max(labels, scores)
    _add_metric_with_ci(result, "image_f1", image_f1, _bootstrap_ci(labels, scores, compute_f1_max))

    if print_image_first:
        print(format_image_metrics(result), flush=True)

    if print_image_first:
        print("Computing pixel-level Slice-Px(abn) metrics and 95CI...", flush=True)
        print("Computing exact full-pixel point estimates with sklearn...", flush=True)
    gt_px_abn = masks[labels == 1].reshape(-1).astype(bool)
    pr_px_abn = maps[labels == 1].reshape(-1).astype(np.float64)
    pixel_auroc = _safe_auroc(gt_px_abn, pr_px_abn)
    pixel_aupr = _safe_ap(gt_px_abn, pr_px_abn)
    px_auroc, px_aupr = _pixel_slice_histogram_bootstrap(
        labels,
        masks,
        maps,
        exact_auroc=pixel_auroc,
        exact_aupr=pixel_aupr,
        progress=print_image_first,
    )
    _add_metric_with_ci(result, "pixel_auroc_abn", px_auroc[0], px_auroc[1])
    _add_metric_with_ci(result, "pixel_aupr_abn", px_aupr[0], px_aupr[1])
    if print_image_first:
        print(format_pixel_metrics(result), flush=True)

    if print_image_first:
        print("Computing patient-level metrics and 95CI...", flush=True)
    patient_scores = defaultdict(list)
    patient_labels = defaultdict(list)
    for path, score, label_value in zip(img_paths, scores, labels):
        pid = patient_id_from_path(path)
        patient_scores[pid].append(float(score))
        patient_labels[pid].append(int(label_value))

    pat_score, pat_label = [], []
    for pid in patient_scores:
        pat_score.append(float(np.max(patient_scores[pid])))
        pat_label.append(int(np.max(patient_labels[pid])))
    pat_score = np.asarray(pat_score, dtype=np.float64)
    pat_label = np.asarray(pat_label, dtype=np.int32)

    patient_auroc = _safe_auroc(pat_label, pat_score)
    _add_metric_with_ci(result, "patient_auroc", patient_auroc, _bootstrap_ci(pat_label, pat_score, _safe_auroc))
    patient_ap = _safe_ap(pat_label, pat_score)
    _add_metric_with_ci(result, "patient_ap", patient_ap, _bootstrap_ci(pat_label, pat_score, _average_precision))
    patient_f1 = compute_f1_max(pat_label, pat_score)
    _add_metric_with_ci(result, "patient_f1", patient_f1, _bootstrap_ci(pat_label, pat_score, compute_f1_max))
    result["n_patients"] = len(pat_label)
    result["n_abn_patients"] = int(pat_label.sum()) if len(pat_label) else 0
    return result


def compute_pro(masks, amaps, num_th=200):
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, index=[0])])

    df = df[df["fpr"] < 0.3]
    df["fpr"] = (df["fpr"] - df["fpr"].min()) / (df["fpr"].max() - df["fpr"].min() + 1e-10)

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc
