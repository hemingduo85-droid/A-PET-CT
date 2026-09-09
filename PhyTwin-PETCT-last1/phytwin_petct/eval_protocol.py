
"""Unified slice, pixel, and patient metrics for PhyTwin."""

import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def f1_score_max(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    p, r, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / (p + r + 1e-7)
    f1 = f1[:-1]
    return float(f1.max()) if len(f1) else 0.0


def patient_id_from_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


def _has_two_classes(y_true):
    return len(np.unique(np.asarray(y_true))) >= 2


def _percentile_ci(values, point):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float(point), float(point))
    low, high = np.percentile(values, [2.5, 97.5])
    low = min(float(low), float(point))
    high = max(float(high), float(point))
    return (low, high)


def _bootstrap_ci(labels, scores, metric_fn, bootstrap_iters, seed):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    point = metric_fn(labels, scores)
    n = len(labels)
    if bootstrap_iters <= 0 or n == 0 or not _has_two_classes(labels):
        return float(point), (float(point), float(point))

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(bootstrap_iters)):
        idx = rng.integers(0, n, size=n)
        if not _has_two_classes(labels[idx]):
            continue
        values.append(metric_fn(labels[idx], scores[idx]))
    return float(point), _percentile_ci(values, point)


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
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    auroc = float(trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)
    aupr = float(np.sum(precision * (tp / total_pos)))
    return auroc, aupr


def _pixel_slice_bootstrap(labels, masks, maps, bootstrap_iters, seed, hist_bins=16384, progress_callback=None):
    keep = np.asarray(labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        return (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0))

    if progress_callback is not None:
        progress_callback("Computing exact full-pixel point estimates with sklearn...")
    point_labels = masks[keep].reshape(-1).astype(bool)
    point_scores = maps[keep].reshape(-1).astype(np.float64)
    auroc_value = _safe_auroc(point_labels, point_scores)
    aupr_value = _safe_ap(point_labels, point_scores)

    if bootstrap_iters <= 0:
        return (auroc_value, (auroc_value, auroc_value)), (aupr_value, (aupr_value, aupr_value))

    score_min = float(np.min(maps[keep]))
    score_max = float(np.max(maps[keep]))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    scale = (hist_bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), hist_bins), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), hist_bins), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        scores = maps[idx].reshape(-1)
        mask = masks[idx].reshape(-1).astype(bool)
        bin_idx = np.floor((scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, hist_bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[mask], minlength=hist_bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~mask], minlength=hist_bins).astype(np.uint32)

    rng = np.random.default_rng(seed)
    aurocs, auprs = [], []
    n = len(abn_idx)
    values = []
    for _ in range(int(bootstrap_iters)):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if progress_callback is not None and ((_ + 1) % 50 == 0 or (_ + 1) == bootstrap_iters):
            progress_callback(f"Pixel histogram slice bootstrap: {_ + 1}/{bootstrap_iters}")

    auroc_ci = tuple(float(v) for v in np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = tuple(float(v) for v in np.percentile(auprs, [2.5, 97.5]))
    return (auroc_value, auroc_ci), (aupr_value, aupr_ci)


def _patient_groups(paths):
    grouped = defaultdict(list)
    for idx, path in enumerate(paths):
        grouped[patient_id_from_path(path)].append(idx)
    return [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]


def compute_metrics(
    gt_labels,
    gt_masks,
    anomaly_maps,
    image_scores,
    paths,
    bootstrap_iters=500,
    bootstrap_seed=0,
    hist_bins=16384,
    progress_callback=None,
):
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    image_scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    if progress_callback is not None:
        progress_callback("Computing image-level metrics and 95CI...")
    img_auroc_ci = _bootstrap_ci(gt_labels, image_scores, _safe_auroc, bootstrap_iters, bootstrap_seed)
    img_ap_ci = _bootstrap_ci(gt_labels, image_scores, _safe_ap, bootstrap_iters, bootstrap_seed + 1)
    img_f1_ci = _bootstrap_ci(gt_labels, image_scores, f1_score_max, bootstrap_iters, bootstrap_seed + 2)
    slice_metrics = {
        "img_auroc": img_auroc_ci[0],
        "img_auroc_ci": img_auroc_ci[1],
        "img_ap": img_ap_ci[0],
        "img_ap_ci": img_ap_ci[1],
        "img_f1": img_f1_ci[0],
        "img_f1_ci": img_f1_ci[1],
        "_pr_sp": image_scores,
        "_gt_sp": gt_labels,
    }
    if progress_callback is not None:
        progress_callback(format_slice_image_metrics(slice_metrics))

    if progress_callback is not None:
        progress_callback("Computing pixel-level Slice-Px(abn) metrics and 95CI...")
    px_auroc_ci, px_aupr_ci = _pixel_slice_bootstrap(
        gt_labels,
        gt_masks,
        maps,
        bootstrap_iters,
        bootstrap_seed + 10,
        hist_bins=hist_bins,
        progress_callback=progress_callback,
    )
    slice_metrics.update({
        "px_auroc_abn": px_auroc_ci[0],
        "px_auroc_abn_ci": px_auroc_ci[1],
        "px_aupr_abn": px_aupr_ci[0],
        "px_aupr_abn_ci": px_aupr_ci[1],
    })
    if progress_callback is not None:
        progress_callback(format_pixel_metrics(slice_metrics))

    if progress_callback is not None:
        progress_callback("Computing patient-level metrics and 95CI...")
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
    pat_auroc_ci = _bootstrap_ci(pat_label, pat_score, _safe_auroc, bootstrap_iters, bootstrap_seed + 5)
    pat_ap_ci = _bootstrap_ci(pat_label, pat_score, _safe_ap, bootstrap_iters, bootstrap_seed + 6)
    pat_f1_ci = _bootstrap_ci(pat_label, pat_score, f1_score_max, bootstrap_iters, bootstrap_seed + 7)
    pat_metrics = {
        "pat_auroc": pat_auroc_ci[0],
        "pat_auroc_ci": pat_auroc_ci[1],
        "pat_ap": pat_ap_ci[0],
        "pat_ap_ci": pat_ap_ci[1],
        "pat_f1": pat_f1_ci[0],
        "pat_f1_ci": pat_f1_ci[1],
        "n_patients": len(pat_label),
        "n_abn_patients": int(pat_label.sum()),
    }
    return slice_metrics, pat_metrics


def _format_metric(metrics, name, label, scale):
    value = metrics[name] * scale
    ci = metrics.get(f"{name}_ci")
    if ci is None:
        return f"{label}={value:.2f}%"
    low, high = ci
    return f"{label}={value:.2f}% (95% CI {low*scale:.2f}-{high*scale:.2f}%)"


def format_slice_image_metrics(slice_metrics, as_percent=True):
    s = 100.0 if as_percent else 1.0
    return "[Slice-Img]  " + "  ".join([
        _format_metric(slice_metrics, "img_auroc", "AUROC", s),
        _format_metric(slice_metrics, "img_ap", "AUPR", s),
        _format_metric(slice_metrics, "img_f1", "F1", s),
    ])


def format_pixel_metrics(slice_metrics, as_percent=True):
    s = 100.0 if as_percent else 1.0
    return "[Slice-Px(abn)]  " + "  ".join([
        _format_metric(slice_metrics, "px_auroc_abn", "AUROC", s),
        _format_metric(slice_metrics, "px_aupr_abn", "AUPR", s),
    ])


def format_metrics(slice_metrics, pat_metrics, as_percent=True):
    s = 100.0 if as_percent else 1.0
    m = slice_metrics
    lines = [
        format_slice_image_metrics(m, as_percent=as_percent),
        format_pixel_metrics(m, as_percent=as_percent),
    ]
    p = pat_metrics
    lines.append(
        f"[Patient({p['n_abn_patients']}/{p['n_patients']}abn)]  "
        + "  ".join([
            _format_metric(p, "pat_auroc", "AUROC", s),
            _format_metric(p, "pat_ap", "AUPR", s),
            _format_metric(p, "pat_f1", "F1", s),
        ])
    )
    return "\n".join(lines)
