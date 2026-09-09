from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def top_percent_mean(score_maps, percent=1.0):
    score_maps = np.asarray(score_maps, dtype=np.float32)
    if score_maps.ndim == 2:
        score_maps = score_maps[None]
    flat = score_maps.reshape(score_maps.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * percent / 100.0)))
    return np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)


def patient_id_from_path(path):
    path_str = str(path)
    if "__" in path_str:
        parts = Path(path_str).name.split("__")
        if len(parts) >= 2:
            return parts[1]
    return Path(path_str).parent.parent.name


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _f1_score_max(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-7)
    return float(np.nanmax(f1)) if f1.size else 0.0


def _bootstrap_ci(y_true, y_score, metric_fn, n_boot=500, seed=0):
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        point = metric_fn(y_true, y_score)
        return point, point

    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        sample_true = y_true[idx]
        if len(np.unique(sample_true)) < 2:
            continue
        values.append(metric_fn(sample_true, y_score[idx]))
    if not values:
        point = metric_fn(y_true, y_score)
        return point, point
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _pixel_vectors(gt_labels, gt_masks, maps):
    keep = np.asarray(gt_labels, dtype=np.int32).reshape(-1) == 1
    if not np.any(keep):
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
    y_true = gt_masks[keep].reshape(-1).astype(np.int32)
    y_score = maps[keep].reshape(-1).astype(np.float64)
    return y_true, y_score


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
    integrate = getattr(np, "trapezoid", np.trapz)
    auroc = float(integrate(np.r_[0.0, tpr], np.r_[0.0, fpr]))

    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)
    aupr = float(np.sum(precision * (tp / total_pos)))
    return auroc, aupr


def _pixel_histogram_slice_bootstrap_ci(
    gt_labels,
    gt_masks,
    maps,
    n_boot=500,
    seed=0,
    bins=16384,
    exact_auroc=None,
    exact_aupr=None,
    progress_callback=None,
):
    keep = np.asarray(gt_labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if abn_idx.size == 0:
        return (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0))

    masks = np.asarray(gt_masks)
    scores = np.asarray(maps, dtype=np.float32)
    score_min = float(np.min(scores[keep]))
    score_max = float(np.max(scores[keep]))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    scale = (int(bins) - 1) / (score_max - score_min)

    pos_hists = np.zeros((abn_idx.size, int(bins)), dtype=np.uint32)
    neg_hists = np.zeros((abn_idx.size, int(bins)), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        slice_scores = scores[idx].reshape(-1)
        slice_mask = masks[idx].reshape(-1) > 0.5
        bin_idx = np.floor((slice_scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, int(bins) - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[slice_mask], minlength=int(bins)).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~slice_mask], minlength=int(bins)).astype(np.uint32)

    rng = np.random.default_rng(seed)
    aurocs, auprs = [], []
    n_slices = abn_idx.size
    n_boot = int(n_boot)
    for i in range(1, n_boot + 1):
        sample = rng.integers(0, n_slices, size=n_slices)
        weights = np.bincount(sample, minlength=n_slices).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if progress_callback is not None and (i % 50 == 0 or i == n_boot):
            progress_callback(f"Pixel histogram slice bootstrap: {i}/{n_boot}")

    binned_auroc, binned_aupr = _metrics_from_hist(pos_hists.sum(axis=0), neg_hists.sum(axis=0))
    auroc_value = binned_auroc if exact_auroc is None else float(exact_auroc)
    aupr_value = binned_aupr if exact_aupr is None else float(exact_aupr)
    if not aurocs:
        return (auroc_value, (auroc_value, auroc_value)), (aupr_value, (aupr_value, aupr_value))
    auroc_ci = tuple(float(v) for v in np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = tuple(float(v) for v in np.percentile(auprs, [2.5, 97.5]))
    return (auroc_value, auroc_ci), (aupr_value, aupr_ci)


def _add_bootstrap_ci(metrics, key, y_true, y_score, metric_fn, n_boot=500, seed=0):
    metrics[f"{key}_ci"] = _bootstrap_ci(y_true, y_score, metric_fn, n_boot=n_boot, seed=seed)


def _format_pixel_metrics(slice_metrics, as_percent=True):
    scale = 100.0 if as_percent else 1.0
    return "[Slice-Px(abn)]  " + "  ".join([
        _metric_ci_text(slice_metrics, "px_auroc_abn", "AUROC", scale),
        _metric_ci_text(slice_metrics, "px_aupr_abn", "AUPR", scale),
    ])


def eval_protocol_compute_metrics(
    gt_labels,
    gt_masks,
    anomaly_maps,
    image_scores,
    paths,
    bootstrap_iters=500,
    ci_seed=0,
    ci_pixel_max_samples=200000,
    ci_hist_bins=16384,
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
    slice_metrics = {
        "img_auroc": _safe_auroc(gt_labels, image_scores),
        "img_ap": _safe_ap(gt_labels, image_scores),
        "img_f1": _f1_score_max(gt_labels, image_scores),
    }
    _add_bootstrap_ci(slice_metrics, "img_auroc", gt_labels, image_scores, _safe_auroc, bootstrap_iters, ci_seed)
    _add_bootstrap_ci(slice_metrics, "img_ap", gt_labels, image_scores, _safe_ap, bootstrap_iters, ci_seed + 1)
    _add_bootstrap_ci(slice_metrics, "img_f1", gt_labels, image_scores, _f1_score_max, bootstrap_iters, ci_seed + 2)
    if progress_callback is not None:
        progress_callback(eval_protocol_format_image_metrics(slice_metrics))

    if progress_callback is not None:
        progress_callback("Computing pixel-level Slice-Px(abn) metrics and 95CI...")
        progress_callback("Computing exact full-pixel point estimates with sklearn...")
    px_true_full, px_score_full = _pixel_vectors(gt_labels, gt_masks, maps)
    slice_metrics["px_auroc_abn"] = _safe_auroc(px_true_full, px_score_full)
    slice_metrics["px_aupr_abn"] = _safe_ap(px_true_full, px_score_full)
    px_auroc, px_aupr = _pixel_histogram_slice_bootstrap_ci(
        gt_labels,
        gt_masks,
        maps,
        n_boot=bootstrap_iters,
        seed=ci_seed + 3,
        bins=ci_hist_bins,
        exact_auroc=slice_metrics["px_auroc_abn"],
        exact_aupr=slice_metrics["px_aupr_abn"],
        progress_callback=progress_callback,
    )
    slice_metrics["px_auroc_abn"], slice_metrics["px_auroc_abn_ci"] = px_auroc
    slice_metrics["px_aupr_abn"], slice_metrics["px_aupr_abn_ci"] = px_aupr
    if progress_callback is not None:
        progress_callback(_format_pixel_metrics(slice_metrics))

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

    pat_metrics = {
        "pat_auroc": _safe_auroc(pat_label, pat_score),
        "pat_ap": _safe_ap(pat_label, pat_score),
        "pat_f1": _f1_score_max(pat_label, pat_score),
        "n_patients": len(pat_label),
        "n_abn_patients": int(pat_label.sum()),
    }
    _add_bootstrap_ci(pat_metrics, "pat_auroc", pat_label, pat_score, _safe_auroc, bootstrap_iters, ci_seed + 4)
    _add_bootstrap_ci(pat_metrics, "pat_ap", pat_label, pat_score, _safe_ap, bootstrap_iters, ci_seed + 5)
    _add_bootstrap_ci(pat_metrics, "pat_f1", pat_label, pat_score, _f1_score_max, bootstrap_iters, ci_seed + 6)

    return slice_metrics, pat_metrics


def _metric_ci_text(metrics, key, label, scale):
    ci = metrics.get(f"{key}_ci", (metrics[key], metrics[key]))
    return f"{label}={metrics[key]*scale:.2f}% (95% CI {ci[0]*scale:.2f}-{ci[1]*scale:.2f}%)"


def eval_protocol_format_image_metrics(slice_metrics, as_percent=True):
    scale = 100.0 if as_percent else 1.0
    return "[Slice-Img]  " + "  ".join([
        _metric_ci_text(slice_metrics, "img_auroc", "AUROC", scale),
        _metric_ci_text(slice_metrics, "img_ap", "AUPR", scale),
        _metric_ci_text(slice_metrics, "img_f1", "F1", scale),
    ])


def eval_protocol_format_metrics(slice_metrics, pat_metrics, as_percent=True):
    scale = 100.0 if as_percent else 1.0
    lines = [
        eval_protocol_format_image_metrics(slice_metrics, as_percent=as_percent),
        _format_pixel_metrics(slice_metrics, as_percent=as_percent),
    ]
    lines.append(
        f"[Patient({pat_metrics['n_abn_patients']}/{pat_metrics['n_patients']}abn)]  "
        + "  ".join([
            _metric_ci_text(pat_metrics, "pat_auroc", "AUROC", scale),
            _metric_ci_text(pat_metrics, "pat_ap", "AUPR", scale),
            _metric_ci_text(pat_metrics, "pat_f1", "F1", scale),
        ])
    )
    return "\n".join(lines)
