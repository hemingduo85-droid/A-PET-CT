import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _clip_ci(low, high):
    return float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0))


def _with_ci(value, ci):
    return {"value": float(value), "ci": (float(ci[0]), float(ci[1]))}


def _metric_value(metric):
    return float(metric["value"]) if isinstance(metric, dict) else float(metric)


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _aupr_score(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _f1_score_max(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    f1 = f1[:-1]
    return float(f1.max()) if len(f1) else 0.0


def _bootstrap_ci(y_true, y_score, metric_fn, n_bootstrap=500, seed=42):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    value = metric_fn(y_true, y_score)
    if n_bootstrap <= 0 or len(np.unique(y_true)) < 2:
        return value, value

    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[sample_idx])) < 2:
            continue
        values.append(metric_fn(y_true[sample_idx], y_score[sample_idx]))
    if not values:
        return value, value
    return tuple(np.percentile(values, [2.5, 97.5]).astype(float))


def _pixel_slice_metric(gt_labels, gt_masks, maps, metric_fn):
    keep = np.asarray(gt_labels, dtype=np.int32).reshape(-1) == 1
    masks = np.asarray(gt_masks, dtype=np.float32)[keep]
    scores = np.asarray(maps, dtype=np.float32)[keep]
    if masks.size == 0:
        return 0.0
    y_true = masks.reshape(-1).astype(np.uint8)
    y_score = scores.reshape(-1).astype(np.float64)
    return metric_fn(y_true, y_score)


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


def _pixel_slice_bootstrap_metric(
    gt_labels,
    gt_masks,
    maps,
    metric_fn,
    n_bootstrap=500,
    seed=42,
    progress_fn=None,
    progress_name="pixel",
):
    keep = np.asarray(gt_labels, dtype=np.int32).reshape(-1) == 1
    masks = np.asarray(gt_masks, dtype=np.float32)[keep]
    scores = np.asarray(maps, dtype=np.float32)[keep]
    value = _pixel_slice_metric(gt_labels, gt_masks, maps, metric_fn)
    if masks.size == 0 or n_bootstrap <= 0:
        return _with_ci(value, (value, value))

    rng = np.random.default_rng(seed)
    values = []
    n = masks.shape[0]
    for boot_idx in range(1, int(n_bootstrap) + 1):
        idx = rng.integers(0, n, size=n)
        y_true = masks[idx].reshape(-1).astype(np.uint8)
        y_score = scores[idx].reshape(-1).astype(np.float64)
        if len(np.unique(y_true)) < 2:
            continue
        values.append(float(metric_fn(y_true, y_score)))
        if progress_fn is not None and (boot_idx % 50 == 0 or boot_idx == int(n_bootstrap)):
            progress_fn(f"{progress_name} bootstrap: {boot_idx}/{int(n_bootstrap)}")
    if not values:
        return _with_ci(value, (value, value))
    return _with_ci(value, _clip_ci(*np.percentile(values, [2.5, 97.5])))


def _pixel_slice_hist_bootstrap_metrics(
    gt_labels,
    gt_masks,
    maps,
    n_bootstrap=500,
    seed=42,
    bins=16384,
    exact_point=False,
    exact_auroc=None,
    exact_aupr=None,
    progress_fn=None,
):
    keep = np.asarray(gt_labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        empty = _with_ci(0.0, (0.0, 0.0))
        return empty, empty

    masks = np.asarray(gt_masks, dtype=np.float32)
    scores = np.asarray(maps, dtype=np.float32)

    score_min = float(np.min(scores[keep]))
    score_max = float(np.max(scores[keep]))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    scale = (bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        slice_scores = scores[idx].reshape(-1)
        slice_mask = masks[idx].reshape(-1).astype(bool)
        bin_idx = np.floor((slice_scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[slice_mask], minlength=bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~slice_mask], minlength=bins).astype(np.uint32)

    binned_auroc, binned_aupr = _metrics_from_hist(pos_hists.sum(axis=0), neg_hists.sum(axis=0))
    if exact_auroc is not None or exact_aupr is not None:
        if exact_auroc is None or exact_aupr is None:
            raise ValueError("exact_auroc and exact_aupr must be provided together.")
        auroc_value = float(exact_auroc)
        aupr_value = float(exact_aupr)
    elif exact_point:
        if progress_fn is not None:
            progress_fn("Computing exact full-pixel point estimates with sklearn...")
        auroc_value = _pixel_slice_metric(gt_labels, masks, scores, _safe_auroc)
        aupr_value = _pixel_slice_metric(gt_labels, masks, scores, _aupr_score)
    else:
        auroc_value = binned_auroc
        aupr_value = binned_aupr

    rng = np.random.default_rng(seed)
    aurocs, auprs = [], []
    n = len(abn_idx)
    for boot_idx in range(1, int(n_bootstrap) + 1):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if progress_fn is not None and (boot_idx % 50 == 0 or boot_idx == int(n_bootstrap)):
            progress_fn(f"Pixel histogram slice bootstrap: {boot_idx}/{int(n_bootstrap)}")

    auroc_ci = _clip_ci(*np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = _clip_ci(*np.percentile(auprs, [2.5, 97.5]))
    return _with_ci(auroc_value, auroc_ci), _with_ci(aupr_value, aupr_ci)


def _auroc_metric(y_true, y_score, n_bootstrap, seed):
    value = _safe_auroc(y_true, y_score)
    ci = _bootstrap_ci(y_true, y_score, _safe_auroc, n_bootstrap=n_bootstrap, seed=seed)
    return _with_ci(value, ci)


def _bootstrap_metric(y_true, y_score, metric_fn, n_bootstrap, seed):
    value = metric_fn(y_true, y_score)
    ci = _bootstrap_ci(y_true, y_score, metric_fn, n_bootstrap=n_bootstrap, seed=seed)
    return _with_ci(value, ci)


def patient_id_from_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


def _patient_scores(paths, image_scores, gt_labels):
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
    return np.asarray(pat_label, dtype=np.int32), np.asarray(pat_score, dtype=np.float64)


def compute_petct_metrics(
    gt_labels,
    gt_masks,
    anomaly_maps,
    image_scores,
    paths,
    n_bootstrap=500,
    seed=42,
    pixel_bootstrap_max_samples=None,
    pixel_hist_bins=16384,
    exact_pixel_points=True,
    exact_pixel_auroc=None,
    exact_pixel_aupr=None,
    progress_fn=None,
):
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    image_scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    if progress_fn is not None:
        progress_fn("Computing image-level metrics and 95CI...")
    slice_metrics = {
        "img_auroc": _auroc_metric(gt_labels, image_scores, n_bootstrap, seed),
        "img_aupr": _bootstrap_metric(gt_labels, image_scores, _aupr_score, n_bootstrap, seed),
        "img_f1": _bootstrap_metric(gt_labels, image_scores, _f1_score_max, n_bootstrap, seed + 1),
    }
    if progress_fn is not None:
        progress_fn(format_image_metrics(slice_metrics))
        progress_fn("Computing pixel-level Slice-Px(abn) metrics and 95CI...")

    slice_metrics.update({
        "px_auroc_abn": None,
        "px_aupr_abn": None,
    })
    slice_metrics["px_auroc_abn"], slice_metrics["px_aupr_abn"] = _pixel_slice_hist_bootstrap_metrics(
            gt_labels,
            gt_masks,
            maps,
            n_bootstrap,
            seed + 2,
            bins=pixel_hist_bins,
            exact_point=exact_pixel_points,
            exact_auroc=exact_pixel_auroc,
            exact_aupr=exact_pixel_aupr,
            progress_fn=progress_fn,
        )
    if progress_fn is not None:
        progress_fn(
            f"[Slice-Px(abn)]  AUROC={_fmt_metric(slice_metrics['px_auroc_abn'])}  "
            f"AUPR={_fmt_metric(slice_metrics['px_aupr_abn'])}"
        )
        progress_fn("Computing patient-level metrics and 95CI...")

    pat_label, pat_score = _patient_scores(paths, image_scores, gt_labels)
    patient_metrics = {
        "pat_auroc": _auroc_metric(pat_label, pat_score, n_bootstrap, seed + 4),
        "pat_aupr": _bootstrap_metric(pat_label, pat_score, _aupr_score, n_bootstrap, seed + 5),
        "pat_f1": _bootstrap_metric(pat_label, pat_score, _f1_score_max, n_bootstrap, seed + 6),
        "n_patients": len(pat_label),
        "n_abn_patients": int(pat_label.sum()),
    }
    return slice_metrics, patient_metrics


def _fmt_metric(metric, scale=100.0):
    value = _metric_value(metric) * scale
    low, high = metric["ci"]
    return f"{value:.2f}% (95% CI {low * scale:.2f}-{high * scale:.2f}%)"


def format_image_metrics(slice_metrics):
    m = slice_metrics
    return (
        f"[Slice-Img]  AUROC={_fmt_metric(m['img_auroc'])}  "
        f"AUPR={_fmt_metric(m['img_aupr'])}  F1={_fmt_metric(m['img_f1'])}"
    )


def format_petct_metrics(slice_metrics, patient_metrics):
    m = slice_metrics
    lines = [
        format_image_metrics(m),
        f"[Slice-Px(abn)]  AUROC={_fmt_metric(m['px_auroc_abn'])}  AUPR={_fmt_metric(m['px_aupr_abn'])}",
    ]
    p = patient_metrics
    lines.append(
        f"[Patient({p['n_abn_patients']}/{p['n_patients']}abn)]  "
        f"AUROC={_fmt_metric(p['pat_auroc'])}  "
        f"AUPR={_fmt_metric(p['pat_aupr'])}  "
        f"F1={_fmt_metric(p['pat_f1'])}"
    )
    return "\n".join(lines)
