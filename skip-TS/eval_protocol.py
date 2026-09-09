"""Unified slice, pixel, and patient metrics with bootstrap 95% confidence intervals."""

import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


DEFAULT_BOOTSTRAP_ITERS = 500
DEFAULT_CI_PIXEL_MAX_SAMPLES = 0
DEFAULT_CI_SEED = 20260717
DEFAULT_HIST_BINS = 16384


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_aupr(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def f1_score_max(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    f1 = f1[:-1]
    return float(f1.max()) if len(f1) else 0.0


def _ci_tuple(value):
    value = float(value)
    return value, value


def _bootstrap_ci(y_true, y_score, metric_fn, bootstrap_iters=DEFAULT_BOOTSTRAP_ITERS, seed=DEFAULT_CI_SEED):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    point = metric_fn(y_true, y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2 or bootstrap_iters <= 0:
        return point, _ci_tuple(point)

    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(int(bootstrap_iters)):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_fn(y_true[idx], y_score[idx]))
    if not values:
        return point, _ci_tuple(point)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return point, (float(lo), float(hi))


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


def _pixel_slice_hist_bootstrap(
    gt_labels,
    gt_masks,
    maps,
    bootstrap_iters=DEFAULT_BOOTSTRAP_ITERS,
    seed=DEFAULT_CI_SEED,
    bins=DEFAULT_HIST_BINS,
    progress_every=50,
):
    keep = np.asarray(gt_labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        return (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0))

    masks_abn = np.asarray(gt_masks)[keep]
    maps_abn = np.asarray(maps)[keep]
    y_true = masks_abn.reshape(-1).astype(bool)
    y_score = maps_abn.reshape(-1).astype(np.float64)

    print("Computing exact full-pixel point estimates with sklearn...", flush=True)
    auroc_value = _safe_auroc(y_true, y_score)
    aupr_value = _safe_aupr(y_true, y_score)
    if len(np.unique(y_true)) < 2 or bootstrap_iters <= 0:
        return (auroc_value, _ci_tuple(auroc_value)), (aupr_value, _ci_tuple(aupr_value))

    score_min = float(np.min(maps_abn))
    score_max = float(np.max(maps_abn))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    scale = (bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    for row in range(len(abn_idx)):
        scores = maps_abn[row].reshape(-1)
        mask = masks_abn[row].reshape(-1).astype(bool)
        bin_idx = np.floor((scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[mask], minlength=bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~mask], minlength=bins).astype(np.uint32)

    rng = np.random.default_rng(seed)
    aurocs, auprs = [], []
    n = len(abn_idx)
    for i in range(int(bootstrap_iters)):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"Pixel histogram slice bootstrap: {i + 1}/{bootstrap_iters}", flush=True)

    auroc_ci = tuple(float(v) for v in np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = tuple(float(v) for v in np.percentile(auprs, [2.5, 97.5]))
    return (auroc_value, auroc_ci), (aupr_value, aupr_ci)


def patient_id_from_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


def _patient_arrays(paths, image_scores, gt_labels):
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


def _format_metric(name, value, ci, scale):
    return f"{name}={value * scale:.2f}% (95% CI {ci[0] * scale:.2f}-{ci[1] * scale:.2f}%)"


def format_image_metrics(slice_metrics, as_percent=True):
    scale = 100.0 if as_percent else 1.0
    return (
        "[Slice-Img]  "
        + "  ".join([
            _format_metric("AUROC", slice_metrics["img_auroc"], slice_metrics["img_auroc_ci"], scale),
            _format_metric("AUPR", slice_metrics["img_aupr"], slice_metrics["img_aupr_ci"], scale),
            _format_metric("F1", slice_metrics["img_f1"], slice_metrics["img_f1_ci"], scale),
        ])
    )


def compute_metrics(
    gt_labels,
    gt_masks,
    anomaly_maps,
    image_scores,
    paths,
    bootstrap_iters=DEFAULT_BOOTSTRAP_ITERS,
    ci_pixel_max_samples=DEFAULT_CI_PIXEL_MAX_SAMPLES,
    ci_seed=DEFAULT_CI_SEED,
    hist_bins=DEFAULT_HIST_BINS,
    print_image_first=False,
):
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    image_scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    print("Computing image-level metrics and 95CI...", flush=True)
    img_auroc, img_auroc_ci = _bootstrap_ci(gt_labels, image_scores, _safe_auroc, bootstrap_iters, ci_seed)
    img_aupr, img_aupr_ci = _bootstrap_ci(gt_labels, image_scores, _safe_aupr, bootstrap_iters, ci_seed + 1)
    img_f1, img_f1_ci = _bootstrap_ci(gt_labels, image_scores, f1_score_max, bootstrap_iters, ci_seed + 2)
    slice_metrics = {
        "img_auroc": img_auroc,
        "img_auroc_ci": img_auroc_ci,
        "img_aupr": img_aupr,
        "img_aupr_ci": img_aupr_ci,
        "img_ap": img_aupr,
        "img_ap_ci": img_aupr_ci,
        "img_f1": img_f1,
        "img_f1_ci": img_f1_ci,
        "_pr_sp": image_scores,
        "_gt_sp": gt_labels,
    }
    if print_image_first:
        print(format_image_metrics(slice_metrics, as_percent=True), flush=True)

    print("Computing pixel-level Slice-Px(abn) metrics and 95CI...", flush=True)
    px_auroc, px_aupr = _pixel_slice_hist_bootstrap(
        gt_labels,
        gt_masks,
        maps,
        bootstrap_iters=bootstrap_iters,
        seed=ci_seed + 10,
        bins=hist_bins,
    )
    slice_metrics.update({
        "px_auroc_abn": px_auroc[0],
        "px_auroc_abn_ci": px_auroc[1],
        "px_aupr_abn": px_aupr[0],
        "px_aupr_abn_ci": px_aupr[1],
    })
    print(
        "[Slice-Px(abn)]  "
        + "  ".join([
            _format_metric("AUROC", slice_metrics["px_auroc_abn"], slice_metrics["px_auroc_abn_ci"], 100.0),
            _format_metric("AUPR", slice_metrics["px_aupr_abn"], slice_metrics["px_aupr_abn_ci"], 100.0),
        ]),
        flush=True,
    )

    print("Computing patient-level metrics and 95CI...", flush=True)
    pat_label, pat_score = _patient_arrays(paths, image_scores, gt_labels)
    pat_auroc, pat_auroc_ci = _bootstrap_ci(pat_label, pat_score, _safe_auroc, bootstrap_iters, ci_seed + 5)
    pat_aupr, pat_aupr_ci = _bootstrap_ci(pat_label, pat_score, _safe_aupr, bootstrap_iters, ci_seed + 6)
    pat_f1, pat_f1_ci = _bootstrap_ci(pat_label, pat_score, f1_score_max, bootstrap_iters, ci_seed + 7)
    pat_metrics = {
        "pat_auroc": pat_auroc,
        "pat_auroc_ci": pat_auroc_ci,
        "pat_aupr": pat_aupr,
        "pat_aupr_ci": pat_aupr_ci,
        "pat_ap": pat_aupr,
        "pat_ap_ci": pat_aupr_ci,
        "pat_f1": pat_f1,
        "pat_f1_ci": pat_f1_ci,
        "n_patients": len(pat_label),
        "n_abn_patients": int(pat_label.sum()),
    }
    return slice_metrics, pat_metrics


def format_metrics(slice_metrics, pat_metrics, as_percent=True):
    scale = 100.0 if as_percent else 1.0
    lines = [
        format_image_metrics(slice_metrics, as_percent=as_percent),
        "[Slice-Px(abn)]  "
        + "  ".join([
            _format_metric("AUROC", slice_metrics["px_auroc_abn"], slice_metrics["px_auroc_abn_ci"], scale),
            _format_metric("AUPR", slice_metrics["px_aupr_abn"], slice_metrics["px_aupr_abn_ci"], scale),
        ]),
    ]
    p = pat_metrics
    lines.append(
        f"[Patient({p['n_abn_patients']}/{p['n_patients']}abn)]  "
        + "  ".join([
            _format_metric("AUROC", p["pat_auroc"], p["pat_auroc_ci"], scale),
            _format_metric("AUPR", p["pat_aupr"], p["pat_aupr_ci"], scale),
            _format_metric("F1", p["pat_f1"], p["pat_f1_ci"], scale),
        ])
    )
    return "\n".join(lines)
