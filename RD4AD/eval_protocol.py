"""Unified metrics and bootstrap 95% CI for RD4AD comparison runs."""

import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


BOOTSTRAP_ITERS = 500
BOOTSTRAP_SEED = 20260717
HIST_BINS = 16384


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_aupr(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
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


def _bootstrap_ci(y_true, y_score, metric_fn, iters, seed):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    value = metric_fn(y_true, y_score)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_fn(y_true[idx], y_score[idx]))
    if not values:
        return value, (value, value)
    low, high = np.percentile(values, [2.5, 97.5])
    return value, (float(low), float(high))


def patient_id_from_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


def _patient_arrays(paths, image_scores, labels):
    patient_scores = defaultdict(list)
    patient_labels = defaultdict(list)
    for path, score, label in zip(paths, image_scores, labels):
        pid = patient_id_from_path(path)
        patient_scores[pid].append(float(score))
        patient_labels[pid].append(int(label))
    scores, labels_out = [], []
    for pid in patient_scores:
        scores.append(float(np.max(patient_scores[pid])))
        labels_out.append(int(np.max(patient_labels[pid])))
    return np.asarray(labels_out, dtype=np.int32), np.asarray(scores, dtype=np.float64)


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
    auroc = float(np.trapz(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)
    aupr = float(np.sum(precision * (tp / total_pos)))
    return auroc, aupr


def _pixel_slice_bootstrap(labels, masks, maps, iters, seed, bins, progress=True):
    keep = np.asarray(labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        empty = (0.0, (0.0, 0.0))
        return empty, empty

    pixel_gt = masks[keep].reshape(-1).astype(bool)
    pixel_score = maps[keep].reshape(-1).astype(np.float64)
    exact_auroc = _safe_auroc(pixel_gt, pixel_score)
    exact_aupr = _safe_aupr(pixel_gt, pixel_score)

    score_min = float(np.min(maps[keep]))
    score_max = float(np.max(maps[keep]))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    scale = (bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        scores = maps[idx].reshape(-1)
        mask = masks[idx].reshape(-1).astype(bool)
        bin_idx = np.floor((scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[mask], minlength=bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~mask], minlength=bins).astype(np.uint32)

    rng = np.random.default_rng(seed)
    aurocs, auprs = [], []
    n = len(abn_idx)
    for boot_idx in range(1, iters + 1):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if progress and boot_idx % 50 == 0:
            print(f"Pixel histogram slice bootstrap: {boot_idx}/{iters}", flush=True)

    auroc_ci = tuple(float(v) for v in np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = tuple(float(v) for v in np.percentile(auprs, [2.5, 97.5]))
    return (exact_auroc, auroc_ci), (exact_aupr, aupr_ci)


def _fmt(name, value_ci):
    value, ci = value_ci
    return f"{name}={value * 100:.2f}% (95% CI {ci[0] * 100:.2f}-{ci[1] * 100:.2f}%)"


def format_image_metrics(slice_metrics):
    return "[Slice-Img]  " + "  ".join([
        _fmt("AUROC", slice_metrics["img_auroc"]),
        _fmt("AUPR", slice_metrics["img_aupr"]),
        _fmt("F1", slice_metrics["img_f1"]),
    ])


def format_pixel_metrics(slice_metrics):
    return "[Slice-Px(abn)]  " + "  ".join([
        _fmt("AUROC", slice_metrics["px_auroc"]),
        _fmt("AUPR", slice_metrics["px_aupr"]),
    ])


def format_patient_metrics(patient_metrics):
    return (
        f"[Patient({patient_metrics['n_abn_patients']}/{patient_metrics['n_patients']}abn)]  "
        + "  ".join([
            _fmt("AUROC", patient_metrics["pat_auroc"]),
            _fmt("AUPR", patient_metrics["pat_aupr"]),
            _fmt("F1", patient_metrics["pat_f1"]),
        ])
    )


def compute_metrics(gt_labels, gt_masks, anomaly_maps, image_scores, paths,
                    bootstrap_iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED,
                    hist_bins=HIST_BINS, progress=True):
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    image_scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    if progress:
        print("Computing image-level metrics and 95CI...", flush=True)
    slice_metrics = {
        "img_auroc": _bootstrap_ci(gt_labels, image_scores, _safe_auroc, bootstrap_iters, seed),
        "img_aupr": _bootstrap_ci(gt_labels, image_scores, _safe_aupr, bootstrap_iters, seed + 1),
        "img_f1": _bootstrap_ci(gt_labels, image_scores, f1_score_max, bootstrap_iters, seed + 2),
        "_pr_sp": image_scores,
        "_gt_sp": gt_labels,
    }
    if progress:
        print(format_image_metrics(slice_metrics), flush=True)

    if progress:
        print("Computing pixel-level Slice-Px(abn) metrics and 95CI...", flush=True)
        print("Computing exact full-pixel point estimates with sklearn...", flush=True)
    px_auroc, px_aupr = _pixel_slice_bootstrap(
        gt_labels,
        gt_masks,
        maps,
        bootstrap_iters,
        seed + 10,
        hist_bins,
        progress=progress,
    )
    slice_metrics["px_auroc"] = px_auroc
    slice_metrics["px_aupr"] = px_aupr
    if progress:
        print(format_pixel_metrics(slice_metrics), flush=True)

    if progress:
        print("Computing patient-level metrics and 95CI...", flush=True)
    pat_label, pat_score = _patient_arrays(paths, image_scores, gt_labels)
    patient_metrics = {
        "pat_auroc": _bootstrap_ci(pat_label, pat_score, _safe_auroc, bootstrap_iters, seed + 5),
        "pat_aupr": _bootstrap_ci(pat_label, pat_score, _safe_aupr, bootstrap_iters, seed + 6),
        "pat_f1": _bootstrap_ci(pat_label, pat_score, f1_score_max, bootstrap_iters, seed + 7),
        "n_patients": len(pat_label),
        "n_abn_patients": int(pat_label.sum()),
    }
    return slice_metrics, patient_metrics


def format_metrics(slice_metrics, patient_metrics):
    return "\n".join([
        format_image_metrics(slice_metrics),
        format_pixel_metrics(slice_metrics),
        format_patient_metrics(patient_metrics),
    ])
