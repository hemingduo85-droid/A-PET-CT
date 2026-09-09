from collections import defaultdict
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
        parts = path_str.split("/")[-1].split("__")
        if len(parts) >= 2:
            return parts[1]
    parts = path_str.replace("\\", "/").split("/")
    return parts[-3] if len(parts) >= 3 else path_str


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_aupr(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _max_f1(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-7)
    return float(np.max(f1)) if f1.size else 0.0


def _bootstrap_metric(y_true, y_score, metric_fn, iters, seed):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    value = metric_fn(y_true, y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return value, (value, value)
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
    ci = tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))
    return value, ci


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
    auc_integral = getattr(np, "trapezoid", np.trapz)
    auroc = float(auc_integral(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)
    aupr = float(np.sum(precision * (tp / total_pos)))
    return auroc, aupr


def _pixel_slice_bootstrap(labels, masks, maps, iters, seed, bins, exact_auroc, exact_aupr, progress_callback=None):
    keep = np.asarray(labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        return (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0))

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
    aurocs = []
    auprs = []
    n = len(abn_idx)
    for _ in range(iters):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        done = len(aurocs)
        if progress_callback is not None and (done % 50 == 0 or done == iters):
            progress_callback(f"Pixel histogram slice bootstrap: {done}/{iters}")

    auroc_ci = tuple(float(v) for v in np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = tuple(float(v) for v in np.percentile(auprs, [2.5, 97.5]))
    return (float(exact_auroc), auroc_ci), (float(exact_aupr), aupr_ci)


def _patient_arrays(paths, image_scores, labels):
    patient_scores = defaultdict(list)
    patient_labels = defaultdict(list)
    for path, score, label_value in zip(paths, image_scores, labels):
        pid = patient_id_from_path(path)
        patient_scores[pid].append(float(score))
        patient_labels[pid].append(int(label_value))
    pat_score, pat_label = [], []
    for pid in patient_scores:
        pat_score.append(float(np.max(patient_scores[pid])))
        pat_label.append(int(np.max(patient_labels[pid])))
    return np.asarray(pat_label, dtype=np.int32), np.asarray(pat_score, dtype=np.float64)


def _metric(value, ci=None, p_value=None):
    return {"value": float(value), "ci": ci, "p": p_value}


def eval_protocol_compute_metrics(
    gt_labels,
    gt_masks,
    anomaly_maps,
    image_scores,
    paths,
    bootstrap_iters=500,
    hist_bins=16384,
    random_state=20260717,
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
    img_auroc = _bootstrap_metric(gt_labels, image_scores, _safe_auroc, bootstrap_iters, random_state)
    img_aupr = _bootstrap_metric(gt_labels, image_scores, _safe_aupr, bootstrap_iters, random_state + 1)
    img_f1 = _bootstrap_metric(gt_labels, image_scores, _max_f1, bootstrap_iters, random_state + 2)
    slice_metrics = {
        "img_auroc": _metric(img_auroc[0], img_auroc[1]),
        "img_aupr": _metric(img_aupr[0], img_aupr[1]),
        "img_f1": _metric(img_f1[0], img_f1[1]),
    }
    if progress_callback is not None:
        progress_callback(eval_protocol_format_image_metrics(slice_metrics))

    if progress_callback is not None:
        progress_callback("Computing pixel-level Slice-Px(abn) metrics and 95CI...")
        progress_callback("Computing exact full-pixel point estimates with sklearn...")
    keep = gt_labels == 1
    px_true = gt_masks[keep].reshape(-1).astype(bool)
    px_score = maps[keep].reshape(-1).astype(np.float64)
    exact_px_auroc = _safe_auroc(px_true, px_score)
    exact_px_aupr = _safe_aupr(px_true, px_score)
    px_auroc, px_aupr = _pixel_slice_bootstrap(
        gt_labels,
        gt_masks,
        maps,
        bootstrap_iters,
        random_state + 10,
        hist_bins,
        exact_px_auroc,
        exact_px_aupr,
        progress_callback=progress_callback,
    )
    slice_metrics.update(
        {
            "px_auroc_abn": _metric(px_auroc[0], px_auroc[1]),
            "px_aupr_abn": _metric(px_aupr[0], px_aupr[1]),
        }
    )
    if progress_callback is not None:
        progress_callback(eval_protocol_format_pixel_metrics(slice_metrics))

    if progress_callback is not None:
        progress_callback("Computing patient-level metrics and 95CI...")
    pat_label, pat_score = _patient_arrays(paths, image_scores, gt_labels)
    pat_auroc = _bootstrap_metric(pat_label, pat_score, _safe_auroc, bootstrap_iters, random_state + 5)
    pat_aupr = _bootstrap_metric(pat_label, pat_score, _safe_aupr, bootstrap_iters, random_state + 6)
    pat_f1 = _bootstrap_metric(pat_label, pat_score, _max_f1, bootstrap_iters, random_state + 7)
    pat_metrics = {
        "pat_auroc": _metric(pat_auroc[0], pat_auroc[1]),
        "pat_aupr": _metric(pat_aupr[0], pat_aupr[1]),
        "pat_f1": _metric(pat_f1[0], pat_f1[1]),
        "n_patients": len(pat_label),
        "n_abn_patients": int(pat_label.sum()),
    }
    return slice_metrics, pat_metrics


def _value(metric):
    return float(metric["value"]) if isinstance(metric, dict) else float(metric)


def _fmt_metric(name, metric, scale):
    value = _value(metric) * scale
    ci = metric.get("ci") if isinstance(metric, dict) else None
    if ci is None or np.any(np.isnan(ci)):
        return f"{name}={value:.2f}% (95% CI NA)"
    return f"{name}={value:.2f}% (95% CI {ci[0] * scale:.2f}-{ci[1] * scale:.2f}%)"


def eval_protocol_format_image_metrics(slice_metrics, as_percent=True):
    s = 100.0 if as_percent else 1.0
    return (
        "[Slice-Img]  "
        + "  ".join(
            [
                _fmt_metric("AUROC", slice_metrics["img_auroc"], s),
                _fmt_metric("AUPR", slice_metrics["img_aupr"], s),
                _fmt_metric("F1", slice_metrics["img_f1"], s),
            ]
        )
    )


def eval_protocol_format_pixel_metrics(slice_metrics, as_percent=True):
    s = 100.0 if as_percent else 1.0
    return (
        "[Slice-Px(abn)]  "
        + "  ".join(
            [
                _fmt_metric("AUROC", slice_metrics["px_auroc_abn"], s),
                _fmt_metric("AUPR", slice_metrics["px_aupr_abn"], s),
            ]
        )
    )


def eval_protocol_format_metrics(slice_metrics, pat_metrics, as_percent=True):
    s = 100.0 if as_percent else 1.0
    lines = [
        eval_protocol_format_image_metrics(slice_metrics, as_percent=as_percent),
        eval_protocol_format_pixel_metrics(slice_metrics, as_percent=as_percent),
    ]
    lines.append(
        f"[Patient({pat_metrics['n_abn_patients']}/{pat_metrics['n_patients']}abn)]  "
        + "  ".join(
            [
                _fmt_metric("AUROC", pat_metrics["pat_auroc"], s),
                _fmt_metric("AUPR", pat_metrics["pat_aupr"], s),
                _fmt_metric("F1", pat_metrics["pat_f1"], s),
            ]
        )
    )
    return "\n".join(lines)
