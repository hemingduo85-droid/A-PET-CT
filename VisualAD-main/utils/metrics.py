"""Unified slice, pixel, and patient metrics with bootstrap 95% CI."""
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


N_BOOTSTRAP = 500
HIST_BINS = 16384


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
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    f1 = f1[:-1]
    return float(f1.max()) if len(f1) else 0.0


def patient_id_from_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


def _bootstrap_ci(y_true, y_score, metric_fn, iters=N_BOOTSTRAP, seed=20260717):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    value = metric_fn(y_true, y_score)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    if n == 0:
        return value, (value, value)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_fn(y_true[idx], y_score[idx]))
    if not values:
        return value, (value, value)
    low, high = np.percentile(values, [2.5, 97.5])
    return value, (float(low), float(high))


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
    auroc = float(np.trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)
    aupr = float(np.sum(precision * (tp / total_pos)))
    return auroc, aupr


def _pixel_slice_bootstrap(
    labels,
    masks,
    maps,
    iters=N_BOOTSTRAP,
    seed=20260727,
    bins=HIST_BINS,
    exact_auroc=None,
    exact_aupr=None,
    logger=None,
):
    keep = np.asarray(labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        return (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0)), (0.0, 0.0)

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

    binned_auroc, binned_aupr = _metrics_from_hist(pos_hists.sum(axis=0), neg_hists.sum(axis=0))
    auroc_value = binned_auroc if exact_auroc is None else float(exact_auroc)
    aupr_value = binned_aupr if exact_aupr is None else float(exact_aupr)

    rng = np.random.default_rng(seed)
    aurocs, auprs = [], []
    n = len(abn_idx)
    for i in range(iters):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if logger is not None and (i + 1) % 50 == 0:
            logger.info(f"Pixel histogram slice bootstrap: {i + 1}/{iters}")

    auroc_ci = tuple(float(v) for v in np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = tuple(float(v) for v in np.percentile(auprs, [2.5, 97.5]))
    return (auroc_value, auroc_ci), (aupr_value, aupr_ci), (binned_auroc, binned_aupr)


def _fmt(name, value_ci):
    value, ci = value_ci
    return f"{name}={value * 100:.2f}% (95% CI {ci[0] * 100:.2f}-{ci[1] * 100:.2f}%)"


def _image_line(img_auroc, img_aupr, img_f1):
    return "[Slice-Img]  " + "  ".join([
        _fmt("AUROC", img_auroc),
        _fmt("AUPR", img_aupr),
        _fmt("F1", img_f1),
    ])


def _pixel_line(px_auroc, px_aupr):
    return "[Slice-Px(abn)]  " + "  ".join([
        _fmt("AUROC", px_auroc),
        _fmt("AUPR", px_aupr),
    ])


def _patient_line(pat_labels, pat_auroc, pat_aupr, pat_f1):
    return f"[Patient({int(pat_labels.sum())}/{len(pat_labels)}abn)]  " + "  ".join([
        _fmt("AUROC", pat_auroc),
        _fmt("AUPR", pat_aupr),
        _fmt("F1", pat_f1),
    ])


def compute_protocol_metrics(labels, masks, maps, image_scores, paths, logger=None):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    masks = np.asarray(masks, dtype=np.float32)
    maps = np.asarray(maps, dtype=np.float32)
    image_scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    if logger is not None:
        logger.info("Computing image-level metrics and 95CI...")
    img_auroc = _bootstrap_ci(labels, image_scores, _safe_auroc, N_BOOTSTRAP, 20260717)
    img_aupr = _bootstrap_ci(labels, image_scores, _safe_ap, N_BOOTSTRAP, 20260718)
    img_f1 = _bootstrap_ci(labels, image_scores, f1_score_max, N_BOOTSTRAP, 20260719)
    if logger is not None:
        logger.info(_image_line(img_auroc, img_aupr, img_f1))

    if logger is not None:
        logger.info("Computing pixel-level Slice-Px(abn) metrics and 95CI...")
        logger.info("Computing exact full-pixel point estimates with sklearn...")
    keep = labels == 1
    if np.any(keep):
        pixel_gt = masks[keep].reshape(-1).astype(np.int32)
        pixel_score = maps[keep].reshape(-1).astype(np.float64)
        exact_px_auroc = _safe_auroc(pixel_gt, pixel_score)
        exact_px_aupr = _safe_ap(pixel_gt, pixel_score)
    else:
        exact_px_auroc = 0.0
        exact_px_aupr = 0.0
    px_auroc, px_aupr, binned = _pixel_slice_bootstrap(
        labels,
        masks,
        maps,
        N_BOOTSTRAP,
        20260727,
        HIST_BINS,
        exact_px_auroc,
        exact_px_aupr,
        logger,
    )
    if logger is not None:
        logger.info(_pixel_line(px_auroc, px_aupr))

    if logger is not None:
        logger.info("Computing patient-level metrics and 95CI...")
    pat_labels, pat_scores = _patient_arrays(paths, image_scores, labels)
    pat_auroc = _bootstrap_ci(pat_labels, pat_scores, _safe_auroc, N_BOOTSTRAP, 20260722)
    pat_aupr = _bootstrap_ci(pat_labels, pat_scores, _safe_ap, N_BOOTSTRAP, 20260723)
    pat_f1 = _bootstrap_ci(pat_labels, pat_scores, f1_score_max, N_BOOTSTRAP, 20260724)

    return {
        "img_auroc": img_auroc,
        "img_aupr": img_aupr,
        "img_f1": img_f1,
        "px_auroc": px_auroc,
        "px_aupr": px_aupr,
        "pat_labels": pat_labels,
        "pat_auroc": pat_auroc,
        "pat_aupr": pat_aupr,
        "pat_f1": pat_f1,
        "binned_px_auroc": binned[0],
        "binned_px_aupr": binned[1],
    }


def format_metrics(metrics):
    return "\n".join([
        _image_line(metrics["img_auroc"], metrics["img_aupr"], metrics["img_f1"]),
        _pixel_line(metrics["px_auroc"], metrics["px_aupr"]),
        _patient_line(metrics["pat_labels"], metrics["pat_auroc"], metrics["pat_aupr"], metrics["pat_f1"]),
    ])


def compute_metrics(results, obj_list, logger):
    labels, masks, maps, image_scores, paths = [], [], [], [], []

    for obj in obj_list:
        d = results[obj]
        labels.extend(int(x) for x in d["gt_sp"])
        image_scores.extend(float(x) for x in d["pr_sp"])
        paths.extend(d.get("img_paths", [""] * len(d["gt_sp"])))
        for mask_batch in d["imgs_masks"]:
            masks.append(mask_batch.squeeze().cpu().numpy().astype(np.float32))
        for amap_batch in d["anomaly_maps"]:
            maps.append(amap_batch.squeeze().cpu().numpy().astype(np.float32))

    metrics = compute_protocol_metrics(labels, masks, maps, image_scores, paths, logger)
    logger.info(format_metrics(metrics))

    return {
        "img_auroc": metrics["img_auroc"][0],
        "img_aupr": metrics["img_aupr"][0],
        "img_ap": metrics["img_aupr"][0],
        "img_f1": metrics["img_f1"][0],
        "px_auroc": metrics["px_auroc"][0],
        "px_aupr": metrics["px_aupr"][0],
        "pat_auroc": metrics["pat_auroc"][0],
        "pat_aupr": metrics["pat_aupr"][0],
        "pat_ap": metrics["pat_aupr"][0],
        "pat_f1": metrics["pat_f1"][0],
        "n_patients": len(metrics["pat_labels"]),
        "n_abn_patients": int(metrics["pat_labels"].sum()),
    }
