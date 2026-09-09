"""
统一 PET-CT 异常检测评估协议

Slice-level 指标:
  image score  = top-1% pixel mean (或 max)，不用全图 mean
  Img AUROC/AUPR = 原始 image score vs 0/1 标签
  Img F1       = 原始 image score，PR 曲线取最大 F1
  Px  AUROC/AUPR = 原始 pixel score，仅异常切片 flatten
  每个指标报告 95% CI: 全部使用 bootstrap

Patient-level 指标（需传入 paths）:
  patient score = 该患者所有切片的 image score 取 max（默认）或 mean
  patient label = 该患者任一切片为异常则为 1
  Pat AUROC / Pat AUPR / Pat F1 = patient-level 指标

Best model 建议按 Image AUPR 选取。
"""

import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def image_score_from_map(anomaly_map_2d, agg='top1pct'):
    """
    从单张 anomaly map 聚合 image-level 分数。
    agg: 'top1pct' -> top-1% 像素均值 (默认) | 'max' -> 像素最大值
    """
    flat = np.asarray(anomaly_map_2d, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    if agg == 'max':
        return float(flat.max())
    k = max(1, int(len(flat) * 0.01))
    return float(np.partition(flat, -k)[-k:].mean())


def f1_score_max(y_true, y_score):
    """PR 曲线上取最大 F1 (F1-Max)."""
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.0
    precs, recs, _ = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return float(f1s.max()) if len(f1s) > 0 else 0.0


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _empty_ci():
    return (0.0, 0.0)


def _percentile_ci(values, alpha=0.95):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return _empty_ci()
    lo = (1.0 - alpha) / 2.0 * 100.0
    hi = (1.0 + alpha) / 2.0 * 100.0
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def _bootstrap_ci(y_true, y_score, metric_fn, n_bootstraps=500, seed=1203):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    point = metric_fn(y_true, y_score) if len(np.unique(y_true)) >= 2 else 0.0
    if len(np.unique(y_true)) < 2:
        return _empty_ci()
    if n_bootstraps <= 0:
        return (float(point), float(point))

    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_fn(y_true[idx], y_score[idx]))
    return _percentile_ci(values) if values else (float(point), float(point))


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


def _pixel_histogram_slice_bootstrap(gt_labels, gt_masks, anomaly_maps,
                                     n_bootstraps=500, seed=1203, bins=16384,
                                     exact_auroc=None, exact_aupr=None,
                                     progress_callback=None):
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    gt_masks = np.asarray(gt_masks)
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float64)
    abn_idx = np.flatnonzero(gt_labels == 1)
    if len(abn_idx) == 0:
        zero = _metric_dict(0.0, _empty_ci())
        return zero, zero

    score_min = float(np.min(anomaly_maps[abn_idx]))
    score_max = float(np.max(anomaly_maps[abn_idx]))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    scale = (bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        scores = anomaly_maps[idx].reshape(-1)
        mask = gt_masks[idx].reshape(-1).astype(bool)
        bin_idx = np.floor((scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[mask], minlength=bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~mask], minlength=bins).astype(np.uint32)

    binned_auroc, binned_aupr = _metrics_from_hist(
        pos_hists.sum(axis=0), neg_hists.sum(axis=0))
    auroc_value = binned_auroc if exact_auroc is None else float(exact_auroc)
    aupr_value = binned_aupr if exact_aupr is None else float(exact_aupr)
    if n_bootstraps <= 0:
        return (
            _metric_dict(auroc_value, (auroc_value, auroc_value)),
            _metric_dict(aupr_value, (aupr_value, aupr_value)),
        )

    rng = np.random.default_rng(seed)
    aurocs, auprs = [], []
    n = len(abn_idx)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        weights = np.bincount(idx, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if progress_callback is not None and (len(aurocs) % 50 == 0 or len(aurocs) == n_bootstraps):
            progress_callback(f"Pixel histogram slice bootstrap: {len(aurocs)}/{n_bootstraps}")

    return (
        _metric_dict(auroc_value, _percentile_ci(aurocs)),
        _metric_dict(aupr_value, _percentile_ci(auprs)),
    )


def _metric_dict(value, ci):
    return {'value': float(value), 'ci': ci}


def _extract_patient_id(path):
    """
    从切片路径提取患者 ID。
    路径结构: .../test/abnormal/<patient_id>/pet/0000.png
    patient_id = basename(dirname(dirname(path)))
    """
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


# ---------------------------------------------------------------------------
# Slice-level 统一指标
# ---------------------------------------------------------------------------

def _prepare_metric_arrays(gt_labels, gt_masks, anomaly_maps):
    gt_labels    = np.asarray(gt_labels,    dtype=np.int32).ravel()
    gt_masks     = np.asarray(gt_masks,     dtype=np.float32)
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)

    if anomaly_maps.ndim == 4:
        anomaly_maps = anomaly_maps.squeeze(1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)

    n = len(gt_labels)
    assert anomaly_maps.shape[0] == n and gt_masks.shape[0] == n
    return gt_labels, gt_masks, anomaly_maps


def compute_image_metrics(gt_labels, anomaly_maps, image_agg='top1pct',
                          n_bootstraps=500, seed=1203):
    gt_labels = np.asarray(gt_labels, dtype=np.int32).ravel()
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)
    if anomaly_maps.ndim == 4:
        anomaly_maps = anomaly_maps.squeeze(1)

    assert anomaly_maps.shape[0] == len(gt_labels)
    # image scores (per slice)
    pr_sp = np.array([image_score_from_map(anomaly_maps[i], agg=image_agg)
                      for i in range(len(gt_labels))], dtype=np.float64)
    gt_sp = gt_labels.astype(np.float32)

    img_auroc = _safe_auroc(gt_sp, pr_sp)
    img_aupr  = _safe_ap(gt_sp, pr_sp)
    img_f1    = f1_score_max(gt_sp, pr_sp)

    return {
        'img_auroc':    _metric_dict(
            img_auroc,
            _bootstrap_ci(gt_sp, pr_sp, _safe_auroc, n_bootstraps, seed)
        ),
        'img_aupr':     _metric_dict(
            img_aupr,
            _bootstrap_ci(gt_sp, pr_sp, _safe_ap, n_bootstraps, seed + 1)
        ),
        'img_f1':       _metric_dict(
            img_f1,
            _bootstrap_ci(gt_sp, pr_sp, f1_score_max, n_bootstraps, seed + 2)
        ),
        '_pr_sp':       pr_sp,
        '_gt_sp':       gt_sp,
    }


def compute_pixel_metrics(gt_labels, gt_masks, anomaly_maps, n_bootstraps=500,
                          seed=1203, hist_bins=16384, progress_callback=None):
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)
    if anomaly_maps.ndim == 4:
        anomaly_maps = anomaly_maps.squeeze(1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)

    keep = gt_labels == 1
    gt_masks_abn = gt_masks[keep]
    anomaly_maps_abn = anomaly_maps[keep]
    gt_px = gt_masks_abn.reshape(-1).astype(bool)
    pr_px = anomaly_maps_abn.reshape(-1).astype(np.float64)

    px_auroc = _safe_auroc(gt_px, pr_px)
    px_aupr  = _safe_ap(gt_px, pr_px)

    px_auroc_metric, px_aupr_metric = _pixel_histogram_slice_bootstrap(
        gt_labels,
        gt_masks,
        anomaly_maps,
        n_bootstraps=n_bootstraps,
        seed=seed + 10,
        bins=hist_bins,
        exact_auroc=px_auroc,
        exact_aupr=px_aupr,
        progress_callback=progress_callback,
    )
    return {'px_auroc': px_auroc_metric, 'px_aupr': px_aupr_metric}


def compute_unified_metrics(gt_labels, gt_masks, anomaly_maps, image_agg='top1pct',
                            n_bootstraps=500, seed=1203,
                            hist_bins=16384, progress_callback=None):
    """
    计算 slice-level 统一协议指标。

    返回 dict (值域 [0, 1]):
        Slice Image : img_auroc, img_aupr, img_f1
        Slice Pixel (abn only): px_auroc, px_aupr
        及 pr_sp (per-slice image score array, 供 patient-level 使用)
    """
    gt_labels, gt_masks, anomaly_maps = _prepare_metric_arrays(
        gt_labels, gt_masks, anomaly_maps)
    metrics = compute_image_metrics(
        gt_labels, anomaly_maps, image_agg=image_agg,
        n_bootstraps=n_bootstraps, seed=seed)
    metrics.update(compute_pixel_metrics(
        gt_labels, gt_masks, anomaly_maps, n_bootstraps=n_bootstraps,
        seed=seed, hist_bins=hist_bins,
        progress_callback=progress_callback))
    return metrics


# ---------------------------------------------------------------------------
# Patient-level 指标
# ---------------------------------------------------------------------------

def compute_patient_metrics(paths, pr_sp, gt_labels, patient_agg='max',
                            n_bootstraps=500, seed=1203):
    """
    聚合 slice-level 分数到 patient-level，计算患者级别指标。

    参数:
        paths       : list of str, 每个切片的文件路径 (用于提取患者 ID)
        pr_sp       : (N,) per-slice image scores
        gt_labels   : (N,) per-slice 0/1 标签
        patient_agg : 'max'  患者分数 = 所有切片分数的最大值 (默认, 更灵敏)
                      'mean' 患者分数 = 所有切片分数的均值

    返回 dict:
        pat_auroc     : 患者级 AUROC
        pat_aupr      : 患者级 AUPR
        pat_f1        : 患者级 F1-Max
        n_patients    : 总患者数
        n_abn_patients: 异常患者数
    """
    pr_sp     = np.asarray(pr_sp,     dtype=np.float64)
    gt_labels = np.asarray(gt_labels, dtype=np.int32)

    patient_scores = defaultdict(list)
    patient_labels = defaultdict(list)

    for path, score, label in zip(paths, pr_sp, gt_labels):
        pid = _extract_patient_id(path)
        patient_scores[pid].append(float(score))
        patient_labels[pid].append(int(label))

    pat_pr, pat_gt = [], []
    for pid in patient_scores:
        scores = np.array(patient_scores[pid])
        labels = np.array(patient_labels[pid])
        if patient_agg == 'max':
            pat_pr.append(float(scores.max()))
        else:
            pat_pr.append(float(scores.mean()))
        # 患者有任一异常切片则为异常患者
        pat_gt.append(int(labels.max()))

    pat_pr = np.array(pat_pr, dtype=np.float64)
    pat_gt = np.array(pat_gt, dtype=np.int32)

    return {
        'pat_auroc':      _metric_dict(
            _safe_auroc(pat_gt, pat_pr),
            _bootstrap_ci(pat_gt, pat_pr, _safe_auroc, n_bootstraps, seed + 200)
        ),
        'pat_aupr':       _metric_dict(
            _safe_ap(pat_gt, pat_pr),
            _bootstrap_ci(pat_gt, pat_pr, _safe_ap, n_bootstraps, seed + 201)
        ),
        'pat_f1':         _metric_dict(
            f1_score_max(pat_gt, pat_pr),
            _bootstrap_ci(pat_gt, pat_pr, f1_score_max, n_bootstraps, seed + 202)
        ),
        'n_patients':     len(pat_gt),
        'n_abn_patients': int(pat_gt.sum()),
    }


# ---------------------------------------------------------------------------
# 格式化输出
# ---------------------------------------------------------------------------

def _format_metric(metric, as_percent=True):
    s = 100.0 if as_percent else 1.0
    value = metric['value'] if isinstance(metric, dict) else float(metric)
    ci = metric.get('ci', _empty_ci()) if isinstance(metric, dict) else _empty_ci()
    return f"{value*s:.2f}% (95% CI {ci[0]*s:.2f}-{ci[1]*s:.2f}%)"


def format_image_metrics(slice_metrics, as_percent=True):
    m = slice_metrics
    return (
        f"[Slice-Img]  "
        f"AUROC={_format_metric(m['img_auroc'], as_percent)}  "
        f"AUPR={_format_metric(m['img_aupr'], as_percent)}  "
        f"F1={_format_metric(m['img_f1'], as_percent)}"
    )


def format_pixel_metrics(slice_metrics, as_percent=True):
    m = slice_metrics
    return (
        f"[Slice-Px(abn)]  "
        f"AUROC={_format_metric(m['px_auroc'], as_percent)}  "
        f"AUPR={_format_metric(m['px_aupr'], as_percent)}"
    )


def format_metrics(slice_metrics, pat_metrics=None, as_percent=True):
    """
    格式化为多行日志字符串。

    slice_metrics : compute_unified_metrics 的返回值
    pat_metrics   : compute_patient_metrics 的返回值 (可选)
    """
    m = slice_metrics

    line1 = format_image_metrics(m, as_percent=as_percent)
    line2 = format_pixel_metrics(m, as_percent=as_percent)
    lines = [line1, line2]

    if pat_metrics is not None:
        p = pat_metrics
        line3 = (
            f"[Patient({p['n_abn_patients']}/{p['n_patients']}abn)]  "
            f"AUROC={_format_metric(p['pat_auroc'], as_percent)}  "
            f"AUPR={_format_metric(p['pat_aupr'], as_percent)}  "
            f"F1={_format_metric(p['pat_f1'], as_percent)}"
        )
        lines.append(line3)

    return '\n'.join(lines)
