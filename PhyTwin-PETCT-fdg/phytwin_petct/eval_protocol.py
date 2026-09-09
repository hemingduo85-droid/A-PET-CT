
"""Unified slice, pixel, and patient metrics for PhyTwin."""

import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


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


def f1_score_max(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.0
    p, r, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / (p + r + 1e-7)
    f1 = f1[:-1]
    return float(f1.max()) if len(f1) else 0.0


def _sens_at_spec(y_true, y_score, spec):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= (1.0 - spec)
    return float(tpr[ok].max()) if ok.any() else 0.0


def patient_id_from_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


def compute_metrics(gt_labels, gt_masks, anomaly_maps, image_scores, paths):
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
    px_auroc_abn = _safe_auroc(gt_px_abn, pr_px_abn)
    px_ap_abn = _safe_ap(gt_px_abn, pr_px_abn)
    if pr_px_abn.size:
        pmin, pmax = pr_px_abn.min(), pr_px_abn.max()
        px_f1_abn = f1_score_max(gt_px_abn, (pr_px_abn - pmin) / (pmax - pmin + 1e-8))
    else:
        px_f1_abn = 0.0

    slice_metrics = {
        "img_auroc": _safe_auroc(gt_labels, image_scores),
        "img_ap": _safe_ap(gt_labels, image_scores),
        "img_f1": f1_score_max(gt_labels, image_scores),
        "px_auroc_abn": px_auroc_abn,
        "px_ap_abn": px_ap_abn,
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
        "pat_auroc": _safe_auroc(pat_label, pat_score),
        "pat_ap": _safe_ap(pat_label, pat_score),
        "pat_f1": f1_score_max(pat_label, pat_score),
        "pat_sens90": _sens_at_spec(pat_label, pat_score, 0.90),
        "pat_sens95": _sens_at_spec(pat_label, pat_score, 0.95),
        "n_patients": len(pat_label),
        "n_abn_patients": int(pat_label.sum()),
    }
    return slice_metrics, pat_metrics


def format_metrics(slice_metrics, pat_metrics, as_percent=True):
    s = 100.0 if as_percent else 1.0
    m = slice_metrics
    lines = [
        f"[Slice-Img]  AUROC={m['img_auroc']*s:.2f}%  AP={m['img_ap']*s:.2f}%  F1={m['img_f1']*s:.2f}%",
        f"[Slice-Px(abn)]  AUROC={m['px_auroc_abn']*s:.2f}%  AP={m['px_ap_abn']*s:.2f}%  F1={m['px_f1_abn']*s:.2f}%",
    ]
    p = pat_metrics
    lines.append(
        f"[Patient({p['n_abn_patients']}/{p['n_patients']}abn)]  AUROC={p['pat_auroc']*s:.2f}%  "
        f"AP={p['pat_ap']*s:.2f}%  F1={p['pat_f1']*s:.2f}%  "
        f"Sens@90Spec={p['pat_sens90']*s:.2f}%  Sens@95Spec={p['pat_sens95']*s:.2f}%"
    )
    return "\n".join(lines)
