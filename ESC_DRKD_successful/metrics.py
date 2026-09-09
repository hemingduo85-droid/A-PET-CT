"""Evaluation metrics for ESC-DRKD."""

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def auc_safe(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def ap_safe(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def best_f1_and_thr(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.0, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    idx = int(np.argmax(f1))
    return float(f1[idx]), float(thresholds[idx])


def calculate_dsc(pred_bin, gt_bin):
    intersection = np.sum(pred_bin * gt_bin)
    return (2.0 * intersection) / (np.sum(pred_bin) + np.sum(gt_bin) + 1e-8)


def calculate_hd95(pred_bin, gt_bin):
    if np.sum(pred_bin) == 0 and np.sum(gt_bin) == 0:
        return 0.0
    if np.sum(pred_bin) == 0 or np.sum(gt_bin) == 0:
        return 100.0

    def distances(mask_a, mask_b):
        dt = distance_transform_edt(~mask_b.astype(bool))
        return dt[mask_a.astype(bool)]

    d1 = distances(pred_bin, gt_bin)
    d2 = distances(gt_bin, pred_bin)
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def calculate_assd(pred_bin, gt_bin):
    if np.sum(pred_bin) == 0 and np.sum(gt_bin) == 0:
        return 0.0
    if np.sum(pred_bin) == 0 or np.sum(gt_bin) == 0:
        return 100.0

    def surface_distances(mask_a, mask_b):
        mask_a = mask_a.astype(bool)
        mask_b = mask_b.astype(bool)
        if not np.any(mask_a) or not np.any(mask_b):
            return np.array([0.0])
        return np.concatenate(
            [
                distance_transform_edt(~mask_a)[mask_b],
                distance_transform_edt(~mask_b)[mask_a],
            ]
        )

    return float(np.mean(surface_distances(pred_bin, gt_bin)))


def calculate_ppv(pred_bin, gt_bin):
    tp = np.sum(pred_bin * gt_bin)
    fp = np.sum(pred_bin * (1 - gt_bin))
    return tp / (tp + fp + 1e-8)


def calculate_sensitive(pred_bin, gt_bin):
    tp = np.sum(pred_bin * gt_bin)
    fn = np.sum((1 - pred_bin) * gt_bin)
    return tp / (tp + fn + 1e-8)


def calculate_aupro(anomaly_maps, ground_truth_masks, num_thresholds=100):
    anomaly_maps = [np.asarray(am) for am in anomaly_maps]
    ground_truth_masks = [np.asarray(gt) for gt in ground_truth_masks]
    all_scores = np.concatenate([am.reshape(-1) for am in anomaly_maps])
    if all_scores.size == 0 or np.max(all_scores) == np.min(all_scores):
        return 0.0

    thresholds = np.linspace(np.min(all_scores), np.max(all_scores), num_thresholds)
    recalls, fprs = [], []
    for threshold in thresholds:
        total_tp = total_fn = total_fpr = valid = 0
        for anomaly_map, gt_mask in zip(anomaly_maps, ground_truth_masks):
            pred_mask = (anomaly_map > threshold).astype(np.float32)
            gt_mask = (gt_mask > 0.5).astype(np.float32)
            total_tp += np.sum((pred_mask == 1) & (gt_mask == 1))
            total_fn += np.sum((pred_mask == 0) & (gt_mask == 1))
            normal_pixels = np.prod(gt_mask.shape) - np.sum(gt_mask)
            if normal_pixels > 0:
                total_fpr += np.sum((pred_mask == 1) & (gt_mask == 0)) / normal_pixels
                valid += 1
        recalls.append(total_tp / (total_tp + total_fn + 1e-8))
        fprs.append(total_fpr / (valid + 1e-8))

    order = np.argsort(fprs)
    return float(np.trapz(np.asarray(recalls)[order], np.asarray(fprs)[order]))
