"""Anomaly metrics."""
import numpy as np
from sklearn import metrics


def _f1_score_max(y_true, y_score):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    f1_scores = f1_scores[:-1]
    if len(f1_scores) == 0:
        return 0.0, 0.5
    idx = np.argmax(f1_scores)
    return f1_scores[idx], thresholds[idx]


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes image-level retrieval statistics (AUROC, AP, F1).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    labels = np.asarray(anomaly_ground_truth_labels)
    scores = np.asarray(anomaly_prediction_weights)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    auroc = metrics.roc_auc_score(labels, scores)
    ap = metrics.average_precision_score(labels, scores)
    f1, optimal_threshold = _f1_score_max(labels, scores)

    return {
        "auroc": auroc,
        "ap": ap,
        "f1": f1,
        "optimal_threshold": optimal_threshold,
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    }


def compute_pixelwise_dice(anomaly_segmentations, ground_truth_masks, threshold):
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    pred_bin = (anomaly_segmentations >= threshold).astype(np.float32)
    gt_bin = ground_truth_masks.astype(np.float32)

    dice_scores = []
    for i in range(gt_bin.shape[0]):
        if np.sum(gt_bin[i]) == 0:
            continue
        intersection = np.sum(pred_bin[i] * gt_bin[i])
        union = np.sum(pred_bin[i]) + np.sum(gt_bin[i])
        dice_scores.append((2 * intersection) / (union + 1e-8))

    return float(np.mean(dice_scores)) if dice_scores else 0.0


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, AP, F1, Dice) for anomaly
    segmentations and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel().astype(int)

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )
    ap = metrics.average_precision_score(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )
    f1, optimal_threshold = _f1_score_max(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )
    dice = compute_pixelwise_dice(
        anomaly_segmentations, ground_truth_masks, optimal_threshold
    )

    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "ap": ap,
        "f1": f1,
        "dice": dice,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


def compute_all_evaluation_metrics(scores, anomaly_labels, segmentations, masks_gt):
    """Compute image-level and pixel-level evaluation metrics."""
    image_metrics = compute_imagewise_retrieval_metrics(scores, anomaly_labels)
    pixel_metrics = compute_pixelwise_retrieval_metrics(segmentations, masks_gt)

    sel_idxs = [i for i in range(len(masks_gt)) if np.sum(masks_gt[i]) > 0]
    anomaly_pixel_metrics = compute_pixelwise_retrieval_metrics(
        [segmentations[i] for i in sel_idxs],
        [masks_gt[i] for i in sel_idxs],
    )

    return {
        "instance_auroc": image_metrics["auroc"],
        "instance_ap": image_metrics["ap"],
        "instance_f1": image_metrics["f1"],
        "pixel_auroc": pixel_metrics["auroc"],
        "pixel_ap": pixel_metrics["ap"],
        "pixel_f1": pixel_metrics["f1"],
        "pixel_dice": pixel_metrics["dice"],
        "anomaly_pixel_auroc": anomaly_pixel_metrics["auroc"],
    }
