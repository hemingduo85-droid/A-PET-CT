import os
from skimage import measure
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
import numpy as np


def evaluate(labels, scores, metric='roc'):
    if metric == 'pro':
        return pro(labels, scores)
    if metric == 'roc':
        return roc(labels, scores)
    if metric == 'ap':
        return ap(labels, scores)
    if metric == 'f1':
        return f1(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")


def roc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def ap(labels, scores):
    """
    计算Average Precision
    :param labels: 真实标签
    :param scores: 预测分数
    :return: AP值
    """
    return average_precision_score(labels, scores)


def f1(labels, scores):
    """
    计算F1分数
    :param labels: 真实标签
    :param scores: 预测分数
    :return: 最佳F1值
    """
    thresholds = np.linspace(0, 1, 101)  # 生成多个阈值
    best_f1 = 0
    for threshold in thresholds:
        binary_scores = (scores > threshold).astype(int)
        current_f1 = f1_score(labels, binary_scores)
        best_f1 = max(best_f1, current_f1)  # 记录最佳F1值
    return best_f1


def rescale(x):
    denom = x.max() - x.min()
    if denom == 0:
        return np.zeros_like(x)
    return (x - x.min()) / denom


def pro(masks, scores):
    '''
        https://github.com/YoungGod/DFR/blob/a942f344570db91bc7feefc6da31825cf15ba3f9/DFR-source/anoseg_dfr.py#L447
    '''
    # per region overlap
    max_step = 4000
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(scores, dtype=bool)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1

        pro = []
        for i in range(len(binary_score_maps)):
            label_map = measure.label(masks[i], connectivity=2)
            props = measure.regionprops(label_map, binary_score_maps[i])
            for prop in props:
                pro.append(prop.intensity_image.sum() / prop.area)
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr
        masks_neg = ~masks
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    expect_fpr = 0.3
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr    # # rescale fpr [0, 0.3] -> [0, 1]
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)
    pros_mean_selected = rescale(pros_mean[idx])    # need scale
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score
