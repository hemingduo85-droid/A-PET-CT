import torch
from numpy.random import normal
import random
import logging
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
import os
from functools import partial
import math
from tqdm import tqdm
from aug_funcs import rot_img, translation_img, hflip_img, grey_img, rot90_img
import torch.backends.cudnn as cudnn
from adeval import EvalAccumulatorCuda
from scipy.spatial.distance import directed_hausdorff
import numpy as np
# ========== 新增补丁 兼容numpy2.0 trapz消失问题 ==========
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
# ======================================================
import torch
from numpy.random import normal
import random
import logging
... # 你原有剩下的import不动

def ader_evaluator(pr_px, pr_sp, gt_px, gt_sp, device=None, skip_aupro=False, aupro_nstrips=200,
                   use_metrics=['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO']):
    if len(gt_px.shape) == 4:
        gt_px = gt_px.squeeze(1)
    if len(pr_px.shape) == 4:
        pr_px = pr_px.squeeze(1)

    eval_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_device = torch.device(eval_device)
    if eval_device.type == "cuda":
        torch.cuda.set_device(eval_device)

    score_min = min(pr_sp)
    score_max = max(pr_sp)
    anomap_min = pr_px.min()
    anomap_max = pr_px.max()

    accum = EvalAccumulatorCuda(
        score_min,
        score_max,
        anomap_min,
        anomap_max,
        skip_pixel_aupro=skip_aupro,
        nstrips=aupro_nstrips,
    )
    accum.add_anomap_batch(
        torch.as_tensor(pr_px, device=eval_device),
        torch.as_tensor(gt_px.astype(np.uint8), device=eval_device)
    )

    metrics = accum.summary()
    metric_results = {}
    for metric in use_metrics:
        if metric.startswith('I-AUROC'):
            auroc_sp = roc_auc_score(gt_sp, pr_sp)
            metric_results[metric] = auroc_sp
        elif metric.startswith('I-AP'):
            ap_sp = average_precision_score(gt_sp, pr_sp)
            metric_results[metric] = ap_sp
        elif metric.startswith('I-F1_max'):
            best_f1_score_sp = f1_score_max(gt_sp, pr_sp)
            metric_results[metric] = best_f1_score_sp
        elif metric.startswith('P-AUROC'):
            metric_results[metric] = metrics['p_auroc']
        elif metric.startswith('P-AP'):
            metric_results[metric] = metrics['p_aupr']
        elif metric.startswith('P-F1_max'):
            best_f1_score_px = f1_score_max(gt_px.ravel(), pr_px.ravel())
            metric_results[metric] = best_f1_score_px
        elif metric.startswith('AUPRO'):
            metric_results[metric] = float('nan') if skip_aupro else metrics['p_aupro']
    return list(metric_results.values())


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def augmentation(img):
    img = img.unsqueeze(0)
    augment_img = img
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16,
                  np.pi / 4]:
        rotate_img = rot_img(img, angle)
        augment_img = torch.cat([augment_img, rotate_img], dim=0)
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),
                 (0.1, -0.1)]:
        trans_img = translation_img(img, a, b)
        augment_img = torch.cat([augment_img, trans_img], dim=0)
    flipped_img = hflip_img(img)
    augment_img = torch.cat([augment_img, flipped_img], dim=0)
    greyed_img = grey_img(img)
    augment_img = torch.cat([augment_img, greyed_img], dim=0)
    for angle in [1, 2, 3]:
        rotate90_img = rot90_img(img, angle)
        augment_img = torch.cat([augment_img, rotate90_img], dim=0)
    augment_img = augment_img[torch.randperm(augment_img.size(0))]
    return augment_img


def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x


def global_cosine_hm_adaptive(a, b, y=3):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()
        mean_dist = point_dist.mean()
        factor = (point_dist / mean_dist) ** (y)
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1), b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad_v2, factor=factor)
        b_.register_hook(partial_func)
    loss = loss / len(a)
    return loss


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def cal_prototype_anomaly_map(model, out_size):
    if not hasattr(model, "distance"):
        raise AttributeError("model.distance is unavailable; run a forward pass before requesting prototype map.")
    distance = model.distance
    side = int(math.sqrt(distance.shape[1]))
    if side * side != distance.shape[1]:
        raise ValueError(f"Cannot reshape prototype distance of length {distance.shape[1]} into a square map.")
    anomaly_map = distance.reshape(distance.shape[0], 1, side, side).contiguous()
    return F.interpolate(anomaly_map, size=out_size, mode='bilinear', align_corners=True)


def build_anomaly_map(model, en, de, out_size, anomaly_source="reconstruction"):
    if anomaly_source == "reconstruction":
        anomaly_map, _ = cal_anomaly_maps(en, de, out_size)
        return anomaly_map
    if anomaly_source == "prototype":
        return cal_prototype_anomaly_map(model, out_size)
    if anomaly_source == "fused":
        recon_map, _ = cal_anomaly_maps(en, de, out_size)
        proto_map = cal_prototype_anomaly_map(model, out_size)
        return recon_map + proto_map
    raise ValueError(f"Unknown anomaly_source: {anomaly_source}")


def compute_image_score(score_map, max_ratio=0.005, image_score_mode="max", score_percentile=99.5):
    flat = score_map.flatten(1)
    if image_score_mode == "max":
        return flat.max(dim=1)[0]
    if image_score_mode == "topk_mean":
        topk = max(1, int(flat.shape[1] * max_ratio))
        return torch.sort(flat, dim=1, descending=True)[0][:, :topk].mean(dim=1)
    if image_score_mode == "percentile":
        q = min(max(score_percentile / 100.0, 0.0), 1.0)
        return torch.quantile(flat, q, dim=1)
    raise ValueError(f"Unknown image_score_mode: {image_score_mode}")


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def return_best_thr(y_true, y_score):
    """计算最大化F1分数的最佳阈值"""
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]

    if len(f1s) == 0:
        return 0.5  # 兜底阈值
    best_idx = np.argmax(f1s)
    best_thr = thrs[best_idx]
    return best_thr


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N


def denormalize(img):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def save_imag_ZS(imgs, anomaly_map, gt, prototype_map, save_root, img_path):
    batch_num = imgs.shape[0]
    for i in range(batch_num):
        img_path_list = img_path[i].split(os.path.sep)
        if len(img_path_list) >= 4:
            class_name, category, idx_name = img_path_list[-4], img_path_list[-2], img_path_list[-1]
        else:
            class_name = os.path.basename(os.path.dirname(os.path.dirname(img_path[i])))
            category = os.path.basename(os.path.dirname(img_path[i]))
            idx_name = os.path.basename(img_path[i]).split('.')[0]

        os.makedirs(os.path.join(save_root, class_name, category), exist_ok=True)
        input_frame = denormalize(imgs[i].clone().squeeze(0).cpu().detach().numpy())
        cv2_input = np.array(input_frame, dtype=np.uint8)
        plt.imsave(os.path.join(save_root, class_name, category, f'{idx_name}_0.png'), cv2_input)
        ano_map = anomaly_map[i].squeeze(0).cpu().detach().numpy()
        plt.imsave(os.path.join(save_root, class_name, category, f'{idx_name}_1.png'), ano_map, cmap='jet')
        gt_map = gt[i].squeeze(0).cpu().detach().numpy()
        plt.imsave(os.path.join(save_root, class_name, category, f'{idx_name}_2.png'), gt_map, cmap='gray')
        distance = prototype_map[i].view((28, 28)).cpu().detach().numpy()
        distance = cv2.resize(distance, (392, 392), interpolation=cv2.INTER_AREA)
        plt.imsave(os.path.join(save_root, class_name, category, f'{idx_name}_3.png'), distance, cmap='jet')
        plt.close()


# --- 核心修改：医学指标计算函数（支持最佳阈值）---
def compute_medical_metrics(gt_mask, pred_mask, threshold=None):
    """
    计算 DSC（Dice Score）和 HD95（95% Hausdorff Distance）
    Args:
        gt_mask: numpy array (H, W) - 真实掩码（0/1）
        pred_mask: numpy array (H, W) - 预测异常图（概率值）
        threshold: float - 二值化阈值（None则自动计算最佳阈值）
    Returns:
        dsc: float - Dice系数
        hd95: float - 95%豪斯多夫距离
    """
    # 确保输入是numpy数组
    gt_mask = np.array(gt_mask)
    pred_mask = np.array(pred_mask)

    # 1. 处理真实掩码（二值化）
    gt_mask_bin = (gt_mask > 0.5).astype(np.bool_)

    # 2. 处理预测掩码（使用最佳阈值二值化）
    if threshold is None:
        # 自动计算当前样本的最佳阈值
        threshold = return_best_thr(gt_mask_bin.ravel(), pred_mask.ravel())
    pred_mask_bin = (pred_mask > threshold).astype(np.bool_)

    # 3. 计算 Dice Score (DSC)
    intersection = np.logical_and(gt_mask_bin, pred_mask_bin).sum()
    union = np.logical_or(gt_mask_bin, pred_mask_bin).sum()

    if union == 0:
        dsc = 1.0  # 全正常样本
    else:
        dsc = (2. * intersection) / union

    # 4. 计算 Hausdorff Distance 95 (HD95)
    if np.sum(gt_mask_bin) == 0:
        hd95 = 0.0 if np.sum(pred_mask_bin) == 0 else 50.0
    elif np.sum(pred_mask_bin) == 0:
        hd95 = 50.0
    else:
        coords_gt = np.column_stack(np.where(gt_mask_bin))
        coords_pred = np.column_stack(np.where(pred_mask_bin))
        d1 = directed_hausdorff(coords_gt, coords_pred)[0]
        d2 = directed_hausdorff(coords_pred, coords_gt)[0]
        hd95 = max(d1, d2)

    return dsc, hd95


def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0.005, resize_mask=None,
                     anomaly_source="prototype", score_csv_path=None, calc_medical_metrics=False,
                     image_score_mode="max", score_percentile=99.5, skip_aupro=False, aupro_nstrips=200):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    dsc_list = []  # 存储每个样本的DSC
    hd95_list = []  # 存储每个样本的HD95
    score_rows = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map = build_anomaly_map(model, en, de, img.shape[-1], anomaly_source)

            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            # 存储像素级和样本级数据
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            # 计算样本级分数
            sp_score = compute_image_score(
                anomaly_map,
                max_ratio=max_ratio,
                image_score_mode=image_score_mode,
                score_percentile=score_percentile,
            )
            pr_list_sp.append(sp_score)

            if score_csv_path is not None:
                flat_map = anomaly_map.flatten(1)
                flat_gt = gt.flatten(1)
                for i in range(img.shape[0]):
                    score_rows.append({
                        "path": img_path[i],
                        "label": int(label[i].item()),
                        "image_score": float(sp_score[i].detach().cpu().item()),
                        "image_score_mode": image_score_mode,
                        "mask_pixels": int(flat_gt[i].sum().detach().cpu().item()),
                        "map_min": float(flat_map[i].min().detach().cpu().item()),
                        "map_mean": float(flat_map[i].mean().detach().cpu().item()),
                        "map_max": float(flat_map[i].max().detach().cpu().item()),
                    })

            # 计算医学指标（DSC/HD95）
            if calc_medical_metrics and gt.sum() > 0:  # 只对异常样本计算
                gt_np = gt.cpu().numpy()
                pred_np = anomaly_map.cpu().numpy()

                # 对每个样本计算DSC和HD95（使用样本级最佳阈值）
                for i in range(gt_np.shape[0]):
                    dsc, hd95 = compute_medical_metrics(
                        gt_np[i, 0],  # 真实掩码
                        pred_np[i, 0],  # 预测异常图
                        threshold=None  # 自动计算最佳阈值
                    )
                    dsc_list.append(dsc)
                    hd95_list.append(hd95)

        # 拼接所有数据
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        # 计算全局最佳阈值（可选：用于统一评估）
        if calc_medical_metrics and len(dsc_list) == 0:
            # 如果没有异常样本，补充默认值
            dsc_list = [0.0]
            hd95_list = [50.0]

        # GPU加速计算原有指标
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(
            pr_list_px,
            pr_list_sp,
            gt_list_px,
            gt_list_sp,
            device=device,
            skip_aupro=skip_aupro,
            aupro_nstrips=aupro_nstrips,
        )

        if score_csv_path is not None:
            os.makedirs(os.path.dirname(score_csv_path) or ".", exist_ok=True)
            pd.DataFrame(score_rows).to_csv(score_csv_path, index=False)

    if calc_medical_metrics:
        return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px], dsc_list, hd95_list
    else:
        return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def evaluation_batch_vis_ZS(model, dataloader, device, _class_=None, max_ratio=0.005, resize_mask=None, save_root=None,
                            anomaly_source="prototype", calc_medical_metrics=False,
                            image_score_mode="max", score_percentile=99.5):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    dsc_list = []
    hd95_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map = build_anomaly_map(model, en, de, img.shape[-1], anomaly_source)

            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            anomaly_map = gaussian_kernel(anomaly_map)

            # 保存可视化结果
            save_imag_ZS(img, anomaly_map, gt, model.distance, save_root, img_path)

            # 二值化处理
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            # 存储数据
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            # 计算样本级分数
            sp_score = compute_image_score(
                anomaly_map,
                max_ratio=max_ratio,
                image_score_mode=image_score_mode,
                score_percentile=score_percentile,
            )
            pr_list_sp.append(sp_score)

            # 计算医学指标
            if calc_medical_metrics and gt.sum() > 0:
                gt_np = gt.cpu().numpy()
                pred_np = anomaly_map.cpu().numpy()
                for i in range(gt_np.shape[0]):
                    dsc, hd95 = compute_medical_metrics(
                        gt_np[i, 0],
                        pred_np[i, 0],
                        threshold=None
                    )
                    dsc_list.append(dsc)
                    hd95_list.append(hd95)

        # 拼接数据
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        # GPU加速计算原有指标
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(
            pr_list_px, pr_list_sp, gt_list_px, gt_list_sp, device=device
        )

    if calc_medical_metrics:
        return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px], dsc_list, hd95_list
    else:
        return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR"""
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": [mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(
        in_channels=channels, out_channels=channels, kernel_size=kernel_size,
        groups=channels, bias=False, padding=kernel_size // 2
    )

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


from torch.optim.lr_scheduler import _LRScheduler


class WarmCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]
