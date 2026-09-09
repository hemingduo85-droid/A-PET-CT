import math
import os
from functools import partial
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from scipy.spatial.distance import directed_hausdorff
from skimage import measure

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_curve,
    auc,
    f1_score,
)

# pandas 只用于 compute_pro（可选，但保留以便更稳定复现原行为）
try:
    import pandas as pd
except Exception:
    pd = None

# OpenCV 只用于可视化（可选）
try:
    import cv2
except Exception:
    cv2 = None


def modify_grad(x, inds, factor=0.0):
    """
    根据 inds 做梯度/特征按 mask 缩放（原工程逻辑）
    """
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def global_cosine(a, b, stop_grad=True):
    """
    a, b: feature list, 每个元素形状通常为 [B,C,H,W]
    """
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    weight = [1, 1, 1]

    for item in range(len(a)):
        if stop_grad:
            loss += (
                torch.mean(
                    1 - cos_loss(a[item].view(a[item].shape[0], -1).detach(),
                                 b[item].view(b[item].shape[0], -1))
                )
                * weight[item]
            )
        else:
            loss += (
                torch.mean(
                    1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                 b[item].view(b[item].shape[0], -1))
                )
                * weight[item]
            )
    return loss


def global_cosine_hm(a, b, alpha=1.0, factor=0.0):
    """
    重点：hard mining 逻辑 + register_hook（原工程常用）
    """
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    weight = [1, 1, 1]

    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]

        with torch.no_grad():
            # point_dist: [B,1,H,W] 或 [B,H,W] 取决于 cosine_similarity 输出，后续会广播
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)

        mean_dist = point_dist.mean()
        std_dist = point_dist.reshape(-1).std()

        # 主损失（全局）
        loss += (
            torch.mean(
                1 - cos_loss(a_.view(a_.shape[0], -1), b_.view(b_.shape[0], -1))
            )
            * weight[item]
        )

        # 硬挖掘阈值
        thresh = mean_dist + alpha * std_dist
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    return loss


def region_cosine(a, b):
    """
    原工程保留（少用）
    """
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += 1 - cos_loss(a[item].detach(), b[item]).mean()
    return loss


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode="mul", log=False):
    """
    单样本 anomaly map（返回 numpy）
    - fs_list, ft_list: list of encoder/decoder features
    - 输出：anomaly_map: [out_size, out_size] numpy
    """
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    if amap_mode == "mul":
        anomaly_map = np.ones(out_size, dtype=np.float32)
    else:
        anomaly_map = np.zeros(out_size, dtype=np.float32)

    a_map_list = []

    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        # [B,H,W]
        a_map = 1 - F.cosine_similarity(fs, ft, dim=1)
        # [B,1,H,W]
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)

        a_map_np = a_map[0, 0].detach().cpu().numpy()
        a_map_list.append(a_map_np)

        if amap_mode == "mul":
            anomaly_map *= a_map_np
        else:
            anomaly_map += a_map_np

    return anomaly_map, a_map_list


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    """
    批量 anomaly map（返回 torch）
    - 输出 anomaly_map: [B,1,out,out]
    """
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft, dim=1)  # [B,H,W]
        a_map = torch.unsqueeze(a_map, dim=1)          # [B,1,H,W]
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
        a_map_list.append(a_map)

    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)  # [B,1,out,out]
    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    """
    可视化：把 heatmap 叠到图像上
    img: numpy uint8, shape [H,W,3] 或单通道（按你的图而定）
    anomaly_map: numpy uint8 [H,W]
    """
    if cv2 is None:
        raise ImportError("cv2 not found. Please install opencv-python to use visualization.")

    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min + 1e-8)


def cvt2heatmap(gray):
    if cv2 is None:
        raise ImportError("cv2 not found. Please install opencv-python to use visualization.")
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / (N + 1e-8)


def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200) -> float:
    """
    计算 PRO（原工程常用版本）
    masks: [N,H,W]，binary {0,1}
    amaps: [N,H,W]，float anomaly map
    """
    assert isinstance(amaps, np.ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, np.ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3 and masks.ndim == 3, "amaps/masks must be [N,H,W]"
    assert amaps.shape == masks.shape, "amaps.shape must equal masks.shape"
    assert set(masks.flatten()) <= {0, 1}, "masks must be binary"
    assert isinstance(num_th, int)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th if max_th > min_th else 1e-6

    # 用 pandas 计算曲线更方便（但也可无 pandas）
    if pd is not None:
        df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    else:
        rows = []

    # 为了效率，先准备
    for th in np.arange(min_th, max_th, delta):
        binary_amaps = (amaps > th).astype(np.uint8)

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            labeled_mask = measure.label(mask)
            regions = measure.regionprops(labeled_mask)
            for region in regions:
                coords = region.coords
                tp_pixels = binary_amap[coords[:, 0], coords[:, 1]].sum()
                pros.append(tp_pixels / (region.area + 1e-8))

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / (inverse_masks.sum() + 1e-8)

        pro_val = mean(pros) if len(pros) > 0 else 0.0

        if pd is not None:
            df = pd.concat([df, pd.DataFrame([{"pro": pro_val, "fpr": fpr, "threshold": th}])],
                           ignore_index=True)
        else:
            rows.append((pro_val, fpr, th))

    if pd is not None:
        df = df[df["fpr"] < 0.3]
        if df.empty:
            return 0.0
        df["fpr"] = df["fpr"] / (df["fpr"].max() + 1e-8)
        return auc(df["fpr"], df["pro"])
    else:
        # fallback：没有 pandas 时仍尽可能计算
        filtered = [r for r in rows if r[1] < 0.3]  # r = (pro, fpr, th)
        if len(filtered) < 2:
            return 0.0
        fprs = np.array([r[1] for r in filtered], dtype=np.float64)
        pros = np.array([r[0] for r in filtered], dtype=np.float64)
        fprs = fprs / (fprs.max() + 1e-8)
        # 简易排序后积分
        idx = np.argsort(fprs)
        return auc(fprs[idx], pros[idx])


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    """
    depthwise gaussian blur (Conv2d 固定权重)
    """
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2

    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2.0 * variance)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,
        bias=False,
        padding=kernel_size // 2
    )
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


def replace_layers(model, old, new):
    """
    递归替换模型中的某类模块 old -> new
    old/new 通常是 nn.Module 类（例如 nn.ReLU -> nn.GELU）
    """
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, old, new)

        if isinstance(module, old):
            setattr(model, n, new)