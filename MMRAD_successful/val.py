import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os
from torch.utils.data import DataLoader, Dataset
from dataloader import MultimodalGrayDataset
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import glob
import matplotlib.pyplot as plt
from model import MultiModalAnomalyDetector
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
# 新增：计算模型复杂度所需库
from thop import profile, clever_format


# ------------------- 扩展指标计算函数 -------------------
def calculate_dsc(pred_mask, gt_mask, threshold=0.5):
    """计算Dice Similarity Coefficient（DSC）"""
    pred_bin = (pred_mask >= threshold).astype(np.float32)
    gt_bin = (gt_mask > 0).astype(np.float32)

    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)

    if union == 0:
        return 1.0
    dice = (2.0 * intersection) / (union + 1e-8)
    return dice


def calculate_hd95(pred_mask, gt_mask, threshold=0.5):
    """计算95th Percentile Hausdorff Distance（HD95）"""
    pred_bin = (pred_mask >= threshold).astype(np.bool_)
    gt_bin = (gt_mask > 0).astype(np.bool_)

    if not np.any(pred_bin) and not np.any(gt_bin):
        return 0.0
    if not np.any(pred_bin) or not np.any(gt_bin):
        return np.sqrt(256 ** 2 + 256 ** 2)  # 替换inf为图像对角线长度

    pred_coords = np.argwhere(pred_bin)
    gt_coords = np.argwhere(gt_bin)

    # 计算HD95（改进版）
    pred_dist = distance_transform_edt(~pred_bin)
    gt_dist_values = pred_dist[gt_bin]
    hd95_1 = np.percentile(gt_dist_values, 95)

    gt_dist = distance_transform_edt(~gt_bin)
    pred_dist_values = gt_dist[pred_bin]
    hd95_2 = np.percentile(pred_dist_values, 95)

    hd95 = max(hd95_1, hd95_2)
    return hd95


def calculate_assd(pred_mask, gt_mask, threshold=0.5):
    """计算Average Symmetric Surface Distance（ASSD）"""
    pred_bin = (pred_mask >= threshold).astype(np.bool_)
    gt_bin = (gt_mask > 0).astype(np.bool_)

    if not np.any(pred_bin) and not np.any(gt_bin):
        return 0.0
    if not np.any(pred_bin) or not np.any(gt_bin):
        return 100.0  # 惩罚值

    def surface_distances(mask_a, mask_b):
        mask_a = mask_a.astype(bool)
        mask_b = mask_b.astype(bool)
        if not np.any(mask_a) or not np.any(mask_b):
            return np.array([0.0])
        dist_a = distance_transform_edt(~mask_a)[mask_b]
        dist_b = distance_transform_edt(~mask_b)[mask_a]
        return np.concatenate([dist_a, dist_b])

    distances = surface_distances(pred_bin, gt_bin)
    return np.mean(distances)


def calculate_ppv(pred_mask, gt_mask, threshold=0.5):
    """计算Positive Predictive Value（PPV/精确率）"""
    pred_bin = (pred_mask >= threshold).astype(np.float32)
    gt_bin = (gt_mask > 0).astype(np.float32)

    tp = np.sum(pred_bin * gt_bin)
    fp = np.sum(pred_bin * (1 - gt_bin))
    return tp / (tp + fp + 1e-8)


def calculate_sensitive(pred_mask, gt_mask, threshold=0.5):
    """计算Sensitivity（敏感度/召回率）"""
    pred_bin = (pred_mask >= threshold).astype(np.float32)
    gt_bin = (gt_mask > 0).astype(np.float32)

    tp = np.sum(pred_bin * gt_bin)
    fn = np.sum((1 - pred_bin) * gt_bin)
    return tp / (tp + fn + 1e-8)


# ------------------- 自定义Collate函数（保持不变） -------------------
def custom_collate_fn(batch):
    processed_batch = {'image': [], 'label': [], 'mask': []}

    for item in batch:
        # 处理图像
        img = item['image']
        if img.shape[1:] != (256, 256):
            img = F.interpolate(
                img.unsqueeze(0),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        processed_batch['image'].append(img)

        processed_batch['label'].append(item['label'])

        # 处理mask
        mask = item['mask']
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[1:] != (256, 256):
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=(256, 256),
                mode='nearest'
            ).squeeze(0)
        processed_batch['mask'].append(mask.squeeze(0) if len(mask.shape) == 3 else mask)

    processed_batch['image'] = torch.stack(processed_batch['image'])
    processed_batch['label'] = torch.tensor(processed_batch['label'])
    processed_batch['mask'] = torch.stack(processed_batch['mask'])

    return processed_batch


# ------------------- 模型复杂度计算函数 -------------------
def calculate_model_complexity(model, device, input_size=(1, 256, 256)):
    """计算模型的FLOPs和Params"""
    try:
        # 构建dummy input（匹配模型输入维度）
        dummy_input = torch.randn(1, *input_size).to(device)

        # 计算FLOPs和Params
        flops, params = profile(
            model,
            inputs=(dummy_input,),
            verbose=False,
            custom_ops={MultiModalAnomalyDetector: None}
        )

        # 格式化输出（单位：GFlops, MParams）
        flops, params = clever_format([flops, params], "%.2f")
        return flops, params
    except Exception as e:
        print(f"计算模型复杂度失败: {str(e)}")
        return "计算失败", "计算失败"


# ------------------- 纯评估函数（重构自原有训练函数） -------------------
def evaluate_model(weight_path, data_dir, modalities=['ct'], device='cuda:1'):
    """加载预训练权重并执行完整评估"""
    # 1. 初始化模型
    model = MultiModalAnomalyDetector(
        num_modalities=len(modalities),
        img_size=256,
        patch_size=8,
        embed_dim=32,
        depth=12
    ).to(device)

    # 2. 加载预训练权重
    print(f"Loading pretrained weights from: {weight_path}")
    try:
        checkpoint = torch.load(weight_path, map_location=device)

        # 兼容不同的权重格式
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print("Pretrained weights loaded successfully!")
    except Exception as e:
        print(f"加载权重失败: {str(e)}")
        return

    # 3. 加载测试数据集
    test_dataset = MultimodalGrayDataset(data_dir, modalities=modalities, mode='test')
    assert len(test_dataset) > 0, "测试集为空，请检查数据路径！"

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn
    )

    # 4. 模型评估
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()

    # 存储评估数据
    all_image_labels, all_image_scores = [], []
    all_pixel_labels, all_pixel_scores = [], []
    pro_curve_data = []

    # 存储所有像素级指标（用于计算标准差）
    dice_list, hd95_list, assd_list, ppv_list, sensitive_list = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            image_labels = batch['label'].cpu().numpy()
            pixel_labels = batch['mask'].cpu().numpy()

            # 前向传播
            reconstructions = model(images)
            loss = criterion(reconstructions, images)
            test_loss += loss.item()

            # 计算图像级异常分数
            image_anomaly_scores = F.mse_loss(reconstructions, images, reduction='none')
            image_anomaly_scores = image_anomaly_scores.mean(dim=(1, 2, 3)).cpu().numpy()

            # 计算像素级异常分数
            pixel_anomaly_scores = F.mse_loss(reconstructions, images, reduction='none')
            pixel_anomaly_scores = pixel_anomaly_scores.mean(dim=1, keepdim=True)
            pixel_anomaly_scores = F.interpolate(
                pixel_anomaly_scores,
                size=pixel_labels.shape[1:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1).cpu().numpy()

            # 收集图像级数据
            if len(image_labels) > 0:
                all_image_labels.append(image_labels)
                all_image_scores.append(image_anomaly_scores)

            # 收集像素级数据
            for b in range(pixel_anomaly_scores.shape[0]):
                flat_pixel_scores = pixel_anomaly_scores[b].flatten()
                flat_pixel_labels = pixel_labels[b].flatten()

                if len(flat_pixel_labels) > 0:
                    all_pixel_labels.append(flat_pixel_labels)
                    all_pixel_scores.append(flat_pixel_scores)

                    if image_labels[b] == 1:
                        pro_curve_data.append((flat_pixel_scores, flat_pixel_labels))

                # 计算所有像素级指标（仅对异常样本）
                if image_labels[b] == 1:
                    pred_score = pixel_anomaly_scores[b]
                    gt_mask = pixel_labels[b]

                    # 使用像素级分数的最佳阈值（基于F1最大化）
                    all_flat_scores = np.concatenate(all_pixel_scores) if all_pixel_scores else flat_pixel_scores
                    all_flat_labels = np.concatenate(all_pixel_labels) if all_pixel_labels else flat_pixel_labels

                    if len(np.unique(all_flat_labels)) > 1:
                        precision, recall, thresholds = precision_recall_curve(all_flat_labels, all_flat_scores)
                        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                        best_idx = np.argmax(f1_scores[:-1])
                        best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
                    else:
                        best_thr = np.median(all_flat_scores) if len(all_flat_scores) > 0 else 0.5

                    # 计算所有指标
                    dice = calculate_dsc(pred_score, gt_mask, threshold=best_thr)
                    hd95 = calculate_hd95(pred_score, gt_mask, threshold=best_thr)
                    assd = calculate_assd(pred_score, gt_mask, threshold=best_thr)
                    ppv = calculate_ppv(pred_score, gt_mask, threshold=best_thr)
                    sensitive = calculate_sensitive(pred_score, gt_mask, threshold=best_thr)

                    # 添加到列表
                    dice_list.append(dice)
                    hd95_list.append(hd95)
                    assd_list.append(assd)
                    ppv_list.append(ppv)
                    sensitive_list.append(sensitive)

    # 5. 计算所有指标
    avg_test_loss = test_loss / len(test_loader)

    # 图像级指标
    if len(all_image_labels) == 0:
        image_auroc = 0.0
        image_ap = 0.0
        image_f1 = 0.0
    else:
        all_image_labels = np.concatenate(all_image_labels)
        all_image_scores = np.concatenate(all_image_scores)
        image_auroc = roc_auc_score(all_image_labels, all_image_scores)
        image_ap = average_precision_score(all_image_labels, all_image_scores)

        image_precision, image_recall, _ = precision_recall_curve(all_image_labels, all_image_scores)
        image_f1_scores = 2 * (image_precision * image_recall) / (image_precision + image_recall + 1e-8)
        image_f1 = np.max(image_f1_scores)

    # 像素级指标
    if len(all_pixel_labels) == 0:
        pixel_auroc = 0.0
        pixel_ap = 0.0
        pixel_f1 = 0.0
        aupro = 0.0
    else:
        all_pixel_labels = np.concatenate(all_pixel_labels)
        all_pixel_scores = np.concatenate(all_pixel_scores)
        pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
        pixel_ap = average_precision_score(all_pixel_labels, all_pixel_scores)

        pixel_precision, pixel_recall, _ = precision_recall_curve(all_pixel_labels, all_pixel_scores)
        pixel_f1_scores = 2 * (pixel_precision * pixel_recall) / (pixel_precision + pixel_recall + 1e-8)
        pixel_f1 = np.max(pixel_f1_scores)

        aupro = calculate_aupro(pro_curve_data)

    # 医学指标（均值 + 标准差）
    metrics = {
        'dice_mean': np.mean(dice_list) if dice_list else 0.0,
        'dice_std': np.std(dice_list) if dice_list else 0.0,
        'hd95_mean': np.mean(hd95_list) if hd95_list else 0.0,
        'hd95_std': np.std(hd95_list) if hd95_list else 0.0,
        'assd_mean': np.mean(assd_list) if assd_list else 0.0,
        'assd_std': np.std(assd_list) if assd_list else 0.0,
        'ppv_mean': np.mean(ppv_list) if ppv_list else 0.0,
        'ppv_std': np.std(ppv_list) if ppv_list else 0.0,
        'sensitive_mean': np.mean(sensitive_list) if sensitive_list else 0.0,
        'sensitive_std': np.std(sensitive_list) if sensitive_list else 0.0
    }

    # 模型复杂度
    flops, params = calculate_model_complexity(model, device, input_size=(len(modalities), 256, 256))

    # 6. 打印完整结果
    print("\n" + "=" * 80)
    print("Final Evaluation Results")
    print("=" * 80)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"\nImage-Level Metrics:")
    print(f"  AUROC: {image_auroc:.4f}")
    print(f"  AP:    {image_ap:.4f}")
    print(f"  F1:    {image_f1:.4f}")
    print(f"\nPixel-Level Metrics:")
    print(f"  AUROC: {pixel_auroc:.4f}")
    print(f"  AP:    {pixel_ap:.4f}")
    print(f"  F1:    {pixel_f1:.4f}")
    print(f"  AUPRO: {aupro:.4f}")
    print(f"\nMedical Metrics (Mean ± Std):")
    print(f"  Dice:     {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"  HD95:     {metrics['hd95_mean']:.4f} ± {metrics['hd95_std']:.4f}")
    print(f"  ASSD:     {metrics['assd_mean']:.4f} ± {metrics['assd_std']:.4f}")
    print(f"  PPV:      {metrics['ppv_mean']:.4f} ± {metrics['ppv_std']:.4f}")
    print(f"  Sensitive:{metrics['sensitive_mean']:.4f} ± {metrics['sensitive_std']:.4f}")
    print(f"\nModel Complexity:")
    print(f"  FLOPs: {flops}")
    print(f"  Params: {params}")
    print("=" * 80 + "\n")

    # 7. 保存结果
    results = {
        'test_loss': avg_test_loss,
        'image_level': {
            'auroc': float(image_auroc),
            'ap': float(image_ap),
            'f1': float(image_f1)
        },
        'pixel_level': {
            'auroc': float(pixel_auroc),
            'ap': float(pixel_ap),
            'f1': float(pixel_f1),
            'aupro': float(aupro)
        },
        'medical_metrics': metrics,
        'model_complexity': {
            'flops': flops,
            'params': params
        }
    }

    # 保存为JSON文件
    with open('evaluation_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=4)

    print(f"评估结果已保存至: evaluation_results.json")

    return model, results


def calculate_aupro(pro_curve_data):
    """AUPRO计算方法（保持不变）"""
    if not pro_curve_data:
        return 0.0

    all_fpr_recall_pairs = []

    for scores, labels in pro_curve_data:
        if np.sum(labels) == 0:
            continue

        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]

        tp = np.cumsum(sorted_labels)
        fp = np.cumsum(1 - sorted_labels)

        total_positives = np.sum(sorted_labels)
        total_negatives = len(sorted_labels) - total_positives

        if total_positives == 0 or total_negatives == 0:
            continue

        recall = tp / total_positives
        fpr = fp / total_negatives

        recall = np.concatenate(([0], recall))
        fpr = np.concatenate(([0], fpr))

        all_fpr_recall_pairs.append((fpr, recall))

    if not all_fpr_recall_pairs:
        return 0.0

    max_fpr = 0.3
    fpr_grid = np.linspace(0, max_fpr, 100)

    interp_recalls = []
    for fpr, recall in all_fpr_recall_pairs:
        unique_fpr, indices = np.unique(fpr, return_index=True)
        unique_recall = recall[indices]

        interp_recall = np.interp(fpr_grid, unique_fpr, unique_recall,
                                  left=0.0, right=unique_recall[-1] if len(unique_recall) > 0 else 0.0)
        interp_recalls.append(interp_recall)

    mean_recall = np.mean(interp_recalls, axis=0)
    aupro = np.trapz(mean_recall, fpr_grid) / max_fpr

    return aupro


if __name__ == '__main__':
    # 配置参数
    weight_path = '/home/wuchangwei/dyx/successful/MMRAD_successful/best_model_ct.pth'
    data_dir = '/home/wuchangwei/dyx/successful/brain_data4'
    device = 'cuda:1'
    modalities = ['ct']

    # 执行纯评估（无训练）
    model, results = evaluate_model(
        weight_path=weight_path,
        data_dir=data_dir,
        modalities=modalities,
        device=device
    )


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# import numpy as np
# from datetime import datetime
# import os
# from torch.utils.data import DataLoader, Dataset
# from dataloader import MultimodalGrayDataset
# from PIL import Image
# from sklearn.metrics import average_precision_score, roc_auc_score
# import glob
# import matplotlib.pyplot as plt
# from model import MultiModalAnomalyDetector
# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
# # 新增：导入计算HD95需要的库
# from scipy.ndimage import distance_transform_edt
# from scipy.spatial.distance import directed_hausdorff
#
#
# # ------------------- 新增：DSC和HD95计算函数 -------------------
# def calculate_dsc(pred_mask, gt_mask, threshold=0.5):
#     """
#     计算Dice Similarity Coefficient（DSC）
#     Args:
#         pred_mask: 预测的像素级异常分数图 (numpy array), shape (H, W)
#         gt_mask: 真实掩码 (numpy array), shape (H, W)
#         threshold: 异常分数二值化阈值
#     Returns:
#         dice: DSC值（0-1之间，越高越好）
#     """
#     # 将异常分数二值化得到预测掩码
#     pred_bin = (pred_mask >= threshold).astype(np.float32)
#     gt_bin = (gt_mask > 0).astype(np.float32)  # 真实掩码非0即为异常
#
#     intersection = np.sum(pred_bin * gt_bin)
#     union = np.sum(pred_bin) + np.sum(gt_bin)
#
#     if union == 0:
#         return 1.0  # 无异常时DSC为1
#     dice = (2.0 * intersection) / (union + 1e-8)  # 防止除零
#     return dice
#
#
# def calculate_hd95(pred_mask, gt_mask, threshold=0.5):
#     """
#     计算95th Percentile Hausdorff Distance（HD95）
#     Args:
#         pred_mask: 预测的像素级异常分数图 (numpy array), shape (H, W)
#         gt_mask: 真实掩码 (numpy array), shape (H, W)
#         threshold: 异常分数二值化阈值
#     Returns:
#         hd95: HD95值（越低越好，单位为像素）
#     """
#     # 二值化
#     pred_bin = (pred_mask >= threshold).astype(np.bool_)
#     gt_bin = (gt_mask > 0).astype(np.bool_)
#
#     # 处理全0/全1的情况
#     if not np.any(pred_bin) and not np.any(gt_bin):
#         return 0.0
#     if not np.any(pred_bin) or not np.any(gt_bin):
#         return np.inf  # 无重叠时距离无穷大
#
#     # 获取掩码的坐标点
#     pred_coords = np.argwhere(pred_bin)
#     gt_coords = np.argwhere(gt_bin)
#
#     # 计算双向豪斯多夫距离
#     d1 = directed_hausdorff(pred_coords, gt_coords)[0]
#     d2 = directed_hausdorff(gt_coords, pred_coords)[0]
#     hd = max(d1, d2)
#
#     # 计算HD95（更鲁棒的版本）
#     # 步骤1：计算预测掩码到真实掩码的距离变换
#     pred_dist = distance_transform_edt(~pred_bin)
#     # 步骤2：取真实掩码区域内的距离值，计算95分位数
#     gt_dist_values = pred_dist[gt_bin]
#     hd95_1 = np.percentile(gt_dist_values, 95)
#
#     # 步骤3：反向计算真实掩码到预测掩码的距离变换
#     gt_dist = distance_transform_edt(~gt_bin)
#     # 步骤4：取预测掩码区域内的距离值，计算95分位数
#     pred_dist_values = gt_dist[pred_bin]
#     hd95_2 = np.percentile(pred_dist_values, 95)
#
#     # 最终HD95取两者最大值
#     hd95 = max(hd95_1, hd95_2)
#     return hd95
#
#
# # ------------------- 关键修改：自定义Collate函数确保尺寸统一 -------------------
# def custom_collate_fn(batch):
#     processed_batch = {'image': [], 'label': [], 'mask': []}
#
#     for item in batch:
#         # 处理图像（保持不变）
#         img = item['image']
#         if img.shape[1:] != (256, 256):
#             img = F.interpolate(
#                 img.unsqueeze(0),
#                 size=(256, 256),
#                 mode='bilinear',
#                 align_corners=False
#             ).squeeze(0)
#         processed_batch['image'].append(img)
#
#         processed_batch['label'].append(item['label'])
#
#         # 处理mask：移除nearest模式下的align_corners参数
#         mask = item['mask']
#         if len(mask.shape) == 2:
#             mask = mask.unsqueeze(0)
#         if mask.shape[1:] != (256, 256):
#             mask = F.interpolate(
#                 mask.unsqueeze(0),
#                 size=(256, 256),
#                 mode='nearest'  # 仅保留mode，移除align_corners
#             ).squeeze(0)
#         processed_batch['mask'].append(mask.squeeze(0) if len(mask.shape) == 3 else mask)
#
#     processed_batch['image'] = torch.stack(processed_batch['image'])
#     processed_batch['label'] = torch.tensor(processed_batch['label'])
#     processed_batch['mask'] = torch.stack(processed_batch['mask'])
#
#     return processed_batch
#
#
# # ------------------- 原有训练函数（新增DSC/HD95指标计算） -------------------
# def train_model(data_dir, modalities=['ct'], epochs=50, batch_size=8, lr=1e-4, device='cuda:1'):
#     model = MultiModalAnomalyDetector(
#         num_modalities=len(modalities),
#         img_size=256,
#         patch_size=8,
#         embed_dim=32,
#         depth=12
#     ).to(device)
#
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#
#     # 加载数据集（保持原有导入逻辑）
#     train_dataset = MultimodalGrayDataset(data_dir, modalities=modalities, mode='train')
#     test_dataset = MultimodalGrayDataset(data_dir, modalities=modalities, mode='test')
#
#     # 校验数据集长度，避免空数据报错
#     assert len(train_dataset) > 0, "训练集为空，请检查数据路径！"
#     assert len(test_dataset) > 0, "测试集为空，请检查数据路径！"
#
#     viz_samples = [test_dataset[-i - 1] for i in range(min(5, len(test_dataset)))]
#
#     # ------------------- 关键修改：DataLoader添加自定义Collate函数 -------------------
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=4,
#         collate_fn=custom_collate_fn  # 强制resize到256x256
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=2,
#         collate_fn=custom_collate_fn  # 强制resize到256x256
#     )
#
#     writer = SummaryWriter(log_dir=os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S')))
#     best_ap = 0.0
#
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0.0
#         for batch in tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{epochs}'):
#             images = batch['image'].to(device)
#             # 校验图像尺寸（可选，用于调试）
#             assert images.shape[2:] == (256, 256), f"图像尺寸错误：{images.shape}，预期256x256"
#
#             reconstructions = model(images)
#             loss = criterion(reconstructions, images)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#
#         model.eval()
#         test_loss = 0.0
#         all_image_labels, all_image_scores = [], []
#         all_pixel_labels, all_pixel_scores = [], []
#         # 新增：收集DSC和HD95计算所需数据
#         all_dsc = []
#         all_hd95 = []
#
#         # 用于计算AUPRO
#         pro_curve_data = []
#
#         with torch.no_grad():
#             for batch in tqdm(test_loader, desc=f'Test Epoch {epoch + 1}'):
#                 images = batch['image'].to(device)
#                 image_labels = batch['label'].cpu().numpy()
#                 pixel_labels = batch['mask'].cpu().numpy()
#
#                 reconstructions = model(images)
#                 loss = criterion(reconstructions, images)
#                 test_loss += loss.item()
#
#                 # 计算图像级异常分数
#                 image_anomaly_scores = F.mse_loss(reconstructions, images, reduction='none')
#                 image_anomaly_scores = image_anomaly_scores.mean(dim=(1, 2, 3)).cpu().numpy()
#
#                 # 计算像素级异常分数
#                 pixel_anomaly_scores = F.mse_loss(reconstructions, images, reduction='none')
#                 pixel_anomaly_scores = pixel_anomaly_scores.mean(dim=1, keepdim=True)  # 合并通道维度
#                 pixel_anomaly_scores = F.interpolate(
#                     pixel_anomaly_scores,
#                     size=pixel_labels.shape[1:],
#                     mode='bilinear',
#                     align_corners=False
#                 ).squeeze(1).cpu().numpy()
#
#                 # 收集图像级评估数据（添加非空校验）
#                 if len(image_labels) > 0:
#                     all_image_labels.append(image_labels)
#                     all_image_scores.append(image_anomaly_scores)
#
#                 # 收集像素级评估数据（添加非空校验）
#                 for b in range(pixel_anomaly_scores.shape[0]):
#                     flat_pixel_scores = pixel_anomaly_scores[b].flatten()
#                     flat_pixel_labels = pixel_labels[b].flatten()
#
#                     if len(flat_pixel_labels) > 0:
#                         all_pixel_labels.append(flat_pixel_labels)
#                         all_pixel_scores.append(flat_pixel_scores)
#
#                         # 为AUPRO计算收集数据
#                         if image_labels[b] == 1:  # 只处理异常样本
#                             pro_curve_data.append((flat_pixel_scores, flat_pixel_labels))
#
#                     # 新增：计算单样本的DSC和HD95
#                     pred_score = pixel_anomaly_scores[b]
#                     gt_mask = pixel_labels[b]
#                     # 使用像素级分数的中位数作为阈值（更鲁棒）
#                     threshold = np.median(all_pixel_scores) if len(all_pixel_scores) > 0 else 0.5
#                     dsc = calculate_dsc(pred_score, gt_mask, threshold=threshold)
#                     hd95 = calculate_hd95(pred_score, gt_mask, threshold=threshold)
#
#                     all_dsc.append(dsc)
#                     all_hd95.append(hd95)
#
#         # ------------------- 新增：计算DSC和HD95的平均值 -------------------
#         # 处理空数据情况
#         if len(all_dsc) == 0:
#             mean_dsc = 0.0
#             mean_hd95 = 0.0
#             print("警告：未收集到足够数据计算DSC/HD95，指标置0")
#         else:
#             mean_dsc = np.mean(all_dsc)
#             # 替换无穷大值为合理上限（如图像对角线长度）
#             all_hd95_clean = [x if x != np.inf else np.sqrt(256 ** 2 + 256 ** 2) for x in all_hd95]
#             mean_hd95 = np.mean(all_hd95_clean)
#
#         # ------------------- 原有指标计算（保持不变） -------------------
#         # 计算图像级指标
#         if len(all_image_labels) == 0:
#             print("警告：未收集到图像级标签数据，指标置0")
#             image_auroc = 0.0
#             image_ap = 0.0
#             image_f1 = 0.0
#         else:
#             all_image_labels = np.concatenate(all_image_labels)
#             all_image_scores = np.concatenate(all_image_scores)
#             image_auroc = roc_auc_score(all_image_labels, all_image_scores)
#             image_ap = average_precision_score(all_image_labels, all_image_scores)
#
#             # 计算图像级F1分数
#             image_precision, image_recall, _ = precision_recall_curve(all_image_labels, all_image_scores)
#             image_f1_scores = 2 * (image_precision * image_recall) / (image_precision + image_recall + 1e-8)
#             image_f1 = np.max(image_f1_scores)
#
#         # 计算像素级指标
#         if len(all_pixel_labels) == 0:
#             print("警告：未收集到像素级标签数据，指标置0")
#             pixel_auroc = 0.0
#             pixel_ap = 0.0
#             pixel_f1 = 0.0
#         else:
#             all_pixel_labels = np.concatenate(all_pixel_labels)
#             all_pixel_scores = np.concatenate(all_pixel_scores)
#             pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
#             pixel_ap = average_precision_score(all_pixel_labels, all_pixel_scores)
#
#             # 计算像素级F1分数
#             pixel_precision, pixel_recall, _ = precision_recall_curve(all_pixel_labels, all_pixel_scores)
#             pixel_f1_scores = 2 * (pixel_precision * pixel_recall) / (pixel_precision + pixel_recall + 1e-8)
#             pixel_f1 = np.max(pixel_f1_scores)
#
#         # 计算AUPRO
#         aupro = calculate_aupro(pro_curve_data)
#
#         # 记录指标
#         avg_train_loss = train_loss / len(train_loader)
#         avg_test_loss = test_loss / len(test_loader)
#
#         writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
#         writer.add_scalar('Loss/Test', avg_test_loss, epoch + 1)
#         writer.add_scalar('Metrics/Image_AUROC', image_auroc, epoch + 1)
#         writer.add_scalar('Metrics/Image_AP', image_ap, epoch + 1)
#         writer.add_scalar('Metrics/Image_F1', image_f1, epoch + 1)
#         writer.add_scalar('Metrics/Pixel_AUROC', pixel_auroc, epoch + 1)
#         writer.add_scalar('Metrics/Pixel_AP', pixel_ap, epoch + 1)
#         writer.add_scalar('Metrics/Pixel_F1', pixel_f1, epoch + 1)
#         writer.add_scalar('Metrics/AUPRO', aupro, epoch + 1)
#         # 新增：记录DSC和HD95
#         writer.add_scalar('Metrics/DSC', mean_dsc, epoch + 1)
#         writer.add_scalar('Metrics/HD95', mean_hd95, epoch + 1)
#         writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch + 1)
#
#         print(f"\nResults at Epoch {epoch + 1}:")
#         print(f"Image-Level-AUROC:{image_auroc:.4f}, AP:{image_ap:.4f}, F1:{image_f1:.4f}")
#         print(f"Pixel-Level-AUROC:{pixel_auroc:.4f}, AP:{pixel_ap:.4f}, F1:{pixel_f1:.4f}, AUPRO:{aupro:.4f}")
#         # 新增：打印DSC和HD95
#         print(f"Segmentation Metrics - DSC:{mean_dsc:.4f}, HD95:{mean_hd95:.4f}")
#
#         if image_ap > best_ap:
#             best_ap = image_ap
#             torch.save({
#                 'state_dict': model.state_dict(),
#                 'modalities': modalities,
#                 'metrics': {
#                     'Image_AUROC': image_auroc,
#                     'Image_AP': image_ap,
#                     'Image_F1': image_f1,
#                     'Pixel_AUROC': pixel_auroc,
#                     'Pixel_AP': pixel_ap,
#                     'Pixel_F1': pixel_f1,
#                     'AUPRO': aupro,
#                     # 新增：保存DSC和HD95
#                     'DSC': mean_dsc,
#                     'HD95': mean_hd95
#                 },
#                 'epoch': epoch + 1
#             }, f'best_model_{"_".join(modalities)}.pth')
#
#         scheduler.step()
#
#     writer.close()
#     return model
#
#
# def calculate_aupro(pro_curve_data):
#     """
#     更精确的AUPRO计算方法
#     """
#     if not pro_curve_data:
#         return 0.0
#
#     # 收集所有样本的PRO曲线数据
#     all_fpr_recall_pairs = []
#
#     for scores, labels in pro_curve_data:
#         if np.sum(labels) == 0:
#             continue
#
#         # 按分数排序
#         sorted_indices = np.argsort(scores)[::-1]  # 高分在前
#         sorted_scores = scores[sorted_indices]
#         sorted_labels = labels[sorted_indices]
#
#         # 计算累积统计量
#         tp = np.cumsum(sorted_labels)
#         fp = np.cumsum(1 - sorted_labels)
#
#         total_positives = np.sum(sorted_labels)
#         total_negatives = len(sorted_labels) - total_positives
#
#         if total_positives == 0 or total_negatives == 0:
#             continue
#
#         # 计算召回率和假阳性率
#         recall = tp / total_positives
#         fpr = fp / total_negatives
#
#         # 确保曲线从(0,0)开始
#         recall = np.concatenate(([0], recall))
#         fpr = np.concatenate(([0], fpr))
#
#         all_fpr_recall_pairs.append((fpr, recall))
#
#     if not all_fpr_recall_pairs:
#         return 0.0
#
#     # 在0-1范围内均匀采样100个fpr点
#     max_fpr = 0.3  # 通常AUPRO只计算到0.3 FPR
#     fpr_grid = np.linspace(0, max_fpr, 100)
#
#     # 为每个样本在fpr_grid上插值召回率
#     interp_recalls = []
#
#     for fpr, recall in all_fpr_recall_pairs:
#         # 去除重复的fpr值
#         unique_fpr, indices = np.unique(fpr, return_index=True)
#         unique_recall = recall[indices]
#
#         # 插值
#         interp_recall = np.interp(fpr_grid, unique_fpr, unique_recall,
#                                   left=0.0, right=unique_recall[-1] if len(unique_recall) > 0 else 0.0)
#         interp_recalls.append(interp_recall)
#
#     # 计算平均召回率
#     mean_recall = np.mean(interp_recalls, axis=0)
#
#     aupro = np.trapz(mean_recall, fpr_grid) / max_fpr
#
#     return aupro
#
#
# if __name__ == '__main__':
#     data_dir = '/home/wuchangwei/dyx/successful/brain_data2'
#     device = 'cuda:1'
#
#     trained_model = train_model(
#         data_dir=data_dir,
#         modalities=['ct'],
#         epochs=30,
#         batch_size=16,
#         lr=1e-4,
#         device=device
#     )




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# import numpy as np
# from datetime import datetime
# import os
# from torch.utils.data import DataLoader, Dataset
# from dataloader import MultimodalGrayDataset
# from PIL import Image
# from sklearn.metrics import average_precision_score, roc_auc_score
# import glob
# import matplotlib.pyplot as plt
# from model import MultiModalAnomalyDetector
# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
#
#
# # ------------------- 关键修改：自定义Collate函数确保尺寸统一 -------------------
# def custom_collate_fn(batch):
#     processed_batch = {'image': [], 'label': [], 'mask': []}
#
#     for item in batch:
#         # 处理图像（保持不变）
#         img = item['image']
#         if img.shape[1:] != (256, 256):
#             img = F.interpolate(
#                 img.unsqueeze(0),
#                 size=(256, 256),
#                 mode='bilinear',
#                 align_corners=False
#             ).squeeze(0)
#         processed_batch['image'].append(img)
#
#         processed_batch['label'].append(item['label'])
#
#         # 处理mask：移除nearest模式下的align_corners参数
#         mask = item['mask']
#         if len(mask.shape) == 2:
#             mask = mask.unsqueeze(0)
#         if mask.shape[1:] != (256, 256):
#             mask = F.interpolate(
#                 mask.unsqueeze(0),
#                 size=(256, 256),
#                 mode='nearest'  # 仅保留mode，移除align_corners
#             ).squeeze(0)
#         processed_batch['mask'].append(mask.squeeze(0) if len(mask.shape) == 3 else mask)
#
#     processed_batch['image'] = torch.stack(processed_batch['image'])
#     processed_batch['label'] = torch.tensor(processed_batch['label'])
#     processed_batch['mask'] = torch.stack(processed_batch['mask'])
#
#     return processed_batch
#
# # ------------------- 原有训练函数（仅修改DataLoader部分） -------------------
# def train_model(data_dir, modalities=['ct'], epochs=50, batch_size=8, lr=1e-4, device='cuda:1'):
#     model = MultiModalAnomalyDetector(
#         num_modalities=len(modalities),
#         img_size=256,
#         patch_size=8,
#         embed_dim=32,
#         depth=12
#     ).to(device)
#
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#
#     # 加载数据集（保持原有导入逻辑）
#     train_dataset = MultimodalGrayDataset(data_dir, modalities=modalities, mode='train')
#     test_dataset = MultimodalGrayDataset(data_dir, modalities=modalities, mode='test')
#
#     # 校验数据集长度，避免空数据报错
#     assert len(train_dataset) > 0, "训练集为空，请检查数据路径！"
#     assert len(test_dataset) > 0, "测试集为空，请检查数据路径！"
#
#     viz_samples = [test_dataset[-i - 1] for i in range(min(5, len(test_dataset)))]
#
#     # ------------------- 关键修改：DataLoader添加自定义Collate函数 -------------------
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=4,
#         collate_fn=custom_collate_fn  # 强制resize到256x256
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=2,
#         collate_fn=custom_collate_fn  # 强制resize到256x256
#     )
#
#     writer = SummaryWriter(log_dir=os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S')))
#     best_ap = 0.0
#
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0.0
#         for batch in tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{epochs}'):
#             images = batch['image'].to(device)
#             # 校验图像尺寸（可选，用于调试）
#             assert images.shape[2:] == (256, 256), f"图像尺寸错误：{images.shape}，预期256x256"
#
#             reconstructions = model(images)
#             loss = criterion(reconstructions, images)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#
#         model.eval()
#         test_loss = 0.0
#         all_image_labels, all_image_scores = [], []
#         all_pixel_labels, all_pixel_scores = [], []
#
#         # 用于计算AUPRO
#         pro_curve_data = []
#
#         with torch.no_grad():
#             for batch in tqdm(test_loader, desc=f'Test Epoch {epoch + 1}'):
#                 images = batch['image'].to(device)
#                 image_labels = batch['label'].cpu().numpy()
#                 pixel_labels = batch['mask'].cpu().numpy()
#
#                 reconstructions = model(images)
#                 loss = criterion(reconstructions, images)
#                 test_loss += loss.item()
#
#                 # 计算图像级异常分数
#                 image_anomaly_scores = F.mse_loss(reconstructions, images, reduction='none')
#                 image_anomaly_scores = image_anomaly_scores.mean(dim=(1, 2, 3)).cpu().numpy()
#
#                 # 计算像素级异常分数
#                 pixel_anomaly_scores = F.mse_loss(reconstructions, images, reduction='none')
#                 pixel_anomaly_scores = pixel_anomaly_scores.mean(dim=1, keepdim=True)  # 合并通道维度
#                 pixel_anomaly_scores = F.interpolate(
#                     pixel_anomaly_scores,
#                     size=pixel_labels.shape[1:],
#                     mode='bilinear',
#                     align_corners=False
#                 ).squeeze(1).cpu().numpy()
#
#                 # 收集图像级评估数据（添加非空校验）
#                 if len(image_labels) > 0:
#                     all_image_labels.append(image_labels)
#                     all_image_scores.append(image_anomaly_scores)
#
#                 # 收集像素级评估数据（添加非空校验）
#                 for b in range(pixel_anomaly_scores.shape[0]):
#                     flat_pixel_scores = pixel_anomaly_scores[b].flatten()
#                     flat_pixel_labels = pixel_labels[b].flatten()
#
#                     if len(flat_pixel_labels) > 0:
#                         all_pixel_labels.append(flat_pixel_labels)
#                         all_pixel_scores.append(flat_pixel_scores)
#
#                         # 为AUPRO计算收集数据
#                         if image_labels[b] == 1:  # 只处理异常样本
#                             pro_curve_data.append((flat_pixel_scores, flat_pixel_labels))
#
#         # ------------------- 新增：处理空数据情况，避免拼接报错 -------------------
#         # 计算图像级指标
#         if len(all_image_labels) == 0:
#             print("警告：未收集到图像级标签数据，指标置0")
#             image_auroc = 0.0
#             image_ap = 0.0
#             image_f1 = 0.0
#         else:
#             all_image_labels = np.concatenate(all_image_labels)
#             all_image_scores = np.concatenate(all_image_scores)
#             image_auroc = roc_auc_score(all_image_labels, all_image_scores)
#             image_ap = average_precision_score(all_image_labels, all_image_scores)
#
#             # 计算图像级F1分数
#             image_precision, image_recall, _ = precision_recall_curve(all_image_labels, all_image_scores)
#             image_f1_scores = 2 * (image_precision * image_recall) / (image_precision + image_recall + 1e-8)
#             image_f1 = np.max(image_f1_scores)
#
#         # 计算像素级指标
#         if len(all_pixel_labels) == 0:
#             print("警告：未收集到像素级标签数据，指标置0")
#             pixel_auroc = 0.0
#             pixel_ap = 0.0
#             pixel_f1 = 0.0
#         else:
#             all_pixel_labels = np.concatenate(all_pixel_labels)
#             all_pixel_scores = np.concatenate(all_pixel_scores)
#             pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
#             pixel_ap = average_precision_score(all_pixel_labels, all_pixel_scores)
#
#             # 计算像素级F1分数
#             pixel_precision, pixel_recall, _ = precision_recall_curve(all_pixel_labels, all_pixel_scores)
#             pixel_f1_scores = 2 * (pixel_precision * pixel_recall) / (pixel_precision + pixel_recall + 1e-8)
#             pixel_f1 = np.max(pixel_f1_scores)
#
#         # 计算AUPRO
#         aupro = calculate_aupro(pro_curve_data)
#
#         # 记录指标
#         avg_train_loss = train_loss / len(train_loader)
#         avg_test_loss = test_loss / len(test_loader)
#
#         writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
#         writer.add_scalar('Loss/Test', avg_test_loss, epoch + 1)
#         writer.add_scalar('Metrics/Image_AUROC', image_auroc, epoch + 1)
#         writer.add_scalar('Metrics/Image_AP', image_ap, epoch + 1)
#         writer.add_scalar('Metrics/Image_F1', image_f1, epoch + 1)
#         writer.add_scalar('Metrics/Pixel_AUROC', pixel_auroc, epoch + 1)
#         writer.add_scalar('Metrics/Pixel_AP', pixel_ap, epoch + 1)
#         writer.add_scalar('Metrics/Pixel_F1', pixel_f1, epoch + 1)
#         writer.add_scalar('Metrics/AUPRO', aupro, epoch + 1)
#         writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch + 1)
#
#         print(f"\nResults at Epoch {epoch + 1}:")
#         print(f"Image-Level-AUROC:{image_auroc:.4f}, AP:{image_ap:.4f}, F1:{image_f1:.4f}")
#         print(f"Pixel-Level-AUROC:{pixel_auroc:.4f}, AP:{pixel_ap:.4f}, F1:{pixel_f1:.4f}, AUPRO:{aupro:.4f}")
#
#         if image_ap > best_ap:
#             best_ap = image_ap
#             torch.save({
#                 'state_dict': model.state_dict(),
#                 'modalities': modalities,
#                 'metrics': {
#                     'Image_AUROC': image_auroc,
#                     'Image_AP': image_ap,
#                     'Image_F1': image_f1,
#                     'Pixel_AUROC': pixel_auroc,
#                     'Pixel_AP': pixel_ap,
#                     'Pixel_F1': pixel_f1,
#                     'AUPRO': aupro
#                 },
#                 'epoch': epoch + 1
#             }, f'best_model_{"_".join(modalities)}.pth')
#
#         scheduler.step()
#
#     writer.close()
#     return model
#
#
# def calculate_aupro(pro_curve_data):
#     """
#     更精确的AUPRO计算方法
#     """
#     if not pro_curve_data:
#         return 0.0
#
#     # 收集所有样本的PRO曲线数据
#     all_fpr_recall_pairs = []
#
#     for scores, labels in pro_curve_data:
#         if np.sum(labels) == 0:
#             continue
#
#         # 按分数排序
#         sorted_indices = np.argsort(scores)[::-1]  # 高分在前
#         sorted_scores = scores[sorted_indices]
#         sorted_labels = labels[sorted_indices]
#
#         # 计算累积统计量
#         tp = np.cumsum(sorted_labels)
#         fp = np.cumsum(1 - sorted_labels)
#
#         total_positives = np.sum(sorted_labels)
#         total_negatives = len(sorted_labels) - total_positives
#
#         if total_positives == 0 or total_negatives == 0:
#             continue
#
#         # 计算召回率和假阳性率
#         recall = tp / total_positives
#         fpr = fp / total_negatives
#
#         # 确保曲线从(0,0)开始
#         recall = np.concatenate(([0], recall))
#         fpr = np.concatenate(([0], fpr))
#
#         all_fpr_recall_pairs.append((fpr, recall))
#
#     if not all_fpr_recall_pairs:
#         return 0.0
#
#     # 在0-1范围内均匀采样100个fpr点
#     max_fpr = 0.3  # 通常AUPRO只计算到0.3 FPR
#     fpr_grid = np.linspace(0, max_fpr, 100)
#
#     # 为每个样本在fpr_grid上插值召回率
#     interp_recalls = []
#
#     for fpr, recall in all_fpr_recall_pairs:
#         # 去除重复的fpr值
#         unique_fpr, indices = np.unique(fpr, return_index=True)
#         unique_recall = recall[indices]
#
#         # 插值
#         interp_recall = np.interp(fpr_grid, unique_fpr, unique_recall,
#                                   left=0.0, right=unique_recall[-1] if len(unique_recall) > 0 else 0.0)
#         interp_recalls.append(interp_recall)
#
#     # 计算平均召回率
#     mean_recall = np.mean(interp_recalls, axis=0)
#
#     aupro = np.trapz(mean_recall, fpr_grid) / max_fpr
#
#     return aupro
#
#
# if __name__ == '__main__':
#     data_dir = '/home/wuchangwei/dyx/successful/brain_data'
#     device = 'cuda:1'
#
#     trained_model = train_model(
#         data_dir=data_dir,
#         modalities=['ct'],
#         epochs=30,
#         batch_size=16,
#         lr=1e-4,
#         device=device
#     )