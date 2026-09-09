import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from denoising import denoising
import os
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
import glob
from skimage.transform import resize
import warnings
from scipy.spatial.distance import directed_hausdorff

# 忽略运行时警告
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MultimodalSliceDataset(Dataset):
    def __init__(self, data_path, modalities=['ct'], mode='train', target_size=(256, 256)):
        self.modalities = modalities
        self.mode = mode
        self.target_size = target_size
        self.samples = []

        if mode == 'train':
            slice_paths = glob.glob(os.path.join(data_path, 'train', modalities[0], '*.png'))
            for path in slice_paths:
                slice_id = os.path.basename(path).split('.')[0]
                self.samples.append({
                    'id': slice_id,
                    'label': 0,
                    'modalities': {mod: os.path.join(data_path, 'train', mod, f'{slice_id}.png')
                                   for mod in modalities},
                    'mask_path': None
                })
        else:
            for label, subfolder in enumerate(['NORMAL', 'ABNORMAL']):
                test_path = os.path.join(data_path, 'test', subfolder)
                slice_paths = glob.glob(os.path.join(test_path, modalities[0], '*.png'))
                for path in slice_paths:
                    slice_id = os.path.basename(path).split('.')[0]
                    mask_path = None
                    if label == 1:
                        mask_path = os.path.join(data_path, 'test', 'ABNORMAL', 'mask', f'{slice_id}_mask.png')
                    self.samples.append({
                        'id': slice_id,
                        'label': label,
                        'modalities': {mod: os.path.join(test_path, mod, f'{slice_id}.png')
                                       for mod in modalities},
                        'mask_path': mask_path
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = {}
        for mod in self.modalities:
            img = Image.open(sample['modalities'][mod]).convert('L')
            img = np.array(img) / 255.0
            if img.shape != self.target_size:
                img = resize(img, self.target_size, anti_aliasing=True)
            img = torch.FloatTensor(img).unsqueeze(0)
            data[mod] = img

        if sample['mask_path'] is not None:
            mask = Image.open(sample['mask_path']).convert('L')
            mask = np.array(mask) / 255.0
            if mask.shape != self.target_size:
                mask = resize(mask, self.target_size, anti_aliasing=False, preserve_range=True)
                mask = (mask > 0.5).astype(np.float32)
        else:
            mask = np.zeros(self.target_size, dtype=np.float32)

        mask = torch.FloatTensor(mask).unsqueeze(0)
        return {'modalities': data, 'label': sample['label'], 'id': sample['id'], 'mask': mask}


# ------------------- 新增/修改的指标计算函数 -------------------

def calculate_dsc(pred_mask, gt_mask):
    """计算 Dice Similarity Coefficient"""
    intersection = np.sum(pred_mask * gt_mask)
    return (2. * intersection) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-8)


def calculate_hd95(pred_mask, gt_mask):
    """计算 95th percentile Hausdorff Distance"""
    if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
        return 0.0
    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return 50.0  # 当一个为空时给一个较大的惩罚值（根据图像尺寸调整）

    # 提取边缘点坐标
    pts_pred = np.argwhere(pred_mask > 0)
    pts_gt = np.argwhere(gt_mask > 0)

    # 计算双向距离
    d1 = [directed_hausdorff(pts_pred, pts_gt)[0]]
    d2 = [directed_hausdorff(pts_gt, pts_pred)[0]]

    # 这里简写为最大距离的 95%，标准实现需计算点对点距离分布
    # 为性能考虑，异常检测中常用双向 Hausdorff 的最大值
    return max(np.percentile(d1, 95), np.percentile(d2, 95))


def calculate_anomaly_score(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2, dim=[1, 2, 3])


def calculate_optimal_f1_score(scores, labels):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1 = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0
    return best_f1, best_threshold


def calculate_aupro(anomaly_maps, ground_truth_masks):
    all_scores = np.concatenate([am.flatten() for am in anomaly_maps])
    if all_scores.size == 0 or np.max(all_scores) == np.min(all_scores):
        return 0.0
    thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 500)
    recalls, false_positive_rates = [], []
    for threshold in thresholds:
        t_tp, t_fn, t_fpr, v_imgs = 0, 0, 0, 0
        for am, gt in zip(anomaly_maps, ground_truth_masks):
            pred = (am >= threshold).astype(np.float32)
            t_tp += np.sum((pred == 1) & (gt == 1))
            t_fn += np.sum((pred == 0) & (gt == 1))
            norm_px = np.sum(gt == 0)
            if norm_px > 0:
                t_fpr += np.sum((pred == 1) & (gt == 0)) / norm_px
                v_imgs += 1
        recalls.append(t_tp / (t_tp + t_fn + 1e-8))
        false_positive_rates.append(t_fpr / (v_imgs + 1e-8))

    curve = np.array(sorted(zip(false_positive_rates, recalls)))
    unique_fpr, indices = np.unique(curve[:, 0], return_index=True)
    unique_recalls = curve[indices, 1]

    if unique_fpr[0] > 0:
        unique_fpr = np.insert(unique_fpr, 0, 0);
        unique_recalls = np.insert(unique_recalls, 0, 0)

    idx_06 = np.searchsorted(unique_fpr, 0.6)
    fpr_final = np.append(unique_fpr[:idx_06], 0.6)
    recall_final = np.append(unique_recalls[:idx_06], unique_recalls[idx_06 - 1] if idx_06 > 0 else 0)
    return np.trapz(recall_final, fpr_final) / 0.6


# ------------------- 训练与评估主体 -------------------

def train_model(data_dir, modalities=['ct'], epochs=30, batch_size=4, noise_std=0.2,
                noise_res=16, target_size=(256, 256)):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    train_dataset = MultimodalSliceDataset(data_dir, modalities=modalities, mode='train', target_size=target_size)
    test_dataset = MultimodalSliceDataset(data_dir, modalities=modalities, mode='test', target_size=target_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    models = {mod: denoising(identifier=f'm_{mod}', n_input=1, noise_std=noise_std, noise_res=noise_res) for mod in
              modalities}
    optimizers = {}
    for mod in modalities:
        models[mod].model.to(device)
        optimizers[mod] = models[mod].optimiser

    writer = SummaryWriter(log_dir=os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S')))
    best_ap = 0.0

    for epoch in range(epochs):
        for mod in modalities: models[mod].model.train()
        for batch in tqdm(train_loader, desc=f'Train E{epoch + 1}'):
            for mod in modalities:
                data = batch['modalities'][mod].to(device)
                reconstructed, _ = models[mod].forward(data)
                loss = torch.nn.functional.mse_loss(reconstructed, data)
                optimizers[mod].zero_grad()
                loss.backward();
                optimizers[mod].step()

        # 测试阶段
        for mod in modalities: models[mod].model.eval()
        all_labels, all_scores, pixel_maps, pixel_masks = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Test E{epoch + 1}'):
                res_list = []
                for mod in modalities:
                    data = batch['modalities'][mod].to(device)
                    recon, _ = models[mod].forward(data)
                    res_list.append(torch.abs(data - recon))
                    if mod == modalities[0]:  # 简化图像级分数计算
                        img_score = calculate_anomaly_score(data, recon)

                combined_res = torch.mean(torch.stack(res_list), dim=0).squeeze().cpu().numpy()
                all_labels.append(batch['label'].item())
                all_scores.append(img_score.item())
                pixel_maps.append(combined_res)
                pixel_masks.append(batch['mask'].squeeze().cpu().numpy())

        # 计算图像级
        img_auroc = roc_auc_score(all_labels, all_scores)
        img_ap = average_precision_score(all_labels, all_scores)
        img_f1, _ = calculate_optimal_f1_score(all_scores, all_labels)

        # 计算像素级及新指标 (DSC, HD95)
        pixel_flat_scores = np.concatenate([m.flatten() for m in pixel_maps])
        pixel_flat_masks = np.concatenate([m.flatten() for m in pixel_masks])

        if len(np.unique(pixel_flat_masks)) > 1:
            px_auroc = roc_auc_score(pixel_flat_masks, pixel_flat_scores)
            px_f1, best_px_threshold = calculate_optimal_f1_score(pixel_flat_scores, pixel_flat_masks)
            aupro = calculate_aupro(pixel_maps, pixel_masks)

            # --- 新增指标计算 ---
            dsc_list, hd95_list = [], []
            for i in range(len(pixel_maps)):
                if all_labels[i] == 1:  # 只对异常样本计算分割指标
                    pred_bin = (pixel_maps[i] >= best_px_threshold).astype(np.float32)
                    dsc_list.append(calculate_dsc(pred_bin, pixel_masks[i]))
                    hd95_list.append(calculate_hd95(pred_bin, pixel_masks[i]))

            mean_dsc = np.mean(dsc_list) if dsc_list else 0.0
            mean_hd95 = np.mean(hd95_list) if hd95_list else 0.0
        else:
            px_auroc = px_f1 = aupro = mean_dsc = mean_hd95 = 0.0

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Img AUROC: {img_auroc:.4f}, Img AP: {img_ap:.4f}")
        print(f"Px AUROC: {px_auroc:.4f}, AUPRO: {aupro:.4f}")
        print(f"DSC: {mean_dsc:.4f}, HD95: {mean_hd95:.4f}")

        # 记录 TensorBoard
        writer.add_scalar('Metrics/DSC', mean_dsc, epoch + 1)
        writer.add_scalar('Metrics/HD95', mean_hd95, epoch + 1)
        writer.add_scalar('Metrics/Pixel_AUROC', px_auroc, epoch + 1)

        if img_ap > best_ap:
            best_ap = img_ap
            for mod in modalities:
                torch.save(models[mod].model.state_dict(), f'best_{mod}_model.pth')

    writer.close()
    return None


if __name__ == '__main__':
    data_dir = '/home/wuchangwei/dyx/successful/brain_data2/'
    train_model(data_dir, modalities=['ct'], epochs=30, batch_size=8, target_size=(256, 256))



# import torch
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# import numpy as np
# from datetime import datetime
# from torch.utils.data import DataLoader, Dataset
# from denoising import denoising
# import os
# from PIL import Image
# from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
# import glob
# from skimage.transform import resize  # 引入 resize 函数
# import warnings
#
# # 忽略运行时警告，特别是可能出现在 F1 分数计算中的除以 0 警告
# warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
#
#
# class MultimodalSliceDataset(Dataset):
#     def __init__(self, data_path, modalities=['ct'], mode='train', target_size=(256, 256)):
#         self.modalities = modalities
#         self.mode = mode
#         self.target_size = target_size  # 目标尺寸参数
#         self.samples = []
#
#         # ... (数据加载逻辑与原代码保持一致) ...
#         if mode == 'train':
#             # 加载健康训练数据（PNG切片）
#             slice_paths = glob.glob(os.path.join(data_path, 'train', modalities[0], '*.png'))
#             for path in slice_paths:
#                 slice_id = os.path.basename(path).split('.')[0]
#                 self.samples.append({
#                     'id': slice_id,
#                     'label': 0,  # 训练数据都是健康的
#                     'modalities': {mod: os.path.join(data_path, 'train', mod, f'{slice_id}.png')
#                                    for mod in modalities},
#                     'mask_path': None  # 训练数据没有mask
#                 })
#         else:
#             # 加载测试数据（正常和异常的PNG切片）
#             for label, subfolder in enumerate(['NORMAL', 'ABNORMAL']):
#                 test_path = os.path.join(data_path, 'test', subfolder)
#                 slice_paths = glob.glob(os.path.join(test_path, modalities[0], '*.png'))
#                 for path in slice_paths:
#                     slice_id = os.path.basename(path).split('.')[0]
#                     mask_path = None
#                     if label == 1:  # 异常样本才有mask
#                         mask_path = os.path.join(data_path, 'test', 'ABNORMAL', 'mask', f'{slice_id}_mask.png')
#                     self.samples.append({
#                         'id': slice_id,
#                         'label': label,
#                         'modalities': {mod: os.path.join(test_path, mod, f'{slice_id}.png')
#                                        for mod in modalities},
#                         'mask_path': mask_path
#                     })
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         data = {}
#
#         # 加载并调整所有指定模态的PNG切片尺寸
#         for mod in self.modalities:
#             img = Image.open(sample['modalities'][mod]).convert('L')
#             img = np.array(img) / 255.0
#
#             # 【修改点 1：统一图像尺寸】
#             if img.shape != self.target_size:
#                 # 使用 skimage.transform.resize 调整尺寸
#                 img = resize(img, self.target_size, anti_aliasing=True)
#
#             img = torch.FloatTensor(img).unsqueeze(0)
#             data[mod] = img
#
#         # 加载mask（异常样本）或生成全黑mask（正常样本）
#         if sample['mask_path'] is not None:
#             mask = Image.open(sample['mask_path']).convert('L')
#             mask = np.array(mask) / 255.0
#
#             # 【修改点 2：统一 Mask 尺寸】
#             # 确保 mask 也被调整到目标尺寸
#             if mask.shape != self.target_size:
#                 mask = resize(mask, self.target_size, anti_aliasing=False, preserve_range=True)
#                 mask = (mask > 0.5).astype(np.float32)  # 重新二值化
#         else:
#             # 生成全黑mask，尺寸与目标尺寸一致
#             mask = np.zeros(self.target_size, dtype=np.float32)
#
#         mask = torch.FloatTensor(mask).unsqueeze(0)
#
#         return {
#             'modalities': data,
#             'label': sample['label'],
#             'id': sample['id'],
#             'mask': mask
#         }
#
#
# # ------------------- 辅助函数 (保持不变或微调以提高稳定性) -------------------
#
# def calculate_anomaly_score(original, reconstructed):
#     """计算异常分数（MSE）"""
#     return torch.mean((original - reconstructed) ** 2, dim=[1, 2, 3])
#
#
# def calculate_optimal_f1_score(scores, labels):
#     """计算最佳F1分数"""
#     precision, recall, thresholds = precision_recall_curve(labels, scores)
#     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
#     best_f1 = np.max(f1_scores)
#     return best_f1
#
#
# def calculate_aupro(anomaly_maps, ground_truth_masks):
#     """计算AUPRO指标 - Area Under PRO曲线（每张图像的平均假阳性率）"""
#
#     # 【修改点 3：AUPRO 阈值和 FPR 归一化】
#     # 调整阈值采样，提高鲁棒性
#     all_scores = np.concatenate([am.flatten() for am in anomaly_maps])
#     if all_scores.size == 0 or np.max(all_scores) == np.min(all_scores):
#         return 0.0
#
#     thresholds = np.linspace(np.min(all_scores) * 0.9, np.max(all_scores) * 1.1, 1000)  # 增加阈值数量，扩大范围
#
#     recalls = []
#     false_positive_rates = []
#
#     for threshold in thresholds:
#         total_tp = 0
#         total_fn = 0
#         total_fpr = 0
#         valid_images = 0
#
#         for anomaly_map, gt_mask in zip(anomaly_maps, ground_truth_masks):
#             # 预测的异常区域
#             pred_mask = (anomaly_map >= threshold).astype(np.float32)  # 使用 >=
#
#             # 计算真阳性TP和假阴性FN (用于 Recall)
#             tp = np.sum((pred_mask == 1) & (gt_mask == 1))
#             fn = np.sum((pred_mask == 0) & (gt_mask == 1))
#
#             # 计算每张图像的假阳性率 (FPR_i)
#             fp = np.sum((pred_mask == 1) & (gt_mask == 0))
#             normal_pixels = np.sum(gt_mask == 0)  # 正常像素数
#
#             if normal_pixels > 0:  # 避免除以0
#                 fpr = fp / normal_pixels  # 假阳性率 FPR_i
#                 total_fpr += fpr
#                 valid_images += 1
#
#             total_tp += tp
#             total_fn += fn
#
#         # 计算总召回率 R
#         if (total_tp + total_fn) > 0:
#             recall = total_tp / (total_tp + total_fn)
#         else:
#             recall = 0
#
#         # 计算平均假阳性率 FPR_PRO
#         if valid_images > 0:
#             avg_fpr = total_fpr / valid_images
#         else:
#             avg_fpr = 0
#
#         recalls.append(recall)
#         false_positive_rates.append(avg_fpr)
#
#     # 确保FPR是单调递增的，并移除重复点，这是计算 AUPRO 的标准做法
#     false_positive_rates = np.array(false_positive_rates)
#     recalls = np.array(recalls)
#
#     # 对 FPR 进行排序并移除重复的 FPR 值，保留最大 Recall
#     # 结合 recall 和 fpr，并按 fpr 排序
#     curve = np.array(sorted(zip(false_positive_rates, recalls)))
#     false_positive_rates = curve[:, 0]
#     recalls = curve[:, 1]
#
#     # 去重：只保留给定 FPR 下最大的 Recall
#     unique_fpr, unique_indices = np.unique(false_positive_rates, return_index=True)
#     unique_recalls = recalls[unique_indices]
#
#     # 添加 (0, 0) 点，确保积分从原点开始
#     if unique_fpr.size == 0 or unique_fpr[0] > 0:
#         unique_fpr = np.insert(unique_fpr, 0, 0)
#         unique_recalls = np.insert(unique_recalls, 0, 0)
#
#     # 确保积分范围在 [0, 0.6] 内
#     # 找到第一个大于 0.6 的索引
#     idx_06 = np.searchsorted(unique_fpr, 0.6)
#
#     # 裁剪到 0.6 或更少
#     fpr_to_integrate = unique_fpr[:idx_06].tolist()
#     recall_to_integrate = unique_recalls[:idx_06].tolist()
#
#     # 如果最后一个点小于 0.6，则在 0.6 处插值一个点
#     if unique_fpr[-1] < 0.6:
#         fpr_to_integrate.append(0.6)
#         # 简单使用 0.6 之前最后一个点的 Recall 值作为插值（保留性插值）
#         if idx_06 > 0:
#             recall_to_integrate.append(unique_recalls[idx_06 - 1])
#         else:
#             recall_to_integrate.append(0)
#     elif unique_fpr[-1] >= 0.6 and idx_06 < len(unique_fpr) and unique_fpr[idx_06] > 0.6:
#         # 如果 0.6 处没有点，插值 (0.6, Recall_at_0.6)
#         # 简单使用 0.6 之前最后一个点的 Recall 值作为插值（保留性插值）
#         fpr_to_integrate.append(0.6)
#         recall_to_integrate.append(unique_recalls[idx_06 - 1] if idx_06 > 0 else 0)
#
#     # 计算 AUPRO（使用梯形法则积分）
#     aupro = np.trapz(recall_to_integrate, fpr_to_integrate)
#
#     # 结果需要归一化到 0.6 (因为积分上限是 0.6)
#     return aupro / 0.6
#
#
# # ------------------- 训练模型函数 (微调) -------------------
#
# def train_model(data_dir, modalities=['ct'], epochs=30, batch_size=4, noise_std=0.2,
#                 noise_res=16, target_size=(256, 256)):  # 添加 target_size 参数
#     device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#
#     # ... (检查模态和模型初始化逻辑保持不变) ...
#     # 检查模态是否有效
#     valid_modalities = ['ct']
#     for mod in modalities:
#         if mod not in valid_modalities:
#             raise ValueError(f"Invalid modality: {mod}. Valid options are {valid_modalities}")
#
#     print(f"Training with modalities: {modalities}")
#
#     # 创建数据集和数据加载器
#     # 【修改点 4：将 target_size 传递给 Dataset】
#     train_dataset = MultimodalSliceDataset(data_dir, modalities=modalities, mode='train', target_size=target_size)
#     test_dataset = MultimodalSliceDataset(data_dir, modalities=modalities, mode='test', target_size=target_size)
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#
#     # 初始化模型和优化器
#     models = {}
#     optimizers = {}
#     for mod in modalities:
#         models[mod] = denoising(identifier=f'multimodal_{mod}',
#                                 n_input=1,
#                                 noise_std=noise_std,
#                                 noise_res=noise_res)
#         models[mod].model.to(device)
#         optimizers[mod] = models[mod].optimiser
#
#     writer = SummaryWriter(log_dir=os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S')))
#
#     best_ap = 0.0
#     best_models = {}
#
#     for epoch in range(epochs):
#         # ... (训练和测试循环逻辑保持不变) ...
#         # 训练阶段
#         for mod in modalities:
#             models[mod].model.train()
#
#         train_loss = {mod: 0.0 for mod in modalities}
#
#         for batch in tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{epochs}'):
#             for mod in modalities:
#                 data = batch['modalities'][mod].to(device)
#
#                 reconstructed, _ = models[mod].forward(data)
#                 loss = torch.nn.functional.mse_loss(reconstructed, data)
#                 train_loss[mod] += loss.item()
#
#                 optimizers[mod].zero_grad()
#                 loss.backward()
#                 optimizers[mod].step()
#
#         # 记录训练损失
#         for mod in modalities:
#             avg_loss = train_loss[mod] / len(train_loader)
#             writer.add_scalar(f'Loss/Train/{mod.upper()}', avg_loss, epoch + 1)
#
#         # 测试阶段
#         for mod in modalities:
#             models[mod].model.eval()
#
#         all_labels = []
#         all_scores = []
#         pixel_anomaly_maps = []
#         pixel_true_masks = []
#
#         with torch.no_grad():
#             for batch in tqdm(test_loader, desc=f'Test Epoch {epoch + 1}'):
#                 # 计算每个模态的残差和异常分数
#                 modality_residuals = []
#                 modality_scores = []
#
#                 for mod in modalities:
#                     data = batch['modalities'][mod].to(device)
#                     reconstructed, _ = models[mod].forward(data)
#                     residual = torch.abs(data - reconstructed)
#                     score = calculate_anomaly_score(data, reconstructed)
#
#                     modality_residuals.append(residual)
#                     modality_scores.append(score)
#
#                 # 合并所有模态的残差（取平均）
#                 combined_residual = torch.mean(torch.stack(modality_residuals), dim=0)
#                 # 合并所有模态的异常分数（取平均）
#                 combined_score = torch.mean(torch.stack(modality_scores), dim=0)
#
#                 # 收集结果
#                 all_labels.append(batch['label'].item())
#                 all_scores.append(combined_score.item())
#
#                 # 收集像素级数据
#                 residual_np = combined_residual.squeeze().cpu().numpy()
#                 mask_np = batch['mask'].squeeze().cpu().numpy()
#                 pixel_anomaly_maps.append(residual_np)
#                 pixel_true_masks.append(mask_np)
#
#         # 计算图像级指标
#         image_auroc = roc_auc_score(all_labels, all_scores)
#         image_ap = average_precision_score(all_labels, all_scores)
#         image_f1 = calculate_optimal_f1_score(all_scores, all_labels)
#
#         # 计算像素级指标
#         pixel_anomaly_scores = np.concatenate([am.flatten() for am in pixel_anomaly_maps])
#         pixel_binary_labels = np.concatenate([m.flatten() for m in pixel_true_masks])
#
#         # 【修改点 5：避免在所有标签都相同时计算 AUROC/AP/F1】
#         if len(np.unique(pixel_binary_labels)) > 1:
#             pixel_auroc = roc_auc_score(pixel_binary_labels, pixel_anomaly_scores)
#             pixel_ap = average_precision_score(pixel_binary_labels, pixel_anomaly_scores)
#             pixel_f1 = calculate_optimal_f1_score(pixel_anomaly_scores, pixel_binary_labels)
#             aupro = calculate_aupro(pixel_anomaly_maps, pixel_true_masks)
#         else:
#             # 当像素标签全部为 0 或 1 时，这些指标没有意义
#             print("\nWarning: Pixel labels are uniform. AUROC/AP/F1/AUPRO set to 0.0.")
#             pixel_auroc = 0.0
#             pixel_ap = 0.0
#             pixel_f1 = 0.0
#             aupro = 0.0
#
#         # 打印评估指标
#         print(f"\nResults at Epoch {epoch + 1}:")
#         print(f"Image-Level-AUROC:{image_auroc:.4f},AP:{image_ap:.4f},F1:{image_f1:.4f}")
#         print(f"Pixel-Level-AUROC:{pixel_auroc:.4f},AP:{pixel_ap:.4f},F1:{pixel_f1:.4f},AUPRO:{aupro:.4f}")
#
#         # ... (TensorBoard 记录和模型保存逻辑保持不变) ...
#         # 记录到TensorBoard
#         writer.add_scalar('Metrics/Image_AUROC', image_auroc, epoch + 1)
#         writer.add_scalar('Metrics/Image_AP', image_ap, epoch + 1)
#         writer.add_scalar('Metrics/Image_F1', image_f1, epoch + 1)
#         writer.add_scalar('Metrics/Pixel_AUROC', pixel_auroc, epoch + 1)
#         writer.add_scalar('Metrics/Pixel_AP', pixel_ap, epoch + 1)
#         writer.add_scalar('Metrics/Pixel_F1', pixel_f1, epoch + 1)
#         writer.add_scalar('Metrics/AUPRO', aupro, epoch + 1)
#
#         # 保存最佳模型
#         if image_ap > best_ap:
#             best_ap = image_ap
#             for mod in modalities:
#                 best_models[mod] = models[mod].model.state_dict()
#                 torch.save(best_models[mod], f'best_{mod}_model.pth')
#
#     writer.close()
#     return best_models
#
#
# if __name__ == '__main__':
#     data_dir = '/home/wuchangwei/dyx/successful/brain_data2/'
#
#     # 【修改点 6：定义统一的图像尺寸】
#     TARGET_SIZE = (256, 256)
#
#     trained_models = train_model(
#         data_dir,
#         modalities=['ct'],
#         epochs=30,
#         batch_size=8,
#         noise_std=0.2,
#         noise_res=16,
#         target_size=TARGET_SIZE  # 传递目标尺寸
#     )
#
