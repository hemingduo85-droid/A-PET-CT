import os
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from new_model import ESC_DRKD_Multimodal
from utils import AverageMeter, EarlyStop, print_log
import torch.nn.functional as F
import warnings
import csv  # 新增：CSV写入库

warnings.filterwarnings('ignore')

# 新增：计算FLOPs和参数量的库
from thop import profile, clever_format


# 配置参数
class Config:
    data_root = '/home/wuchangwei/dyx/successful/brain_data3'
    modalities = ['ct']
    batch_size = 8
    epochs = 30  # 保留但不再使用
    lr = 1e-4  # 保留但不再使用
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    save_dir = 'checkpoints/'
    log_file = 'evaluation.log'  # 修改日志文件名
    early_stop_patience = 20  # 保留但不再使用
    img_size = 256
    # 新增：预训练权重路径
    pretrained_weight_path = '/home/wuchangwei/dyx/successful/ESC_DRKD_successful/checkpoints/best_model.pt'
    # 新增：CSV结果保存路径
    csv_result_path = os.path.join(save_dir, 'BHSD.csv')


# 初始化配置
config = Config()
os.makedirs(config.save_dir, exist_ok=True)
log = open(os.path.join(config.save_dir, config.log_file), 'w')


# --- 新增/完善指标计算函数 ---

def calculate_dsc(pred_bin, gt_bin):
    """计算 Dice 系数 (与 F1-score 相同)"""
    intersection = np.sum(pred_bin * gt_bin)
    return (2. * intersection) / (np.sum(pred_bin) + np.sum(gt_bin) + 1e-8)


def calculate_hd95(pred_bin, gt_bin):
    """计算 95% 豪斯多夫距离 (HD95)"""
    if np.sum(pred_bin) == 0 and np.sum(gt_bin) == 0:
        return 0.0
    if np.sum(pred_bin) == 0 or np.sum(gt_bin) == 0:
        return 100.0  # 惩罚值

    def get_distances(mask_a, mask_b):
        dt = distance_transform_edt(~mask_b.astype(bool))
        distances = dt[mask_a.astype(bool)]
        return distances

    dists_1 = get_distances(pred_bin, gt_bin)
    dists_2 = get_distances(gt_bin, pred_bin)
    res = np.percentile(np.concatenate([dists_1, dists_2]), 95)
    return res


def calculate_assd(pred_bin, gt_bin):
    """计算平均对称表面距离 (ASSD)"""
    if np.sum(pred_bin) == 0 and np.sum(gt_bin) == 0:
        return 0.0
    if np.sum(pred_bin) == 0 or np.sum(gt_bin) == 0:
        return 100.0

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


def calculate_ppv(pred_bin, gt_bin):
    """计算阳性预测值 (PPV/精确率)"""
    tp = np.sum(pred_bin * gt_bin)
    fp = np.sum(pred_bin * (1 - gt_bin))
    return tp / (tp + fp + 1e-8)


def calculate_sensitive(pred_bin, gt_bin):
    """计算敏感度 (召回率/Recall)"""
    tp = np.sum(pred_bin * gt_bin)
    fn = np.sum((1 - pred_bin) * gt_bin)
    return tp / (tp + fn + 1e-8)


# --- 数据加载类 (保持不变) ---

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, mode='train'):
        self.mode = mode
        self.samples = []

        if mode == 'train':
            for mod in config.modalities:
                mod_path = os.path.join(data_path, 'train', mod)
                for img_name in os.listdir(mod_path):
                    img_id = img_name.split('.')[0]
                    if not any(s['id'] == img_id for s in self.samples):
                        self.samples.append({'id': img_id})
                    sample = next(s for s in self.samples if s['id'] == img_id)
                    sample[mod] = os.path.join(mod_path, img_name)
        else:
            for label, subfolder in enumerate(['NORMAL', 'ABNORMAL']):
                for mod in config.modalities:
                    mod_path = os.path.join(data_path, 'test', subfolder, mod)
                    for img_name in os.listdir(mod_path):
                        img_id = img_name.split('.')[0]
                        if not any(s['id'] == img_id for s in self.samples):
                            self.samples.append({
                                'id': img_id,
                                'label': label,
                                'mask': os.path.join(data_path, 'test', 'ABNORMAL', 'mask',
                                                     f'{img_id}_mask.png') if label == 1 else None
                            })
                        sample = next(s for s in self.samples if s['id'] == img_id)
                        sample[mod] = os.path.join(mod_path, img_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = {}

        for mod in config.modalities:
            if mod in sample:
                img = cv2.imread(sample[mod], cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (config.img_size, config.img_size))
                img = img / 255.0
                img = torch.FloatTensor(img).unsqueeze(0)
            else:
                img = torch.zeros(1, config.img_size, config.img_size)
            data[mod] = img

        if self.mode != 'train':
            mask = np.zeros((config.img_size, config.img_size))
            if sample['mask']:
                mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (config.img_size, config.img_size)) / 255.0
            return {
                'modalities': data,
                'label': sample['label'],
                'mask': torch.FloatTensor(mask),
                'id': sample['id']
            }
        return {'modalities': data}


def collate_fn(batch):
    modalities = config.modalities
    if 'label' in batch[0]:
        return {
            'modalities': {mod: torch.stack([item['modalities'][mod] for item in batch], dim=0) for mod in modalities},
            'label': torch.tensor([item['label'] for item in batch]),
            'mask': torch.stack([item['mask'] for item in batch], dim=0),
            'id': [item['id'] for item in batch]
        }
    else:
        return {
            'modalities': {mod: torch.stack([item['modalities'][mod] for item in batch], dim=0) for mod in modalities}
        }


def calculate_optimal_threshold_and_f1(scores, labels):
    """计算最佳阈值及其对应的 F1 (DSC)"""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1], f1_scores[best_idx]


def calculate_aupro(anomaly_maps, ground_truth_masks):
    anomaly_maps = [np.array(am) for am in anomaly_maps]
    ground_truth_masks = [np.array(gt) for gt in ground_truth_masks]
    all_scores = np.concatenate([am.flatten() for am in anomaly_maps])
    thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 100)

    recalls, false_positive_rates = [], []
    for threshold in thresholds:
        total_tp, total_fn, total_fpr, valid_images = 0, 0, 0, 0
        for anomaly_map, gt_mask in zip(anomaly_maps, ground_truth_masks):
            pred_mask = (anomaly_map > threshold).astype(np.float32)
            total_tp += np.sum((pred_mask == 1) & (gt_mask == 1))
            total_fn += np.sum((pred_mask == 0) & (gt_mask == 1))
            fp = np.sum((pred_mask == 1) & (gt_mask == 0))
            normal_pixels = np.prod(gt_mask.shape) - np.sum(gt_mask)
            if normal_pixels > 0:
                total_fpr += fp / normal_pixels
                valid_images += 1
        recalls.append(total_tp / (total_tp + total_fn + 1e-8))
        false_positive_rates.append(total_fpr / (valid_images + 1e-8))

    sort_idx = np.argsort(false_positive_rates)
    return np.trapz(np.array(recalls)[sort_idx], np.array(false_positive_rates)[sort_idx])


def get_anomaly_map(t_features, s_recon):
    anomaly_maps = []
    with torch.no_grad():
        s_recon_features = model.teacher.backbone(s_recon)
        for t_feat, s_feat in zip(t_features[:3], s_recon_features[:3]):
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode='bilinear')
            cos_sim = F.cosine_similarity(t_feat, s_feat, dim=1)
            anomaly_map = 1 - cos_sim.unsqueeze(1)
            anomaly_maps.append(F.interpolate(anomaly_map, size=(config.img_size, config.img_size), mode='bilinear'))
    return torch.mean(torch.cat(anomaly_maps, dim=1), dim=1, keepdim=True)


def test():
    """修改后的测试函数：计算所有新增指标并返回标准差，同时逐样本保存到CSV"""
    model.eval()
    gt_labels, gt_masks, pred_scores, pred_maps = [], [], [], []

    # 新增：存储每个异常图像的像素级指标
    dice_list, hd95_list, assd_list, ppv_list, sensitive_list = [], [], [], [], []
    # 新增：存储所有样本的CSV行数据
    csv_rows = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            x_dict = {mod: batch['modalities'][mod].to(config.device) for mod in config.modalities}
            x = torch.stack(list(x_dict.values()), dim=0).mean(dim=0)
            t_features = model.teacher.backbone(x)
            reconstruction = model.student(t_features[-1], t_features[:-1])
            anomaly_map = get_anomaly_map(t_features, reconstruction)

            gt_labels.append(batch['label'].item())
            gt_mask = batch['mask'].squeeze().cpu().numpy()
            gt_masks.append(gt_mask)
            pred_scores.append(anomaly_map.max().item())
            pred_map = anomaly_map.squeeze().cpu().numpy()
            pred_maps.append(pred_map)

            # 仅对异常图像计算像素级指标
            if batch['label'].item() == 1:
                # 使用像素级最佳阈值二值化
                pixel_threshold, _ = calculate_optimal_threshold_and_f1(
                    np.concatenate([m.flatten() for m in pred_maps]),
                    np.concatenate([m.flatten() for m in gt_masks])
                )
                pred_bin = (pred_map > pixel_threshold).astype(np.uint8)
                gt_bin = (gt_mask > 0.5).astype(np.uint8)

                # 计算单张图像的像素级指标
                dice = calculate_dsc(pred_bin, gt_bin)
                hd95 = calculate_hd95(pred_bin, gt_bin)
                assd = calculate_assd(pred_bin, gt_bin)
                ppv = calculate_ppv(pred_bin, gt_bin)
                sensitive = calculate_sensitive(pred_bin, gt_bin)

                dice_list.append(dice)
                hd95_list.append(hd95)
                assd_list.append(assd)
                ppv_list.append(ppv)
                sensitive_list.append(sensitive)

                # 保存到CSV行（与图片格式一致：filename, dataset, method, PPV, DSC, HD95, ASSD）
                csv_rows.append({
                    'filename': batch['id'][0],  # batch_size=1，直接取第一个
                    'dataset': 'INSTANCE2022',  # 与图片一致
                    'method': 'ESC_DRKD',   # 与图片一致
                    'PPV': ppv,
                    'DSC': dice,
                    'HD95': hd95,
                    'ASSD': assd
                })
            # else:
            #     # 正常样本也写入CSV，指标填0（可选，可根据需求修改）
            #     csv_rows.append({
            #         'filename': batch['id'][0],
            #         'dataset': 'CT-ICH',
            #         'method': 'ESC_DRKD',
            #         'PPV': 0.0,
            #         'DSC': 0.0,
            #         'HD95': 0.0,
            #         'ASSD': 0.0
            #     })

    # 图像级指标
    gt_labels = np.array(gt_labels)
    pred_scores = np.array(pred_scores)
    image_bin_labels = (gt_labels > 0).astype(int)
    img_auroc = roc_auc_score(image_bin_labels, pred_scores)
    img_ap = average_precision_score(image_bin_labels, pred_scores)
    _, img_f1 = calculate_optimal_threshold_and_f1(pred_scores, image_bin_labels)

    # 像素级指标（含标准差）
    pixel_scores = np.concatenate([am.flatten() for am in pred_maps])
    pixel_gt = np.concatenate([m.flatten() for m in gt_masks])

    pix_auroc = pix_ap = pix_dsc = aupro = 0.0
    avg_dice = avg_hd95 = avg_assd = avg_ppv = avg_sensitive = 0.0
    std_dice = std_hd95 = std_assd = std_ppv = std_sensitive = 0.0

    if len(np.unique(pixel_gt)) > 1:
        pix_auroc = roc_auc_score(pixel_gt, pixel_scores)
        pix_ap = average_precision_score(pixel_gt, pixel_scores)
        best_threshold, pix_dsc = calculate_optimal_threshold_and_f1(pixel_scores, pixel_gt)
        aupro = calculate_aupro(pred_maps, gt_masks)

        # 计算像素级指标的均值和标准差
        if dice_list:
            avg_dice = np.mean(dice_list)
            std_dice = np.std(dice_list)
            avg_hd95 = np.mean(hd95_list)
            std_hd95 = np.std(hd95_list)
            avg_assd = np.mean(assd_list)
            std_assd = np.std(assd_list)
            avg_ppv = np.mean(ppv_list)
            std_ppv = np.std(ppv_list)
            avg_sensitive = np.mean(sensitive_list)
            std_sensitive = np.std(sensitive_list)

    # 计算FLOPs和Params（修复维度不匹配问题）
    try:
        # 构建与student模型实际输入维度匹配的dummy input
        # 基于报错信息推断的维度：t_features[-1]是 [1,2048,16,16]
        # t_features[:-1] 包含 [1,64,256,256], [1,512,64,64], [1,1024,32,32]
        dummy_main = torch.randn(1, 2048, 16, 16).to(config.device)  # t_features[-1]
        dummy_skip1 = torch.randn(1, 64, 256, 256).to(config.device)  # t_features[0]
        dummy_skip2 = torch.randn(1, 512, 64, 64).to(config.device)  # t_features[2]
        dummy_skip3 = torch.randn(1, 1024, 32, 32).to(config.device)  # t_features[3]

        # 计算student模型的FLOPs和Params
        flops, params = profile(
            model.student,
            inputs=(dummy_main, [dummy_skip1, dummy_skip2, dummy_skip3]),
            verbose=False
        )
        flops, params = clever_format([flops, params], "%.2f")
    except Exception as e:
        # 如果仍有维度问题，降级处理
        print_log(f"计算FLOPs/Params失败: {str(e)}", log)
        flops, params = "计算失败", "计算失败"

    # --- 新增：写入CSV文件 ---
    with open(config.csv_result_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'dataset', 'method', 'PPV', 'DSC', 'HD95', 'ASSD']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print_log(f"CSV results saved to: {config.csv_result_path}", log)

    return (img_auroc, img_ap, img_f1,
            pix_auroc, pix_ap, pix_dsc, aupro,
            avg_dice, std_dice, avg_hd95, std_hd95,
            avg_assd, std_assd, avg_ppv, std_ppv,
            avg_sensitive, std_sensitive,
            flops, params)


# --- 主程序 ---
if __name__ == '__main__':
    # 初始化模型
    model = ESC_DRKD_Multimodal(modalities=config.modalities).to(config.device)

    # 加载预训练权重（兼容不同键名）
    print_log(f"Loading pretrained weights from: {config.pretrained_weight_path}", log)
    try:
        checkpoint = torch.load(config.pretrained_weight_path, map_location=config.device)

        # 自动检测权重键名
        if 'student' in checkpoint.keys():
            model.student.load_state_dict(checkpoint['student'])
            print_log("使用键名 'student' 加载权重", log)
        elif 'model' in checkpoint.keys():
            model.student.load_state_dict(checkpoint['model'])
            print_log("使用键名 'model' 加载权重", log)
        elif 'state_dict' in checkpoint.keys():
            model.student.load_state_dict(checkpoint['state_dict'])
            print_log("使用键名 'state_dict' 加载权重", log)
        else:
            # 如果只有模型参数，直接加载
            model.student.load_state_dict(checkpoint)
            print_log("权重文件无外层键，直接加载", log)

        # print_log(f"Pretrained weights loaded successfully! Best AP: {checkpoint.get('best_ap', 'N/A')}", log)
    except Exception as e:
        print_log(f"加载权重失败: {str(e)}", log)
        exit(1)

    # 只创建测试数据集和加载器
    test_dataset = MultimodalDataset(config.data_root, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 执行测试
    print_log("Starting evaluation...", log)
    res = test()

    # 解析结果
    img_auroc, img_ap, img_f1 = res[0], res[1], res[2]
    pix_auroc, pix_ap, pix_dsc, aupro = res[3], res[4], res[5], res[6]
    avg_dice, std_dice = res[7], res[8]
    avg_hd95, std_hd95 = res[9], res[10]
    avg_assd, std_assd = res[11], res[12]
    avg_ppv, std_ppv = res[13], res[14]
    avg_sensitive, std_sensitive = res[15], res[16]
    flops, params = res[17], res[18]

    # 打印日志
    print_log("=" * 80, log)
    print_log("Evaluation Results (Complete)", log)
    print_log("=" * 80, log)
    print_log(f'Image-Level Metrics:', log)
    print_log(f'  AUROC: {img_auroc:.4f}', log)
    print_log(f'  AP:    {img_ap:.4f}', log)
    print_log(f'  F1:    {img_f1:.4f}', log)
    print_log("-" * 50, log)
    print_log(f'Pixel-Level Metrics:', log)
    print_log(f'  AUROC: {pix_auroc:.4f}', log)
    print_log(f'  AP:    {pix_ap:.4f}', log)
    print_log(f'  DSC:   {pix_dsc:.4f}', log)
    print_log(f'  AUPRO: {aupro:.4f}', log)
    print_log("-" * 50, log)
    print_log(f'Detailed Pixel Metrics (Mean ± Std):', log)
    print_log(f'  Dice:     {avg_dice:.4f} ± {std_dice:.4f}', log)
    print_log(f'  HD95:     {avg_hd95:.4f} ± {std_hd95:.4f}', log)
    print_log(f'  ASSD:     {avg_assd:.4f} ± {std_assd:.4f}', log)
    print_log(f'  PPV:      {avg_ppv:.4f} ± {std_ppv:.4f}', log)
    print_log(f'  Sensitive:{avg_sensitive:.4f} ± {std_sensitive:.4f}', log)
    print_log("-" * 50, log)
    print_log(f'Model Complexity:', log)
    print_log(f'  FLOPs: {flops}', log)
    print_log(f'  Params: {params}', log)
    print_log("=" * 80, log)

    # 关闭日志文件
    log.close()
    print("Evaluation completed! Results saved to evaluation.log and BHSD.csv")


# import os
# import cv2
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
# import numpy as np
# from tqdm import tqdm
# from scipy.ndimage import distance_transform_edt
# from new_model import ESC_DRKD_Multimodal
# from utils import AverageMeter, EarlyStop, print_log
# import torch.nn.functional as F
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # 新增：计算FLOPs和参数量的库
# from thop import profile, clever_format
#
#
# # 配置参数
# class Config:
#     data_root = '/home/wuchangwei/dyx/successful/brain_data3'
#     modalities = ['ct']
#     batch_size = 8
#     epochs = 30  # 保留但不再使用
#     lr = 1e-4  # 保留但不再使用
#     device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
#     save_dir = 'checkpoints/'
#     log_file = 'evaluation.log'  # 修改日志文件名
#     early_stop_patience = 20  # 保留但不再使用
#     img_size = 256
#     # 新增：预训练权重路径
#     pretrained_weight_path = '/home/wuchangwei/dyx/successful/ESC_DRKD_successful/checkpoints/best_model.pt'
#
#
# # 初始化配置
# config = Config()
# os.makedirs(config.save_dir, exist_ok=True)
# log = open(os.path.join(config.save_dir, config.log_file), 'w')
#
#
# # --- 新增/完善指标计算函数 ---
#
# def calculate_dsc(pred_bin, gt_bin):
#     """计算 Dice 系数 (与 F1-score 相同)"""
#     intersection = np.sum(pred_bin * gt_bin)
#     return (2. * intersection) / (np.sum(pred_bin) + np.sum(gt_bin) + 1e-8)
#
#
# def calculate_hd95(pred_bin, gt_bin):
#     """计算 95% 豪斯多夫距离 (HD95)"""
#     if np.sum(pred_bin) == 0 and np.sum(gt_bin) == 0:
#         return 0.0
#     if np.sum(pred_bin) == 0 or np.sum(gt_bin) == 0:
#         return 100.0  # 惩罚值
#
#     def get_distances(mask_a, mask_b):
#         dt = distance_transform_edt(~mask_b.astype(bool))
#         distances = dt[mask_a.astype(bool)]
#         return distances
#
#     dists_1 = get_distances(pred_bin, gt_bin)
#     dists_2 = get_distances(gt_bin, pred_bin)
#     res = np.percentile(np.concatenate([dists_1, dists_2]), 95)
#     return res
#
#
# def calculate_assd(pred_bin, gt_bin):
#     """计算平均对称表面距离 (ASSD)"""
#     if np.sum(pred_bin) == 0 and np.sum(gt_bin) == 0:
#         return 0.0
#     if np.sum(pred_bin) == 0 or np.sum(gt_bin) == 0:
#         return 100.0
#
#     def surface_distances(mask_a, mask_b):
#         mask_a = mask_a.astype(bool)
#         mask_b = mask_b.astype(bool)
#         if not np.any(mask_a) or not np.any(mask_b):
#             return np.array([0.0])
#         dist_a = distance_transform_edt(~mask_a)[mask_b]
#         dist_b = distance_transform_edt(~mask_b)[mask_a]
#         return np.concatenate([dist_a, dist_b])
#
#     distances = surface_distances(pred_bin, gt_bin)
#     return np.mean(distances)
#
#
# def calculate_ppv(pred_bin, gt_bin):
#     """计算阳性预测值 (PPV/精确率)"""
#     tp = np.sum(pred_bin * gt_bin)
#     fp = np.sum(pred_bin * (1 - gt_bin))
#     return tp / (tp + fp + 1e-8)
#
#
# def calculate_sensitive(pred_bin, gt_bin):
#     """计算敏感度 (召回率/Recall)"""
#     tp = np.sum(pred_bin * gt_bin)
#     fn = np.sum((1 - pred_bin) * gt_bin)
#     return tp / (tp + fn + 1e-8)
#
#
# # --- 数据加载类 (保持不变) ---
#
# class MultimodalDataset(torch.utils.data.Dataset):
#     def __init__(self, data_path, mode='train'):
#         self.mode = mode
#         self.samples = []
#
#         if mode == 'train':
#             for mod in config.modalities:
#                 mod_path = os.path.join(data_path, 'train', mod)
#                 for img_name in os.listdir(mod_path):
#                     img_id = img_name.split('.')[0]
#                     if not any(s['id'] == img_id for s in self.samples):
#                         self.samples.append({'id': img_id})
#                     sample = next(s for s in self.samples if s['id'] == img_id)
#                     sample[mod] = os.path.join(mod_path, img_name)
#         else:
#             for label, subfolder in enumerate(['NORMAL', 'ABNORMAL']):
#                 for mod in config.modalities:
#                     mod_path = os.path.join(data_path, 'test', subfolder, mod)
#                     for img_name in os.listdir(mod_path):
#                         img_id = img_name.split('.')[0]
#                         if not any(s['id'] == img_id for s in self.samples):
#                             self.samples.append({
#                                 'id': img_id,
#                                 'label': label,
#                                 'mask': os.path.join(data_path, 'test', 'ABNORMAL', 'mask',
#                                                      f'{img_id}_mask.png') if label == 1 else None
#                             })
#                         sample = next(s for s in self.samples if s['id'] == img_id)
#                         sample[mod] = os.path.join(mod_path, img_name)
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         data = {}
#
#         for mod in config.modalities:
#             if mod in sample:
#                 img = cv2.imread(sample[mod], cv2.IMREAD_GRAYSCALE)
#                 img = cv2.resize(img, (config.img_size, config.img_size))
#                 img = img / 255.0
#                 img = torch.FloatTensor(img).unsqueeze(0)
#             else:
#                 img = torch.zeros(1, config.img_size, config.img_size)
#             data[mod] = img
#
#         if self.mode != 'train':
#             mask = np.zeros((config.img_size, config.img_size))
#             if sample['mask']:
#                 mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
#                 mask = cv2.resize(mask, (config.img_size, config.img_size)) / 255.0
#             return {
#                 'modalities': data,
#                 'label': sample['label'],
#                 'mask': torch.FloatTensor(mask),
#                 'id': sample['id']
#             }
#         return {'modalities': data}
#
#
# def collate_fn(batch):
#     modalities = config.modalities
#     if 'label' in batch[0]:
#         return {
#             'modalities': {mod: torch.stack([item['modalities'][mod] for item in batch], dim=0) for mod in modalities},
#             'label': torch.tensor([item['label'] for item in batch]),
#             'mask': torch.stack([item['mask'] for item in batch], dim=0),
#             'id': [item['id'] for item in batch]
#         }
#     else:
#         return {
#             'modalities': {mod: torch.stack([item['modalities'][mod] for item in batch], dim=0) for mod in modalities}
#         }
#
#
# def calculate_optimal_threshold_and_f1(scores, labels):
#     """计算最佳阈值及其对应的 F1 (DSC)"""
#     precision, recall, thresholds = precision_recall_curve(labels, scores)
#     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
#     best_idx = np.argmax(f1_scores)
#     return thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1], f1_scores[best_idx]
#
#
# def calculate_aupro(anomaly_maps, ground_truth_masks):
#     anomaly_maps = [np.array(am) for am in anomaly_maps]
#     ground_truth_masks = [np.array(gt) for gt in ground_truth_masks]
#     all_scores = np.concatenate([am.flatten() for am in anomaly_maps])
#     thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 100)
#
#     recalls, false_positive_rates = [], []
#     for threshold in thresholds:
#         total_tp, total_fn, total_fpr, valid_images = 0, 0, 0, 0
#         for anomaly_map, gt_mask in zip(anomaly_maps, ground_truth_masks):
#             pred_mask = (anomaly_map > threshold).astype(np.float32)
#             total_tp += np.sum((pred_mask == 1) & (gt_mask == 1))
#             total_fn += np.sum((pred_mask == 0) & (gt_mask == 1))
#             fp = np.sum((pred_mask == 1) & (gt_mask == 0))
#             normal_pixels = np.prod(gt_mask.shape) - np.sum(gt_mask)
#             if normal_pixels > 0:
#                 total_fpr += fp / normal_pixels
#                 valid_images += 1
#         recalls.append(total_tp / (total_tp + total_fn + 1e-8))
#         false_positive_rates.append(total_fpr / (valid_images + 1e-8))
#
#     sort_idx = np.argsort(false_positive_rates)
#     return np.trapz(np.array(recalls)[sort_idx], np.array(false_positive_rates)[sort_idx])
#
#
# def get_anomaly_map(t_features, s_recon):
#     anomaly_maps = []
#     with torch.no_grad():
#         s_recon_features = model.teacher.backbone(s_recon)
#         for t_feat, s_feat in zip(t_features[:3], s_recon_features[:3]):
#             if s_feat.shape[-2:] != t_feat.shape[-2:]:
#                 s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode='bilinear')
#             cos_sim = F.cosine_similarity(t_feat, s_feat, dim=1)
#             anomaly_map = 1 - cos_sim.unsqueeze(1)
#             anomaly_maps.append(F.interpolate(anomaly_map, size=(config.img_size, config.img_size), mode='bilinear'))
#     return torch.mean(torch.cat(anomaly_maps, dim=1), dim=1, keepdim=True)
#
#
# def test():
#     """修改后的测试函数：计算所有新增指标并返回标准差"""
#     model.eval()
#     gt_labels, gt_masks, pred_scores, pred_maps = [], [], [], []
#
#     # 新增：存储每个异常图像的像素级指标
#     dice_list, hd95_list, assd_list, ppv_list, sensitive_list = [], [], [], [], []
#
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc='Evaluating'):
#             x_dict = {mod: batch['modalities'][mod].to(config.device) for mod in config.modalities}
#             x = torch.stack(list(x_dict.values()), dim=0).mean(dim=0)
#             t_features = model.teacher.backbone(x)
#             reconstruction = model.student(t_features[-1], t_features[:-1])
#             anomaly_map = get_anomaly_map(t_features, reconstruction)
#
#             gt_labels.append(batch['label'].item())
#             gt_mask = batch['mask'].squeeze().cpu().numpy()
#             gt_masks.append(gt_mask)
#             pred_scores.append(anomaly_map.max().item())
#             pred_map = anomaly_map.squeeze().cpu().numpy()
#             pred_maps.append(pred_map)
#
#             # 仅对异常图像计算像素级指标
#             if batch['label'].item() == 1:
#                 # 使用像素级最佳阈值二值化
#                 pixel_threshold, _ = calculate_optimal_threshold_and_f1(
#                     np.concatenate([m.flatten() for m in pred_maps]),
#                     np.concatenate([m.flatten() for m in gt_masks])
#                 )
#                 pred_bin = (pred_map > pixel_threshold).astype(np.uint8)
#                 gt_bin = (gt_mask > 0.5).astype(np.uint8)
#
#                 # 计算单张图像的像素级指标
#                 dice_list.append(calculate_dsc(pred_bin, gt_bin))
#                 hd95_list.append(calculate_hd95(pred_bin, gt_bin))
#                 assd_list.append(calculate_assd(pred_bin, gt_bin))
#                 ppv_list.append(calculate_ppv(pred_bin, gt_bin))
#                 sensitive_list.append(calculate_sensitive(pred_bin, gt_bin))
#
#     # 图像级指标
#     gt_labels = np.array(gt_labels)
#     pred_scores = np.array(pred_scores)
#     image_bin_labels = (gt_labels > 0).astype(int)
#     img_auroc = roc_auc_score(image_bin_labels, pred_scores)
#     img_ap = average_precision_score(image_bin_labels, pred_scores)
#     _, img_f1 = calculate_optimal_threshold_and_f1(pred_scores, image_bin_labels)
#
#     # 像素级指标（含标准差）
#     pixel_scores = np.concatenate([am.flatten() for am in pred_maps])
#     pixel_gt = np.concatenate([m.flatten() for m in gt_masks])
#
#     pix_auroc = pix_ap = pix_dsc = aupro = 0.0
#     avg_dice = avg_hd95 = avg_assd = avg_ppv = avg_sensitive = 0.0
#     std_dice = std_hd95 = std_assd = std_ppv = std_sensitive = 0.0
#
#     if len(np.unique(pixel_gt)) > 1:
#         pix_auroc = roc_auc_score(pixel_gt, pixel_scores)
#         pix_ap = average_precision_score(pixel_gt, pixel_scores)
#         best_threshold, pix_dsc = calculate_optimal_threshold_and_f1(pixel_scores, pixel_gt)
#         aupro = calculate_aupro(pred_maps, gt_masks)
#
#         # 计算像素级指标的均值和标准差
#         if dice_list:
#             avg_dice = np.mean(dice_list)
#             std_dice = np.std(dice_list)
#             avg_hd95 = np.mean(hd95_list)
#             std_hd95 = np.std(hd95_list)
#             avg_assd = np.mean(assd_list)
#             std_assd = np.std(assd_list)
#             avg_ppv = np.mean(ppv_list)
#             std_ppv = np.std(ppv_list)
#             avg_sensitive = np.mean(sensitive_list)
#             std_sensitive = np.std(sensitive_list)
#
#     # 计算FLOPs和Params（修复维度不匹配问题）
#     try:
#         # 构建与student模型实际输入维度匹配的dummy input
#         # 基于报错信息推断的维度：t_features[-1]是 [1,2048,16,16]
#         # t_features[:-1] 包含 [1,64,256,256], [1,512,64,64], [1,1024,32,32]
#         dummy_main = torch.randn(1, 2048, 16, 16).to(config.device)  # t_features[-1]
#         dummy_skip1 = torch.randn(1, 64, 256, 256).to(config.device)  # t_features[0]
#         dummy_skip2 = torch.randn(1, 512, 64, 64).to(config.device)  # t_features[2]
#         dummy_skip3 = torch.randn(1, 1024, 32, 32).to(config.device)  # t_features[3]
#
#         # 计算student模型的FLOPs和Params
#         flops, params = profile(
#             model.student,
#             inputs=(dummy_main, [dummy_skip1, dummy_skip2, dummy_skip3]),
#             verbose=False
#         )
#         flops, params = clever_format([flops, params], "%.2f")
#     except Exception as e:
#         # 如果仍有维度问题，降级处理
#         print_log(f"计算FLOPs/Params失败: {str(e)}", log)
#         flops, params = "计算失败", "计算失败"
#
#     return (img_auroc, img_ap, img_f1,
#             pix_auroc, pix_ap, pix_dsc, aupro,
#             avg_dice, std_dice, avg_hd95, std_hd95,
#             avg_assd, std_assd, avg_ppv, std_ppv,
#             avg_sensitive, std_sensitive,
#             flops, params)
#
#
# # --- 主程序 ---
# if __name__ == '__main__':
#     # 初始化模型
#     model = ESC_DRKD_Multimodal(modalities=config.modalities).to(config.device)
#
#     # 加载预训练权重（兼容不同键名）
#     print_log(f"Loading pretrained weights from: {config.pretrained_weight_path}", log)
#     try:
#         checkpoint = torch.load(config.pretrained_weight_path, map_location=config.device)
#
#         # 自动检测权重键名
#         if 'student' in checkpoint.keys():
#             model.student.load_state_dict(checkpoint['student'])
#             print_log("使用键名 'student' 加载权重", log)
#         elif 'model' in checkpoint.keys():
#             model.student.load_state_dict(checkpoint['model'])
#             print_log("使用键名 'model' 加载权重", log)
#         elif 'state_dict' in checkpoint.keys():
#             model.student.load_state_dict(checkpoint['state_dict'])
#             print_log("使用键名 'state_dict' 加载权重", log)
#         else:
#             # 如果只有模型参数，直接加载
#             model.student.load_state_dict(checkpoint)
#             print_log("权重文件无外层键，直接加载", log)
#
#         # print_log(f"Pretrained weights loaded successfully! Best AP: {checkpoint.get('best_ap', 'N/A')}", log)
#     except Exception as e:
#         print_log(f"加载权重失败: {str(e)}", log)
#         exit(1)
#
#     # 只创建测试数据集和加载器
#     test_dataset = MultimodalDataset(config.data_root, mode='test')
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
#
#     # 执行测试
#     print_log("Starting evaluation...", log)
#     res = test()
#
#     # 解析结果
#     img_auroc, img_ap, img_f1 = res[0], res[1], res[2]
#     pix_auroc, pix_ap, pix_dsc, aupro = res[3], res[4], res[5], res[6]
#     avg_dice, std_dice = res[7], res[8]
#     avg_hd95, std_hd95 = res[9], res[10]
#     avg_assd, std_assd = res[11], res[12]
#     avg_ppv, std_ppv = res[13], res[14]
#     avg_sensitive, std_sensitive = res[15], res[16]
#     flops, params = res[17], res[18]
#
#     # 打印日志
#     print_log("=" * 80, log)
#     print_log("Evaluation Results (Complete)", log)
#     print_log("=" * 80, log)
#     print_log(f'Image-Level Metrics:', log)
#     print_log(f'  AUROC: {img_auroc:.4f}', log)
#     print_log(f'  AP:    {img_ap:.4f}', log)
#     print_log(f'  F1:    {img_f1:.4f}', log)
#     print_log("-" * 50, log)
#     print_log(f'Pixel-Level Metrics:', log)
#     print_log(f'  AUROC: {pix_auroc:.4f}', log)
#     print_log(f'  AP:    {pix_ap:.4f}', log)
#     print_log(f'  DSC:   {pix_dsc:.4f}', log)
#     print_log(f'  AUPRO: {aupro:.4f}', log)
#     print_log("-" * 50, log)
#     print_log(f'Detailed Pixel Metrics (Mean ± Std):', log)
#     print_log(f'  Dice:     {avg_dice:.4f} ± {std_dice:.4f}', log)
#     print_log(f'  HD95:     {avg_hd95:.4f} ± {std_hd95:.4f}', log)
#     print_log(f'  ASSD:     {avg_assd:.4f} ± {std_assd:.4f}', log)
#     print_log(f'  PPV:      {avg_ppv:.4f} ± {std_ppv:.4f}', log)
#     print_log(f'  Sensitive:{avg_sensitive:.4f} ± {std_sensitive:.4f}', log)
#     print_log("-" * 50, log)
#     print_log(f'Model Complexity:', log)
#     print_log(f'  FLOPs: {flops}', log)
#     print_log(f'  Params: {params}', log)
#     print_log("=" * 80, log)
#
#     # 关闭日志文件
#     log.close()
#     print("Evaluation completed! Results saved to evaluation.log")
#
