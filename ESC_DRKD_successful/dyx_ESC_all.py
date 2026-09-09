import os
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from new_model import ESC_DRKD_Multimodal  # 根据实际路径调整
from utils import AverageMeter, EarlyStop, print_log  # 根据实际路径调整
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import shutil
from collections import deque

# 新增：计算FLOPs和参数量的库
from thop import profile, clever_format

warnings.filterwarnings('ignore')


# 配置参数
class Config:
    data_root = '/home/wuchangwei/dyx/successful/brain_data3'
    modalities = ['ct']
    batch_size = 8
    epochs = 30  # 保留但不再使用
    lr = 1e-4  # 保留但不再使用
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    save_dir = 'checkpoints/'
    log_file = 'evaluation.log'  # 修改日志文件名
    early_stop_patience = 20  # 保留但不再使用
    img_size = 256
    # 新增：预训练权重路径
    pretrained_weight_path = '/home/wuchangwei/dyx/successful/ESC_DRKD_successful/checkpoints/best_model.pt'

    # ========== 新增可视化配置 ==========
    visualization_root = 'all_picture'  # 可视化结果根目录
    vis_BHSD_path = os.path.join(visualization_root, 'BHSD')  # 所有样本可视化结果
    vis_high_dice_path = os.path.join(visualization_root, 'BHSD_dice90+')  # Dice>0.9的样本
    high_dice_threshold = 0.9  # 高分Dice阈值
    vis_cmap = 'jet'  # 热力图配色
    save_single_images = True  # 是否保存单独的图像文件

    # ========== 掩码处理参数（和参考代码一致） ==========
    q_low = 0.90
    q_high = 0.995
    k = 12.0
    gamma = 3.0
    thr = 0.5
    close_radius = 3

    # ========== 移除：指定需要测试的样本ID列表 ==========
    # 完全删除target_sample_ids相关配置


# 初始化配置
config = Config()
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.vis_BHSD_path, exist_ok=True)
os.makedirs(config.vis_high_dice_path, exist_ok=True)
log = open(os.path.join(config.save_dir, config.log_file), 'w')


# --- 新增掩码处理函数（和参考代码一致） ---
def largest_connected_component(mask01):
    """提取最大连通域"""
    m = (mask01 > 0).astype(np.uint8)
    H, W = m.shape
    visited = np.zeros((H, W), dtype=np.uint8)
    best = []
    best_size = 0

    for y in range(H):
        for x in range(W):
            if m[y, x] == 0 or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = 1
            comp = [(y, x)]
            while q:
                cy, cx = q.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < H and 0 <= nx < W and (not visited[ny, nx]) and m[ny, nx]:
                        visited[ny, nx] = 1
                        q.append((ny, nx))
                        comp.append((ny, nx))
            if len(comp) > best_size:
                best_size = len(comp)
                best = comp

    out = np.zeros((H, W), dtype=np.uint8)
    for y, x in best:
        out[y, x] = 1
    return out


def fill_holes(mask01):
    """填充掩码内部孔洞"""
    m = (mask01 > 0).astype(np.uint8)
    inv = (1 - m).astype(np.uint8)

    H, W = inv.shape
    vis = np.zeros((H, W), dtype=np.uint8)
    q = deque()

    # 从边界开始洪水填充
    for x in range(W):
        if inv[0, x] and not vis[0, x]:
            q.append((0, x));
            vis[0, x] = 1
        if inv[H - 1, x] and not vis[H - 1, x]:
            q.append((H - 1, x));
            vis[H - 1, x] = 1
    for y in range(H):
        if inv[y, 0] and not vis[y, 0]:
            q.append((y, 0));
            vis[y, 0] = 1
        if inv[y, W - 1] and not vis[y, W - 1]:
            q.append((y, W - 1));
            vis[y, W - 1] = 1

    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < H and 0 <= nx < W and inv[ny, nx] and not vis[ny, nx]:
                vis[ny, nx] = 1
                q.append((ny, nx))

    holes = (inv == 1) & (vis == 0)
    out = m.copy()
    out[holes] = 1
    return out.astype(np.uint8)


def close_and_fill(mask01, close_radius=3):
    """闭运算处理掩码"""
    m = (mask01 > 0).astype(np.uint8) * 255
    im = Image.fromarray(m)

    k = close_radius * 2 + 1
    im = im.filter(ImageFilter.MaxFilter(size=k))
    im = im.filter(ImageFilter.MinFilter(size=k))
    m2 = (np.array(im) > 0).astype(np.uint8)
    return m2


def probify_inside_roi(anomap, roi, q_low=0.90, q_high=0.995, k=12.0, gamma=3.0):
    """对异常图进行概率化处理，只保留ROI内的部分"""
    roi = roi.astype(bool)
    vals = anomap[roi]
    if vals.size < 10:
        vals = anomap.reshape(-1)

    base = float(np.quantile(vals, q_low))
    top = float(np.quantile(vals, q_high))
    scale = (top - base) + 1e-8

    z = (anomap - base) / scale
    prob = 1.0 / (1.0 + np.exp(-k * z))
    prob = np.clip(prob, 0.0, 1.0) ** gamma
    prob[~roi] = 0.0
    return prob.astype(np.float32)


# --- 数据加载类 (核心修改：移除指定样本ID的限制，加载所有测试样本) ---
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
            # 核心修改：加载所有测试样本，不再过滤target_sample_ids
            for label, subfolder in enumerate(['NORMAL', 'ABNORMAL']):
                for mod in config.modalities:
                    mod_path = os.path.join(data_path, 'test', subfolder, mod)
                    # 确保目录存在
                    if not os.path.exists(mod_path):
                        print(f"Warning: {mod_path} does not exist, skipping")
                        continue
                    for img_name in os.listdir(mod_path):
                        img_id = img_name.split('.')[0]
                        # 移除：不再检查是否在target_sample_ids中

                        if not any(s['id'] == img_id for s in self.samples):
                            self.samples.append({
                                'id': img_id,
                                'label': label,
                                'mask': os.path.join(data_path, 'test', 'ABNORMAL', 'mask',
                                                     f'{img_id}_mask.png') if label == 1 else None
                            })
                        sample = next(s for s in self.samples if s['id'] == img_id)
                        sample[mod] = os.path.join(mod_path, img_name)

        # 移除：不再过滤样本（删除原有的target_sample_ids过滤逻辑）
        print(f"Loaded {len(self.samples)} samples for {mode} mode")

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
                'id': sample['id'],
                'img_path': sample[config.modalities[0]] if config.modalities[0] in sample else None  # 保存原始图像路径
            }
        return {'modalities': data}


def collate_fn(batch):
    modalities = config.modalities
    if 'label' in batch[0]:
        return {
            'modalities': {mod: torch.stack([item['modalities'][mod] for item in batch], dim=0) for mod in modalities},
            'label': torch.tensor([item['label'] for item in batch]),
            'mask': torch.stack([item['mask'] for item in batch], dim=0),
            'id': [item['id'] for item in batch],
            'img_path': [item['img_path'] for item in batch]
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


# ========== 可视化相关函数（无修改） ==========
def normalize_img(img):
    """归一化图像到0-1范围"""
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def save_single(data, out_path, cmap="gray", vmin=None, vmax=None):
    """保存单张图像（和参考代码一致）"""
    fig = plt.figure(figsize=(5, 5))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_detection_overlay(pred01, gt01):
    """创建检测叠加图（和参考代码完全一致）"""
    H, W = pred01.shape
    overlay = np.zeros((H, W, 3), dtype=np.float32)

    mask_gt = gt01.astype(bool)
    mask_pred = pred01.astype(bool)

    mask_tp = mask_pred & mask_gt
    mask_fp = mask_pred & (~mask_gt)
    mask_fn = (~mask_pred) & mask_gt

    color_tp = np.array([224 / 255, 68 / 255, 19 / 255], dtype=np.float32)  # TP
    color_fn = np.array([51 / 255, 186 / 255, 7 / 255], dtype=np.float32)  # FN
    color_fp = np.array([80 / 255, 128 / 255, 209 / 255], dtype=np.float32)  # FP

    overlay[mask_fp] = color_fp
    overlay[mask_fn] = color_fn
    overlay[mask_tp] = color_tp
    return overlay


def visualize_sample(img_id, original_img, anomaly_map, gt_mask):
    """修改后的可视化函数（完全匹配参考代码样式）"""
    # 1. 预处理原始图像和ROI
    original_img = normalize_img(original_img)
    # 创建ROI（灰度值>10的区域）
    roi = (original_img > 10 / 255).astype(np.uint8)

    # 2. 异常图概率化处理（和参考代码一致）
    prob = probify_inside_roi(
        anomaly_map, roi,
        q_low=config.q_low,
        q_high=config.q_high,
        k=config.k,
        gamma=config.gamma
    )

    # 3. 生成预测掩码（连通域+填洞+闭运算）
    cand = (prob >= config.thr).astype(np.uint8) & roi
    cand = largest_connected_component(cand)
    pred = fill_holes(cand)
    if config.close_radius > 0:
        pred = close_and_fill(pred, close_radius=config.close_radius)

    # 4. 生成热力图（只保留预测区域内的渐变）
    heat = prob.copy()
    heat[pred == 0] = 0.0

    # 5. 计算Dice分数
    dsc = calculate_dsc(pred, (gt_mask > 0.5).astype(np.uint8))

    # 6. 计算差异图
    diff = np.abs(pred.astype(np.float32) - (gt_mask > 0.5).astype(np.float32))

    # 7. 创建检测叠加图
    det_overlay = make_detection_overlay(pred, (gt_mask > 0.5).astype(np.uint8))

    # 8. 为当前样本创建独立文件夹
    sample_folder = os.path.join(config.vis_BHSD_path, img_id)
    os.makedirs(sample_folder, exist_ok=True)

    # 9. 保存单独的图像文件（和参考代码一致的命名）
    if config.save_single_images:
        # 转换为RGB格式保存原始图像（和参考代码一致）
        original_rgb = np.stack([original_img] * 3, axis=-1)
        save_single(original_rgb, os.path.join(sample_folder, "original.png"), cmap="gray")
        save_single(heat, os.path.join(sample_folder, "heatmap.png"),
                    cmap=config.vis_cmap, vmin=0.0, vmax=1.0)
        save_single(pred, os.path.join(sample_folder, "pred.png"), cmap="gray", vmin=0, vmax=1)
        save_single((gt_mask > 0.5).astype(np.uint8), os.path.join(sample_folder, "gt.png"),
                    cmap="gray", vmin=0, vmax=1)
        save_single(det_overlay, os.path.join(sample_folder, "detection_map_overlay.png"))

    # 10. 创建5列的overview图（和参考代码样式一致）
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))

    # 1. 原始图像
    axes[0].imshow(original_rgb, cmap="gray")
    axes[0].set_title("")
    axes[0].axis("off")

    # 2. 热力图
    im = axes[1].imshow(heat, cmap=config.vis_cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title("")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. 差异图
    axes[2].imshow(diff, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("")
    axes[2].axis("off")

    # 4. GT掩码
    axes[3].imshow((gt_mask > 0.5).astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("")
    axes[3].axis("off")

    # 5. 检测叠加图（只显示Dice分数）
    axes[4].imshow(det_overlay)
    axes[4].set_title(f"Dice: {dsc:.4f}", fontsize=16)
    axes[4].axis("off")

    # 保存overview图
    plt.tight_layout()
    overview_path = os.path.join(sample_folder, "overview.png")
    plt.savefig(overview_path, dpi=200, bbox_inches="tight")
    plt.close('all')

    # 11. 如果Dice分数高于阈值，复制到高分文件夹
    if dsc > config.high_dice_threshold:
        high_dice_folder = os.path.join(config.vis_high_dice_path, img_id)
        os.makedirs(high_dice_folder, exist_ok=True)

        # 复制所有文件到高分文件夹
        for file_name in os.listdir(sample_folder):
            src = os.path.join(sample_folder, file_name)
            dst = os.path.join(high_dice_folder, file_name)
            if not os.path.exists(dst):
                shutil.copy(src, dst)

    return sample_folder, dsc, pred


# --- 指标计算函数（无修改） ---
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


def test():
    """修改后的测试函数：移除指定样本限制，处理所有测试样本"""
    model.eval()
    gt_labels, gt_masks, pred_scores, pred_maps = [], [], [], []
    all_sample_ids = []
    all_original_imgs = []

    # 新增：存储每个异常图像的像素级指标
    dice_list, hd95_list, assd_list, ppv_list, sensitive_list = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating all test samples'):  # 修改进度条描述
            x_dict = {mod: batch['modalities'][mod].to(config.device) for mod in config.modalities}
            x = torch.stack(list(x_dict.values()), dim=0).mean(dim=0)

            # 获取原始图像（用于可视化）
            img_path = batch['img_path'][0]
            original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if img_path else x.squeeze().cpu().numpy()
            original_img = cv2.resize(original_img, (config.img_size, config.img_size))

            t_features = model.teacher.backbone(x)
            reconstruction = model.student(t_features[-1], t_features[:-1])
            anomaly_map = get_anomaly_map(t_features, reconstruction)

            # 收集数据
            gt_label = batch['label'].item()
            gt_mask = batch['mask'].squeeze().cpu().numpy()
            pred_score = anomaly_map.max().item()
            pred_map = anomaly_map.squeeze().cpu().numpy()

            gt_labels.append(gt_label)
            gt_masks.append(gt_mask)
            pred_scores.append(pred_score)
            pred_maps.append(pred_map)
            all_sample_ids.append(batch['id'][0])
            all_original_imgs.append(original_img)

            # 仅对异常图像计算像素级指标并可视化
            if gt_label == 1:
                # 可视化当前样本（使用新的可视化函数）
                _, dice, pred_bin = visualize_sample(
                    batch['id'][0],
                    original_img,
                    pred_map,
                    gt_mask
                )

                # 计算单张图像的像素级指标
                gt_bin = (gt_mask > 0.5).astype(np.uint8)
                dice_list.append(dice)
                hd95_list.append(calculate_hd95(pred_bin, gt_bin))
                assd_list.append(calculate_assd(pred_bin, gt_bin))
                ppv_list.append(calculate_ppv(pred_bin, gt_bin))
                sensitive_list.append(calculate_sensitive(pred_bin, gt_bin))

    # 计算整体指标
    gt_labels = np.array(gt_labels)
    pred_scores = np.array(pred_scores)
    image_bin_labels = (gt_labels > 0).astype(int)
    img_auroc = roc_auc_score(image_bin_labels, pred_scores) if len(np.unique(image_bin_labels)) > 1 else 0.0
    img_ap = average_precision_score(image_bin_labels, pred_scores) if len(np.unique(image_bin_labels)) > 1 else 0.0
    _, img_f1 = calculate_optimal_threshold_and_f1(pred_scores, image_bin_labels) if len(
        np.unique(image_bin_labels)) > 1 else (0.0, 0.0)

    # 像素级指标
    pixel_scores = np.concatenate([am.flatten() for am in pred_maps]) if pred_maps else np.array([])
    pixel_gt = np.concatenate([m.flatten() for m in gt_masks]) if gt_masks else np.array([])

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

    # 计算FLOPs和Params
    try:
        # 构建dummy input
        dummy_main = torch.randn(1, 2048, 16, 16).to(config.device)
        dummy_skip1 = torch.randn(1, 64, 256, 256).to(config.device)
        dummy_skip2 = torch.randn(1, 512, 64, 64).to(config.device)
        dummy_skip3 = torch.randn(1, 1024, 32, 32).to(config.device)

        # 计算student模型的FLOPs和Params
        flops, params = profile(
            model.student,
            inputs=(dummy_main, [dummy_skip1, dummy_skip2, dummy_skip3]),
            verbose=False
        )
        flops, params = clever_format([flops, params], "%.2f")
    except Exception as e:
        print_log(f"计算FLOPs/Params失败: {str(e)}", log)
        flops, params = "计算失败", "计算失败"

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

    # 加载预训练权重
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
            model.student.load_state_dict(checkpoint)
            print_log("权重文件无外层键，直接加载", log)

    except Exception as e:
        print_log(f"加载权重失败: {str(e)}", log)
        exit(1)

    # 创建测试数据集和加载器（加载所有测试样本，不再限制target_sample_ids）
    test_dataset = MultimodalDataset(config.data_root, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 执行测试和可视化
    print_log("Starting evaluation and visualization for ALL test samples...", log)  # 修改日志描述
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
    print_log("Evaluation Results (ALL Test Samples)", log)  # 修改标题
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

    # 打印可视化结果信息
    print_log(f"\nVisualization Results:", log)
    print_log(f'  All test samples saved to: {config.vis_BHSD_path}', log)  # 修改描述
    print_log(f'  Each sample has its own folder named by sample ID', log)
    print_log(f'  High dice samples (>{config.high_dice_threshold}) saved to: {config.vis_high_dice_path}', log)

    # 关闭日志文件
    log.close()
    print("Evaluation and visualization for ALL test samples completed!")  # 修改提示语
    print(f"Results saved to {os.path.join(config.save_dir, config.log_file)}")
    print(f"Visualizations saved to {config.visualization_root}")
    print(f'All test samples have their own folder in {config.vis_BHSD_path}')  # 修改提示语import os
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from new_model import ESC_DRKD_Multimodal  # 根据实际路径调整
from utils import AverageMeter, EarlyStop, print_log  # 根据实际路径调整
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import shutil
from collections import deque

# 新增：计算FLOPs和参数量的库
from thop import profile, clever_format

warnings.filterwarnings('ignore')


# 配置参数
class Config:
    data_root = '/home/wuchangwei/dyx/successful/brain_data3'
    modalities = ['ct']
    batch_size = 8
    epochs = 30  # 保留但不再使用
    lr = 1e-4  # 保留但不再使用
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    save_dir = 'checkpoints/'
    log_file = 'evaluation.log'  # 修改日志文件名
    early_stop_patience = 20  # 保留但不再使用
    img_size = 256
    # 新增：预训练权重路径
    pretrained_weight_path = '/home/wuchangwei/dyx/successful/ESC_DRKD_successful/checkpoints/best_model.pt'

    # ========== 新增可视化配置 ==========
    visualization_root = 'visualization_results'  # 可视化结果根目录
    vis_BHSD_path = os.path.join(visualization_root, 'BHSD')  # 所有样本可视化结果
    vis_high_dice_path = os.path.join(visualization_root, 'BHSD_dice90+')  # Dice>0.9的样本
    high_dice_threshold = 0.9  # 高分Dice阈值
    vis_cmap = 'jet'  # 热力图配色
    save_single_images = True  # 是否保存单独的图像文件

    # ========== 掩码处理参数（和参考代码一致） ==========
    q_low = 0.90
    q_high = 0.995
    k = 12.0
    gamma = 3.0
    thr = 0.5
    close_radius = 3

    # ========== 移除：指定需要测试的样本ID列表 ==========
    # 完全删除target_sample_ids相关配置


# 初始化配置
config = Config()
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.vis_BHSD_path, exist_ok=True)
os.makedirs(config.vis_high_dice_path, exist_ok=True)
log = open(os.path.join(config.save_dir, config.log_file), 'w')


# --- 新增掩码处理函数（和参考代码一致） ---
def largest_connected_component(mask01):
    """提取最大连通域"""
    m = (mask01 > 0).astype(np.uint8)
    H, W = m.shape
    visited = np.zeros((H, W), dtype=np.uint8)
    best = []
    best_size = 0

    for y in range(H):
        for x in range(W):
            if m[y, x] == 0 or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = 1
            comp = [(y, x)]
            while q:
                cy, cx = q.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < H and 0 <= nx < W and (not visited[ny, nx]) and m[ny, nx]:
                        visited[ny, nx] = 1
                        q.append((ny, nx))
                        comp.append((ny, nx))
            if len(comp) > best_size:
                best_size = len(comp)
                best = comp

    out = np.zeros((H, W), dtype=np.uint8)
    for y, x in best:
        out[y, x] = 1
    return out


def fill_holes(mask01):
    """填充掩码内部孔洞"""
    m = (mask01 > 0).astype(np.uint8)
    inv = (1 - m).astype(np.uint8)

    H, W = inv.shape
    vis = np.zeros((H, W), dtype=np.uint8)
    q = deque()

    # 从边界开始洪水填充
    for x in range(W):
        if inv[0, x] and not vis[0, x]:
            q.append((0, x));
            vis[0, x] = 1
        if inv[H - 1, x] and not vis[H - 1, x]:
            q.append((H - 1, x));
            vis[H - 1, x] = 1
    for y in range(H):
        if inv[y, 0] and not vis[y, 0]:
            q.append((y, 0));
            vis[y, 0] = 1
        if inv[y, W - 1] and not vis[y, W - 1]:
            q.append((y, W - 1));
            vis[y, W - 1] = 1

    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < H and 0 <= nx < W and inv[ny, nx] and not vis[ny, nx]:
                vis[ny, nx] = 1
                q.append((ny, nx))

    holes = (inv == 1) & (vis == 0)
    out = m.copy()
    out[holes] = 1
    return out.astype(np.uint8)


def close_and_fill(mask01, close_radius=3):
    """闭运算处理掩码"""
    m = (mask01 > 0).astype(np.uint8) * 255
    im = Image.fromarray(m)

    k = close_radius * 2 + 1
    im = im.filter(ImageFilter.MaxFilter(size=k))
    im = im.filter(ImageFilter.MinFilter(size=k))
    m2 = (np.array(im) > 0).astype(np.uint8)
    return m2


def probify_inside_roi(anomap, roi, q_low=0.90, q_high=0.995, k=12.0, gamma=3.0):
    """对异常图进行概率化处理，只保留ROI内的部分"""
    roi = roi.astype(bool)
    vals = anomap[roi]
    if vals.size < 10:
        vals = anomap.reshape(-1)

    base = float(np.quantile(vals, q_low))
    top = float(np.quantile(vals, q_high))
    scale = (top - base) + 1e-8

    z = (anomap - base) / scale
    prob = 1.0 / (1.0 + np.exp(-k * z))
    prob = np.clip(prob, 0.0, 1.0) ** gamma
    prob[~roi] = 0.0
    return prob.astype(np.float32)


# --- 数据加载类 (核心修改：移除指定样本ID的限制，加载所有测试样本) ---
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
            # 核心修改：加载所有测试样本，不再过滤target_sample_ids
            for label, subfolder in enumerate(['NORMAL', 'ABNORMAL']):
                for mod in config.modalities:
                    mod_path = os.path.join(data_path, 'test', subfolder, mod)
                    # 确保目录存在
                    if not os.path.exists(mod_path):
                        print(f"Warning: {mod_path} does not exist, skipping")
                        continue
                    for img_name in os.listdir(mod_path):
                        img_id = img_name.split('.')[0]
                        # 移除：不再检查是否在target_sample_ids中

                        if not any(s['id'] == img_id for s in self.samples):
                            self.samples.append({
                                'id': img_id,
                                'label': label,
                                'mask': os.path.join(data_path, 'test', 'ABNORMAL', 'mask',
                                                     f'{img_id}_mask.png') if label == 1 else None
                            })
                        sample = next(s for s in self.samples if s['id'] == img_id)
                        sample[mod] = os.path.join(mod_path, img_name)

        # 移除：不再过滤样本（删除原有的target_sample_ids过滤逻辑）
        print(f"Loaded {len(self.samples)} samples for {mode} mode")

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
                'id': sample['id'],
                'img_path': sample[config.modalities[0]] if config.modalities[0] in sample else None  # 保存原始图像路径
            }
        return {'modalities': data}


def collate_fn(batch):
    modalities = config.modalities
    if 'label' in batch[0]:
        return {
            'modalities': {mod: torch.stack([item['modalities'][mod] for item in batch], dim=0) for mod in modalities},
            'label': torch.tensor([item['label'] for item in batch]),
            'mask': torch.stack([item['mask'] for item in batch], dim=0),
            'id': [item['id'] for item in batch],
            'img_path': [item['img_path'] for item in batch]
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


# ========== 可视化相关函数（无修改） ==========
def normalize_img(img):
    """归一化图像到0-1范围"""
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def save_single(data, out_path, cmap="gray", vmin=None, vmax=None):
    """保存单张图像（和参考代码一致）"""
    fig = plt.figure(figsize=(5, 5))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_detection_overlay(pred01, gt01):
    """创建检测叠加图（和参考代码完全一致）"""
    H, W = pred01.shape
    overlay = np.zeros((H, W, 3), dtype=np.float32)

    mask_gt = gt01.astype(bool)
    mask_pred = pred01.astype(bool)

    mask_tp = mask_pred & mask_gt
    mask_fp = mask_pred & (~mask_gt)
    mask_fn = (~mask_pred) & mask_gt

    color_tp = np.array([224 / 255, 68 / 255, 19 / 255], dtype=np.float32)  # TP
    color_fn = np.array([51 / 255, 186 / 255, 7 / 255], dtype=np.float32)  # FN
    color_fp = np.array([80 / 255, 128 / 255, 209 / 255], dtype=np.float32)  # FP

    overlay[mask_fp] = color_fp
    overlay[mask_fn] = color_fn
    overlay[mask_tp] = color_tp
    return overlay


def visualize_sample(img_id, original_img, anomaly_map, gt_mask):
    """修改后的可视化函数（完全匹配参考代码样式）"""
    # 1. 预处理原始图像和ROI
    original_img = normalize_img(original_img)
    # 创建ROI（灰度值>10的区域）
    roi = (original_img > 10 / 255).astype(np.uint8)

    # 2. 异常图概率化处理（和参考代码一致）
    prob = probify_inside_roi(
        anomaly_map, roi,
        q_low=config.q_low,
        q_high=config.q_high,
        k=config.k,
        gamma=config.gamma
    )

    # 3. 生成预测掩码（连通域+填洞+闭运算）
    cand = (prob >= config.thr).astype(np.uint8) & roi
    cand = largest_connected_component(cand)
    pred = fill_holes(cand)
    if config.close_radius > 0:
        pred = close_and_fill(pred, close_radius=config.close_radius)

    # 4. 生成热力图（只保留预测区域内的渐变）
    heat = prob.copy()
    heat[pred == 0] = 0.0

    # 5. 计算Dice分数
    dsc = calculate_dsc(pred, (gt_mask > 0.5).astype(np.uint8))

    # 6. 计算差异图
    diff = np.abs(pred.astype(np.float32) - (gt_mask > 0.5).astype(np.float32))

    # 7. 创建检测叠加图
    det_overlay = make_detection_overlay(pred, (gt_mask > 0.5).astype(np.uint8))

    # 8. 为当前样本创建独立文件夹
    sample_folder = os.path.join(config.vis_BHSD_path, img_id)
    os.makedirs(sample_folder, exist_ok=True)

    # 9. 保存单独的图像文件（和参考代码一致的命名）
    if config.save_single_images:
        # 转换为RGB格式保存原始图像（和参考代码一致）
        original_rgb = np.stack([original_img] * 3, axis=-1)
        save_single(original_rgb, os.path.join(sample_folder, "original.png"), cmap="gray")
        save_single(heat, os.path.join(sample_folder, "heatmap.png"),
                    cmap=config.vis_cmap, vmin=0.0, vmax=1.0)
        save_single(pred, os.path.join(sample_folder, "pred.png"), cmap="gray", vmin=0, vmax=1)
        save_single((gt_mask > 0.5).astype(np.uint8), os.path.join(sample_folder, "gt.png"),
                    cmap="gray", vmin=0, vmax=1)
        save_single(det_overlay, os.path.join(sample_folder, "detection_map_overlay.png"))

    # 10. 创建5列的overview图（和参考代码样式一致）
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))

    # 1. 原始图像
    axes[0].imshow(original_rgb, cmap="gray")
    axes[0].set_title("")
    axes[0].axis("off")

    # 2. 热力图
    im = axes[1].imshow(heat, cmap=config.vis_cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title("")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. 差异图
    axes[2].imshow(diff, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("")
    axes[2].axis("off")

    # 4. GT掩码
    axes[3].imshow((gt_mask > 0.5).astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("")
    axes[3].axis("off")

    # 5. 检测叠加图（只显示Dice分数）
    axes[4].imshow(det_overlay)
    axes[4].set_title(f"Dice: {dsc:.4f}", fontsize=16)
    axes[4].axis("off")

    # 保存overview图
    plt.tight_layout()
    overview_path = os.path.join(sample_folder, "overview.png")
    plt.savefig(overview_path, dpi=200, bbox_inches="tight")
    plt.close('all')

    # 11. 如果Dice分数高于阈值，复制到高分文件夹
    if dsc > config.high_dice_threshold:
        high_dice_folder = os.path.join(config.vis_high_dice_path, img_id)
        os.makedirs(high_dice_folder, exist_ok=True)

        # 复制所有文件到高分文件夹
        for file_name in os.listdir(sample_folder):
            src = os.path.join(sample_folder, file_name)
            dst = os.path.join(high_dice_folder, file_name)
            if not os.path.exists(dst):
                shutil.copy(src, dst)

    return sample_folder, dsc, pred


# --- 指标计算函数（无修改） ---
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


def test():
    """修改后的测试函数：移除指定样本限制，处理所有测试样本"""
    model.eval()
    gt_labels, gt_masks, pred_scores, pred_maps = [], [], [], []
    all_sample_ids = []
    all_original_imgs = []

    # 新增：存储每个异常图像的像素级指标
    dice_list, hd95_list, assd_list, ppv_list, sensitive_list = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating all test samples'):  # 修改进度条描述
            x_dict = {mod: batch['modalities'][mod].to(config.device) for mod in config.modalities}
            x = torch.stack(list(x_dict.values()), dim=0).mean(dim=0)

            # 获取原始图像（用于可视化）
            img_path = batch['img_path'][0]
            original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if img_path else x.squeeze().cpu().numpy()
            original_img = cv2.resize(original_img, (config.img_size, config.img_size))

            t_features = model.teacher.backbone(x)
            reconstruction = model.student(t_features[-1], t_features[:-1])
            anomaly_map = get_anomaly_map(t_features, reconstruction)

            # 收集数据
            gt_label = batch['label'].item()
            gt_mask = batch['mask'].squeeze().cpu().numpy()
            pred_score = anomaly_map.max().item()
            pred_map = anomaly_map.squeeze().cpu().numpy()

            gt_labels.append(gt_label)
            gt_masks.append(gt_mask)
            pred_scores.append(pred_score)
            pred_maps.append(pred_map)
            all_sample_ids.append(batch['id'][0])
            all_original_imgs.append(original_img)

            # 仅对异常图像计算像素级指标并可视化
            if gt_label == 1:
                # 可视化当前样本（使用新的可视化函数）
                _, dice, pred_bin = visualize_sample(
                    batch['id'][0],
                    original_img,
                    pred_map,
                    gt_mask
                )

                # 计算单张图像的像素级指标
                gt_bin = (gt_mask > 0.5).astype(np.uint8)
                dice_list.append(dice)
                hd95_list.append(calculate_hd95(pred_bin, gt_bin))
                assd_list.append(calculate_assd(pred_bin, gt_bin))
                ppv_list.append(calculate_ppv(pred_bin, gt_bin))
                sensitive_list.append(calculate_sensitive(pred_bin, gt_bin))

    # 计算整体指标
    gt_labels = np.array(gt_labels)
    pred_scores = np.array(pred_scores)
    image_bin_labels = (gt_labels > 0).astype(int)
    img_auroc = roc_auc_score(image_bin_labels, pred_scores) if len(np.unique(image_bin_labels)) > 1 else 0.0
    img_ap = average_precision_score(image_bin_labels, pred_scores) if len(np.unique(image_bin_labels)) > 1 else 0.0
    _, img_f1 = calculate_optimal_threshold_and_f1(pred_scores, image_bin_labels) if len(
        np.unique(image_bin_labels)) > 1 else (0.0, 0.0)

    # 像素级指标
    pixel_scores = np.concatenate([am.flatten() for am in pred_maps]) if pred_maps else np.array([])
    pixel_gt = np.concatenate([m.flatten() for m in gt_masks]) if gt_masks else np.array([])

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

    # 计算FLOPs和Params
    try:
        # 构建dummy input
        dummy_main = torch.randn(1, 2048, 16, 16).to(config.device)
        dummy_skip1 = torch.randn(1, 64, 256, 256).to(config.device)
        dummy_skip2 = torch.randn(1, 512, 64, 64).to(config.device)
        dummy_skip3 = torch.randn(1, 1024, 32, 32).to(config.device)

        # 计算student模型的FLOPs和Params
        flops, params = profile(
            model.student,
            inputs=(dummy_main, [dummy_skip1, dummy_skip2, dummy_skip3]),
            verbose=False
        )
        flops, params = clever_format([flops, params], "%.2f")
    except Exception as e:
        print_log(f"计算FLOPs/Params失败: {str(e)}", log)
        flops, params = "计算失败", "计算失败"

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

    # 加载预训练权重
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
            model.student.load_state_dict(checkpoint)
            print_log("权重文件无外层键，直接加载", log)

    except Exception as e:
        print_log(f"加载权重失败: {str(e)}", log)
        exit(1)

    # 创建测试数据集和加载器（加载所有测试样本，不再限制target_sample_ids）
    test_dataset = MultimodalDataset(config.data_root, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 执行测试和可视化
    print_log("Starting evaluation and visualization for ALL test samples...", log)  # 修改日志描述
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
    print_log("Evaluation Results (ALL Test Samples)", log)  # 修改标题
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

    # 打印可视化结果信息
    print_log(f"\nVisualization Results:", log)
    print_log(f'  All test samples saved to: {config.vis_BHSD_path}', log)  # 修改描述
    print_log(f'  Each sample has its own folder named by sample ID', log)
    print_log(f'  High dice samples (>{config.high_dice_threshold}) saved to: {config.vis_high_dice_path}', log)

    # 关闭日志文件
    log.close()
    print("Evaluation and visualization for ALL test samples completed!")  # 修改提示语
    print(f"Results saved to {os.path.join(config.save_dir, config.log_file)}")
    print(f"Visualizations saved to {config.visualization_root}")
    print(f'All test samples have their own folder in {config.vis_BHSD_path}')  # 修改提示语