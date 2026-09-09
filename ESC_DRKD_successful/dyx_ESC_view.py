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
from PIL import Image
import shutil

# 新增：计算FLOPs和参数量的库
from thop import profile, clever_format

warnings.filterwarnings('ignore')


# 配置参数
class Config:
    data_root = '/home/wuchangwei/dyx/successful/brain_data2'
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
    vis_miccai_path = os.path.join(visualization_root, 'miccai')  # 所有样本可视化结果
    vis_high_dice_path = os.path.join(visualization_root, 'scans_ct_dice90+')  # Dice>0.9的样本
    high_dice_threshold = 0.9  # 高分Dice阈值
    vis_cmap = 'jet'  # 热力图配色
    save_single_images = True  # 是否保存单独的图像文件

    # ========== 指定需要测试的样本ID列表 ==========

    # target_sample_ids = [
    #     'slice_05638a3a_0024', 'slice_40c1c697_0019', 'slice_5e8ae04c_0023', 'slice_6d01fdbb_0021',
    #     'slice_9228b183_0012', 'slice_b2a32890_0020', 'slice_c5e78b87_0025', 'slice_dc848389_0022',
    #     'slice_e7a64421_0028', 'slice_f5499bd8_0029', 'slice_f771eee0_0027', 'slice_ff4b2d5d_0018'
    # ]
    
    # target_sample_ids = [
    #     'slice_006e12e2_0021', 'slice_01563e80_0020', 'slice_0322d2ac_0019', 'slice_070b0a19_0017',
    #     'slice_0f6d75b4_0012', 'slice_1363571c_0020', 'slice_13ac1633_0008', 'slice_154e802e_0015',
    #     'slice_166bca38_0014', 'slice_1681d8fc_0021', 'slice_16d67ead_0017', 'slice_17b3be3f_0019',
    #     'slice_180bdb1b_0017', 'slice_18815933_0020', 'slice_1a6f32c7_0014', 'slice_256c2936_0019',
    #     'slice_266c306c_0024', 'slice_2670052f_0014', 'slice_2941dd4a_0017', 'slice_2b3d7810_0028',
    #     'slice_2b4662c3_0018', 'slice_2d522b7a_0018', 'slice_32807ffc_0015', 'slice_32be9794_0016',
    #     'slice_353c0df2_0011', 'slice_357c2feb_0020', 'slice_3667b5a1_0018', 'slice_3a89df4f_0014',
    #     'slice_3abdc21a_0024', 'slice_3ba59a29_0015', 'slice_3d79d93c_0016', 'slice_3ee97572_0011',
    #     'slice_3fedaa87_0015', 'slice_49de2df6_0016', 'slice_49ec823c_0018', 'slice_4bae9a78_0016',
    #     'slice_4e194f44_0014', 'slice_5024286a_0022', 'slice_5256b30f_0016', 'slice_5948f54b_0019',
    #     'slice_5d74ba84_0019', 'slice_5f06d13b_0016', 'slice_5f6d2b0b_0013', 'slice_60d097f0_0023',
    #     'slice_60e9197f_0018', 'slice_62c06d0a_0018', 'slice_6bfc5c7c_0026', 'slice_72649977_0013',
    #     'slice_734dcf05_0014', 'slice_7368a779_0017', 'slice_7731d243_0009', 'slice_7dd06f75_0017',
    #     'slice_7e4729a1_0015', 'slice_8145bf7f_0021', 'slice_81b4a440_0016', 'slice_83f4831c_0019',
    #     'slice_86a887fd_0013', 'slice_875a58a9_0018', 'slice_883e03a6_0018', 'slice_8a2ad1ae_0013',
    #     'slice_8c2751d3_0022', 'slice_8cb34379_0015', 'slice_8eccbd81_0009', 'slice_90d9a594_0017',
    #     'slice_930101f9_0017', 'slice_94b292f1_0015', 'slice_963da3a9_0010', 'slice_a17a1d94_0030',
    #     'slice_a1b3ed34_0015', 'slice_a79b3ee0_0017', 'slice_a7bf0f78_0013', 'slice_a8bfa199_0018',
    #     'slice_a8e78647_0013', 'slice_b21c2ce0_0020', 'slice_b4abdd37_0010', 'slice_bd90e498_0016',
    #     'slice_c0671d69_0021', 'slice_c1e17cbf_0017', 'slice_c9cafdb8_0022', 'slice_ca821592_0012',
    #     'slice_cc249a20_0027', 'slice_ce9256d7_0029', 'slice_d4a8452b_0015', 'slice_dc0d6bdc_0018',
    #     'slice_dcef378a_0011', 'slice_dd27525b_0025', 'slice_de774a12_0020', 'slice_df8b2cdc_0015',
    #     'slice_e09da3c1_0011', 'slice_e1e35396_0032', 'slice_e27882e8_0019', 'slice_e5a9cd35_0017',
    #     'slice_e60b6188_0018', 'slice_e7b5774c_0014', 'slice_e7c5e5e9_0016', 'slice_e7f3261a_0016',
    #     'slice_e9f71e55_0009', 'slice_ea7fad41_0012', 'slice_f0d1208c_0016', 'slice_f1907cd5_0027',
    #     'slice_f23b1bd8_0016', 'slice_f375efde_0020', 'slice_f3c55402_0017', 'slice_f41b6ae4_0025',
    #     'slice_f428e187_0019', 'slice_f4465d56_0026', 'slice_f6b18464_0016', 'slice_fad16b4b_0018'
    # ]
    target_sample_ids = [
        'slice_01196f9d_0013', 'slice_0137fce4_0012', 'slice_015d44d6_0013', 'slice_02923166_0006',
        'slice_038b40b2_0018', 'slice_050f877a_0020', 'slice_051a9307_0020', 'slice_096ad7fd_0012',
        'slice_0aa3f3a2_0011', 'slice_0dcbf71e_0012', 'slice_0deda7f3_0014', 'slice_0e5f5c40_0013',
        'slice_105453ea_0019', 'slice_112242bd_0015', 'slice_11d79618_0017', 'slice_13ba1113_0015',
        'slice_14973c0a_0015', 'slice_160d0eb3_0017', 'slice_1716318e_0016', 'slice_18e4ad8a_0014',
        'slice_1cb1dc21_0016', 'slice_21e7bad5_0018', 'slice_22c3b025_0012', 'slice_22df0034_0011',
        'slice_27306637_0021', 'slice_27b7e1eb_0009', 'slice_2ab7880f_0013', 'slice_2cc37ed4_0014',
        'slice_3140102c_0017', 'slice_33ca91cd_0015', 'slice_355049fc_0014', 'slice_37727098_0007',
        'slice_3871ef17_0015', 'slice_39c400da_0022', 'slice_3b738497_0012', 'slice_3fee348f_0015',
        'slice_40b5edde_0016', 'slice_418ba45c_0013', 'slice_41c8e9f9_0016', 'slice_42eead85_0012',
        'slice_44daa30f_0015', 'slice_45ac3f97_0015', 'slice_45c6c7c5_0011', 'slice_467ea1c3_0018',
        'slice_46c0885a_0015', 'slice_478ad42c_0019', 'slice_48df79a4_0011', 'slice_4937fc18_0012',
        'slice_4a55c909_0015', 'slice_4c1c9fe9_0008', 'slice_4e4a5d47_0013', 'slice_53942ba0_0021',
        'slice_54c11144_0016', 'slice_55d665d5_0015', 'slice_575481ac_0009', 'slice_5a3790c7_0013',
        'slice_5c7beb9f_0014', 'slice_5cedf3bd_0019', 'slice_5d72847c_0020', 'slice_5e479025_0012',
        'slice_5e686e89_0013', 'slice_5e9d4093_0011', 'slice_628169b5_0013', 'slice_63154bd6_0010',
        'slice_6419d2b2_0012', 'slice_64a76a08_0016', 'slice_65d52028_0013', 'slice_662568dc_0016',
        'slice_664eb6a4_0013', 'slice_6781aae2_0011', 'slice_678e41b9_0015', 'slice_67b85b55_0012',
        'slice_6b340054_0016', 'slice_6b8753ce_0015', 'slice_6d971b3d_0013', 'slice_6f919857_0015',
        'slice_707a8f8b_0014', 'slice_7378af21_0017', 'slice_756a4c41_0017', 'slice_7730f7d1_0018',
        'slice_77377dad_0011', 'slice_7762b64a_0014', 'slice_7951e2d7_0012', 'slice_7a726fb7_0013',
        'slice_7cffaa4c_0020', 'slice_7d595407_0011', 'slice_7e45f399_0015', 'slice_7e53c409_0020',
        'slice_7e7373d1_0014', 'slice_82b1295d_0017', 'slice_8440a0e1_0020', 'slice_856f3a6c_0019',
        'slice_86669f86_0013', 'slice_866b222d_0019', 'slice_872d7429_0016', 'slice_8771fa61_0016',
        'slice_878045d2_0016', 'slice_880ec344_0019', 'slice_888802fb_0017', 'slice_88f80874_0018',
        'slice_8c085c6b_0014', 'slice_8c5753ed_0016', 'slice_8c81f151_0012', 'slice_8cedfc55_0013',
        'slice_8efd6c7f_0004', 'slice_91f35038_0012', 'slice_92e7be1a_0017', 'slice_948ff989_0009',
        'slice_95925aa0_0021', 'slice_95cecdd9_0017', 'slice_987da749_0012', 'slice_9a0ca8d8_0017',
        'slice_9e2b374e_0013', 'slice_9f205e68_0012', 'slice_9fdd85b2_0010', 'slice_a058b83c_0013',
        'slice_a2ef8b00_0012', 'slice_a3e34fe0_0020', 'slice_a4198f95_0021', 'slice_a5a9f4a4_0015',
        'slice_a753ca6a_0009', 'slice_a7afab63_0018', 'slice_aa5b9ff7_0022', 'slice_acd5ecad_0014',
        'slice_ae123f7f_0014', 'slice_ae4f7e1b_0015', 'slice_af685a6d_0014', 'slice_afa4f311_0016',
        'slice_b06f6abb_0010', 'slice_b193af7b_0011', 'slice_b24f0d5c_0015', 'slice_b43b256b_0019',
        'slice_b4f9bc2a_0017', 'slice_b6c92184_0013', 'slice_b7f44e2a_0010', 'slice_b9e0e666_0012',
        'slice_ba6c49e3_0014', 'slice_bc0a578a_0019', 'slice_bc0f002b_0016', 'slice_bd0c2878_0013',
        'slice_bed215f9_0020', 'slice_c01a8456_0021', 'slice_c0e317e9_0013', 'slice_c2659670_0016',
        'slice_c905c6b8_0016', 'slice_ca61f7c6_0012', 'slice_cb60878c_0018', 'slice_ceaf20bc_0009',
        'slice_ceb717c7_0009', 'slice_cf453a41_0014', 'slice_cf978fc6_0018', 'slice_d53f6667_0014',
        'slice_d756260b_0014', 'slice_d7f71e0b_0012', 'slice_d8ed8c7d_0018', 'slice_daaa9392_0013',
        'slice_db90c4a4_0019', 'slice_dd06d487_0011', 'slice_dd074d45_0014', 'slice_de348060_0016',
        'slice_df073da5_0018', 'slice_e103aa4c_0010', 'slice_e3612c66_0016', 'slice_e36bbece_0011',
        'slice_e51f5e19_0016', 'slice_e58c2777_0013', 'slice_ea512a0c_0013', 'slice_eab4a91d_0017',
        'slice_ebf0e326_0015', 'slice_eee6eb05_0014', 'slice_f3032946_0019', 'slice_f32312ed_0014',
        'slice_f33733d1_0016', 'slice_f3f8dc93_0018', 'slice_f615c3df_0017', 'slice_f6f3ca71_0017',
        'slice_f729ad96_0010', 'slice_f7347206_0012', 'slice_f7a8d7f1_0015', 'slice_f9ab53f4_0018',
        'slice_fc5a31c7_0019', 'slice_fc5f5625_0018', 'slice_fc9ba057_0011', 'slice_fd9975bf_0012',
        'slice_fe6e589c_0014', 'slice_ff7f071e_0016'
    ]


# 初始化配置
config = Config()
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.vis_miccai_path, exist_ok=True)
os.makedirs(config.vis_high_dice_path, exist_ok=True)
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


# --- 数据加载类 (修改：只加载指定样本) ---
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
            # 只加载指定的样本ID
            target_ids = set(config.target_sample_ids)
            for label, subfolder in enumerate(['NORMAL', 'ABNORMAL']):
                for mod in config.modalities:
                    mod_path = os.path.join(data_path, 'test', subfolder, mod)
                    for img_name in os.listdir(mod_path):
                        img_id = img_name.split('.')[0]
                        # 只处理指定的样本ID
                        if img_id not in target_ids:
                            continue

                        if not any(s['id'] == img_id for s in self.samples):
                            self.samples.append({
                                'id': img_id,
                                'label': label,
                                'mask': os.path.join(data_path, 'test', 'ABNORMAL', 'mask',
                                                     f'{img_id}_mask.png') if label == 1 else None
                            })
                        sample = next(s for s in self.samples if s['id'] == img_id)
                        sample[mod] = os.path.join(mod_path, img_name)

        # 过滤掉不在目标列表中的样本（双重保险）
        if mode != 'train':
            self.samples = [s for s in self.samples if s['id'] in target_ids]
        print(f"Loaded {len(self.samples)} target samples for {mode} mode")

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


# ========== 新增可视化函数 ==========
def normalize_img(img):
    """归一化图像到0-1范围"""
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def save_single_image(data, save_path, cmap='gray', vmin=None, vmax=None):
    """保存单张图像"""
    fig = plt.figure(figsize=(5, 5))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_detection_overlay(pred_bin, gt_bin):
    """创建检测叠加图（TP:红, FP:蓝, FN:绿）"""
    # 创建RGB图像
    overlay = np.zeros((pred_bin.shape[0], pred_bin.shape[1], 3), dtype=np.float32)

    # 计算TP/FP/FN掩码
    mask_tp = np.logical_and(pred_bin, gt_bin)  # 真阳性
    mask_fp = np.logical_and(pred_bin, 1 - gt_bin)  # 假阳性
    mask_fn = np.logical_and(1 - pred_bin, gt_bin)  # 假阴性

    # 定义配色（医学分割常用配色）
    color_tp = [224 / 255, 68 / 255, 19 / 255]  # 红色
    color_fp = [80 / 255, 128 / 255, 209 / 255]  # 蓝色
    color_fn = [51 / 255, 186 / 255, 7 / 255]  # 绿色

    # 应用颜色
    overlay[mask_fp] = color_fp
    overlay[mask_fn] = color_fn
    overlay[mask_tp] = color_tp

    return overlay


def visualize_sample(img_id, original_img, anomaly_map, gt_mask, pred_mask, dice_score, best_threshold):
    """可视化单个样本：为每个样本创建独立文件夹"""
    # 数据预处理
    original_img = normalize_img(original_img)
    anomaly_map = anomaly_map.squeeze()
    gt_mask = gt_mask.squeeze()
    pred_mask = (anomaly_map > best_threshold).astype(np.float32)  # 使用最佳阈值二值化

    # 为当前样本创建独立文件夹
    sample_folder = os.path.join(config.vis_miccai_path, img_id)
    os.makedirs(sample_folder, exist_ok=True)

    # 创建6列可视化图
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))

    # 1. 原始图像
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 2. 异常热力图
    im2 = axes[1].imshow(anomaly_map, cmap=config.vis_cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Anomaly Heatmap\n(min:{anomaly_map.min():.3f}, max:{anomaly_map.max():.3f})")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. 差异图
    diff = np.abs(pred_mask - gt_mask)
    axes[2].imshow(diff, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Difference Map\n(Grayscale)")
    axes[2].axis('off')

    # 4. GT掩码
    axes[3].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title("GT Mask")
    axes[3].axis('off')

    # 5. 预测掩码
    axes[4].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    axes[4].set_title(f"Pred Mask\nDice: {dice_score:.4f}")
    axes[4].axis('off')

    # 6. 检测叠加图
    detection_overlay = create_detection_overlay(pred_mask, gt_mask)
    axes[5].imshow(detection_overlay)
    axes[5].set_title("Detection Overlay\n(TP:Red, FP:Blue, FN:Green)")
    axes[5].axis('off')

    # 保存综合图到样本独立文件夹
    overview_path = os.path.join(sample_folder, f"{img_id}_overview.png")
    plt.suptitle(f"{img_id} | Dice Score: {dice_score:.4f} | Threshold: {best_threshold:.4f}")
    plt.savefig(overview_path, bbox_inches='tight', dpi=150)
    plt.close('all')

    # 保存单独的图像文件到样本文件夹
    if config.save_single_images:
        save_single_image(original_img, os.path.join(sample_folder, "original.png"), cmap='gray')
        save_single_image(anomaly_map, os.path.join(sample_folder, "heatmap.png"),
                          cmap=config.vis_cmap, vmin=0.0, vmax=1.0)
        save_single_image(gt_mask, os.path.join(sample_folder, "gt_mask.png"), cmap='gray')
        save_single_image(pred_mask, os.path.join(sample_folder, "pred_mask.png"), cmap='gray')
        save_single_image(diff, os.path.join(sample_folder, "difference_map.png"), cmap='gray')
        save_single_image(detection_overlay, os.path.join(sample_folder, "detection_overlay.png"))

    # 如果Dice分数高于阈值，复制到高分文件夹
    if dice_score > config.high_dice_threshold:
        high_dice_folder = os.path.join(config.vis_high_dice_path, img_id)
        os.makedirs(high_dice_folder, exist_ok=True)

        # 复制所有文件到高分文件夹
        for file_name in os.listdir(sample_folder):
            src = os.path.join(sample_folder, file_name)
            dst = os.path.join(high_dice_folder, file_name)
            if not os.path.exists(dst):
                shutil.copy(src, dst)

    return sample_folder


def test():
    """修改后的测试函数：包含可视化功能"""
    model.eval()
    gt_labels, gt_masks, pred_scores, pred_maps = [], [], [], []
    all_sample_ids = []
    all_original_imgs = []

    # 新增：存储每个异常图像的像素级指标
    dice_list, hd95_list, assd_list, ppv_list, sensitive_list = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating target samples'):
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
                # 使用像素级最佳阈值二值化
                pixel_threshold, _ = calculate_optimal_threshold_and_f1(
                    np.concatenate([m.flatten() for m in pred_maps]),
                    np.concatenate([m.flatten() for m in gt_masks])
                )
                pred_bin = (pred_map > pixel_threshold).astype(np.uint8)
                gt_bin = (gt_mask > 0.5).astype(np.uint8)

                # 计算单张图像的像素级指标
                dice = calculate_dsc(pred_bin, gt_bin)
                dice_list.append(dice)
                hd95_list.append(calculate_hd95(pred_bin, gt_bin))
                assd_list.append(calculate_assd(pred_bin, gt_bin))
                ppv_list.append(calculate_ppv(pred_bin, gt_bin))
                sensitive_list.append(calculate_sensitive(pred_bin, gt_bin))

                # 可视化当前样本（每个样本创建独立文件夹）
                visualize_sample(batch['id'][0], original_img, pred_map, gt_mask, pred_bin, dice, pixel_threshold)

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
    # 注意：请确保ESC_DRKD_Multimodal类的导入路径正确
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

    # 创建测试数据集和加载器（只加载指定样本）
    test_dataset = MultimodalDataset(config.data_root, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 执行测试和可视化
    print_log("Starting evaluation and visualization for target samples...", log)
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
    print_log("Evaluation Results (Target Samples)", log)
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
    print_log(f'  Target samples saved to: {config.vis_miccai_path}', log)
    print_log(f'  Each sample has its own folder named by sample ID', log)
    print_log(f'  High dice samples (>{config.high_dice_threshold}) saved to: {config.vis_high_dice_path}', log)

    # 关闭日志文件
    log.close()
    print("Evaluation and visualization for target samples completed!")
    print(f"Results saved to {os.path.join(config.save_dir, config.log_file)}")
    print(f"Visualizations saved to {config.visualization_root}")
    print(f"Each target sample has its own folder in {config.vis_miccai_path}")

# import os
# import cv2
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
# import numpy as np
# from tqdm import tqdm
# from scipy.ndimage import distance_transform_edt
# from new_model import ESC_DRKD_Multimodal  # 根据实际路径调整
# from utils import AverageMeter, EarlyStop, print_log  # 根据实际路径调整
# import torch.nn.functional as F
# import warnings
# import matplotlib.pyplot as plt
# from PIL import Image
# import shutil
#
# # 新增：计算FLOPs和参数量的库
# from thop import profile, clever_format
#
# warnings.filterwarnings('ignore')
#
#
# # 配置参数
# class Config:
#     data_root = '/home/wuchangwei/dyx/successful/brain_data2'
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
#     # ========== 新增可视化配置 ==========
#     visualization_root = 'visualization_results'  # 可视化结果根目录
#     vis_miccai_path = os.path.join(visualization_root, 'miccai')  # 所有样本可视化结果
#     vis_high_dice_path = os.path.join(visualization_root, 'miccai_dice90+')  # Dice>0.9的样本
#     high_dice_threshold = 0.9  # 高分Dice阈值
#     vis_cmap = 'jet'  # 热力图配色
#     save_single_images = True  # 是否保存单独的图像文件
#
#     # ========== 指定需要测试的样本ID列表 ==========
#     target_sample_ids = [
#         'slice_01196f9d_0013', 'slice_0137fce4_0012', 'slice_015d44d6_0013', 'slice_02923166_0006',
#         'slice_038b40b2_0018', 'slice_050f877a_0020', 'slice_051a9307_0020', 'slice_096ad7fd_0012',
#         'slice_0aa3f3a2_0011', 'slice_0dcbf71e_0012', 'slice_0deda7f3_0014', 'slice_0e5f5c40_0013',
#         'slice_105453ea_0019', 'slice_112242bd_0015', 'slice_11d79618_0017', 'slice_13ba1113_0015',
#         'slice_14973c0a_0015', 'slice_160d0eb3_0017', 'slice_1716318e_0016', 'slice_18e4ad8a_0014',
#         'slice_1cb1dc21_0016', 'slice_21e7bad5_0018', 'slice_22c3b025_0012', 'slice_22df0034_0011',
#         'slice_27306637_0021', 'slice_27b7e1eb_0009', 'slice_2ab7880f_0013', 'slice_2cc37ed4_0014',
#         'slice_3140102c_0017', 'slice_33ca91cd_0015', 'slice_355049fc_0014', 'slice_37727098_0007',
#         'slice_3871ef17_0015', 'slice_39c400da_0022', 'slice_3b738497_0012', 'slice_3fee348f_0015',
#         'slice_40b5edde_0016', 'slice_418ba45c_0013', 'slice_41c8e9f9_0016', 'slice_42eead85_0012',
#         'slice_44daa30f_0015', 'slice_45ac3f97_0015', 'slice_45c6c7c5_0011', 'slice_467ea1c3_0018',
#         'slice_46c0885a_0015', 'slice_478ad42c_0019', 'slice_48df79a4_0011', 'slice_4937fc18_0012',
#         'slice_4a55c909_0015', 'slice_4c1c9fe9_0008', 'slice_4e4a5d47_0013', 'slice_53942ba0_0021',
#         'slice_54c11144_0016', 'slice_55d665d5_0015', 'slice_575481ac_0009', 'slice_5a3790c7_0013',
#         'slice_5c7beb9f_0014', 'slice_5cedf3bd_0019', 'slice_5d72847c_0020', 'slice_5e479025_0012',
#         'slice_5e686e89_0013', 'slice_5e9d4093_0011', 'slice_628169b5_0013', 'slice_63154bd6_0010',
#         'slice_6419d2b2_0012', 'slice_64a76a08_0016', 'slice_65d52028_0013', 'slice_662568dc_0016',
#         'slice_664eb6a4_0013', 'slice_6781aae2_0011', 'slice_678e41b9_0015', 'slice_67b85b55_0012',
#         'slice_6b340054_0016', 'slice_6b8753ce_0015', 'slice_6d971b3d_0013', 'slice_6f919857_0015',
#         'slice_707a8f8b_0014', 'slice_7378af21_0017', 'slice_756a4c41_0017', 'slice_7730f7d1_0018',
#         'slice_77377dad_0011', 'slice_7762b64a_0014', 'slice_7951e2d7_0012', 'slice_7a726fb7_0013',
#         'slice_7cffaa4c_0020', 'slice_7d595407_0011', 'slice_7e45f399_0015', 'slice_7e53c409_0020',
#         'slice_7e7373d1_0014', 'slice_82b1295d_0017', 'slice_8440a0e1_0020', 'slice_856f3a6c_0019',
#         'slice_86669f86_0013', 'slice_866b222d_0019', 'slice_872d7429_0016', 'slice_8771fa61_0016',
#         'slice_878045d2_0016', 'slice_880ec344_0019', 'slice_888802fb_0017', 'slice_88f80874_0018',
#         'slice_8c085c6b_0014', 'slice_8c5753ed_0016', 'slice_8c81f151_0012', 'slice_8cedfc55_0013',
#         'slice_8efd6c7f_0004', 'slice_91f35038_0012', 'slice_92e7be1a_0017', 'slice_948ff989_0009',
#         'slice_95925aa0_0021', 'slice_95cecdd9_0017', 'slice_987da749_0012', 'slice_9a0ca8d8_0017',
#         'slice_9e2b374e_0013', 'slice_9f205e68_0012', 'slice_9fdd85b2_0010', 'slice_a058b83c_0013',
#         'slice_a2ef8b00_0012', 'slice_a3e34fe0_0020', 'slice_a4198f95_0021', 'slice_a5a9f4a4_0015',
#         'slice_a753ca6a_0009', 'slice_a7afab63_0018', 'slice_aa5b9ff7_0022', 'slice_acd5ecad_0014',
#         'slice_ae123f7f_0014', 'slice_ae4f7e1b_0015', 'slice_af685a6d_0014', 'slice_afa4f311_0016',
#         'slice_b06f6abb_0010', 'slice_b193af7b_0011', 'slice_b24f0d5c_0015', 'slice_b43b256b_0019',
#         'slice_b4f9bc2a_0017', 'slice_b6c92184_0013', 'slice_b7f44e2a_0010', 'slice_b9e0e666_0012',
#         'slice_ba6c49e3_0014', 'slice_bc0a578a_0019', 'slice_bc0f002b_0016', 'slice_bd0c2878_0013',
#         'slice_bed215f9_0020', 'slice_c01a8456_0021', 'slice_c0e317e9_0013', 'slice_c2659670_0016',
#         'slice_c905c6b8_0016', 'slice_ca61f7c6_0012', 'slice_cb60878c_0018', 'slice_ceaf20bc_0009',
#         'slice_ceb717c7_0009', 'slice_cf453a41_0014', 'slice_cf978fc6_0018', 'slice_d53f6667_0014',
#         'slice_d756260b_0014', 'slice_d7f71e0b_0012', 'slice_d8ed8c7d_0018', 'slice_daaa9392_0013',
#         'slice_db90c4a4_0019', 'slice_dd06d487_0011', 'slice_dd074d45_0014', 'slice_de348060_0016',
#         'slice_df073da5_0018', 'slice_e103aa4c_0010', 'slice_e3612c66_0016', 'slice_e36bbece_0011',
#         'slice_e51f5e19_0016', 'slice_e58c2777_0013', 'slice_ea512a0c_0013', 'slice_eab4a91d_0017',
#         'slice_ebf0e326_0015', 'slice_eee6eb05_0014', 'slice_f3032946_0019', 'slice_f32312ed_0014',
#         'slice_f33733d1_0016', 'slice_f3f8dc93_0018', 'slice_f615c3df_0017', 'slice_f6f3ca71_0017',
#         'slice_f729ad96_0010', 'slice_f7347206_0012', 'slice_f7a8d7f1_0015', 'slice_f9ab53f4_0018',
#         'slice_fc5a31c7_0019', 'slice_fc5f5625_0018', 'slice_fc9ba057_0011', 'slice_fd9975bf_0012',
#         'slice_fe6e589c_0014', 'slice_ff7f071e_0016'
#     ]
#
#
# # 初始化配置
# config = Config()
# os.makedirs(config.save_dir, exist_ok=True)
# os.makedirs(config.vis_miccai_path, exist_ok=True)
# os.makedirs(config.vis_high_dice_path, exist_ok=True)
# log = open(os.path.join(config.save_dir, config.log_file), 'w')
#
#
# # --- 新增/完善指标计算函数 ---
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
# # --- 数据加载类 (修改：只加载指定样本) ---
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
#             # 只加载指定的样本ID
#             target_ids = set(config.target_sample_ids)
#             for label, subfolder in enumerate(['NORMAL', 'ABNORMAL']):
#                 for mod in config.modalities:
#                     mod_path = os.path.join(data_path, 'test', subfolder, mod)
#                     for img_name in os.listdir(mod_path):
#                         img_id = img_name.split('.')[0]
#                         # 只处理指定的样本ID
#                         if img_id not in target_ids:
#                             continue
#
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
#         # 过滤掉不在目标列表中的样本（双重保险）
#         if mode != 'train':
#             self.samples = [s for s in self.samples if s['id'] in target_ids]
#         print(f"Loaded {len(self.samples)} target samples for {mode} mode")
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
#                 'id': sample['id'],
#                 'img_path': sample[config.modalities[0]] if config.modalities[0] in sample else None  # 保存原始图像路径
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
#             'id': [item['id'] for item in batch],
#             'img_path': [item['img_path'] for item in batch]
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
# # ========== 新增可视化函数 ==========
# def normalize_img(img):
#     """归一化图像到0-1范围"""
#     return (img - img.min()) / (img.max() - img.min() + 1e-8)
#
#
# def save_single_image(data, save_path, cmap='gray', vmin=None, vmax=None):
#     """保存单张图像"""
#     fig = plt.figure(figsize=(5, 5))
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#
#
# def create_detection_overlay(pred_bin, gt_bin):
#     """创建检测叠加图（TP:红, FP:蓝, FN:绿）"""
#     # 创建RGB图像
#     overlay = np.zeros((pred_bin.shape[0], pred_bin.shape[1], 3), dtype=np.float32)
#
#     # 计算TP/FP/FN掩码
#     mask_tp = np.logical_and(pred_bin, gt_bin)  # 真阳性
#     mask_fp = np.logical_and(pred_bin, 1 - gt_bin)  # 假阳性
#     mask_fn = np.logical_and(1 - pred_bin, gt_bin)  # 假阴性
#
#     # 定义配色（医学分割常用配色）
#     color_tp = [224 / 255, 68 / 255, 19 / 255]  # 红色
#     color_fp = [80 / 255, 128 / 255, 209 / 255]  # 蓝色
#     color_fn = [51 / 255, 186 / 255, 7 / 255]  # 绿色
#
#     # 应用颜色
#     overlay[mask_fp] = color_fp
#     overlay[mask_fn] = color_fn
#     overlay[mask_tp] = color_tp
#
#     return overlay
#
#
# def visualize_sample(img_id, original_img, anomaly_map, gt_mask, pred_mask, dice_score, best_threshold):
#     """可视化单个样本：为每个样本创建独立文件夹"""
#     # 数据预处理
#     original_img = normalize_img(original_img)
#     anomaly_map = anomaly_map.squeeze()
#     gt_mask = gt_mask.squeeze()
#     pred_mask = (anomaly_map > best_threshold).astype(np.float32)  # 使用最佳阈值二值化
#
#     # 为当前样本创建独立文件夹
#     sample_folder = os.path.join(config.vis_miccai_path, img_id)
#     os.makedirs(sample_folder, exist_ok=True)
#
#     # 创建6列可视化图
#     fig, axes = plt.subplots(1, 6, figsize=(30, 5))
#
#     # 1. 原始图像
#     axes[0].imshow(original_img, cmap='gray')
#     axes[0].set_title("Original Image")
#     axes[0].axis('off')
#
#     # 2. 异常热力图
#     im2 = axes[1].imshow(anomaly_map, cmap=config.vis_cmap, vmin=0.0, vmax=1.0)
#     axes[1].set_title(f"Anomaly Heatmap\n(min:{anomaly_map.min():.3f}, max:{anomaly_map.max():.3f})")
#     axes[1].axis('off')
#     plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
#
#     # 3. 差异图
#     diff = np.abs(pred_mask - gt_mask)
#     axes[2].imshow(diff, cmap='gray', vmin=0, vmax=1)
#     axes[2].set_title("Difference Map\n(Grayscale)")
#     axes[2].axis('off')
#
#     # 4. GT掩码
#     axes[3].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
#     axes[3].set_title("GT Mask")
#     axes[3].axis('off')
#
#     # 5. 预测掩码
#     axes[4].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
#     axes[4].set_title(f"Pred Mask\nDice: {dice_score:.4f}")
#     axes[4].axis('off')
#
#     # 6. 检测叠加图
#     detection_overlay = create_detection_overlay(pred_mask, gt_mask)
#     axes[5].imshow(detection_overlay)
#     axes[5].set_title("Detection Overlay\n(TP:Red, FP:Blue, FN:Green)")
#     axes[5].axis('off')
#
#     # 保存综合图到样本独立文件夹
#     overview_path = os.path.join(sample_folder, f"{img_id}_overview.png")
#     plt.suptitle(f"{img_id} | Dice Score: {dice_score:.4f} | Threshold: {best_threshold:.4f}")
#     plt.savefig(overview_path, bbox_inches='tight', dpi=150)
#     plt.close('all')
#
#     # 保存单独的图像文件到样本文件夹
#     if config.save_single_images:
#         save_single_image(original_img, os.path.join(sample_folder, "original.png"), cmap='gray')
#         save_single_image(anomaly_map, os.path.join(sample_folder, "heatmap.png"),
#                           cmap=config.vis_cmap, vmin=0.0, vmax=1.0)
#         save_single_image(gt_mask, os.path.join(sample_folder, "gt_mask.png"), cmap='gray')
#         save_single_image(pred_mask, os.path.join(sample_folder, "pred_mask.png"), cmap='gray')
#         save_single_image(diff, os.path.join(sample_folder, "difference_map.png"), cmap='gray')
#         save_single_image(detection_overlay, os.path.join(sample_folder, "detection_overlay.png"))
#
#     # 如果Dice分数高于阈值，复制到高分文件夹
#     if dice_score > config.high_dice_threshold:
#         high_dice_folder = os.path.join(config.vis_high_dice_path, img_id)
#         os.makedirs(high_dice_folder, exist_ok=True)
#
#         # 复制所有文件到高分文件夹
#         for file_name in os.listdir(sample_folder):
#             src = os.path.join(sample_folder, file_name)
#             dst = os.path.join(high_dice_folder, file_name)
#             if not os.path.exists(dst):
#                 shutil.copy(src, dst)
#
#     return sample_folder
#
#
# def test():
#     """修改后的测试函数：包含可视化功能"""
#     model.eval()
#     gt_labels, gt_masks, pred_scores, pred_maps = [], [], [], []
#     all_sample_ids = []
#     all_original_imgs = []
#
#     # 新增：存储每个异常图像的像素级指标
#     dice_list, hd95_list, assd_list, ppv_list, sensitive_list = [], [], [], [], []
#
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc='Evaluating target samples'):
#             x_dict = {mod: batch['modalities'][mod].to(config.device) for mod in config.modalities}
#             x = torch.stack(list(x_dict.values()), dim=0).mean(dim=0)
#
#             # 获取原始图像（用于可视化）
#             img_path = batch['img_path'][0]
#             original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if img_path else x.squeeze().cpu().numpy()
#             original_img = cv2.resize(original_img, (config.img_size, config.img_size))
#
#             t_features = model.teacher.backbone(x)
#             reconstruction = model.student(t_features[-1], t_features[:-1])
#             anomaly_map = get_anomaly_map(t_features, reconstruction)
#
#             # 收集数据
#             gt_label = batch['label'].item()
#             gt_mask = batch['mask'].squeeze().cpu().numpy()
#             pred_score = anomaly_map.max().item()
#             pred_map = anomaly_map.squeeze().cpu().numpy()
#
#             gt_labels.append(gt_label)
#             gt_masks.append(gt_mask)
#             pred_scores.append(pred_score)
#             pred_maps.append(pred_map)
#             all_sample_ids.append(batch['id'][0])
#             all_original_imgs.append(original_img)
#
#             # 仅对异常图像计算像素级指标并可视化
#             if gt_label == 1:
#                 # 使用像素级最佳阈值二值化
#                 pixel_threshold, _ = calculate_optimal_threshold_and_f1(
#                     np.concatenate([m.flatten() for m in pred_maps]),
#                     np.concatenate([m.flatten() for m in gt_masks])
#                 )
#                 pred_bin = (pred_map > pixel_threshold).astype(np.uint8)
#                 gt_bin = (gt_mask > 0.5).astype(np.uint8)
#
#                 # 计算单张图像的像素级指标
#                 dice = calculate_dsc(pred_bin, gt_bin)
#                 dice_list.append(dice)
#                 hd95_list.append(calculate_hd95(pred_bin, gt_bin))
#                 assd_list.append(calculate_assd(pred_bin, gt_bin))
#                 ppv_list.append(calculate_ppv(pred_bin, gt_bin))
#                 sensitive_list.append(calculate_sensitive(pred_bin, gt_bin))
#
#                 # 可视化当前样本（每个样本创建独立文件夹）
#                 visualize_sample(batch['id'][0], original_img, pred_map, gt_mask, pred_bin, dice, pixel_threshold)
#
#     # 计算整体指标
#     gt_labels = np.array(gt_labels)
#     pred_scores = np.array(pred_scores)
#     image_bin_labels = (gt_labels > 0).astype(int)
#     img_auroc = roc_auc_score(image_bin_labels, pred_scores) if len(np.unique(image_bin_labels)) > 1 else 0.0
#     img_ap = average_precision_score(image_bin_labels, pred_scores) if len(np.unique(image_bin_labels)) > 1 else 0.0
#     _, img_f1 = calculate_optimal_threshold_and_f1(pred_scores, image_bin_labels) if len(
#         np.unique(image_bin_labels)) > 1 else (0.0, 0.0)
#
#     # 像素级指标
#     pixel_scores = np.concatenate([am.flatten() for am in pred_maps]) if pred_maps else np.array([])
#     pixel_gt = np.concatenate([m.flatten() for m in gt_masks]) if gt_masks else np.array([])
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
#     # 计算FLOPs和Params
#     try:
#         # 构建dummy input
#         dummy_main = torch.randn(1, 2048, 16, 16).to(config.device)
#         dummy_skip1 = torch.randn(1, 64, 256, 256).to(config.device)
#         dummy_skip2 = torch.randn(1, 512, 64, 64).to(config.device)
#         dummy_skip3 = torch.randn(1, 1024, 32, 32).to(config.device)
#
#         # 计算student模型的FLOPs和Params
#         flops, params = profile(
#             model.student,
#             inputs=(dummy_main, [dummy_skip1, dummy_skip2, dummy_skip3]),
#             verbose=False
#         )
#         flops, params = clever_format([flops, params], "%.2f")
#     except Exception as e:
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
#     # 注意：请确保ESC_DRKD_Multimodal类的导入路径正确
#     model = ESC_DRKD_Multimodal(modalities=config.modalities).to(config.device)
#
#     # 加载预训练权重
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
#             model.student.load_state_dict(checkpoint)
#             print_log("权重文件无外层键，直接加载", log)
#
#     except Exception as e:
#         print_log(f"加载权重失败: {str(e)}", log)
#         exit(1)
#
#     # 创建测试数据集和加载器（只加载指定样本）
#     test_dataset = MultimodalDataset(config.data_root, mode='test')
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
#
#     # 执行测试和可视化
#     print_log("Starting evaluation and visualization for target samples...", log)
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
#     print_log("Evaluation Results (Target Samples)", log)
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
#     # 打印可视化结果信息
#     print_log(f"\nVisualization Results:", log)
#     print_log(f'  Target samples saved to: {config.vis_miccai_path}', log)
#     print_log(f'  Each sample has its own folder named by sample ID', log)
#     print_log(f'  High dice samples (>{config.high_dice_threshold}) saved to: {config.vis_high_dice_path}', log)
#
#     # 关闭日志文件
#     log.close()
#     print("Evaluation and visualization for target samples completed!")
#     print(f"Results saved to {os.path.join(config.save_dir, config.log_file)}")
#     print(f"Visualizations saved to {config.visualization_root}")
#     print(f"Each target sample has its own folder in {config.vis_miccai_path}")

# import os
# import cv2
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
# import numpy as np
# from tqdm import tqdm
# from scipy.ndimage import distance_transform_edt
# from new_model import ESC_DRKD_Multimodal  # 根据实际路径调整
# from utils import AverageMeter, EarlyStop, print_log  # 根据实际路径调整
# import torch.nn.functional as F
# import warnings
# import matplotlib.pyplot as plt
# from PIL import Image
# import shutil
#
# # 新增：计算FLOPs和参数量的库
# from thop import profile, clever_format
#
# warnings.filterwarnings('ignore')
#
#
# # 配置参数
# class Config:
#     data_root = '/home/wuchangwei/dyx/successful/brain_data4'
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
#     # ========== 新增可视化配置 ==========
#     visualization_root = 'visualization_results'  # 可视化结果根目录
#     vis_miccai_path = os.path.join(visualization_root, 'miccai')  # 所有样本可视化结果
#     vis_high_dice_path = os.path.join(visualization_root, 'miccai_dice90+')  # Dice>0.9的样本
#     high_dice_threshold = 0.9  # 高分Dice阈值
#     vis_cmap = 'jet'  # 热力图配色
#     save_single_images = True  # 是否保存单独的图像文件
#
#
# # 初始化配置
# config = Config()
# os.makedirs(config.save_dir, exist_ok=True)
# os.makedirs(config.vis_miccai_path, exist_ok=True)
# os.makedirs(config.vis_high_dice_path, exist_ok=True)
# log = open(os.path.join(config.save_dir, config.log_file), 'w')
#
#
# # --- 新增/完善指标计算函数 ---
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
#                 'id': sample['id'],
#                 'img_path': sample[config.modalities[0]] if config.modalities[0] in sample else None  # 保存原始图像路径
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
#             'id': [item['id'] for item in batch],
#             'img_path': [item['img_path'] for item in batch]
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
# # ========== 新增可视化函数 ==========
# def normalize_img(img):
#     """归一化图像到0-1范围"""
#     return (img - img.min()) / (img.max() - img.min() + 1e-8)
#
#
# def save_single_image(data, save_path, cmap='gray', vmin=None, vmax=None):
#     """保存单张图像"""
#     fig = plt.figure(figsize=(5, 5))
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#
#
# def create_detection_overlay(pred_bin, gt_bin):
#     """创建检测叠加图（TP:红, FP:蓝, FN:绿）"""
#     # 创建RGB图像
#     overlay = np.zeros((pred_bin.shape[0], pred_bin.shape[1], 3), dtype=np.float32)
#
#     # 计算TP/FP/FN掩码
#     mask_tp = np.logical_and(pred_bin, gt_bin)  # 真阳性
#     mask_fp = np.logical_and(pred_bin, 1 - gt_bin)  # 假阳性
#     mask_fn = np.logical_and(1 - pred_bin, gt_bin)  # 假阴性
#
#     # 定义配色（医学分割常用配色）
#     color_tp = [224 / 255, 68 / 255, 19 / 255]  # 红色
#     color_fp = [80 / 255, 128 / 255, 209 / 255]  # 蓝色
#     color_fn = [51 / 255, 186 / 255, 7 / 255]  # 绿色
#
#     # 应用颜色
#     overlay[mask_fp] = color_fp
#     overlay[mask_fn] = color_fn
#     overlay[mask_tp] = color_tp
#
#     return overlay
#
#
# def visualize_sample(img_id, original_img, anomaly_map, gt_mask, pred_mask, dice_score, best_threshold):
#     """可视化单个样本"""
#     # 数据预处理
#     original_img = normalize_img(original_img)
#     anomaly_map = anomaly_map.squeeze()
#     gt_mask = gt_mask.squeeze()
#     pred_mask = (anomaly_map > best_threshold).astype(np.float32)  # 使用最佳阈值二值化
#
#     # 创建6列可视化图
#     fig, axes = plt.subplots(1, 6, figsize=(30, 5))
#
#     # 1. 原始图像
#     axes[0].imshow(original_img, cmap='gray')
#     axes[0].set_title("Original Image")
#     axes[0].axis('off')
#
#     # 2. 异常热力图
#     im2 = axes[1].imshow(anomaly_map, cmap=config.vis_cmap, vmin=0.0, vmax=1.0)
#     axes[1].set_title(f"Anomaly Heatmap\n(min:{anomaly_map.min():.3f}, max:{anomaly_map.max():.3f})")
#     axes[1].axis('off')
#     plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
#
#     # 3. 差异图
#     diff = np.abs(pred_mask - gt_mask)
#     axes[2].imshow(diff, cmap='gray', vmin=0, vmax=1)
#     axes[2].set_title("Difference Map\n(Grayscale)")
#     axes[2].axis('off')
#
#     # 4. GT掩码
#     axes[3].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
#     axes[3].set_title("GT Mask")
#     axes[3].axis('off')
#
#     # 5. 预测掩码
#     axes[4].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
#     axes[4].set_title(f"Pred Mask\nDice: {dice_score:.4f}")
#     axes[4].axis('off')
#
#     # 6. 检测叠加图
#     detection_overlay = create_detection_overlay(pred_mask, gt_mask)
#     axes[5].imshow(detection_overlay)
#     axes[5].set_title("Detection Overlay\n(TP:Red, FP:Blue, FN:Green)")
#     axes[5].axis('off')
#
#     # 保存综合图
#     save_name = f"{img_id}_dice{dice_score:.4f}.png"
#     save_path = os.path.join(config.vis_miccai_path, save_name)
#     plt.suptitle(f"{img_id} | Dice Score: {dice_score:.4f} | Threshold: {best_threshold:.4f}")
#     plt.savefig(save_path, bbox_inches='tight', dpi=150)
#     plt.close('all')
#
#     # 如果Dice分数高于阈值，保存到高分文件夹
#     if dice_score > config.high_dice_threshold:
#         high_dice_folder = os.path.join(config.vis_high_dice_path, img_id)
#         os.makedirs(high_dice_folder, exist_ok=True)
#
#         # 复制综合图
#         shutil.copy(save_path, os.path.join(high_dice_folder, "overview.png"))
#
#         # 保存单独的图像文件
#         if config.save_single_images:
#             save_single_image(original_img, os.path.join(high_dice_folder, "original.png"), cmap='gray')
#             save_single_image(anomaly_map, os.path.join(high_dice_folder, "heatmap.png"),
#                               cmap=config.vis_cmap, vmin=0.0, vmax=1.0)
#             save_single_image(gt_mask, os.path.join(high_dice_folder, "gt_mask.png"), cmap='gray')
#             save_single_image(pred_mask, os.path.join(high_dice_folder, "pred_mask.png"), cmap='gray')
#             save_single_image(diff, os.path.join(high_dice_folder, "difference_map.png"), cmap='gray')
#             save_single_image(detection_overlay, os.path.join(high_dice_folder, "detection_overlay.png"))
#
#     return save_path
#
#
# def test():
#     """修改后的测试函数：包含可视化功能"""
#     model.eval()
#     gt_labels, gt_masks, pred_scores, pred_maps = [], [], [], []
#     all_sample_ids = []
#     all_original_imgs = []
#
#     # 新增：存储每个异常图像的像素级指标
#     dice_list, hd95_list, assd_list, ppv_list, sensitive_list = [], [], [], [], []
#
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc='Evaluating'):
#             x_dict = {mod: batch['modalities'][mod].to(config.device) for mod in config.modalities}
#             x = torch.stack(list(x_dict.values()), dim=0).mean(dim=0)
#
#             # 获取原始图像（用于可视化）
#             img_path = batch['img_path'][0]
#             original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if img_path else x.squeeze().cpu().numpy()
#             original_img = cv2.resize(original_img, (config.img_size, config.img_size))
#
#             t_features = model.teacher.backbone(x)
#             reconstruction = model.student(t_features[-1], t_features[:-1])
#             anomaly_map = get_anomaly_map(t_features, reconstruction)
#
#             # 收集数据
#             gt_label = batch['label'].item()
#             gt_mask = batch['mask'].squeeze().cpu().numpy()
#             pred_score = anomaly_map.max().item()
#             pred_map = anomaly_map.squeeze().cpu().numpy()
#
#             gt_labels.append(gt_label)
#             gt_masks.append(gt_mask)
#             pred_scores.append(pred_score)
#             pred_maps.append(pred_map)
#             all_sample_ids.append(batch['id'][0])
#             all_original_imgs.append(original_img)
#
#             # 仅对异常图像计算像素级指标并可视化
#             if gt_label == 1:
#                 # 使用像素级最佳阈值二值化
#                 pixel_threshold, _ = calculate_optimal_threshold_and_f1(
#                     np.concatenate([m.flatten() for m in pred_maps]),
#                     np.concatenate([m.flatten() for m in gt_masks])
#                 )
#                 pred_bin = (pred_map > pixel_threshold).astype(np.uint8)
#                 gt_bin = (gt_mask > 0.5).astype(np.uint8)
#
#                 # 计算单张图像的像素级指标
#                 dice = calculate_dsc(pred_bin, gt_bin)
#                 dice_list.append(dice)
#                 hd95_list.append(calculate_hd95(pred_bin, gt_bin))
#                 assd_list.append(calculate_assd(pred_bin, gt_bin))
#                 ppv_list.append(calculate_ppv(pred_bin, gt_bin))
#                 sensitive_list.append(calculate_sensitive(pred_bin, gt_bin))
#
#                 # 可视化当前样本
#                 visualize_sample(batch['id'][0], original_img, pred_map, gt_mask, pred_bin, dice, pixel_threshold)
#
#     # 计算整体指标
#     gt_labels = np.array(gt_labels)
#     pred_scores = np.array(pred_scores)
#     image_bin_labels = (gt_labels > 0).astype(int)
#     img_auroc = roc_auc_score(image_bin_labels, pred_scores)
#     img_ap = average_precision_score(image_bin_labels, pred_scores)
#     _, img_f1 = calculate_optimal_threshold_and_f1(pred_scores, image_bin_labels)
#
#     # 像素级指标
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
#     # 计算FLOPs和Params
#     try:
#         # 构建dummy input
#         dummy_main = torch.randn(1, 2048, 16, 16).to(config.device)
#         dummy_skip1 = torch.randn(1, 64, 256, 256).to(config.device)
#         dummy_skip2 = torch.randn(1, 512, 64, 64).to(config.device)
#         dummy_skip3 = torch.randn(1, 1024, 32, 32).to(config.device)
#
#         # 计算student模型的FLOPs和Params
#         flops, params = profile(
#             model.student,
#             inputs=(dummy_main, [dummy_skip1, dummy_skip2, dummy_skip3]),
#             verbose=False
#         )
#         flops, params = clever_format([flops, params], "%.2f")
#     except Exception as e:
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
#     # 注意：请确保ESC_DRKD_Multimodal类的导入路径正确
#     model = ESC_DRKD_Multimodal(modalities=config.modalities).to(config.device)
#
#     # 加载预训练权重
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
#             model.student.load_state_dict(checkpoint)
#             print_log("权重文件无外层键，直接加载", log)
#
#     except Exception as e:
#         print_log(f"加载权重失败: {str(e)}", log)
#         exit(1)
#
#     # 创建测试数据集和加载器
#     test_dataset = MultimodalDataset(config.data_root, mode='test')
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
#
#     # 执行测试和可视化
#     print_log("Starting evaluation and visualization...", log)
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
#     # 打印可视化结果信息
#     print_log(f"\nVisualization Results:", log)
#     print_log(f'  All samples saved to: {config.vis_miccai_path}', log)
#     print_log(f'  High dice samples (>{config.high_dice_threshold}) saved to: {config.vis_high_dice_path}', log)
#
#     # 关闭日志文件
#     log.close()
#     print("Evaluation and visualization completed!")
#     print(f"Results saved to {os.path.join(config.save_dir, config.log_file)}")
#     print(f"Visualizations saved to {config.visualization_root}")

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
#     data_root = '/home/wuchangwei/dyx/successful/brain_data4'
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
