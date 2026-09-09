import os
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from skip import ESC_DRKD_Multimodal
from utils import AverageMeter, EarlyStop, print_log
import torch.nn.functional as F


# 配置参数
class Config:
    data_root = '../A_data/2d_equal/PSMA'
    modalities = ['ct,pet']
    batch_size = 8
    epochs = 30
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = 'checkpoints/'
    log_file = 'training.log'
    early_stop_patience = 20
    img_size = 256


# 初始化配置
config = Config()
os.makedirs(config.save_dir, exist_ok=True)
log = open(os.path.join(config.save_dir, config.log_file), 'w')


# 数据加载类
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
                img = torch.FloatTensor(img).unsqueeze(0)  # [1, H, W]
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
    if 'label' in batch[0]:  # 测试数据
        return {
            'modalities': {mod: torch.stack([item['modalities'][mod] for item in batch], dim=0) for mod in modalities},
            'label': torch.tensor([item['label'] for item in batch]),
            'mask': torch.stack([item['mask'] for item in batch], dim=0),
            'id': [item['id'] for item in batch]
        }
    else:  # 训练数据
        return {
            'modalities': {mod: torch.stack([item['modalities'][mod] for item in batch], dim=0) for mod in modalities}
        }


# 初始化模型和优化器
model = ESC_DRKD_Multimodal(modalities=config.modalities).to(config.device)
optimizer = optim.Adam(model.student.parameters(), lr=config.lr)

# 数据加载器
train_dataset = MultimodalDataset(config.data_root, mode='train')
test_dataset = MultimodalDataset(config.data_root, mode='test')
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# 早停机制
early_stop = EarlyStop(patience=config.early_stop_patience, save_name=os.path.join(config.save_dir, 'best_model.pt'))


def get_anomaly_map(t_features, s_recon):
    """计算异常图"""
    anomaly_maps = []

    # 使用教师网络的特征和学生网络的重建特征计算差异
    with torch.no_grad():
        # 将学生网络的重建输入教师网络获取特征
        s_recon_features = model.teacher.backbone(s_recon)

        # 计算各层特征差异
        for t_feat, s_feat in zip(t_features[:3], s_recon_features[:3]):  # 比较前3层特征
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode='bilinear')

            # 使用余弦相似度计算差异
            cos_sim = F.cosine_similarity(t_feat, s_feat, dim=1)
            anomaly_map = 1 - cos_sim.unsqueeze(1)
            anomaly_map = F.interpolate(anomaly_map, size=(config.img_size, config.img_size), mode='bilinear')
            anomaly_maps.append(anomaly_map)

    # 平均所有层的异常图
    return torch.mean(torch.cat(anomaly_maps, dim=1), dim=1, keepdim=True)


def train(epoch):
    model.train()
    losses = AverageMeter()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch in pbar:
        # 准备输入数据
        x_dict = {mod: batch['modalities'][mod].to(config.device) for mod in config.modalities}

        # 简单平均融合多模态输入
        x = torch.stack(list(x_dict.values()), dim=0).mean(dim=0)

        # 教师网络提取特征
        with torch.no_grad():
            t_features = model.teacher.backbone(x)

        # 学生网络重建
        selected_features = [t_features[i] for i in [1, 2]]
        reconstruction = model.student(t_features[-1], selected_features)
        # 计算损失
        loss = model.compute_loss({
            'reconstruction': reconstruction,
            't_features': t_features,
            't_recon_features': model.teacher.backbone(reconstruction)
        }, x)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), x.size(0))
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})

    print_log(f'Epoch: {epoch} | Train Loss: {losses.avg:.4f}', log)
    return losses.avg


def test():
    model.eval()
    gt_labels = []
    gt_masks = []
    pred_scores = []
    pred_maps = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for batch in pbar:
            # 准备输入数据
            x_dict = {mod: batch['modalities'][mod].to(config.device) for mod in config.modalities}
            x = torch.stack(list(x_dict.values()), dim=0).mean(dim=0)

            # 教师网络提取特征
            t_features = model.teacher.backbone(x)

            # 学生网络重建
            reconstruction = model.student(t_features[-1], t_features[:-1])

            # 计算异常图
            anomaly_map = get_anomaly_map(t_features, reconstruction)

            # 收集结果
            gt_labels.append(batch['label'].item())
            gt_masks.append(batch['mask'].squeeze().cpu().numpy())
            pred_scores.append(anomaly_map.max().item())
            pred_maps.append(anomaly_map.squeeze().cpu().numpy())

    # 计算指标
    ap = average_precision_score(gt_labels, pred_scores)
    auroc = roc_auc_score(gt_labels, pred_scores)

    # 像素级AUROC
    pixel_auroc = roc_auc_score(
        np.concatenate([m.flatten() for m in gt_masks]),
        np.concatenate([m.flatten() for m in pred_maps])
    )

    return ap, auroc, pixel_auroc


# 主训练循环
best_ap = 0
for epoch in range(1, config.epochs + 1):
    train_loss = train(epoch)

    if epoch % 1 == 0:
        ap, auroc, pixel_auroc = test()
        print_log(f'Test AP: {ap:.4f} | AUROC: {auroc:.4f} | Pixel-AUROC: {pixel_auroc:.4f}', log)

        if ap > best_ap:
            best_ap = ap
            torch.save({
                'student': model.student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_ap': best_ap
            }, os.path.join(config.save_dir, 'best_model.pt'))
            print_log(f'Best model saved with AP: {best_ap:.4f}', log)

        if early_stop(-ap, model.student, log):
            print_log('Early stopping triggered', log)
            break

log.close()