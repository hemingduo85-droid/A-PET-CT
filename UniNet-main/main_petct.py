"""
UniNet 适配 PET-CT 医学图像异常检测
支持三种输入模态:
  pet   -> [PET, PET, PET] 伪 RGB
  ct    -> [CT,  CT,  CT]  伪 RGB
  petct -> [CT,  PET, PET] 伪 RGB  (默认)

训练策略:
  - 无监督 one-class (仅正常样本训练)
  - 30 个 epoch 后在测试集评估一次
  - 统一评估协议: image score=top-1%/max, 报告 Img/Px 的 AUROC/AP/F1
  - 保存最后一轮权重
  - 无需划分验证集

运行示例:
  python main_petct.py --dataset_name PSMA --modality petct --epochs 30 --batch_size 8
  nohup python main_petct.py --dataset_name PSMA --modality pet   --epochs 30 --gpu 1 --save_dir ./saved_results/pet_psma   > pet_psma_new.log 2>&1 &
  nohup python main_petct.py --dataset_name PSMA --modality petct --epochs 30 --gpu 1 --save_dir ./saved_results/petct_psma > petct_psma_new.log 2>&1 &
  nohup python main_petct.py --dataset_name PSMA --modality ct    --epochs 30 --gpu 1 --save_dir ./saved_results/ct_psma    > ct_psma_new.log 2>&1 &
  python main_petct.py --modality petct --load_ckpts  # 仅推理

  nohup python main_petct.py \
  --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
  --dataset_name PSMA \
  --modality petct \
  --epochs 30 \
  --batch_size 8 \
  --gpu 3 \
  --save_dir ./saved_results/petct_psma \
  > petct_psma_new.log 2>&1 &

  nohup python main_petct.py \
  --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50_ganzkoerper/FDG \
  --dataset_name FDG \
  --modality petct \
  --epochs 30 \
  --batch_size 8 \
  --gpu 7 \
  --save_dir ./saved_results/petct_fdg \
  > petct_fdg_new.log 2>&1 &

test
nohup python3 -u main_petct.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG \
  --dataset_name FDG \
  --modality petct \
  --epochs 30 \
  --gpu 6 \
  --save_dir ./saved_results/petct_fdg \
  --ci_bootstraps 500 \
  --ci_pixel_max_samples 0 \
  --pixel_metric_max_samples 0 \
  --load_ckpts \
  > fdg_ci.log 2>&1 &

nohup python3 -u main_petct.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA \
  --dataset_name PSMA \
  --modality petct \
  --epochs 30 \
  --gpu 5 \
  --save_dir ./saved_results/petct_psma \
  --ci_bootstraps 500 \
  --ci_pixel_max_samples 0 \
  --pixel_metric_max_samples 0 \
  --load_ckpts \
  > psma_ci.log 2>&1 &

95ci
nohup python3 -u main_petct.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA \
  --dataset_name PSMA \
  --modality petct \
  --epochs 30 \
  --gpu 5 \
  --save_dir ./saved_results/petct_psma \
  --ci_bootstraps 500 \
  --load_ckpts \
  > psma_npz.log 2>&1 &

nohup python3 -u main_petct.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG \
  --dataset_name FDG \
  --modality petct \
  --epochs 30 \
  --gpu 6 \
  --save_dir ./saved_results/petct_fdg \
  --ci_bootstraps 500 \
  --load_ckpts \
  > fdg_npz.log 2>&1 &
"""
import argparse
import copy
import os
import re


def _set_cuda_visible_devices_from_argv():
    """Set CUDA visibility before importing torch."""
    for idx, arg in enumerate(os.sys.argv):
        if arg == '--gpu' and idx + 1 < len(os.sys.argv):
            os.environ['CUDA_VISIBLE_DEVICES'] = os.sys.argv[idx + 1]
            return
        if arg.startswith('--gpu='):
            os.environ['CUDA_VISIBLE_DEVICES'] = arg.split('=', 1)[1]
            return


_set_cuda_visible_devices_from_argv()

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from UniNet_lib.DFS import DomainRelated_Feature_Selection
from UniNet_lib.de_resnet import de_wide_resnet50_2
from UniNet_lib.model import UniNet
from UniNet_lib.resnet import wide_resnet50_2
from datasets_petct import get_petct_dataloaders
from eval_protocol import (compute_image_metrics, compute_pixel_metrics,
                           compute_patient_metrics,
                           format_image_metrics, format_pixel_metrics,
                           format_metrics)
from utils import get_logger, save_weights, load_weights, setup_seed, to_device


# -----------------------------------------------------------------------
# 数据集路径
# -----------------------------------------------------------------------
DATASET_ROOTS = {
    'psma': '/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA',
    'fdg': '/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG',
}


# -----------------------------------------------------------------------
# 参数解析
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='UniNet PET-CT 异常检测')

    parser.add_argument('--data_root', type=str, default=None,
                        help='数据集根目录 (含 train/ 和 test/ 子目录); 不传则按 dataset_name 自动选择')
    parser.add_argument('--dataset_name', type=str, default='PSMA',
                        help='数据集名称, 用于区分 ckpt/log 路径, 例如 PSMA 或 FDG')
    parser.add_argument('--modality', type=str, default='petct',
                        choices=['pet', 'ct', 'petct'],
                        help='输入模态: pet / ct / petct (默认 petct)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数 (默认 30)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='训练 batch size')
    parser.add_argument('--image_size', type=int, default=256,
                        help='输入图像边长')
    parser.add_argument('--lr_s', type=float, default=5e-3,
                        help='Student / BN / DFS 学习率')
    parser.add_argument('--lr_t', type=float, default=1e-6,
                        help='Target Teacher 学习率')
    parser.add_argument('--T', type=float, default=2.0,
                        help='对比学习温度系数')
    parser.add_argument('--weighted_decision_mechanism', action='store_true',
                        default=True,
                        help='是否使用加权决策机制计算异常分数')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='加权决策机制超参数 alpha')
    parser.add_argument('--beta', type=float, default=0.00003,
                        help='加权决策机制超参数 beta')
    parser.add_argument('--default', type=float, default=0.3,
                        help='加权决策机制默认权重')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader 工作进程数')
    parser.add_argument('--save_dir', type=str, default='./saved_results',
                        help='结果保存根目录')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts',
                        help='权重保存根目录')
    parser.add_argument('--load_ckpts', action='store_true', default=False,
                        help='跳过训练, 直接从 ckpts 加载权重推理')
    parser.add_argument('--ckpt_suffix', type=str, default=None,
                        help='推理时加载的权重后缀; 默认 EPOCH_{epochs:03d}')
    parser.add_argument('--save_heatmaps', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='最终测试时保存 anomaly map 热力图')
    parser.add_argument('--image_agg', type=str, default='top1pct',
                        choices=['top1pct', 'max'],
                        help='image-level 分数聚合方式: top1pct (top-1%% 均值) 或 max')
    parser.add_argument('--ci_bootstraps', type=int, default=500,
                        help='AUPR/F1 与 pixel-level CI 的 bootstrap 迭代次数')
    parser.add_argument('--ci_pixel_max_samples', type=int, default=200000,
                        help='兼容旧参数; 当前协议下 pixel CI 使用 histogram full-slice bootstrap')
    parser.add_argument('--pixel_metric_max_samples', type=int, default=0,
                        help='兼容旧参数; 当前协议下 pixel 点估计始终全量精确计算')
    parser.add_argument('--hist_bins', type=int, default=16384,
                        help='pixel histogram bootstrap 的 score bin 数')
    parser.add_argument('--ci_seed', type=int, default=1203,
                        help='CI bootstrap 随机种子')
    parser.add_argument('--gpu', type=str, default='0',
                        help='使用的 GPU 编号, 例如 "0" 或 "0,1"')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 便于其他函数直接使用
    args.domain   = 'medical'
    args.setting  = 'oc'
    dataset_tag = args.dataset_name.strip().replace(' ', '_').upper()
    args.dataset_name = dataset_tag
    if args.data_root is None:
        if dataset_tag.lower() not in DATASET_ROOTS:
            parser.error(f'未知 dataset_name={args.dataset_name}; 请显式传入 --data_root')
        args.data_root = DATASET_ROOTS[dataset_tag.lower()]
    args.dataset  = f'PETCT_{dataset_tag}_{args.modality.upper()}'
    args._class_  = args.dataset   # UniNet 内部用于路径与 loss 调参

    return args


def final_ckpt_suffix(c):
    return c.ckpt_suffix or f'EPOCH_{c.epochs:03d}'


def describe_device(device, requested_gpu):
    if device != 'cuda':
        return '使用设备: cpu'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', requested_gpu)
    current_idx = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_idx)
    return (f'使用设备: cuda:{current_idx} (逻辑编号) | '
            f'CUDA_VISIBLE_DEVICES={visible} | GPU={device_name}')


def _safe_heatmap_name(path, index):
    base = os.path.splitext(os.path.basename(str(path)))[0]
    patient = os.path.basename(os.path.dirname(os.path.dirname(str(path))))
    name = f'{index:05d}_{patient}_{base}.png'
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def save_heatmaps(anomaly_map, paths, heatmap_dir):
    os.makedirs(heatmap_dir, exist_ok=True)
    for idx, (single_map, path) in enumerate(zip(anomaly_map, paths)):
        save_path = os.path.join(heatmap_dir, _safe_heatmap_name(path, idx))
        plt.imsave(save_path, single_map, cmap='jet')


# -----------------------------------------------------------------------
# 评估函数 (统一协议)
# -----------------------------------------------------------------------

def evaluate(model, test_loader, device, c, heatmap_dir=None, logger=None):
    """
    收集 pixel-level anomaly map, 按统一协议计算所有 slice-level 和 patient-level 指标。
    """
    model.train_or_eval(type='eval')
    gt_labels, gt_masks, paths, anomaly_maps = [], [], [], []

    with torch.no_grad():
        for batch_idx, (img, label, mask, path) in enumerate(test_loader, start=1):
            img = img.to(device)
            t_tf, de_features = model(img)

            gt_labels.extend(label.numpy().tolist())
            gt_masks.append(mask.numpy())           # (1, 1, H, W)
            # DataLoader 对字符串会打包成 tuple/list，batch_size=1 时是 (str,)
            paths.extend(path if isinstance(path, (list, tuple)) else [path])

            batch_map = None
            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                layer_map = 1 - F.cosine_similarity(t, s)
                layer_map = F.interpolate(
                    layer_map.unsqueeze(1),
                    size=c.image_size,
                    mode='bilinear',
                    align_corners=True,
                )[:, 0, :, :]
                batch_map = layer_map if batch_map is None else batch_map + layer_map
            batch_map = batch_map.detach().cpu().numpy()
            anomaly_maps.extend([
                gaussian_filter(batch_map[i], sigma=4)
                for i in range(batch_map.shape[0])
            ])
            if logger is not None and (batch_idx == 1 or batch_idx % 100 == 0):
                logger.info(f'评估 forward 进度: {batch_idx}/{len(test_loader)}')

    if logger is not None:
        logger.info('anomaly map 已逐 batch 合成完成')
    anomaly_map = np.stack(anomaly_maps, axis=0)

    gt_labels   = np.array(gt_labels, dtype=np.int32)
    gt_masks_np = np.squeeze(np.concatenate(gt_masks, axis=0), axis=1)  # (N, H, W)
    if logger is not None:
        logger.info(
            f"Eval arrays ready: labels={gt_labels.shape}, "
            f"masks={gt_masks_np.shape}, maps={anomaly_map.shape}, "
            f"bootstrap_iters={c.ci_bootstraps}"
        )

    # ---- Image-level 指标：先输出 ----
    if logger is not None:
        logger.info('Computing image-level metrics and 95CI...')
    slice_metrics = compute_image_metrics(
        gt_labels=gt_labels,
        anomaly_maps=anomaly_map,
        image_agg=c.image_agg,
        n_bootstraps=c.ci_bootstraps,
        seed=c.ci_seed,
    )
    if logger is not None:
        logger.info(format_image_metrics(slice_metrics))

    # ---- Pixel-level 指标 ----
    if logger is not None:
        logger.info('Computing pixel-level Slice-Px(abn) metrics and 95CI...')
        logger.info('Computing exact full-pixel point estimates with sklearn...')
    slice_metrics.update(compute_pixel_metrics(
        gt_labels=gt_labels,
        gt_masks=gt_masks_np,
        anomaly_maps=anomaly_map,
        n_bootstraps=c.ci_bootstraps,
        seed=c.ci_seed,
        hist_bins=c.hist_bins,
        progress_callback=logger.info if logger is not None else None,
    ))
    if logger is not None:
        logger.info(format_pixel_metrics(slice_metrics))

    # ---- Patient-level 指标 ----
    if logger is not None:
        logger.info('Computing patient-level metrics and 95CI...')
    pat_metrics = compute_patient_metrics(
        paths=paths,
        pr_sp=slice_metrics['_pr_sp'],
        gt_labels=gt_labels,
        patient_agg='max',
        n_bootstraps=c.ci_bootstraps,
        seed=c.ci_seed,
    )

    if heatmap_dir is not None:
        if logger is not None:
            logger.info(f'开始保存热力图: {heatmap_dir}')
        save_heatmaps(anomaly_map, paths, heatmap_dir)

    return slice_metrics, pat_metrics


# -----------------------------------------------------------------------
# 构建模型
# -----------------------------------------------------------------------

def build_model(c, device):
    Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
    Source_teacher.layer4 = None
    Source_teacher.fc     = None
    student = de_wide_resnet50_2(pretrained=False)
    dfs     = DomainRelated_Feature_Selection()

    [Source_teacher, bn, student, dfs] = to_device(
        [Source_teacher, bn, student, dfs], device)
    Target_teacher = copy.deepcopy(Source_teacher)

    model = UniNet(c, Source_teacher, Target_teacher, bn, student, DFS=dfs)
    return model, bn, student, dfs, Target_teacher


# -----------------------------------------------------------------------
# 训练
# -----------------------------------------------------------------------

def train(c, model, train_loader, test_loader,
          bn, student, dfs, Target_teacher,
          ckpt_path, device, logger):

    params = (list(student.parameters()) +
              list(bn.parameters()) +
              list(dfs.parameters()))
    optimizer  = torch.optim.AdamW(params,
                                   lr=c.lr_s, betas=(0.9, 0.999),
                                   weight_decay=1e-5)
    optimizer1 = torch.optim.AdamW(list(Target_teacher.parameters()),
                                   lr=c.lr_t, betas=(0.9, 0.999),
                                   weight_decay=1e-5)

    modules_list = [model.t.t_t, model.bn.bn, model.s.s1, dfs]

    logger.info(f'开始训练 | modality={c.modality} | epochs={c.epochs} | '
                f'batch_size={c.batch_size} | image_size={c.image_size} | '
                f'image_agg={c.image_agg}')
    logger.info(f'训练集样本数: {len(train_loader.dataset)} | '
                f'测试集样本数: {len(test_loader.dataset)}')
    logger.info('评估协议: Img/Px AUROC+AP 用原始分数; '
                'Img-F1 用原始 image score; Px-F1 用测试集全局 min-max; '
                '训练完成后统一测试一次')

    for epoch in range(1, c.epochs + 1):
        # ---- 训练阶段 ----
        model.train_or_eval(type='train')
        loss_list = []

        for img, _label, _mask, _ in train_loader:
            img = img.to(device)
            # stop_gradient=True: 医学域常用, 防止 target teacher 梯度传导导致崩塌
            loss = model(img, stop_gradient=True)
            optimizer.zero_grad()
            optimizer1.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer1.step()
            loss_list.append(loss.item())

        mean_loss = np.mean(loss_list)
        header = f'Epoch [{epoch:02d}/{c.epochs}] loss={mean_loss:.4f}'
        logger.info(header)

    suffix = final_ckpt_suffix(c)
    save_weights(modules_list, ckpt_path, suffix, device=device)
    logger.info(f'训练结束 | 已保存最后一轮权重: {suffix}.pth')

    heatmap_dir = None
    if c.save_heatmaps:
        heatmap_dir = os.path.join(c.save_dir, c.dataset, 'heatmaps', suffix)
    logger.info(f'开始评估 | ci_bootstraps={c.ci_bootstraps} | '
                f'hist_bins={c.hist_bins}')
    slice_metrics, pat_metrics = evaluate(
        model, test_loader, device, c, heatmap_dir=heatmap_dir, logger=logger)
    logger.info('\n--- 最后一轮权重最终评估 ---')
    logger.info(format_metrics(slice_metrics, pat_metrics))
    if heatmap_dir is not None:
        logger.info(f'热力图已保存到: {heatmap_dir}')
    return slice_metrics, pat_metrics


# -----------------------------------------------------------------------
# 仅推理 (load_ckpts 模式)
# -----------------------------------------------------------------------

def test_only(c, model, test_loader, ckpt_path, device, logger):
    dfs = model.dfs
    modules = [model.t.t_t, model.bn.bn, model.s.s1, dfs]
    suffix = final_ckpt_suffix(c)
    new_state = load_weights(modules, ckpt_path, suffix, device=device)
    logger.info(f'权重加载完成: {suffix}.pth')
    model.t.t_t  = new_state['tt']
    model.bn.bn  = new_state['bn']
    model.s.s1   = new_state['st']
    model.dfs    = new_state['dfs']

    heatmap_dir = None
    if c.save_heatmaps:
        heatmap_dir = os.path.join(c.save_dir, c.dataset, 'heatmaps', suffix)
    logger.info(f'开始评估 | ci_bootstraps={c.ci_bootstraps} | '
                f'hist_bins={c.hist_bins}')
    slice_metrics, pat_metrics = evaluate(
        model, test_loader, device, c, heatmap_dir=heatmap_dir, logger=logger)
    logger.info('[推理模式]')
    logger.info(format_metrics(slice_metrics, pat_metrics))
    if heatmap_dir is not None:
        logger.info(f'热力图已保存到: {heatmap_dir}')
    return slice_metrics, pat_metrics


# -----------------------------------------------------------------------
# 主入口
# -----------------------------------------------------------------------

def main():
    setup_seed(1203)
    c = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(describe_device(device, c.gpu))

    ckpt_path = os.path.join(c.ckpt_dir, c.dataset)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(c.save_dir, exist_ok=True)

    logger = get_logger(c.dataset, os.path.join(c.save_dir, c.dataset))

    # ---- 数据 ----
    train_loader, test_loader = get_petct_dataloaders(
        data_root=c.data_root,
        modality=c.modality,
        image_size=c.image_size,
        batch_size=c.batch_size,
        num_workers=c.num_workers,
    )

    # ---- 模型 ----
    model, bn, student, dfs, Target_teacher = build_model(c, device)

    # ---- 训练 or 推理 ----
    if c.load_ckpts:
        test_only(c, model, test_loader, ckpt_path, device, logger)
    else:
        train(c, model, train_loader, test_loader,
              bn, student, dfs, Target_teacher,
              ckpt_path, device, logger)


if __name__ == '__main__':
    main()
