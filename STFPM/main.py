import os
import argparse
import cv2
import numpy as np
import re

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Torchvision imports
import torchvision.models as models
try:
    from torchvision.models import ResNet18_Weights
except ImportError:
    ResNet18_Weights = None
from torchvision import transforms
# Other imports
from evaluate import evaluate
from pic import save_heatmap
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image

"""
Examples:
python main.py train --dataset fdg --modalities ct pet --cuda 0 --batch-size 32
python main.py test --dataset psma --modalities pet --cuda 1

导出npz

nohup python3 -u main.py test \
  --dataset psma \
  --modalities ct pet \
  --cuda 5 \
  --checkpoint snapshots/psma_ct+pet_pseudo_rgb_epoch30.pth.tar \
  --test-batch-size 8 \
  --num-workers 4 \
  --cache-path results/psma_ct+pet_pseudo_rgb_eval_cache.npz \
  --export-only \
  > stfpm_fdg_ct_pet_export_npz.log 2>&1 &

nohup python3 -u main.py test \
  --dataset fdg \
  --modalities ct pet \
  --cuda 5 \
  --checkpoint snapshots/fdg_ct+pet_pseudo_rgb_epoch30.pth.tar \
  --test-batch-size 8 \
  --num-workers 4 \
  --cache-path results/fdg_ct+pet_pseudo_rgb_eval_cache.npz \
  --export-only \
  > stfpm_fdg_ct_pet_export_npz.log 2>&1 &

计算95
nohup python3 -u compute_cache_metrics.py \
  --cache results/psma_ct+pet_pseudo_rgb_eval_cache.npz \
  --output results/psma_ct+pet_pseudo_rgb_eval_metrics.txt \
  --bootstrap_iters 500 \
  > psma_npz.log 2>&1 &

nohup python3 -u compute_cache_metrics.py \
  --cache results/fdg_ct+pet_pseudo_rgb_eval_cache.npz \
  --output results/fdg_ct+pet_pseudo_rgb_eval_metrics.txt \
  --bootstrap_iters 500 \
  > fdg_npz.log 2>&1 &
"""
DATASET_ROOTS = {
    'fdg': '/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG',
    'psma': '/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA',
}
PIXEL_CI_HIST_BINS = 16384


def gray_to_rgb_tensor(img_l, transform):
    """灰度图复制为 3 通道（R=G=B），以适配 ImageNet 预训练的 3 通道 ResNet。"""
    img_rgb = img_l.convert('RGB')
    return transform(img_rgb)


class SingleModalDataset(object):
    """
    单模态：每个样本仅一张灰度图，复制为 3 通道后送入网络。
    image_list: List[path]
    """
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx]
        img = Image.open(path).convert('L')
        img = gray_to_rgb_tensor(img, self.transform)
        return path, img


class MultiModalDataset(object):
    """
    groups: List[List[path_mod1, path_mod2, ...]]
    每个 group 代表同一位置的多模态图片
    """
    def __init__(self, groups, transform_single=None):
        self.groups = groups
        self.transform_single = transform_single
        if self.transform_single is None:
            from torchvision import transforms
            self.transform_single = transforms.ToTensor()

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        paths = self.groups[idx]
        imgs = []
        for p in paths:
            img = Image.open(p).convert('L')
            img = gray_to_rgb_tensor(img, self.transform_single)
            imgs.append(img)
        # 按通道拼接 (C_total = 3 * num_modalities)
        img = torch.cat(imgs, dim=0)
        # 返回代表路径（第一模态即可）
        return paths[0], img


class PseudoRGBDataset(object):
    """
    PET/CT 双模态原图层面的 3 通道输入: [CT, PET, PET]。
    groups 中必须能找到 ct 和 pet 两个模态路径。
    """
    def __init__(self, groups, modalities, transform_size=(256, 256)):
        self.groups = groups
        self.modalities = [m.lower() for m in modalities]
        self.transform_size = transform_size
        if 'ct' not in self.modalities or 'pet' not in self.modalities:
            raise ValueError("pseudo_rgb 输入需要同时包含 ct 和 pet: --modalities ct pet")

    def __len__(self):
        return len(self.groups)

    def _load_gray(self, path):
        img = Image.open(path).convert('L').resize(self.transform_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        paths = self.groups[idx]
        path_by_modality = {m: p for m, p in zip(self.modalities, paths)}
        ct = self._load_gray(path_by_modality['ct'])
        pet = self._load_gray(path_by_modality['pet'])
        img = torch.stack([ct, pet, pet], dim=0)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype).view(3, 1, 1)
        return path_by_modality['ct'], (img - mean) / std


class ResNet18_MS3(nn.Module):

    def __init__(self, pretrained=False, in_channels=3, weights_path=None):
        super(ResNet18_MS3, self).__init__()     
        if pretrained and weights_path is None:
            if ResNet18_Weights is None:
                net = models.resnet18(pretrained=True)
            else:
                net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            if ResNet18_Weights is None:
                net = models.resnet18(pretrained=False)
            else:
                net = models.resnet18(weights=None)
        if weights_path is not None:
            weight_obj = torch.load(weights_path, map_location='cpu')
            if isinstance(weight_obj, dict) and 'state_dict' in weight_obj:
                weight_obj = weight_obj['state_dict']
            weight_obj = {k.replace('module.', ''): v for k, v in weight_obj.items()}
            net.load_state_dict(weight_obj, strict=False)
        if in_channels != 3:
            orig_conv = net.conv1
            new_conv = nn.Conv2d(in_channels,
                                 orig_conv.out_channels,
                                 kernel_size=orig_conv.kernel_size,
                                 stride=orig_conv.stride,
                                 padding=orig_conv.padding,
                                 bias=(orig_conv.bias is not None))
            if pretrained:
                with torch.no_grad():
                    w = orig_conv.weight  # (64,3,7,7)
                    if in_channels > 3:
                        # 循环复制
                        rep = []
                        for c in range(in_channels):
                            rep.append(w[:, c % 3])
                        new_w = torch.stack(rep, dim=1)  # (64,in_channels,7,7)
                    else:
                        new_w = w[:, :in_channels]
                    new_conv.weight.copy_(new_w)
                    if orig_conv.bias is not None:
                        new_conv.bias.copy_(orig_conv.bias)
            net.conv1 = new_conv
        # ignore the last block and fc
        self.model = torch.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        res = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in ['4', '5', '6']:
                res.append(x)
        return res



def build_single_modal_paths(root_dir, split_type, modality, subset=None):
    """
    单模态：直接收集某一模态下所有 png，无需 pet/ct 文件名对齐。
    目录: {root}/train/normal/{patient}/{modality}/*.png
          {root}/test/{subset}/{patient}/{modality}/*.png
    """
    if split_type == 'train':
        path_pattern = os.path.join(root_dir, 'train', 'normal', '*', modality, '*.png')
    else:
        path_pattern = os.path.join(root_dir, 'test', subset, '*', modality, '*.png')
    files = sorted(glob(path_pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"模态 {modality} 无任何图像: {path_pattern}")
    return files


def multimodal_sample_key(path, modality):
    patient = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return os.path.join(patient, os.path.basename(path))


def build_multimodal_groups(root_dir, split_type, modalities, subset=None, strict=True):
    """
    新增:
      strict=False 时自动取各模态文件名交集构建样本，并报告缺失情况。
      适配病人文件夹结构: {root_dir}/{split_type}/{subset}/{patient_folder}/{modality}/*.png
    """
    groups = []
    modality_file_lists = []
    if split_type == 'train':
        # 训练集: {root_dir}/train/normal/{patient_folder}/{modality}/*.png
        for m in modalities:
            path_pattern = os.path.join(root_dir, 'train', 'normal', '*', m, '*.png')
            files = sorted(glob(path_pattern))
            if len(files) == 0:
                raise FileNotFoundError(f"模态 {m} 无任何图像: {path_pattern}")
            modality_file_lists.append(files)
    else:
        # 测试集: {root_dir}/test/{subset}/{patient_folder}/{modality}/*.png
        for m in modalities:
            path_pattern = os.path.join(root_dir, 'test', subset, '*', m, '*.png')
            files = sorted(glob(path_pattern))
            if len(files) == 0:
                raise FileNotFoundError(f"模态 {m} 在测试集 {subset} 下无任何图像: {path_pattern}")
            modality_file_lists.append(files)

    if len(modality_file_lists) == 0:
        return groups

    # 构建 basename -> path 映射
    maps = []
    for modality, files in zip(modalities, modality_file_lists):
        sample_map = {}
        for p in files:
            key = multimodal_sample_key(p, modality)
            if key in sample_map:
                raise ValueError(f"多模态样本 key 重复: {key} ({sample_map[key]} vs {p})")
            sample_map[key] = p
        maps.append(sample_map)

    # 基准
    all_key_sets = [set(m.keys()) for m in maps]
    if strict:
        # 原有严格模式
        ref_keys = list(all_key_sets[0])
        for ks in all_key_sets[1:]:
            if ks != set(ref_keys):
                raise ValueError("严格模式：多模态文件名不匹配，请确保完全一致或关闭 --strict-modal-match。")
        common_keys = sorted(ref_keys)
    else:
        # 非严格：取交集
        common_keys = sorted(set.intersection(*all_key_sets))
        if len(common_keys) == 0:
            raise ValueError("多模态对齐后交集为空，请检查文件命名。")
        # 报告缺失
        print("[多模态对齐] 公共样本数:", len(common_keys))
        for m_name, ks in zip(modalities, all_key_sets):
            miss = len(ks) - len(common_keys)
            if miss != 0:
                print(f"  模态 {m_name}: 原始 {len(ks)}，可用 {len(common_keys)}，丢弃 {miss}")

    for k in common_keys:
        paths = []
        ok = True
        for mp in maps:
            if k not in mp:
                ok = False
                break
            paths.append(mp[k])
        if ok:
            groups.append(paths)
    return groups


def resolve_dataset_path(args):
    dataset_key = args.dataset.lower()
    if args.dataset_path:
        return args.dataset_path, dataset_key
    if dataset_key not in DATASET_ROOTS:
        raise ValueError(f"未知数据集: {args.dataset}. 可选: {sorted(DATASET_ROOTS)}")
    return DATASET_ROOTS[dataset_key], dataset_key


def checkpoint_stem(dataset_key, modalities, input_mode):
    return f"{dataset_key}_{'+'.join(modalities)}_{input_mode}"


def checkpoint_path(args, modalities_str):
    stem = checkpoint_stem(args.dataset_key, args.modalities, args.input_mode)
    return os.path.join(args.model_save_path, f"{stem}_epoch{args.epochs}.pth.tar")


def make_loader(dataset, batch_size, shuffle, args):
    pin_memory = args.pin_memory and str(args.cuda).lower() not in ('', 'none', 'cpu', '-1')
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )


def resolve_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint
    stem = checkpoint_stem(args.dataset_key, args.modalities, args.input_mode)
    path = os.path.join(args.model_save_path, f"{stem}_epoch{args.epochs}.pth.tar")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未指定 --checkpoint，且自动推断的权重不存在: {path}\n"
            f"请确认 --dataset/--modalities/--input-mode/--epochs 与训练一致，或显式传 --checkpoint。"
        )
    return path


def default_cache_path(args):
    stem = checkpoint_stem(args.dataset_key, args.modalities, args.input_mode)
    return os.path.join(args.result_path, f"{stem}_eval_cache.npz")


def save_eval_cache(path, gt_labels, gt_masks, anomaly_maps, image_scores, paths):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        labels=np.asarray(gt_labels, dtype=np.int32),
        masks=np.asarray(gt_masks, dtype=np.uint8),
        maps=np.asarray(anomaly_maps, dtype=np.float32),
        image_scores=np.asarray(image_scores, dtype=np.float64),
        paths=np.asarray(paths, dtype=str),
    )
    print(f"Saved eval cache: {path}", flush=True)


def select_device(args):
    if args.device:
        return torch.device(args.device)
    if args.cuda is None or str(args.cuda).lower() in ('', 'none', 'cpu', '-1'):
        return torch.device('cpu')
    if not torch.cuda.is_available():
        print("CUDA 不可用，自动使用 CPU。")
        return torch.device('cpu')
    return torch.device(f"cuda:{args.cuda}")


def get_dataset_paths(dataset):
    if hasattr(dataset, 'image_list'):
        return dataset.image_list
    if hasattr(dataset, 'groups'):
        if isinstance(dataset, PseudoRGBDataset):
            ct_index = dataset.modalities.index('ct')
            return [g[ct_index] for g in dataset.groups]
        return [g[0] for g in dataset.groups]
    return []


def load_positive_masks(paths):
    gt = []
    for img_path in paths:
        base = os.path.basename(img_path)
        patient_dir = os.path.dirname(os.path.dirname(img_path))
        mask_path = os.path.join(patient_dir, 'label', base)
        temp = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if temp is None:
            raise FileNotFoundError(f"未找到mask: {mask_path}")
        temp = cv2.resize(temp, (256, 256)).astype(bool)
        gt.append(temp)
    return np.stack(gt, 0)


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_aupr(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def f1_score_max(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.0
    p, r, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / (p + r + 1e-7)
    f1 = f1[:-1]
    return float(f1.max()) if len(f1) else 0.0


def patient_id_from_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


def patient_level_arrays(paths, image_scores, gt_labels):
    patient_scores = {}
    patient_labels = {}
    for path, score, label_value in zip(paths, image_scores, gt_labels):
        pid = patient_id_from_path(path)
        patient_scores.setdefault(pid, []).append(float(score))
        patient_labels.setdefault(pid, []).append(int(label_value))
    pat_score = np.asarray([np.max(patient_scores[pid]) for pid in patient_scores], dtype=np.float64)
    pat_label = np.asarray([np.max(patient_labels[pid]) for pid in patient_scores], dtype=np.int32)
    return pat_label, pat_score


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=500, alpha=0.95, seed=0):
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    point = float(metric_fn(y_true, y_score))
    if len(np.unique(y_true)) < 2 or len(y_true) < 2 or n_boot <= 0:
        return point, (point, point)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(float(metric_fn(y_true[idx], y_score[idx])))
    if not values:
        return point, (point, point)
    low = (1.0 - alpha) / 2.0 * 100.0
    high = (1.0 + alpha) / 2.0 * 100.0
    ci = np.percentile(np.asarray(values), [low, high])
    return point, (float(ci[0]), float(ci[1]))


def metrics_from_hist(pos_hist, neg_hist):
    tp = pos_hist[::-1].astype(np.float64, copy=False)
    fp = neg_hist[::-1].astype(np.float64, copy=False)
    total_pos = float(tp.sum())
    total_neg = float(fp.sum())
    if total_pos <= 0 or total_neg <= 0:
        return 0.0, 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    tpr = cum_tp / total_pos
    fpr = cum_fp / total_neg
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auroc = float(trapz(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)
    aupr = float(np.sum(precision * (tp / total_pos)))
    return auroc, aupr


def pixel_slice_bootstrap_metrics(
    gt_labels,
    gt_masks,
    maps,
    n_boot=500,
    alpha=0.95,
    seed=0,
    bins=PIXEL_CI_HIST_BINS,
    progress=False,
):
    keep = np.asarray(gt_labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        return (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0)), 0

    masks = np.asarray(gt_masks)
    scores = np.asarray(maps)
    y_true = masks[keep].reshape(-1).astype(np.int32)
    y_score = scores[keep].reshape(-1).astype(np.float64)
    point_auroc = _safe_auroc(y_true, y_score)
    point_aupr = _safe_aupr(y_true, y_score)
    if n_boot <= 0:
        return (point_auroc, (point_auroc, point_auroc)), (point_aupr, (point_aupr, point_aupr)), len(y_true)

    score_min = float(np.min(scores[keep]))
    score_max = float(np.max(scores[keep]))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    bins = max(16, int(bins))
    scale = (bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        slice_scores = scores[idx].reshape(-1)
        slice_mask = masks[idx].reshape(-1).astype(bool)
        bin_idx = np.floor((slice_scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[slice_mask], minlength=bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~slice_mask], minlength=bins).astype(np.uint32)

    rng = np.random.default_rng(seed)
    aurocs = []
    auprs = []
    n = len(abn_idx)
    for boot_idx in range(int(n_boot)):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        auroc, aupr = metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if progress and ((boot_idx + 1) % 50 == 0 or boot_idx + 1 == int(n_boot)):
            print(f"Pixel histogram slice bootstrap: {boot_idx + 1}/{n_boot}", flush=True)
    if not aurocs:
        return (point_auroc, (point_auroc, point_auroc)), (point_aupr, (point_aupr, point_aupr)), len(y_true)
    low = (1.0 - alpha) / 2.0 * 100.0
    high = (1.0 + alpha) / 2.0 * 100.0
    auroc_ci = np.percentile(np.asarray(aurocs), [low, high])
    aupr_ci = np.percentile(np.asarray(auprs), [low, high])
    return (
        (point_auroc, (float(auroc_ci[0]), float(auroc_ci[1]))),
        (point_aupr, (float(aupr_ci[0]), float(aupr_ci[1]))),
        len(y_true),
    )


def binary_metric_bundle(y_true, y_score, n_boot=500, alpha=0.95, seed=0, include_f1=True):
    auroc, auroc_ci = bootstrap_ci(y_true, y_score, _safe_auroc, n_boot=n_boot, alpha=alpha, seed=seed)
    aupr, aupr_ci = bootstrap_ci(y_true, y_score, _safe_aupr, n_boot=n_boot, alpha=alpha, seed=seed + 1)
    metrics = {
        "auroc": (auroc, auroc_ci),
        "aupr": (aupr, aupr_ci),
    }
    if include_f1:
        f1, f1_ci = bootstrap_ci(y_true, y_score, f1_score_max, n_boot=n_boot, alpha=alpha, seed=seed + 17)
        metrics["f1"] = (f1, f1_ci)
    return metrics


def pixel_metric_bundle(gt_labels, gt_masks, maps, args):
    px_auroc, px_aupr, pixel_n = pixel_slice_bootstrap_metrics(
        gt_labels,
        gt_masks,
        maps,
        n_boot=args.ci_bootstrap,
        alpha=args.ci_alpha,
        seed=args.ci_seed + 303,
        progress=True,
    )
    return {
        "metrics": {
            "auroc": px_auroc,
            "aupr": px_aupr,
        },
        "n": pixel_n,
        "sampled": False,
    }


def compute_all_metrics(gt_labels, gt_masks, anomaly_maps, image_scores, paths, args, compute_pixel=True):
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    image_scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)

    image_metrics = binary_metric_bundle(
        gt_labels, image_scores, args.ci_bootstrap, args.ci_alpha, args.ci_seed, include_f1=True
    )
    pat_label, pat_score = patient_level_arrays(paths, image_scores, gt_labels)
    patient_metrics = binary_metric_bundle(
        pat_label, pat_score, args.ci_bootstrap, args.ci_alpha, args.ci_seed + 101, include_f1=True
    )

    pixel_metrics = None
    pixel_sampled = False
    pixel_n = 0
    if compute_pixel:
        pixel_result = pixel_metric_bundle(gt_labels, gt_masks, maps, args)
        pixel_metrics = pixel_result["metrics"]
        pixel_sampled = pixel_result["sampled"]
        pixel_n = pixel_result["n"]

    return {
        "image": image_metrics,
        "pixel": pixel_metrics,
        "pixel_sampled": pixel_sampled,
        "pixel_n": pixel_n,
        "patient": patient_metrics,
        "n_patients": len(pat_label),
        "n_abn_patients": int(pat_label.sum()) if len(pat_label) else 0,
    }


def format_metric(value_ci):
    value, ci = value_ci
    return f"{value*100:.2f}% (95% CI {ci[0]*100:.2f}-{ci[1]*100:.2f}%)"


def print_metric_block(title, metrics, extra=None):
    if metrics is None:
        print(f"[{title}] skipped")
        return
    suffix = f"  {extra}" if extra else ""
    parts = [
        f"AUROC={format_metric(metrics['auroc'])}",
        f"AUPR={format_metric(metrics['aupr'])}",
    ]
    if "f1" in metrics:
        parts.append(f"F1={format_metric(metrics['f1'])}")
    print(f"[{title}]{suffix}  " + "  ".join(parts), flush=True)


def print_all_metrics(metrics):
    print_metric_block("Slice-Img", metrics["image"])
    if metrics["pixel"] is None:
        print_metric_block("Slice-Px(abn)", None)
    else:
        print_metric_block("Slice-Px(abn)", metrics["pixel"])
    patient_title = f"Patient({metrics['n_abn_patients']}/{metrics['n_patients']}abn)"
    print_metric_block(patient_title, metrics["patient"])


def save_heatmaps_for_results(scores, paths, args):
    if args.save_all_heatmaps:
        count = len(paths)
    else:
        count = max(0, int(args.heatmap_count))
    if count == 0:
        return
    count = min(count, len(paths))
    save_name = checkpoint_stem(args.dataset_key, args.modalities, args.input_mode)
    for idx in range(count):
        anomaly_map = cv2.resize(scores[idx], (256, 256))
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        save_heatmap(
            result={
                'anomaly_map': anomaly_map,
                'image_path': paths[idx],
                'combination_id': idx + 1,
            },
            save_dir=args.heatmap_dir,
            save_name=save_name,
            threshold=None,
            method='auto'
        )

def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument("split", nargs="?", choices=["train", "test"])
    # required training super-parameters
    parser.add_argument("--checkpoint", type=str, default=None, help="student checkpoint")
    parser.add_argument("--epochs", type=int, default=30, help='number of epochs')
    parser.add_argument("--batch-size", type=int, default=32, help='batch size')
    parser.add_argument("--test-batch-size", type=int, default=32, help='test batch size')
    # trivial parameters
    parser.add_argument("--result-path", type=str, default='results', help="save results")
    parser.add_argument("--dataset", type=str, default='fdg', choices=sorted(DATASET_ROOTS), help="dataset config")
    parser.add_argument("--dataset-path", type=str, default=None, help="Override dataset root path")
    parser.add_argument('--model-save-path', type=str, default='snapshots', help='path where student models are saved')
    parser.add_argument("--cuda", type=str, default='0', help="CUDA id, e.g. 0/1/2. Use -1 or cpu for CPU.")
    parser.add_argument("--device", type=str, default=None, help="Override full torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--input-mode", choices=['auto', 'pseudo_rgb', 'concat'], default='auto',
                        help="auto: ct+pet uses [CT,PET,PET], other multimodal uses channel concat")
    parser.add_argument("--no-test-after-train", action='store_true', help="Only train/save epoch N; do not run final test")
    parser.add_argument("--heatmap-count", type=int, default=0, help="Number of heatmaps to save after testing; default 0")
    parser.add_argument("--save-all-heatmaps", action='store_true', help="Save heatmaps for all test images")
    parser.add_argument("--heatmap-dir", type=str, default='heatmaps', help="Heatmap output directory")
    parser.add_argument("--cache-path", type=str, default=None,
                        help="Optional .npz path to save labels, masks, maps, image_scores, and paths")
    parser.add_argument("--export-only", action='store_true',
                        help="Only export eval cache; skip metric and CI computation")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--pin-memory", action='store_true', default=True, help="Use pinned CPU memory for CUDA loading")
    parser.add_argument("--no-pin-memory", action='store_false', dest='pin_memory', help="Disable pinned memory")
    parser.add_argument("--log-interval", type=int, default=50, help="Print train loss every N batches")
    parser.add_argument("--test-log-interval", type=int, default=50, help="Print test progress every N batches")
    parser.add_argument("--compute-pro", action='store_true', help="Compute slow AUPRO metric during test")
    parser.add_argument("--skip-pixel-metrics", action='store_true', help="Skip pixel-level metrics for quick test")
    parser.add_argument("--ci-bootstrap", type=int, default=500, help="Bootstrap iterations for 95CI")
    parser.add_argument("--ci-alpha", type=float, default=0.95, help="Confidence level")
    parser.add_argument("--ci-seed", type=int, default=0, help="Random seed for CI bootstrap/sampling")
    parser.add_argument("--pixel-max-pixels", type=int, default=2000000,
                        help="Deprecated compatibility option; pixel CI now uses abnormal-slice histogram bootstrap")
    parser.add_argument("--teacher-weights", type=str, default=None, help="Local torchvision ResNet18 weights path for offline servers")
    parser.add_argument("--no-pretrained-teacher", action='store_true', help="Do not load ImageNet weights for teacher")
    # 旧参数 (兼容); 新参数支持多模态
    parser.add_argument("--modality", type=str, help="单模态(兼容旧版本)")
    parser.add_argument("--modalities", type=str, nargs='+', help="1~3 个模态，可用空格/加号/逗号分隔: --modalities flair t1 或 flair+t1 或 flair,t1")
    parser.add_argument('--strict-modal-match', action='store_true', help='启用严格多模态文件名完全一致校验')
    args = parser.parse_args()
    if args.split is None:
        parser.error("需要指定 train 或 test")
    args.dataset_path, args.dataset_key = resolve_dataset_path(args)
    device = select_device(args)

    # -------- 统一解析模态参数 --------
    raw_modalities = []
    if args.modalities:
        raw_modalities = args.modalities
    elif args.modality:
        raw_modalities = [args.modality]
    else:
        raise ValueError("需指定 --modalities 或 --modality")

    parsed = []
    for token in raw_modalities:
        # 支持 flair+t1+t2 或 flair,t1,t2
        parts = re.split(r'[+,]', token)
        for p in parts:
            p = p.strip()
            if p:
                parsed.append(p)
    if len(parsed) == 0:
        raise ValueError("未解析到有效模态，请检查输入。")
    # 保持顺序去重
    modalities = []
    for m in parsed:
        if m not in modalities:
            modalities.append(m)
    if not (1 <= len(modalities) <= 3):
        raise ValueError(f"目前仅支持 1~3 个模态，收到: {modalities}")
    args.modalities = modalities
    modality_set = set(m.lower() for m in args.modalities)
    if args.input_mode == 'auto':
        args.input_mode = 'pseudo_rgb' if modality_set == {'ct', 'pet'} and len(args.modalities) == 2 else 'concat'
    if args.input_mode == 'pseudo_rgb' and not (modality_set == {'ct', 'pet'} and len(args.modalities) == 2):
        raise ValueError("pseudo_rgb 仅支持 --modalities ct pet 或 pet ct")
    modalities_str = '+'.join(args.modalities)
    # -------- 结束模态解析 --------

    print(f'使用模态: {args.modalities}')
    print(f'数据集: {args.dataset_key} -> {args.dataset_path}')
    print(f'输入模式: {args.input_mode}')
    print(f'设备: {device}')

    np.random.seed(0)
    torch.manual_seed(0)
    
    transform_single = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    single_modal = len(args.modalities) == 1

    if args.split == 'train':
        if single_modal:
            paths = build_single_modal_paths(args.dataset_path, 'train', args.modalities[0])
            print(f'单模态训练 [{args.modalities[0]}]: {len(paths)} 张')
            train_paths, _ = train_test_split(paths, test_size=0.2, random_state=0)
            train_dataset = SingleModalDataset(train_paths, transform=transform_single)
        elif args.input_mode == 'pseudo_rgb':
            groups = build_multimodal_groups(args.dataset_path, 'train', args.modalities, strict=args.strict_modal_match)
            if len(groups) == 0:
                raise ValueError(f"未找到任何训练样本，请检查目录: {args.dataset_path}/train/normal/*/<模态>/*.png")
            print(f'PET/CT pseudo RGB 训练: {len(groups)} 组 ([CT,PET,PET])')
            train_groups, _ = train_test_split(groups, test_size=0.2, random_state=0)
            train_dataset = PseudoRGBDataset(train_groups, args.modalities)
        else:
            groups = build_multimodal_groups(args.dataset_path, 'train', args.modalities, strict=args.strict_modal_match)
            if len(groups) == 0:
                raise ValueError(f"未找到任何训练样本，请检查目录: {args.dataset_path}/train/normal/*/<模态>/*.png")
            print(f'多模态训练: {len(groups)} 组 (每样本 {len(args.modalities)} 个模态)')
            train_groups, _ = train_test_split(groups, test_size=0.2, random_state=0)
            train_dataset = MultiModalDataset(train_groups, transform_single=transform_single)
        train_loader = make_loader(train_dataset, args.batch_size, True, args)
    elif args.split == 'test':
        if single_modal:
            modality = args.modalities[0]
            pos_paths = build_single_modal_paths(args.dataset_path, 'test', modality, subset='abnormal')
            neg_paths = build_single_modal_paths(args.dataset_path, 'test', modality, subset='normal')
            print(f'单模态测试 [{modality}]: 阳性 {len(pos_paths)}  阴性 {len(neg_paths)}')
            test_pos_dataset = SingleModalDataset(pos_paths, transform=transform_single)
            test_neg_dataset = SingleModalDataset(neg_paths, transform=transform_single)
        elif args.input_mode == 'pseudo_rgb':
            pos_groups = build_multimodal_groups(args.dataset_path, 'test', args.modalities, subset='abnormal', strict=args.strict_modal_match)
            neg_groups = build_multimodal_groups(args.dataset_path, 'test', args.modalities, subset='normal', strict=args.strict_modal_match)
            if len(pos_groups) == 0:
                raise ValueError("阳性(abnormal) 测试样本为空。")
            if len(neg_groups) == 0:
                raise ValueError("阴性(normal) 测试样本为空。")
            print(f'PET/CT pseudo RGB 测试: 阳性 {len(pos_groups)}  阴性 {len(neg_groups)}')
            test_pos_dataset = PseudoRGBDataset(pos_groups, args.modalities)
            test_neg_dataset = PseudoRGBDataset(neg_groups, args.modalities)
        else:
            pos_groups = build_multimodal_groups(args.dataset_path, 'test', args.modalities, subset='abnormal', strict=args.strict_modal_match)
            neg_groups = build_multimodal_groups(args.dataset_path, 'test', args.modalities, subset='normal', strict=args.strict_modal_match)
            if len(pos_groups) == 0:
                raise ValueError("阳性(abnormal) 测试样本为空。")
            if len(neg_groups) == 0:
                raise ValueError("阴性(normal) 测试样本为空。")
            print(f'多模态测试: 阳性 {len(pos_groups)}  阴性 {len(neg_groups)}')
            test_pos_dataset = MultiModalDataset(pos_groups, transform_single=transform_single)
            test_neg_dataset = MultiModalDataset(neg_groups, transform_single=transform_single)
        test_pos_loader = make_loader(test_pos_dataset, args.test_batch_size, False, args)
        test_neg_loader = make_loader(test_neg_dataset, args.test_batch_size, False, args)

    # 单模态: 3 通道(灰度复制); pseudo RGB: 3 通道; concat: 每模态 3 通道再拼接
    in_channels = 3 if (single_modal or args.input_mode == 'pseudo_rgb') else 3 * len(args.modalities)
    print('初始化 teacher ResNet18...', flush=True)
    teacher = ResNet18_MS3(
        pretrained=not args.no_pretrained_teacher,
        in_channels=in_channels,
        weights_path=args.teacher_weights,
    )
    print('初始化 student ResNet18...', flush=True)
    student = ResNet18_MS3(pretrained=False, in_channels=in_channels)
    print('移动模型到设备...', flush=True)
    teacher.to(device)
    student.to(device)
    print('模型初始化完成。', flush=True)

    if args.split == 'train':
        save_path = train_val(teacher, student, train_loader, args, device, modalities_str)
        if not args.no_test_after_train:
            args.checkpoint = save_path
            test_pos_loader, test_neg_loader, test_pos_dataset, test_neg_dataset = build_test_loaders(args, transform_single)
            run_test(teacher, student, test_pos_loader, test_neg_loader, test_pos_dataset, test_neg_dataset, args, device, modalities_str)
    elif args.split == 'test':
        args.checkpoint = resolve_checkpoint(args)
        run_test(teacher, student, test_pos_loader, test_neg_loader, test_pos_dataset, test_neg_dataset, args, device, modalities_str)


def build_test_loaders(args, transform_single):
    single_modal = len(args.modalities) == 1
    if single_modal:
        modality = args.modalities[0]
        pos_paths = build_single_modal_paths(args.dataset_path, 'test', modality, subset='abnormal')
        neg_paths = build_single_modal_paths(args.dataset_path, 'test', modality, subset='normal')
        print(f'单模态测试 [{modality}]: 阳性 {len(pos_paths)}  阴性 {len(neg_paths)}')
        test_pos_dataset = SingleModalDataset(pos_paths, transform=transform_single)
        test_neg_dataset = SingleModalDataset(neg_paths, transform=transform_single)
    elif args.input_mode == 'pseudo_rgb':
        pos_groups = build_multimodal_groups(args.dataset_path, 'test', args.modalities, subset='abnormal', strict=args.strict_modal_match)
        neg_groups = build_multimodal_groups(args.dataset_path, 'test', args.modalities, subset='normal', strict=args.strict_modal_match)
        print(f'PET/CT pseudo RGB 测试: 阳性 {len(pos_groups)}  阴性 {len(neg_groups)}')
        test_pos_dataset = PseudoRGBDataset(pos_groups, args.modalities)
        test_neg_dataset = PseudoRGBDataset(neg_groups, args.modalities)
    else:
        pos_groups = build_multimodal_groups(args.dataset_path, 'test', args.modalities, subset='abnormal', strict=args.strict_modal_match)
        neg_groups = build_multimodal_groups(args.dataset_path, 'test', args.modalities, subset='normal', strict=args.strict_modal_match)
        print(f'多模态测试: 阳性 {len(pos_groups)}  阴性 {len(neg_groups)}')
        test_pos_dataset = MultiModalDataset(pos_groups, transform_single=transform_single)
        test_neg_dataset = MultiModalDataset(neg_groups, transform_single=transform_single)
    test_pos_loader = make_loader(test_pos_dataset, args.test_batch_size, False, args)
    test_neg_loader = make_loader(test_neg_dataset, args.test_batch_size, False, args)
    return test_pos_loader, test_neg_loader, test_pos_dataset, test_neg_dataset


def run_test(teacher, student, test_pos_loader, test_neg_loader, test_pos_dataset, test_neg_dataset, args, device, modalities_str):
    try:
        saved_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except RuntimeError:
        from torch import serialization as torch_serialization
        torch_serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
        saved_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)

    print('load ' + args.checkpoint)
    student.load_state_dict(saved_dict['state_dict'])
    print(f'Loaded model from epoch {saved_dict.get("epoch", "unknown")}')

    pos = test(teacher, student, test_pos_loader, device, args, name='abnormal')
    neg = test(teacher, student, test_neg_loader, device, args, name='normal')

    scores = []
    for i in range(len(pos)):
        temp = cv2.resize(pos[i], (256, 256))
        scores.append(temp)
    for i in range(len(neg)):
        temp = cv2.resize(neg[i], (256, 256))
        scores.append(temp)
    scores = np.stack(scores)

    pos_base_paths = get_dataset_paths(test_pos_dataset)
    neg_base_paths = get_dataset_paths(test_neg_dataset)
    all_paths = pos_base_paths + neg_base_paths
    gt = load_positive_masks(pos_base_paths)

    neg_gt = np.zeros((len(neg), 256, 256), dtype=bool)
    gt_pixel = np.concatenate((gt, neg_gt), 0)
    gt_image = np.concatenate((np.ones(pos.shape[0], dtype=bool), np.zeros(neg.shape[0], dtype=bool)), 0)
    image_scores_raw = scores.max(-1).max(-1)

    if args.cache_path:
        save_eval_cache(args.cache_path, gt_image, gt_pixel, scores, image_scores_raw, all_paths)
    if args.export_only:
        print("Skipped metric computation (--export-only).", flush=True)
        return

    print(f"Modalities: {args.modalities}")
    print("Computing image-level metrics and 95CI...", flush=True)
    image_metrics = binary_metric_bundle(
        gt_image,
        image_scores_raw,
        n_boot=args.ci_bootstrap,
        alpha=args.ci_alpha,
        seed=args.ci_seed,
        include_f1=True,
    )
    print_metric_block("Slice-Img", image_metrics)

    pro = None
    if args.compute_pro:
        print('Computing AUPRO; this step is slow...', flush=True)
        pro = evaluate(gt_pixel, scores, metric='pro')
    if pro is not None:
        print("AUPRO: {:.4f}".format(pro))
    if args.skip_pixel_metrics:
        print('Skipped pixel-level metrics (--skip-pixel-metrics).', flush=True)
    else:
        print("Computing pixel-level Slice-Px(abn) metrics and 95CI...", flush=True)
        print("Computing exact full-pixel point estimates with sklearn...", flush=True)
    metrics = compute_all_metrics(
        gt_image,
        gt_pixel,
        scores,
        image_scores_raw,
        all_paths,
        args,
        compute_pixel=not args.skip_pixel_metrics,
    )
    print("Computing patient-level metrics and 95CI...", flush=True)
    print_all_metrics(metrics)
    save_heatmaps_for_results(scores, all_paths, args)

    # 保存npy用于AUC对比（图像级分数归一化到0-1）；按数据集、模态、输入模式区分文件名
    os.makedirs('npy', exist_ok=True)
    image_scores = (image_scores_raw - image_scores_raw.min()) / (image_scores_raw.max() - image_scores_raw.min() + 1e-8)
    np.save(os.path.join('npy', f'{args.dataset_key}_{modalities_str}_{args.input_mode}_results.npy'), {
        'labels': gt_image.astype(np.uint8),
        'scores': image_scores
    })
     

def test(teacher, student, loader, device, args, name='test'):
    teacher.eval()
    student.eval()
    loss_map = np.zeros((len(loader.dataset), 64, 64))
    i = 0
    print(f'开始推理 {name}: {len(loader.dataset)} 张, batch_size={loader.batch_size}', flush=True)
    for batch_idx, batch_data in enumerate(loader, start=1):
        _, batch_img = batch_data
        batch_img = batch_img.to(device, non_blocking=True)
        with torch.inference_mode():
            t_feat = teacher(batch_img)
            s_feat = student(batch_img)
        score_map = 1.
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
            # aggregate score map by element-wise product
            score_map = score_map * sm
        loss_map[i: i + batch_img.size(0)] = score_map.squeeze(1).cpu().numpy()
        i += batch_img.size(0)
        if batch_idx == 1 or batch_idx % args.test_log_interval == 0 or batch_idx == len(loader):
            print(f'[{name}] batch {batch_idx}/{len(loader)}', flush=True)
    return loss_map
    

def train_val(teacher, student, train_loader, args, device, modalities_str):
    teacher.eval()
    student.train()

    optimizer = torch.optim.SGD(student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4)
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0
        for batch_idx, batch_data in enumerate(train_loader, start=1):
            _, batch_img = batch_data
            batch_img = batch_img.to(device)

            with torch.no_grad():
                t_feat = teacher(batch_img)
            s_feat = student(batch_img)

            loss = 0
            for i in range(len(t_feat)):
                t_feat[i] = F.normalize(t_feat[i], dim=1)
                s_feat[i] = F.normalize(s_feat[i], dim=1)
                loss += torch.sum((t_feat[i] - s_feat[i]) ** 2, 1).mean()

            if batch_idx == 1 or batch_idx % args.log_interval == 0 or batch_idx == len(train_loader):
                print('[Epoch %d/%d][Batch %d/%d] loss: %f' % (
                    epoch + 1, args.epochs, batch_idx, len(train_loader), loss.item()
                ), flush=True)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        print('Epoch [%d/%d] Average training loss: %f' % (epoch + 1, args.epochs, avg_loss))

    save_name = checkpoint_path(args, modalities_str)
    dir_name = os.path.dirname(save_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    state_dict = {
        'state_dict': student.state_dict(),
        'epoch': args.epochs,
        'dataset': args.dataset_key,
        'modalities': args.modalities,
        'input_mode': args.input_mode,
    }
    torch.save(state_dict, save_name)
    print(f'Saved epoch {args.epochs} model: {save_name}')
    return save_name

if __name__ == "__main__":
    main()
