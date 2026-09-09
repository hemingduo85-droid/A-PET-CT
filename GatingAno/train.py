"""
# 单模态/双模态（训练30轮后在 test 上评估一次）

cd /data/cyf/codes/A-PET-CT/GatingAno

nohup python train.py \
  --dataset PSMA \
  --modality petct \
  --gpu 2 \
  --bootstrap_iters 500 \
  --ci_hist_bins 16384 \
  --checkpoint_dir /data/cyf/codes/A-PET-CT/GatingAno/checkpoints/1e-20.2ep20 \
  > train_psma_petct0.20.2ep20.log 2>&1 &


nohup python train.py \
  --dataset FDG \
  --modality petct \
  --dual_input_mode ctctpet \
  --gpu 6 \
  --bootstrap_iters 500 \
  --ci_hist_bins 16384 \
  --checkpoint_dir /data/cyf/codes/A-PET-CT/GatingAno/checkpoints/ctctpetfdg30 \
  > train_fdg_ctpetpet30.log 2>&1 &


cd /data/cyf/codes/A-PET-CT/GatingAno

nohup python train.py \
  --dataset PSMA \
  --modality petct \
  --dual_input_mode ctctpet \
  --gpu 7 \
  --checkpoint_dir /data/cyf/codes/A-PET-CT/GatingAno/checkpoints/ctctpetepoch20 \
  --bootstrap_iters 500 \
  --ci_hist_bins 16384 \
  > train_psma_ctctpetepoch20.log 2>&1 &

cd /data/cyf/codes/A-PET-CT/GatingAno
nohup python train.py --modality ct --gpu 4 > train_ct_psma.log 2>&1 &

# 单模态 CT
python train.py --modality ct --gpu 4 > train_ct_psma.log 2>&1 &

# 第30轮权重: checkpoints/PSMA_pet_epoch30.pth  (数据集名_模态_epoch30.pth)
python train.py --dataset FDG --modality pet

# 双模态 PET+CT: 默认原图层面伪 RGB [CT, PET, PET]
python train.py --dataset FDG --modality petct
"""
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageOps
from collections import defaultdict
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm
from torchvision import transforms as T
import warnings

from dataloader import PETCTAnomalyDataset
from models import GatingAno, Discriminator, AdversarialLoss

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CI_BOOTSTRAP_ITERS = 500
CI_RANDOM_SEED = 20260707
NORMAL_95_Z = 1.959963984540054


DATASET_CONFIGS = {
    'PSMA': '/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA',
    'FDG': '/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG',
}


def _dataset_name_from_root(data_root):
    """从路径提取数据集名，如 .../PSMA -> PSMA"""
    return os.path.basename(os.path.normpath(data_root))


# ===================== Config =====================
class Config:
    def __init__(
        self,
        modality='pet',
        gpu='2',
        dataset='PSMA',
        data_root=None,
        checkpoint_dir='./checkpoints',
        dual_input_mode='pseudo_rgb',
        bootstrap_iters=CI_BOOTSTRAP_ITERS,
        ci_pixel_max_samples=200000,
        ci_hist_bins=16384,
        ci_seed=CI_RANDOM_SEED,
    ):
        self.modality = modality.lower()
        if self.modality not in ('pet', 'ct', 'petct'):
            raise ValueError("modality must be 'pet', 'ct' or 'petct'")

        dataset_key = dataset.upper()
        if data_root is None and dataset_key not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {', '.join(DATASET_CONFIGS)}")

        root = data_root or DATASET_CONFIGS[dataset_key]
        self.data_root = root
        self.dataset_name = _dataset_name_from_root(root) if data_root else dataset_key
        self.train_root = os.path.join(root, 'train')
        self.test_root = os.path.join(root, 'test')

        self.batch_size = 16
        self.num_workers = 4
        self.lr = 1e-4
        self.num_epochs = 30
        self.image_size = 256
        self.dual_input_mode = dual_input_mode
        if self.modality == 'petct' and self.dual_input_mode not in ('pseudo_rgb', 'ctctpet'):
            raise ValueError("petct dual_input_mode must be 'pseudo_rgb' or 'ctctpet'")
        self.input_channels = 3
        self.output_channels = 3
        self.bootstrap_iters = int(bootstrap_iters)
        self.ci_pixel_max_samples = int(ci_pixel_max_samples)
        self.ci_hist_bins = int(ci_hist_bins)
        self.ci_seed = int(ci_seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} | dataset: {self.dataset_name} | modality: {self.modality.upper()}")

        tag = f'{self.dataset_name}_{self.modality}'
        self.save_dir = f'./results/{tag}'
        self.checkpoint_dir = os.path.expanduser(checkpoint_dir)
        self.final_epoch = self.num_epochs
        self.eval_epochs = [self.final_epoch]
        self.heatmap_dir = os.path.join(self.save_dir, f'heatmaps_epoch{self.final_epoch}')
        self.final_ckpt_path = os.path.join(self.checkpoint_dir, f'{tag}_epoch{self.final_epoch}.pth')
        self.final_metrics_path = os.path.join(
            self.checkpoint_dir, f'{tag}_epoch{self.final_epoch}_metrics.txt'
        )
        self.eval_metrics_path = os.path.join(self.checkpoint_dir, f'{tag}_eval_metrics.txt')

        # 模型参数
        self.alpha = [0.5, 0.5, 0.5, 0.5]
        self.lambda_adv = 0.1
        self.lambda_rec = 1.0


def should_evaluate_epoch(epoch, config):
    return epoch in config.eval_epochs


def save_final_checkpoint(path, epoch, config, generator, metrics):
    """只保存用于推理的 GatingAno（generator），discriminator 不参与测试。"""
    torch.save({
        'epoch': epoch,
        'dataset': config.dataset_name,
        'modality': config.modality,
        'input_channels': config.input_channels,
        'output_channels': config.output_channels,
        'dual_input_mode': config.dual_input_mode,
        'alpha': config.alpha,
        'metrics': metrics,
        'state_dict': generator.state_dict(),
    }, path)


def validate_checkpoint_metadata(checkpoint, expected_dataset, expected_modality, config):
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint is missing GatingAno metadata/state_dict. "
            "Use a checkpoint produced by the current training script."
        )

    expected = {
        "dataset": str(expected_dataset).upper(),
        "modality": str(expected_modality).lower(),
        "input_channels": int(config.input_channels),
        "output_channels": int(config.output_channels),
        "dual_input_mode": str(config.dual_input_mode),
    }
    actual = {
        "dataset": str(checkpoint.get("dataset", "")).upper(),
        "modality": str(checkpoint.get("modality", "")).lower(),
        "input_channels": checkpoint.get("input_channels"),
        "output_channels": checkpoint.get("output_channels"),
        "dual_input_mode": checkpoint.get("dual_input_mode"),
    }
    mismatches = [
        f"{key}: expected {expected[key]!r}, got {actual[key]!r}"
        for key in expected
        if actual[key] != expected[key]
    ]
    if mismatches:
        raise RuntimeError("Checkpoint metadata mismatch: " + "; ".join(mismatches))


def save_metrics(path, epoch, config, metrics):
    with open(path, 'w') as f:
        f.write(f"Dataset: {config.dataset_name}\n")
        f.write(f"Modality: {config.modality}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(format_metrics(metrics['slice'], metrics['patient'], as_percent=True))
        f.write("\n")


def save_heatmap(anomaly_map, label, path):
    values = np.asarray(anomaly_map, dtype=np.float32)
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    if max_v > min_v:
        values = (values - min_v) / (max_v - min_v)
    else:
        values = np.zeros_like(values, dtype=np.float32)

    gray = Image.fromarray((values * 255).astype(np.uint8), mode='L')
    heatmap = ImageOps.colorize(gray, black='#000033', mid='#ffcc00', white='#ff3300')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    heatmap.save(path)


def save_eval_cache(path, gt_labels, gt_masks, anomaly_maps, image_scores, paths):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(
        path,
        labels=np.asarray(gt_labels, dtype=np.int32),
        masks=np.asarray(gt_masks, dtype=np.uint8),
        maps=np.asarray(anomaly_maps, dtype=np.float32),
        image_scores=np.asarray(image_scores, dtype=np.float64),
        paths=np.asarray(paths, dtype=str),
    )


# ===================== 指标函数 =====================
def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score):
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


def _clip_ci(low, high):
    return (float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0)))


def _metric_with_ci(value, ci):
    return {"value": float(value), "ci": (float(ci[0]), float(ci[1]))}


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=CI_BOOTSTRAP_ITERS, seed=CI_RANDOM_SEED):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    value = float(metric_fn(y_true, y_score))
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return _metric_with_ci(value, (value, value))

    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(float(metric_fn(y_true[idx], y_score[idx])))

    if not values:
        return _metric_with_ci(value, (value, value))
    low, high = np.percentile(values, [2.5, 97.5])
    return _metric_with_ci(value, _clip_ci(low, high))


def _metrics_from_hist(pos_hist, neg_hist):
    tp = pos_hist[::-1].astype(np.float64, copy=False)
    fp = neg_hist[::-1].astype(np.float64, copy=False)
    total_pos = float(tp.sum())
    total_neg = float(fp.sum())
    if total_pos <= 0.0 or total_neg <= 0.0:
        return 0.0, 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    tpr = cum_tp / total_pos
    fpr = cum_fp / total_neg
    integrate = getattr(np, "trapezoid", np.trapz)
    auroc = float(integrate(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)
    aupr = float(np.sum(precision * (tp / total_pos)))
    return auroc, aupr


def bootstrap_pixel_hist_ci(
    gt_masks,
    anomaly_maps,
    exact_auroc,
    exact_aupr,
    n_boot=CI_BOOTSTRAP_ITERS,
    seed=CI_RANDOM_SEED,
    bins=16384,
):
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)
    if len(gt_masks) == 0:
        empty = _metric_with_ci(0.0, (0.0, 0.0))
        return empty, empty

    score_min = float(np.min(anomaly_maps))
    score_max = float(np.max(anomaly_maps))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    bins = int(bins)
    scale = (bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(gt_masks), bins), dtype=np.uint32)
    neg_hists = np.zeros((len(gt_masks), bins), dtype=np.uint32)
    for row, (mask, score_map) in enumerate(zip(gt_masks, anomaly_maps)):
        scores = score_map.reshape(-1)
        mask_flat = mask.reshape(-1).astype(bool)
        bin_idx = np.floor((scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[mask_flat], minlength=bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~mask_flat], minlength=bins).astype(np.uint32)

    rng = np.random.default_rng(seed)
    aurocs, auprs = [], []
    n = len(gt_masks)
    for i in range(1, int(n_boot) + 1):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if i % 50 == 0 or i == int(n_boot):
            print(f"Pixel histogram slice bootstrap: {i}/{int(n_boot)}", flush=True)

    auroc_ci = _clip_ci(*np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = _clip_ci(*np.percentile(auprs, [2.5, 97.5]))
    return (
        _metric_with_ci(exact_auroc, auroc_ci),
        _metric_with_ci(exact_aupr, aupr_ci),
    )


def format_metric(name, metric, as_percent=True):
    scale = 100.0 if as_percent else 1.0
    value = metric["value"] * scale
    low = metric["ci"][0] * scale
    high = metric["ci"][1] * scale
    suffix = "%" if as_percent else ""
    return f"{name}={value:.2f}{suffix} (95% CI {low:.2f}-{high:.2f}{suffix})"


def format_image_metrics(slice_metrics, as_percent=True):
    return (
        "[Slice-Img]  "
        + "  ".join([
            format_metric("AUROC", slice_metrics["img_auroc"], as_percent),
            format_metric("AUPR", slice_metrics["img_ap"], as_percent),
            format_metric("F1", slice_metrics["img_f1"], as_percent),
        ])
    )


def compute_metrics(
    gt_labels,
    gt_masks,
    anomaly_maps,
    image_scores,
    paths,
    print_image_first=False,
    bootstrap_iters=CI_BOOTSTRAP_ITERS,
    ci_pixel_max_samples=200000,
    ci_hist_bins=16384,
    ci_seed=CI_RANDOM_SEED,
):
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    image_scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    slice_metrics = {
        "img_auroc": bootstrap_ci(
            gt_labels, image_scores, _safe_auroc, n_boot=bootstrap_iters, seed=ci_seed
        ),
        "img_ap": bootstrap_ci(
            gt_labels, image_scores, _safe_ap, n_boot=bootstrap_iters, seed=ci_seed + 1
        ),
        "img_f1": bootstrap_ci(
            gt_labels, image_scores, f1_score_max, n_boot=bootstrap_iters, seed=ci_seed + 2
        ),
        "_pr_sp": image_scores,
        "_gt_sp": gt_labels,
    }
    if print_image_first:
        print("Computing image-level metrics and 95CI...", flush=True)
        print(format_image_metrics(slice_metrics, as_percent=True))

    gt_px_abn_maps = gt_masks[gt_labels == 1]
    pr_px_abn_maps = maps[gt_labels == 1]
    gt_px_abn = gt_px_abn_maps.reshape(-1).astype(bool)
    pr_px_abn = pr_px_abn_maps.reshape(-1).astype(np.float64)
    print("Computing pixel-level Slice-Px(abn) metrics and 95CI...", flush=True)
    print("Computing exact full-pixel point estimates with sklearn...", flush=True)
    lesion_prevalence = float(gt_px_abn.mean()) if gt_px_abn.size else 0.0
    print(
        f"Pixel AUPR random baseline (lesion-pixel prevalence): "
        f"{lesion_prevalence * 100:.4f}%",
        flush=True,
    )
    exact_px_auroc = _safe_auroc(gt_px_abn, pr_px_abn)
    exact_px_aupr = _safe_ap(gt_px_abn, pr_px_abn)
    px_auroc, px_aupr = bootstrap_pixel_hist_ci(
        gt_px_abn_maps,
        pr_px_abn_maps,
        exact_px_auroc,
        exact_px_aupr,
        n_boot=bootstrap_iters,
        seed=ci_seed + 10,
        bins=ci_hist_bins,
    )
    slice_metrics.update({
        "px_auroc_abn": px_auroc,
        "px_aupr_abn": px_aupr,
    })

    patient_scores = defaultdict(list)
    patient_labels = defaultdict(list)
    for path, score, label_value in zip(paths, image_scores, gt_labels):
        pid = patient_id_from_path(path)
        patient_scores[pid].append(float(score))
        patient_labels[pid].append(int(label_value))
    pat_score, pat_label = [], []
    for pid in patient_scores:
        pat_score.append(float(np.max(patient_scores[pid])))
        pat_label.append(int(np.max(patient_labels[pid])))
    pat_score = np.asarray(pat_score, dtype=np.float64)
    pat_label = np.asarray(pat_label, dtype=np.int32)
    pat_metrics = {
        "pat_auroc": bootstrap_ci(
            pat_label, pat_score, _safe_auroc, n_boot=bootstrap_iters, seed=ci_seed + 5
        ),
        "pat_ap": bootstrap_ci(pat_label, pat_score, _safe_ap, n_boot=bootstrap_iters, seed=ci_seed + 6),
        "pat_f1": bootstrap_ci(
            pat_label, pat_score, f1_score_max, n_boot=bootstrap_iters, seed=ci_seed + 7
        ),
        "n_patients": len(pat_label),
        "n_abn_patients": int(pat_label.sum()),
    }
    return slice_metrics, pat_metrics


def format_metrics(slice_metrics, pat_metrics, as_percent=True):
    m = slice_metrics
    lines = [
        format_image_metrics(m, as_percent),
        "[Slice-Px(abn)]  "
        + "  ".join([
            format_metric("AUROC", m["px_auroc_abn"], as_percent),
            format_metric("AUPR", m["px_aupr_abn"], as_percent),
        ]),
    ]
    p = pat_metrics
    lines.append(
        f"[Patient({p['n_abn_patients']}/{p['n_patients']}abn)]  "
        + "  ".join([
            format_metric("AUROC", p["pat_auroc"], as_percent),
            format_metric("AUPR", p["pat_ap"], as_percent),
            format_metric("F1", p["pat_f1"], as_percent),
        ])
    )
    return "\n".join(lines)


# ===================== 训练函数 =====================
def train_epoch(epoch, config, generator, discriminator, train_loader, 
                optimizer_G, optimizer_D, adversarial_loss, l1_loss):
    """训练一个epoch"""
    generator.train()
    discriminator.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        if len(batch) == 2:
            images, _ = batch
        else:
            images, _, _ = batch
        
        images = images.to(config.device)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        with torch.no_grad():
            _, reconstructed = generator(images, config.alpha)
        
        pred_real, _ = discriminator(images)
        loss_D_real = adversarial_loss(pred_real, True, is_disc=True)
        
        pred_fake, _ = discriminator(reconstructed.detach())
        loss_D_fake = adversarial_loss(pred_fake, False, is_disc=True)
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        latent, reconstructed = generator(images, config.alpha)
        
        loss_rec = l1_loss(reconstructed, images)
        
        pred_fake, _ = discriminator(reconstructed)
        loss_adv = adversarial_loss(pred_fake, True, is_disc=False)
        
        loss_G = config.lambda_rec * loss_rec + config.lambda_adv * loss_adv
        loss_G.backward()
        optimizer_G.step()
        
        total_loss += loss_G.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    return avg_loss


# ===================== 评估函数 =====================
def evaluate(
    epoch,
    config,
    generator,
    test_loader,
    save_heatmaps=False,
    cache_path=None,
    compute_ci=True,
):
    """评估模型"""
    generator.eval()
    
    anomaly_maps = []
    image_scores = []
    gt_masks = []
    gt_labels = []
    paths = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            if len(batch) == 4:
                images, labels, masks, batch_paths = batch
            elif len(batch) == 3:
                images, labels, masks = batch
                batch_paths = [f"unknown/unknown/{batch_idx}_{i}.png" for i in range(len(labels))]
            elif len(batch) == 2:
                images, labels = batch
                masks = torch.zeros_like(images[:, :1, :, :])
                batch_paths = [f"unknown/unknown/{batch_idx}_{i}.png" for i in range(len(labels))]
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            
            images = images.to(config.device)
            labels = labels.cpu().numpy()
            masks = masks.cpu().numpy()
            if isinstance(batch_paths, (str, bytes)):
                batch_paths = [batch_paths]
            
            _, reconstructed = generator(images, config.alpha)
            anomaly_map = torch.abs(images - reconstructed)
            
            scores = anomaly_map.mean(dim=[1, 2, 3]).cpu().numpy()
            image_scores.extend(scores)
            gt_labels.extend(labels)
            paths.extend([str(path) for path in batch_paths])
            
            for i in range(len(labels)):
                anomaly_map_single = anomaly_map[i].mean(dim=0).cpu().numpy()
                anomaly_maps.append(anomaly_map_single)

                if save_heatmaps:
                    label_name = 'abnormal' if int(labels[i]) == 1 else 'normal'
                    heatmap_name = f'{batch_idx:06d}_{i:02d}_{label_name}.png'
                    heatmap_path = os.path.join(config.heatmap_dir, label_name, heatmap_name)
                    save_heatmap(anomaly_map_single, labels[i], heatmap_path)
                
                mask_np = masks[i].squeeze()
                gt_masks.append(mask_np)

    if cache_path:
        save_eval_cache(cache_path, gt_labels, gt_masks, anomaly_maps, image_scores, paths)
        print(f"Saved eval cache: {cache_path}")

    if not compute_ci:
        return None

    slice_metrics, pat_metrics = compute_metrics(
        gt_labels,
        gt_masks,
        anomaly_maps,
        image_scores,
        paths,
        print_image_first=True,
        bootstrap_iters=config.bootstrap_iters,
        ci_pixel_max_samples=config.ci_pixel_max_samples,
        ci_hist_bins=config.ci_hist_bins,
        ci_seed=config.ci_seed,
    )
    
    print(f"\n{'='*70}")
    print(f"📊 Results at Epoch {epoch}:")
    print(f"{'='*70}")
    print(format_metrics(slice_metrics, pat_metrics, as_percent=True))
    print(f"{'='*70}\n")
    
    return {
        'slice': slice_metrics,
        'patient': pat_metrics,
    }


# ===================== 主程序 =====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GatingAno PET/CT anomaly training')
    parser.add_argument('--modality', type=str, default='pet', choices=['pet', 'ct', 'petct'],
                        help='pet, ct, or petct for dual-modality PET+CT')
    parser.add_argument('--gpu', type=str, default='2', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset', type=str.upper, default='PSMA', choices=sorted(DATASET_CONFIGS),
                        help='dataset config key; used to choose data root and output names')
    parser.add_argument('--data_root', type=str, default=None,
                        help='custom dataset root containing train/ and test/; overrides --dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='directory for final checkpoint and metric files')
    parser.add_argument('--dual_input_mode', type=str, default='pseudo_rgb',
                        choices=['pseudo_rgb', 'ctctpet'],
                        help='petct input: pseudo_rgb=[CT,PET,PET], ctctpet=[CT,CT,PET]')
    parser.add_argument('--bootstrap_iters', type=int, default=CI_BOOTSTRAP_ITERS,
                        help='bootstrap iterations for AP/AUPR/F1 95% CI')
    parser.add_argument('--ci_pixel_max_samples', type=int, default=200000,
                        help='deprecated; point estimates use all abnormal pixels')
    parser.add_argument('--ci_hist_bins', type=int, default=16384,
                        help='histogram bins for abnormal-slice pixel bootstrap CI')
    parser.add_argument('--ci_seed', type=int, default=CI_RANDOM_SEED,
                        help='random seed for CI resampling')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = Config(
        modality=args.modality,
        gpu=args.gpu,
        dataset=args.dataset,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        dual_input_mode=args.dual_input_mode,
        bootstrap_iters=args.bootstrap_iters,
        ci_pixel_max_samples=args.ci_pixel_max_samples,
        ci_hist_bins=args.ci_hist_bins,
        ci_seed=args.ci_seed,
    )

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"\nDataset:    {config.dataset_name}")
    print(f"Train root: {config.train_root}")
    print(f"Test root:  {config.test_root}")
    print(f"Modality:   {config.modality.upper()}")
    print(f"Final ckpt: {config.final_ckpt_path}")
    print(f"Heatmaps:   {config.heatmap_dir}")
    print(f"Channels:   {config.input_channels}")
    if config.modality == 'petct':
        channel_order = '[CT, CT, PET]' if config.dual_input_mode == 'ctctpet' else '[CT, PET, PET]'
        print(f"PETCT input: {config.dual_input_mode} {channel_order}")
    print(f"Bootstrap:  {config.bootstrap_iters}")
    print(f"Pixel point estimates: exact full abnormal pixels")
    print(f"Pixel CI histogram bins: {config.ci_hist_bins}")
    print(f"Batch size: {config.batch_size}")
    print(f"Num epochs: {config.num_epochs}\n")
    
    # Transform
    transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load datasets
    print("📦 Loading datasets...\n")
    train_dataset = PETCTAnomalyDataset(
        root=config.train_root,
        mode='train',
        modality=config.modality,
        dual_input_mode=config.dual_input_mode,
        transform=transform,
        image_size=config.image_size,
    )
    test_dataset = PETCTAnomalyDataset(
        root=config.test_root,
        mode='test',
        modality=config.modality,
        dual_input_mode=config.dual_input_mode,
        return_path=True,
        transform=transform,
        image_size=config.image_size,
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True, 
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    # Build models
    print("🧠 Building models...")
    generator = GatingAno(n_channels=config.input_channels, n_classes=config.output_channels).to(config.device)
    discriminator = Discriminator(in_channels=config.input_channels).to(config.device)
    print(f"Generator:     {generator.__class__.__name__}")
    print(f"Discriminator: {discriminator.__class__.__name__}\n")
    
    # Losses and optimizers
    adversarial_loss = AdversarialLoss(type='hinge')
    l1_loss = torch.nn.L1Loss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    
    # Training loop
    print("🚀 Starting training...\n")
    
    for epoch in range(1, config.num_epochs + 1):
        train_epoch(epoch, config, generator, discriminator, train_loader,
                   optimizer_G, optimizer_D, adversarial_loss, l1_loss)
        
        if should_evaluate_epoch(epoch, config):
            metrics = evaluate(epoch, config, generator, test_loader, save_heatmaps=True)
            save_final_checkpoint(config.final_ckpt_path, epoch, config, generator, metrics)
            save_metrics(config.final_metrics_path, epoch, config, metrics)
            print(f"Saved final checkpoint: {config.final_ckpt_path}")
            print(f"Saved final metrics:    {config.final_metrics_path}")
            print(f"Saved heatmaps under:   {config.heatmap_dir}")
    
    print("\n✨ Training completed!")
