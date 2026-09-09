import torch
import numpy as np
import random
import os
import copy
import math
import warnings
import logging
import zipfile
from collections import defaultdict
from PIL import Image

from models.resnet import wide_resnet50_2
from models.de_resnet import de_wide_resnet50_2
from models.recontrast import ReContrast
from utils import global_cosine_hm, replace_layers
from torch.nn import functional as F

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from dataset import get_detection_dataset

"""
cd /data/cyf/codes/A-PET-CT/Recontrast
# PSMA PET-CT 伪 RGB: [CT, PET, PET]，训练 30 epoch 后测试并保存 epoch30 权重
python ReContrast_petct.py --mode train --dataset psma --modality petct --cuda 0

# FDG 单模态 PET
python ReContrast_petct.py --mode train --dataset fdg --modality pet --cuda 1

# 直接测试并保存全部热力图
python ReContrast_petct.py --mode test --dataset psma --modality petct --save_all_heatmaps --cuda 0

cd /data/cyf/codes/A-PET-CT/Recontrast
nohup python -u ReContrast_petct.py --mode test --dataset psma --modality petct --cuda 1 --ci_bootstraps 500 \
     --eval_cache ./saved_results/pet_psma_export/PSMA_petct_eval_cache.npz > recontrast_psma_petct_npz.log 2>&1 &

nohup python -u ReContrast_petct.py --mode test --dataset fdg --modality petct --cuda 0 --ci_bootstraps 500 \
     --eval_cache ./saved_results/pet_fdg_export/FDG_petct_eval_cache.npz > recontrast_fdg_petct_npz.log 2>&1 &
"""

warnings.filterwarnings("ignore")


DATASET_ROOTS = {
    'fdg': '/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG',
    'psma': '/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA',
}


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    if not logger.handlers:
        logger.addHandler(streamHandler)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def f1_score_max(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0, 0.5
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    if len(f1s) == 0:
        return 0.0, 0.5
    idx = np.argmax(f1s)
    return f1s[idx], thrs[idx]


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_aupr(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _max_f1(y_true, y_score):
    return float(f1_score_max(y_true, y_score)[0])


def _bootstrap_ci(y_true, y_score, metric_fn, iters=500, seed=20260717):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    value = metric_fn(y_true, y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2 or iters <= 0:
        return value, (value, value)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(int(iters)):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_fn(y_true[idx], y_score[idx]))
    if not values:
        return value, (value, value)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return value, (float(lo), float(hi))


def _patient_arrays(paths, image_scores, labels):
    patient_scores = defaultdict(list)
    patient_labels = defaultdict(list)
    for path, score, label in zip(paths, image_scores, labels):
        pid = patient_id_from_path(path)
        patient_scores[pid].append(float(score))
        patient_labels[pid].append(int(label))
    scores, labels_out = [], []
    for pid in patient_scores:
        scores.append(float(np.max(patient_scores[pid])))
        labels_out.append(int(np.max(patient_labels[pid])))
    return np.asarray(labels_out, dtype=np.int32), np.asarray(scores, dtype=np.float64)


def _metrics_from_hist(pos_hist, neg_hist):
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
    trapezoid = getattr(np, "trapezoid", np.trapz)
    auroc = float(trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)
    aupr = float(np.sum(precision * (tp / total_pos)))
    return auroc, aupr


def _pixel_slice_bootstrap(labels, masks, maps, iters, seed, bins, exact_auroc, exact_aupr, progress_every=50):
    keep = np.asarray(labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        return (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0))

    score_min = float(np.min(maps[keep]))
    score_max = float(np.max(maps[keep]))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    scale = (bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        scores = maps[idx].reshape(-1)
        mask = masks[idx].reshape(-1).astype(bool)
        bin_idx = np.floor((scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[mask], minlength=bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~mask], minlength=bins).astype(np.uint32)

    rng = np.random.default_rng(seed)
    aurocs, auprs = [], []
    n = len(abn_idx)
    for i in range(int(iters)):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if progress_every and (i + 1) % progress_every == 0:
            print_fn(f"Pixel histogram slice bootstrap: {i + 1}/{iters}")

    auroc_ci = tuple(float(v) for v in np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = tuple(float(v) for v in np.percentile(auprs, [2.5, 97.5]))
    return (float(exact_auroc), auroc_ci), (float(exact_aupr), aupr_ci)


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.nan_to_num(a_map, nan=0.0, posinf=2.0, neginf=0.0)
        a_map = torch.clamp(a_map, min=0.0)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean_val = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean_val) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    gaussian_filter = torch.nn.Conv2d(
        in_channels=channels, out_channels=channels, kernel_size=kernel_size,
        groups=channels, bias=False, padding=kernel_size // 2
    )
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


def patient_id_from_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


def compute_eval_protocol_metrics(gt_labels, gt_masks, anomaly_maps, image_scores, paths,
                                  n_boot=500, seed=20260717, hist_bins=16384):
    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    image_scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    print_fn("Computing image-level metrics and 95CI...")
    img_auroc = _bootstrap_ci(gt_labels, image_scores, _safe_auroc, n_boot, seed)
    img_aupr = _bootstrap_ci(gt_labels, image_scores, _safe_aupr, n_boot, seed + 1)
    img_f1 = _bootstrap_ci(gt_labels, image_scores, _max_f1, n_boot, seed + 2)
    slice_metrics = {
        "img_auroc": img_auroc[0],
        "img_auroc_ci": img_auroc[1],
        "img_aupr": img_aupr[0],
        "img_aupr_ci": img_aupr[1],
        "img_f1": img_f1[0],
        "img_f1_ci": img_f1[1],
    }
    print_fn(format_image_metrics(slice_metrics))

    print_fn("Computing pixel-level Slice-Px(abn) metrics and 95CI...")
    print_fn("Computing exact full-pixel point estimates with sklearn...")
    keep = gt_labels == 1
    px_true = gt_masks[keep].reshape(-1).astype(np.int32)
    px_score = maps[keep].reshape(-1).astype(np.float64)
    exact_auroc = _safe_auroc(px_true, px_score)
    exact_aupr = _safe_aupr(px_true, px_score)
    px_auroc, px_aupr = _pixel_slice_bootstrap(
        gt_labels,
        gt_masks,
        maps,
        n_boot,
        seed + 10,
        hist_bins,
        exact_auroc,
        exact_aupr,
    )
    slice_metrics.update({
        "px_auroc_abn": px_auroc[0],
        "px_auroc_abn_ci": px_auroc[1],
        "px_aupr_abn": px_aupr[0],
        "px_aupr_abn_ci": px_aupr[1],
    })
    print_fn(
        "[Slice-Px(abn)]  "
        + "  ".join([
            _format_metric("AUROC", px_auroc),
            _format_metric("AUPR", px_aupr),
        ])
    )

    print_fn("Computing patient-level metrics and 95CI...")
    pat_label, pat_score = _patient_arrays(paths, image_scores, gt_labels)
    pat_auroc = _bootstrap_ci(pat_label, pat_score, _safe_auroc, n_boot, seed + 5)
    pat_aupr = _bootstrap_ci(pat_label, pat_score, _safe_aupr, n_boot, seed + 6)
    pat_f1 = _bootstrap_ci(pat_label, pat_score, _max_f1, n_boot, seed + 7)
    patient_metrics = {
        "pat_auroc": pat_auroc[0],
        "pat_auroc_ci": pat_auroc[1],
        "pat_aupr": pat_aupr[0],
        "pat_aupr_ci": pat_aupr[1],
        "pat_f1": pat_f1[0],
        "pat_f1_ci": pat_f1[1],
        "n_patients": len(pat_label),
        "n_abn_patients": int(pat_label.sum()),
    }
    return slice_metrics, patient_metrics


def _format_metric(name, value_ci):
    value, ci = value_ci
    return f"{name}={value * 100:.2f}% (95% CI {ci[0] * 100:.2f}-{ci[1] * 100:.2f}%)"


def format_image_metrics(slice_metrics):
    return (
        "[Slice-Img]  "
        + "  ".join([
            _format_metric("AUROC", (slice_metrics["img_auroc"], slice_metrics["img_auroc_ci"])),
            _format_metric("AUPR", (slice_metrics["img_aupr"], slice_metrics["img_aupr_ci"])),
            _format_metric("F1", (slice_metrics["img_f1"], slice_metrics["img_f1_ci"])),
        ])
    )


def format_eval_protocol_metrics(slice_metrics, patient_metrics):
    lines = [
        format_image_metrics(slice_metrics),
        "[Slice-Px(abn)]  "
        + "  ".join([
            _format_metric("AUROC", (slice_metrics["px_auroc_abn"], slice_metrics["px_auroc_abn_ci"])),
            _format_metric("AUPR", (slice_metrics["px_aupr_abn"], slice_metrics["px_aupr_abn_ci"])),
        ]),
    ]
    p = patient_metrics
    lines.append(
        f"[Patient({p['n_abn_patients']}/{p['n_patients']}abn)]  "
        + "  ".join([
            _format_metric("AUROC", (p["pat_auroc"], p["pat_auroc_ci"])),
            _format_metric("AUPR", (p["pat_aupr"], p["pat_aupr_ci"])),
            _format_metric("F1", (p["pat_f1"], p["pat_f1_ci"])),
        ])
    )
    return "\n".join(lines)


def _colorize_heatmap(anomaly_map):
    norm = anomaly_map.astype(np.float32)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
    four = 4.0 * norm
    r = np.clip(np.minimum(four - 1.5, -four + 4.5), 0, 1)
    g = np.clip(np.minimum(four - 0.5, -four + 3.5), 0, 1)
    b = np.clip(np.minimum(four + 0.5, -four + 2.5), 0, 1)
    return np.uint8(np.stack([r, g, b], axis=-1) * 255)


def save_heatmaps(anomaly_maps, paths, labels, save_dir, heatmap_count=0, save_all=False):
    if not save_all and heatmap_count <= 0:
        return 0
    os.makedirs(save_dir, exist_ok=True)
    limit = len(anomaly_maps) if save_all else min(heatmap_count, len(anomaly_maps))
    for idx in range(limit):
        src_path = str(paths[idx])
        patient = patient_id_from_path(src_path)
        stem = os.path.splitext(os.path.basename(src_path))[0]
        label_name = 'abnormal' if int(labels[idx]) == 1 else 'normal'
        out_name = f'{idx:05d}_{label_name}_{patient}_{stem}.png'
        Image.fromarray(_colorize_heatmap(anomaly_maps[idx])).save(os.path.join(save_dir, out_name))
    return limit


def save_eval_cache_atomic(path, labels, masks, maps, image_scores, paths):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.npz"
    np.savez_compressed(
        tmp_path,
        labels=np.asarray(labels, dtype=np.int32),
        masks=np.asarray(masks, dtype=np.uint8),
        maps=np.asarray(maps, dtype=np.float32),
        image_scores=np.asarray(image_scores, dtype=np.float64),
        paths=np.asarray(paths, dtype=str),
    )
    os.replace(tmp_path, path)


def warn_if_cache_name_mismatch(path):
    name = os.path.basename(str(path)).lower()
    dataset = args.dataset.lower()
    modality = args.modality.lower()
    if dataset not in name or modality not in name:
        print_fn(
            f"Warning: eval cache name does not match current config: "
            f"dataset={dataset}, modality={modality}, cache={path}"
        )


def load_eval_cache_or_memory(path, labels, masks, maps, image_scores, paths, n_boot):
    print_fn(f"Loading eval cache from {path}")
    try:
        with np.load(path, allow_pickle=True) as cache:
            labels = cache["labels"].astype(np.int32)
            masks = cache["masks"]
            maps = cache["maps"]
            image_scores = cache["image_scores"].astype(np.float64)
            paths = cache["paths"]
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        print_fn(f"Warning: failed to reload eval cache ({exc}); using in-memory evaluation arrays")

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)
    print_fn(
        f"Cache loaded: labels={labels.shape}, masks={masks.shape}, "
        f"maps={maps.shape}, bootstrap_iters={int(n_boot)}"
    )
    return labels, masks, maps, image_scores, paths


def my_evaluation_batch(model2, dataloader, device, max_ratio=0, resize_mask=None,
                        heatmap_dir=None, heatmap_count=0, save_all_heatmaps=False,
                        n_boot=500, seed=20260717, hist_bins=16384, eval_cache_path=None):
    model2.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    path_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, paths in dataloader:
            img = img.to(device)

            output2 = model2(img)
            en2, de2 = output2[0], output2[1]

            anomaly_map, _ = cal_anomaly_maps(en2, de2, img.shape[-1])

            if resize_mask is not None:
                anomaly_map = F.interpolate(
                    anomaly_map, size=(resize_mask, resize_mask), mode='bilinear', align_corners=False
                )
                gt = F.interpolate(gt, size=(resize_mask, resize_mask), mode='nearest')

            anomaly_map = gaussian_kernel(anomaly_map)
            anomaly_map = torch.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
            gt = gt.bool()

            gt_list_px.append(gt.cpu())
            pr_list_px.append(anomaly_map.cpu())
            gt_list_sp.append(label.cpu())
            path_list.extend([str(p) for p in paths])

            anomaly_map_flat = anomaly_map.flatten(1)
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map_flat, dim=1)[0]
            else:
                num_top = max(1, int(anomaly_map_flat.shape[1] * max_ratio))
                sp_score = torch.topk(anomaly_map_flat, k=num_top, dim=1)[0].mean(dim=1)

            pr_list_sp.append(sp_score.cpu())

        gt_list_px = torch.cat(gt_list_px, dim=0).squeeze(1).numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0).squeeze(1).numpy()
        gt_list_sp = torch.cat(gt_list_sp).numpy().astype(np.int32)
        pr_list_sp = torch.cat(pr_list_sp).numpy()

        if not np.isfinite(pr_list_px).all():
            finite_vals = pr_list_px[np.isfinite(pr_list_px)]
            fill_max = float(finite_vals.max()) if finite_vals.size > 0 else 0.0
            pr_list_px = np.nan_to_num(pr_list_px, nan=0.0, posinf=fill_max, neginf=0.0)
            print_fn('Warning: non-finite values in pixel anomaly maps — replaced before metric computation')

        if not np.isfinite(pr_list_sp).all():
            finite_vals = pr_list_sp[np.isfinite(pr_list_sp)]
            fill_max = float(finite_vals.max()) if finite_vals.size > 0 else 0.0
            pr_list_sp = np.nan_to_num(pr_list_sp, nan=0.0, posinf=fill_max, neginf=0.0)
            print_fn('Warning: non-finite values in image scores — replaced before metric computation')

        if eval_cache_path is None:
            eval_cache_path = os.path.join(args.output_dir, f"{args.dataset.upper()}_{args.modality.lower()}_eval_cache.npz")
        else:
            warn_if_cache_name_mismatch(eval_cache_path)
        save_eval_cache_atomic(
            eval_cache_path,
            gt_list_sp,
            gt_list_px,
            pr_list_px,
            pr_list_sp,
            path_list,
        )
        gt_list_sp, gt_list_px, pr_list_px, pr_list_sp, path_list = load_eval_cache_or_memory(
            eval_cache_path,
            gt_list_sp,
            gt_list_px,
            pr_list_px,
            pr_list_sp,
            path_list,
            n_boot,
        )

        slice_metrics, patient_metrics = compute_eval_protocol_metrics(
            gt_list_sp,
            gt_list_px,
            pr_list_px,
            pr_list_sp,
            path_list,
            n_boot=n_boot,
            seed=seed,
            hist_bins=hist_bins,
        )
        saved_heatmaps = save_heatmaps(
            pr_list_px,
            path_list,
            gt_list_sp,
            heatmap_dir,
            heatmap_count=heatmap_count,
            save_all=save_all_heatmaps,
        ) if heatmap_dir else 0

    return {
        "slice": slice_metrics,
        "patient": patient_metrics,
        "saved_heatmaps": saved_heatmaps,
    }


def resolve_data_dir():
    if args.data_dir:
        return args.data_dir
    return DATASET_ROOTS[args.dataset.lower()]


def resolve_run_name():
    if args.save_name:
        return args.save_name
    return f"recontrast_{args.dataset.lower()}_{args.modality.lower()}"


def model_filename():
    return f"epoch{args.epochs}_{args.dataset.lower()}_{args.modality.lower()}.pth"


def build_model():
    encoder, bn = wide_resnet50_2(pretrained=True, in_channels=args.replicate_channels)
    decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)
    replace_layers(decoder, torch.nn.ReLU, torch.nn.GELU())
    encoder, bn, decoder = encoder.to(device), bn.to(device), decoder.to(device)
    encoder_freeze = copy.deepcopy(encoder)
    return ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder)


def build_dataloader(file_name, shuffle=False, drop_last=False):
    data = get_detection_dataset(
        data_dir=resolve_data_dir(),
        file_name=file_name,
        modality=args.modality.lower(),
        replicate_channels=args.replicate_channels,
    )
    return data, torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )


def print_evaluation(result, prefix='Test'):
    print_fn(format_eval_protocol_metrics(result["slice"], result["patient"]))
    if result["saved_heatmaps"]:
        print_fn(f"Saved heatmaps: {result['saved_heatmaps']}")


def train():
    setup_seed(args.seed)

    data_dir = resolve_data_dir()
    modality = args.modality.lower()
    replicate_channels = args.replicate_channels

    print_fn(
        f'Dataset: {args.dataset.upper()} ({data_dir}) | '
        f'Modality: {modality.upper()} | Input channels: {replicate_channels} | '
        f'Epochs: {args.epochs}'
    )

    train_data, train_dataloader = build_dataloader('train', shuffle=True, drop_last=True)
    test_data, test_dataloader = build_dataloader('test', shuffle=False)
    print_fn(f'Train samples: {len(train_data)}, Test samples: {len(test_data)}')

    model = build_model()

    optimizer = torch.optim.AdamW(
        [{'params': model.decoder.parameters()},
         {'params': model.bottleneck.parameters()},
         {'params': model.encoder.parameters(), 'lr': args.encoder_lr}],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, amsgrad=True
    )
    total_steps = args.epochs * len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[max(1, int(total_steps * 0.8))], gamma=0.2
    )

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train(encoder_bn_train=False)
        epoch_losses = []
        for img, _, _, _ in train_dataloader:
            img = img.to(device)

            en, de = model(img)

            alpha = min(-3 + 4 * step / max(1, total_steps * 0.1), 1)
            loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.) / 2 + \
                   global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))
            step += 1

        print_fn(f'Epoch {epoch}/{args.epochs} loss={np.mean(epoch_losses):.6f}')

    heatmap_dir = os.path.join(args.output_dir, 'heatmaps')
    result = my_evaluation_batch(
        model,
        test_dataloader,
        device,
        max_ratio=args.max_ratio,
        resize_mask=args.resize_mask,
        heatmap_dir=heatmap_dir,
        heatmap_count=args.heatmap_count,
        save_all_heatmaps=args.save_all_heatmaps,
        n_boot=args.ci_bootstraps,
        seed=args.ci_seed,
        hist_bins=args.hist_bins,
        eval_cache_path=args.eval_cache,
    )
    print_evaluation(result, prefix=f'Epoch {args.epochs} Test')

    save_path = os.path.join(args.output_dir, model_filename())
    torch.save(model.state_dict(), save_path)
    print_fn(f'Model saved: {save_path}')


def test():
    setup_seed(args.seed)
    _, test_dataloader = build_dataloader('test', shuffle=False)
    model = build_model()
    checkpoint = args.checkpoint or os.path.join(args.output_dir, model_filename())
    print_fn(f'Loading checkpoint: {checkpoint}')
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    heatmap_dir = os.path.join(args.output_dir, 'heatmaps_test')
    result = my_evaluation_batch(
        model,
        test_dataloader,
        device,
        max_ratio=args.max_ratio,
        resize_mask=args.resize_mask,
        heatmap_dir=heatmap_dir,
        heatmap_count=args.heatmap_count,
        save_all_heatmaps=args.save_all_heatmaps,
        n_boot=args.ci_bootstraps,
        seed=args.ci_seed,
        hist_bins=args.hist_bins,
        eval_cache_path=args.eval_cache,
    )
    print_evaluation(result, prefix='Direct Test')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='psma', choices=['fdg', 'psma'])
    parser.add_argument('--data_dir', type=str, default=None,
                        help='覆盖 dataset 默认路径；不填时按 --dataset 自动选择')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--save_name', type=str, default=None,
                        help='不填时自动使用 recontrast_<dataset>_<modality>')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='直接测试时的权重路径；不填时自动使用结果目录下 epoch<epochs>_<dataset>_<modality>.pth')
    parser.add_argument('--modality', type=str, default='petct', choices=['pet', 'ct', 'petct'],
                        help='pet/ct 为单模态复制通道；petct 为 [CT,PET,PET] 伪 RGB')
    parser.add_argument('--replicate_channels', type=int, default=3,
                        help='输入通道数，petct 固定建议为3')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_ratio', type=float, default=0.01)
    parser.add_argument('--resize_mask', type=int, default=256)
    parser.add_argument('--ci_bootstraps', type=int, default=500,
                        help='AUPR/F1 95% CI bootstrap iterations，默认500')
    parser.add_argument('--ci_seed', type=int, default=20260717,
                        help='95% CI bootstrap seed')
    parser.add_argument('--hist_bins', type=int, default=16384,
                        help='像素级 histogram bootstrap 的 score bins 数')
    parser.add_argument('--eval_cache', type=str, default=None,
                        help='统一 eval cache .npz 路径；不填时保存到当前结果目录')
    parser.add_argument('--heatmap_count', type=int, default=0,
                        help='默认0不保存；设为正数保存前N张热力图')
    parser.add_argument('--save_all_heatmaps', action='store_true',
                        help='保存测试集全部热力图')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', default='0', type=str,
                        help='例如 0、1、cuda:0；设为 cpu 或 -1 使用 CPU')
    parser.add_argument('--gpu', default=None, type=str,
                        help='兼容旧参数；如果设置，会覆盖 --cuda')
    args = parser.parse_args()

    if args.modality == 'petct' and args.replicate_channels != 3:
        raise ValueError("petct modality uses [CT,PET,PET], so --replicate_channels must be 3")

    if args.gpu is not None:
        args.cuda = args.gpu
    args.save_name = resolve_run_name()
    args.output_dir = os.path.join(args.save_dir, args.save_name)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(args.save_name, args.output_dir)
    print_fn = logger.info

    cuda_config = args.cuda.lower()
    if cuda_config in ('cpu', '-1', 'none') or not torch.cuda.is_available():
        device = 'cpu'
    elif cuda_config.startswith('cuda:'):
        device = cuda_config
    else:
        device = 'cuda:' + args.cuda

    print_fn(f'Device: {device}')
    if args.mode == 'train':
        train()
    else:
        test()
