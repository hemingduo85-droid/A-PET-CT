
"""训练和评估 PhyTwin-PETCT。

这是一个独立的 PET/CT 无监督异常检测框架，不是 UniNet 的网络改版。
训练阶段只使用 train/normal；测试标签和 mask 只用于计算指标和生成可视化。

服务器完整训练命令：
默认会按数据集名保存 checkpoint，例如 PSMA -> ./ckpts/PhyTwin_PETCT_PSMA，FDG -> ./ckpts/PhyTwin_PETCT_FDG，不会互相覆盖。

    nohup python train.py \
      --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
      --epochs 30 \
      --batch_size 8 \
      --image_size 256 \
      --gpu 5 \
      --score_mode lesion_z \
      --save_dir ./saved_results_phytwin_psma \
      > phytwin_psma.log 2>&1 &

FDG 完整训练命令：

    nohup python train.py \
      --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/FDG \
      --epochs 30 \
      --batch_size 8 \
      --image_size 256 \
      --gpu 5 \
      --score_mode lesion_z \
      --save_dir ./saved_results_phytwin_fdg \
      > phytwin_fdg.log 2>&1 &

正常分布校准图像级评分 normal_z 实验命令：

    nohup python train.py \
      --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
      --epochs 30 \
      --batch_size 8 \
      --image_size 256 \
      --gpu 5 \
      --score_mode normal_z \
      --save_dir ./saved_results_phytwin_normalz_psma \
      > phytwin_normalz_psma.log 2>&1 &


多尺度病灶相似性图像级评分 lesion_z 实验命令：
会在静态 PHYSIO 热力图基础上，只改变 image-level 排序方式；多尺度寻找弱小/强热点病灶，奖励小病灶/多灶和 PET-CT 不一致，惩罚生理高摄取、体表边缘伪影和大块器官样热点。

    nohup python train.py \
      --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
      --epochs 30 \
      --batch_size 8 \
      --image_size 256 \
      --gpu 5 \
      --score_mode lesion_z \
      --save_dir ./saved_results_phytwin_lesionz_psma \
      > phytwin_lesionz_psma.log 2>&1 &

加载已有 checkpoint，只重新评估并重新生成可视化：

    python train.py \
      --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
      --load_ckpt \
      --score_mode normal_z \
      --pred_mask_quantile 0.99 \
      --gpu 5 \
      --save_dir ./saved_results_phytwin_eval

固定 checkpoint，不重训，只测试多尺度 lesion_z：
会自动按数据集名加载对应 checkpoint。也可以用 --experiment_name 手动指定。
如果旧 checkpoint 里没有当前图像级后处理校准，脚本会自动用 train/normal 重算校准参数，不会重训 Normal Twin 或重建 memory。

    python train.py \
      --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
      --load_ckpt \
      --score_mode lesion_z \
      --gpu 5 \
      --save_dir ./saved_results_phytwin_lesionz_ms_eval

 python train.py \
  --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
  --load_ckpt \
  --experiment_name PhyTwin_PETCT_PSMA  \
  --score_mode lesion_z \
  --adaptive_physio \
  --adaptive_alpha 0.7 \
  --gpu 3 \
  --save_dir ./saved_results_psma_adaptive_a07  > phytwin_psma_adaptive_a07.log 2>&1 &

python train.py \
  --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
  --load_ckpt \
  --score_mode lesion_z \
  --experiment_name PhyTwin_PETCT_PSMA  \
  --adaptive_physio \
  --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 \
  --gpu 4 \
  --save_dir ./saved_results_psma_adaptive_a050_p070

  python train.py \
  --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50_ganzkoerper/FDG \
  --load_ckpt \
  --score_mode lesion_z \
  --experiment_name PhyTwin_PETCT_FDG  \
  --adaptive_physio \
  --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 \
  --gpu 5 \
  --save_dir ./saved_results_fdg_adaptive_a050_p070

训练
 nohup   python train.py \
  --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/FDG \
  --score_mode lesion_z \
  --experiment_name PhyTwin_PETCT_FDG_old  \
  --adaptive_physio \
  --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 \
  --gpu 5 \
  --save_dir ./saved_results_fdg_adaptive_old  > phytwin_fdg_adaptive_old.log 2>&1 &

  python train.py \
  --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50_ganzkoerper/FDG \
  --load_ckpt \
  --score_mode lesion_z \
  --experiment_name PhyTwin_PETCT_FDG  \
  --adaptive_physio \
  --adaptive_alpha 0.65 \
  --gpu 5 \
  --save_dir ./saved_results_fdg_adaptive_a065  > phytwin_fdg_adaptive_a065.log 2>&1 &

Adaptive PHYSIO 实验命令：
这是对原 PHYSIO 的统一升级，PSMA 和 FDG 都用同一套参数；只重新校准 train/normal 正常分布，不使用异常标签。

    python train.py \
      --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
      --load_ckpt \
      --score_mode lesion_z \
      --adaptive_physio \
      --gpu 5 \
      --save_dir ./saved_results_psma_adaptive_physio

    python train.py \
      --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50_ganzkoerper/FDG \
      --load_ckpt \
      --score_mode lesion_z \
      --adaptive_physio \
      --gpu 5 \
      --save_dir ./saved_results_fdg_adaptive_physio

用于分析 image-level 问题的失败案例挖掘命令：
会额外保存正常测试集中分数最高的假阳性，以及异常测试集中分数最低的低置信/漏检样本。

    python train.py \
      --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
      --load_ckpt \
      --score_mode normal_z \
      --pred_mask_quantile 0.99 \
      --failure_vis_num 20 \
      --gpu 5 \
      --save_dir ./saved_results_phytwin_failure

关键消融实验命令：

    # 去掉生理摄取先验抑制模块
    python train.py --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA --no_physio --gpu 5

    # 改变静态生理摄取抑制强度，0.4 更保守，0.8 更强
    python train.py --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA --physio_alpha 0.4 --gpu 5
    python train.py --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA --physio_alpha 0.8 --gpu 5

    # 动态逐图生理热点抑制仅作为消融，默认不开启
    python train.py --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA --dynamic_physio --gpu 5
"""

import argparse
import os
import re

import numpy as np
import torch

from phytwin_petct.data import build_dataloaders, resolve_data_root
from phytwin_petct.dynamic_physio import DynamicPhysioSuppressor
from phytwin_petct.eval_protocol import compute_metrics, format_metrics
from phytwin_petct.memory import ResidualPatchMemory
from phytwin_petct.models.normal_twin import NormalTwinUNet, normal_twin_loss, residual_map
from phytwin_petct.physio import AdaptivePhysioSuppressor, PhysiologicalUptakePrior, pet_tensor_to_gray, save_prior_image
from phytwin_petct.scoring import (extract_normal_hotspot_features, fuse_maps, lesion_component_score,
                                   lesion_likeness_score, normal_calibrated_score, positive_zscore_map,
                                   predict_mask, topk_score)
from phytwin_petct.utils import get_logger, limited, setup_seed
from phytwin_petct.visualization import save_case_visualization


def parse_float_list(text):
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text]
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def safe_name(text):
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "default"


def parse_args():
    parser = argparse.ArgumentParser(description="PhyTwin-PETCT unsupervised PET/CT anomaly detection")
    parser.add_argument("--data_root", type=str, default="/data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--nll_weight", type=float, default=0.05)
    parser.add_argument("--no_uncertainty", action="store_true", default=False,
                        help="Disable the Normal Twin uncertainty head. Default keeps it to reproduce TwinPatch-PHYSIO.")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--memory_max_patches", type=int, default=30000)
    parser.add_argument("--memory_k", type=int, default=3)
    parser.add_argument("--residual_weight", type=float, default=0.45)
    parser.add_argument("--memory_weight", type=float, default=0.55)
    parser.add_argument("--gaussian_sigma", type=float, default=3.0)
    parser.add_argument("--physio_alpha", type=float, default=0.65)
    parser.add_argument("--physio_quantile", type=float, default=0.985)
    parser.add_argument("--physio_sigma", type=float, default=5.0)
    parser.add_argument("--no_physio", action="store_true", default=False)
    parser.add_argument("--adaptive_physio", action="store_true", default=False,
                        help="Enable lesion-preserving adaptive PHYSIO suppression after static PHYSIO.")
    parser.add_argument("--adaptive_alpha", type=float, default=0.35)
    parser.add_argument("--adaptive_pet_quantile", type=float, default=0.985)
    parser.add_argument("--adaptive_min_area", type=int, default=24)
    parser.add_argument("--adaptive_lesion_protect", type=float, default=0.70)
    parser.add_argument("--dynamic_physio", action="store_true", default=False,
                        help="Enable per-slice dynamic physiological hotspot suppression for ablation. Default is disabled.")
    parser.add_argument("--no_dynamic_physio", action="store_true", default=False,
                        help="Deprecated compatibility flag. Dynamic suppression is disabled unless --dynamic_physio is set.")
    parser.add_argument("--dynamic_alpha", type=float, default=0.45)
    parser.add_argument("--dynamic_pet_quantile", type=float, default=0.985)
    parser.add_argument("--dynamic_min_area", type=int, default=24)
    parser.add_argument("--score_mode", type=str, default="top1pct", choices=["top1pct", "component", "normal_z", "lesion_z"],
                        help="Image-level scoring. lesion_z uses lesion-likeness scoring calibrated by normal training maps.")
    parser.add_argument("--component_quantile", type=float, default=0.995)
    parser.add_argument("--lesion_quantile", type=float, default=0.992)
    parser.add_argument("--lesion_quantiles", type=str, default="0.985,0.992,0.997",
                        help="Comma-separated quantiles for multi-scale lesion_z image scoring.")
    parser.add_argument("--lesion_min_area", type=int, default=3)
    parser.add_argument("--lesion_max_components", type=int, default=6)
    parser.add_argument("--edge_penalty", type=float, default=0.55)
    parser.add_argument("--large_area_penalty", type=float, default=0.70)
    parser.add_argument("--fov_penalty", type=float, default=0.65)
    parser.add_argument("--fov_band_fraction", type=float, default=0.06)
    parser.add_argument("--organ_prior_threshold", type=float, default=0.45)
    parser.add_argument("--organ_area_fraction", type=float, default=0.0035)
    parser.add_argument("--use_hotspot_memory", action="store_true", default=False,
                        help="Enable Normal Hotspot Memory as an ablation. Default is disabled; main method uses FOV-only refinement.")
    parser.add_argument("--hotspot_memory_max", type=int, default=6000)
    parser.add_argument("--hotspot_penalty", type=float, default=0.35)
    parser.add_argument("--hotspot_sigma", type=float, default=0.35)
    parser.add_argument("--component_min_area", type=int, default=4)
    parser.add_argument("--pred_mask_quantile", type=float, default=0.985,
                        help="Per-image high-confidence quantile for visualization masks.")
    parser.add_argument("--pred_mask_min_area", type=int, default=8)
    parser.add_argument("--prior_penalty", type=float, default=0.65)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./saved_results_phytwin")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts",
                        help="Root directory for checkpoints. Final path is ckpt_dir/experiment_name.")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment/checkpoint name. Defaults to PhyTwin_PETCT_<data_root_basename>, e.g. PSMA or FDG.")
    parser.add_argument("--seed", type=int, default=1203)
    parser.add_argument("--load_ckpt", action="store_true", default=False)
    parser.add_argument("--vis_num", type=int, default=12)
    parser.add_argument("--failure_vis_num", type=int, default=12,
                        help="Save top normal false positives and low-scored abnormal cases for image-level debugging.")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_memory_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.uncertainty = not args.no_uncertainty
    args.lesion_quantiles_list = parse_float_list(args.lesion_quantiles)
    args.use_dynamic_physio = bool(args.dynamic_physio and not args.no_dynamic_physio)
    data_name = safe_name(os.path.basename(os.path.abspath(os.path.expanduser(args.data_root))))
    args.method_name = safe_name(args.experiment_name) if args.experiment_name else f"PhyTwin_PETCT_{data_name}"
    return args


def train_normal_twin(model, loader, device, args, logger, ckpt_twin):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    logger.info(
        f"Train Normal Twin | epochs={args.epochs} | image_size={args.image_size} | "
        f"base_channels={args.base_channels} | uncertainty={args.uncertainty} | nll_weight={args.nll_weight}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        sums = {"l1": 0.0, "grad": 0.0, "nll": 0.0}
        for _idx, batch in limited(loader, args.max_train_batches):
            pet = batch["pet"].to(device)
            ct = batch["ct"].to(device)
            pred, logvar = model(ct)
            loss, parts = normal_twin_loss(pred, logvar, pet, nll_weight=args.nll_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for k in sums:
                sums[k] += parts[k]
        mean = float(np.mean(losses)) if losses else 0.0
        denom = max(1, len(losses))
        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs}] loss={mean:.4f} "
            f"l1={sums['l1']/denom:.4f} grad={sums['grad']/denom:.4f} nll={sums['nll']/denom:.4f}"
        )
        if mean < best:
            best = mean
            torch.save({"model": model.state_dict(), "args": vars(args), "train_loss": best}, ckpt_twin)
            logger.info(f"  -> Saved best Normal Twin (train loss={best:.4f})")
    return best


@torch.no_grad()
def collect_train_residuals(model, loader, device, args):
    model.eval()
    residuals = []
    pet_maps = []
    ct_maps = []
    for _idx, batch in limited(loader, args.max_memory_batches):
        pet = batch["pet"].to(device)
        ct = batch["ct"].to(device)
        pred, logvar = model(ct)
        residuals.append(residual_map(pet, pred, logvar if args.uncertainty else None).cpu())
        pet_maps.append(pet_tensor_to_gray(batch["pet"]))
        ct_maps.append(pet_tensor_to_gray(batch["ct"]))
    return torch.cat(residuals, dim=0), torch.cat(pet_maps, dim=0), torch.cat(ct_maps, dim=0)


def compute_lesion_score(final, pet_map, ct_map, prior, args, normal_hotspot_memory=None):
    return lesion_likeness_score(
        final,
        pet_map=pet_map,
        ct_map=ct_map,
        physio_prior=None if prior is None else prior.prior,
        threshold_quantile=args.lesion_quantile,
        threshold_quantiles=args.lesion_quantiles_list,
        min_area=args.lesion_min_area,
        max_components=args.lesion_max_components,
        prior_penalty=args.prior_penalty,
        edge_penalty=args.edge_penalty,
        large_area_penalty=args.large_area_penalty,
        fov_penalty=args.fov_penalty,
        fov_band_fraction=args.fov_band_fraction,
        organ_prior_threshold=args.organ_prior_threshold,
        organ_area_fraction=args.organ_area_fraction,
        normal_hotspot_memory=normal_hotspot_memory,
        hotspot_penalty=args.hotspot_penalty,
        hotspot_sigma=args.hotspot_sigma,
    )


def build_adaptive_physio(args):
    if not args.adaptive_physio:
        return None
    return AdaptivePhysioSuppressor(
        alpha=args.adaptive_alpha,
        pet_quantile=args.adaptive_pet_quantile,
        min_area=args.adaptive_min_area,
        lesion_protect=args.adaptive_lesion_protect,
    )


def apply_adaptive_physio(final, pet_gray, ct_gray, prior, adaptive_suppressor):
    if adaptive_suppressor is None:
        return final, None
    return adaptive_suppressor.suppress(
        final,
        pet_gray,
        ct_map=ct_gray,
        static_prior=None if prior is None else prior.prior,
    )


def image_refinement_config(args):
    return {
        "version": "fov_hotspot_v1" if args.use_hotspot_memory else "fov_v1",
        "lesion_quantiles": [float(x) for x in args.lesion_quantiles_list],
        "lesion_min_area": int(args.lesion_min_area),
        "lesion_max_components": int(args.lesion_max_components),
        "prior_penalty": float(args.prior_penalty),
        "edge_penalty": float(args.edge_penalty),
        "large_area_penalty": float(args.large_area_penalty),
        "fov_penalty": float(args.fov_penalty),
        "fov_band_fraction": float(args.fov_band_fraction),
        "organ_prior_threshold": float(args.organ_prior_threshold),
        "organ_area_fraction": float(args.organ_area_fraction),
        "use_hotspot_memory": bool(args.use_hotspot_memory),
        "hotspot_memory_max": int(args.hotspot_memory_max),
        "hotspot_penalty": float(args.hotspot_penalty),
        "hotspot_sigma": float(args.hotspot_sigma),
        "adaptive_physio": bool(args.adaptive_physio),
        "adaptive_alpha": float(args.adaptive_alpha),
        "adaptive_pet_quantile": float(args.adaptive_pet_quantile),
        "adaptive_min_area": int(args.adaptive_min_area),
        "adaptive_lesion_protect": float(args.adaptive_lesion_protect),
    }


@torch.no_grad()
def calibrate_normal_lesion_scores(model, memory, prior, loader, device, args, calibration, logger=None):
    model.eval()
    memory = memory.to(device)
    scores = []
    adaptive_suppressor = build_adaptive_physio(args)
    if args.use_hotspot_memory:
        hotspot_features = []
        for _idx, batch in limited(loader, args.max_memory_batches):
            pet = batch["pet"].to(device)
            ct = batch["ct"].to(device)
            pred, logvar = model(ct)
            residual = residual_map(pet, pred, logvar if args.uncertainty else None)
            mem_map, _ = memory.score_map(residual)
            for b in range(residual.shape[0]):
                residual_np = residual[b].detach().cpu().numpy().astype(np.float32)
                memory_np = mem_map[b].detach().cpu().numpy().astype(np.float32)
                residual_z = positive_zscore_map(residual_np, calibration["residual_mean"], calibration["residual_std"])
                memory_z = positive_zscore_map(memory_np, calibration["memory_mean"], calibration["memory_std"])
                final = fuse_maps(
                    residual_z, memory_z,
                    residual_weight=args.residual_weight,
                    memory_weight=args.memory_weight,
                    physio_prior=prior,
                    sigma=args.gaussian_sigma,
                )
                pet_gray = pet_tensor_to_gray(batch["pet"])[b].numpy().astype(np.float32)
                ct_gray = pet_tensor_to_gray(batch["ct"])[b].numpy().astype(np.float32)
                final, _adaptive_mask = apply_adaptive_physio(final, pet_gray, ct_gray, prior, adaptive_suppressor)
                hotspot_features.extend(extract_normal_hotspot_features(
                    final,
                    pet_map=pet_gray,
                    ct_map=ct_gray,
                    physio_prior=None if prior is None else prior.prior,
                    threshold_quantiles=args.lesion_quantiles_list,
                    min_area=args.lesion_min_area,
                    max_components=args.lesion_max_components,
                    fov_band_fraction=args.fov_band_fraction,
                ))
        if hotspot_features:
            hotspot_memory = np.stack(hotspot_features, axis=0).astype(np.float32)
            if hotspot_memory.shape[0] > int(args.hotspot_memory_max):
                rng = np.random.default_rng(args.seed)
                idx = rng.choice(hotspot_memory.shape[0], size=int(args.hotspot_memory_max), replace=False)
                hotspot_memory = hotspot_memory[idx]
        else:
            hotspot_memory = np.zeros((0, 8), dtype=np.float32)
        calibration["normal_hotspot_memory"] = hotspot_memory
    else:
        hotspot_memory = np.zeros((0, 8), dtype=np.float32)
        calibration.pop("normal_hotspot_memory", None)

    for _idx, batch in limited(loader, args.max_memory_batches):
        pet = batch["pet"].to(device)
        ct = batch["ct"].to(device)
        pred, logvar = model(ct)
        residual = residual_map(pet, pred, logvar if args.uncertainty else None)
        mem_map, _ = memory.score_map(residual)
        for b in range(residual.shape[0]):
            residual_np = residual[b].detach().cpu().numpy().astype(np.float32)
            memory_np = mem_map[b].detach().cpu().numpy().astype(np.float32)
            residual_z = positive_zscore_map(residual_np, calibration["residual_mean"], calibration["residual_std"])
            memory_z = positive_zscore_map(memory_np, calibration["memory_mean"], calibration["memory_std"])
            final = fuse_maps(
                residual_z, memory_z,
                residual_weight=args.residual_weight,
                memory_weight=args.memory_weight,
                physio_prior=prior,
                sigma=args.gaussian_sigma,
            )
            pet_gray = pet_tensor_to_gray(batch["pet"])[b].numpy().astype(np.float32)
            ct_gray = pet_tensor_to_gray(batch["ct"])[b].numpy().astype(np.float32)
            final, _adaptive_mask = apply_adaptive_physio(final, pet_gray, ct_gray, prior, adaptive_suppressor)
            raw_score = compute_lesion_score(
                final,
                pet_gray,
                ct_gray,
                prior,
                args,
                normal_hotspot_memory=hotspot_memory if args.use_hotspot_memory else None,
            )
            scores.append(raw_score)
    scores = np.asarray(scores, dtype=np.float32)
    calibration["normal_lesion_ms_mean"] = float(scores.mean())
    calibration["normal_lesion_ms_std"] = float(scores.std() + 1e-6)
    calibration["image_refinement_config"] = image_refinement_config(args)
    if logger is not None:
        logger.info(
            f"Calibrated multi-scale lesion_z | normal_lesion_ms_mean={calibration['normal_lesion_ms_mean']:.4f} "
            f"normal_lesion_ms_std={calibration['normal_lesion_ms_std']:.4f} | "
            f"normal_hotspots={len(hotspot_memory) if args.use_hotspot_memory else 'disabled'}"
        )
    return calibration


def build_memory_and_prior(model, loader, device, args, logger):
    residuals, pet_maps, ct_maps = collect_train_residuals(model, loader, device, args)
    calibration = {
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std().clamp_min(1e-6)),
    }
    memory = ResidualPatchMemory(
        patch_size=args.patch_size,
        stride=args.patch_stride,
        max_patches=args.memory_max_patches,
        k=args.memory_k,
        seed=args.seed,
    ).fit(residuals)
    mem_maps, _ = memory.score_map(residuals)
    calibration["memory_mean"] = float(mem_maps.mean())
    calibration["memory_std"] = float(mem_maps.std().clamp_min(1e-6))
    prior = None
    if not args.no_physio:
        prior = PhysiologicalUptakePrior(
            alpha=args.physio_alpha,
            uptake_quantile=args.physio_quantile,
            sigma=args.physio_sigma,
        ).fit_from_pet_maps(pet_maps)

    normal_final_maps = []
    normal_image_scores = []
    normal_lesion_scores = []
    adaptive_suppressor = build_adaptive_physio(args)
    calibration_dynamic = None
    if args.use_dynamic_physio:
        calibration_dynamic = DynamicPhysioSuppressor(
            alpha=args.dynamic_alpha,
            pet_quantile=args.dynamic_pet_quantile,
            min_area=args.dynamic_min_area,
        )
    for i in range(residuals.shape[0]):
        residual_np = residuals[i].numpy().astype(np.float32)
        memory_np = mem_maps[i].numpy().astype(np.float32)
        residual_z = positive_zscore_map(residual_np, calibration["residual_mean"], calibration["residual_std"])
        memory_z = positive_zscore_map(memory_np, calibration["memory_mean"], calibration["memory_std"])
        final = fuse_maps(
            residual_z, memory_z,
            residual_weight=args.residual_weight,
            memory_weight=args.memory_weight,
            physio_prior=prior,
            sigma=args.gaussian_sigma,
        )
        if calibration_dynamic is not None:
            final, _dynamic_mask = calibration_dynamic.suppress(
                final,
                pet_maps[i].numpy().astype(np.float32),
                static_prior=None if prior is None else prior.prior,
            )
        pet_gray = pet_maps[i].numpy().astype(np.float32)
        ct_gray = ct_maps[i].numpy().astype(np.float32)
        final, _adaptive_mask = apply_adaptive_physio(final, pet_gray, ct_gray, prior, adaptive_suppressor)
        normal_final_maps.append(final)
        normal_image_scores.append(0.7 * topk_score(final, fraction=0.01) + 0.3 * topk_score(final, fraction=0.001))
        normal_lesion_scores.append(compute_lesion_score(
            final,
            pet_gray,
            ct_gray,
            prior,
            args,
        ))
    normal_final_flat = np.stack(normal_final_maps, axis=0).reshape(-1)
    normal_image_scores = np.asarray(normal_image_scores, dtype=np.float32)
    normal_lesion_scores = np.asarray(normal_lesion_scores, dtype=np.float32)
    calibration["normal_image_mean"] = float(normal_image_scores.mean())
    calibration["normal_image_std"] = float(normal_image_scores.std() + 1e-6)
    calibration["normal_lesion_mean"] = float(normal_lesion_scores.mean())
    calibration["normal_lesion_std"] = float(normal_lesion_scores.std() + 1e-6)
    calibration["normal_lesion_ms_mean"] = calibration["normal_lesion_mean"]
    calibration["normal_lesion_ms_std"] = calibration["normal_lesion_std"]
    calibration["image_refinement_config"] = image_refinement_config(args)
    calibration["pred_mask_threshold"] = float(np.quantile(normal_final_flat, args.pred_mask_quantile))
    calibration = calibrate_normal_lesion_scores(model, memory, prior, loader, device, args, calibration, logger)

    logger.info(
        "Built normal models | "
        f"normal_residuals={len(residuals)} | memory_patches={memory.memory.shape[0]} | "
        f"residual_mean={calibration['residual_mean']:.4f} residual_std={calibration['residual_std']:.4f} | "
        f"memory_mean={calibration['memory_mean']:.4f} memory_std={calibration['memory_std']:.4f} | "
        f"normal_image_mean={calibration['normal_image_mean']:.4f} normal_image_std={calibration['normal_image_std']:.4f} | "
        f"normal_lesion_mean={calibration['normal_lesion_mean']:.4f} normal_lesion_std={calibration['normal_lesion_std']:.4f} | "
        f"pred_mask_thr={calibration['pred_mask_threshold']:.4f} | "
        f"physio={'enabled' if prior is not None else 'disabled'}"
    )
    return memory, prior, calibration


@torch.no_grad()
def evaluate(model, memory, prior, dynamic_suppressor, calibration, loader, device, args, logger, vis_dir):
    model.eval()
    memory = memory.to(device)
    adaptive_suppressor = build_adaptive_physio(args)
    labels, masks, paths = [], [], []
    final_maps, image_scores = [], []
    records = []
    prior_arr = None if prior is None else prior.prior
    for _idx, batch in limited(loader, args.max_test_batches):
        pet = batch["pet"].to(device)
        ct = batch["ct"].to(device)
        pred, logvar = model(ct)
        residual = residual_map(pet, pred, logvar if args.uncertainty else None)
        mem_map, _ = memory.score_map(residual)
        residual_np = residual[0].detach().cpu().numpy().astype(np.float32)
        memory_np = mem_map[0].numpy().astype(np.float32)
        residual_z = positive_zscore_map(residual_np, calibration["residual_mean"], calibration["residual_std"])
        memory_z = positive_zscore_map(memory_np, calibration["memory_mean"], calibration["memory_std"])
        final = fuse_maps(
            residual_z, memory_z,
            residual_weight=args.residual_weight,
            memory_weight=args.memory_weight,
            physio_prior=prior,
            sigma=args.gaussian_sigma,
        )
        pet_gray = pet_tensor_to_gray(batch["pet"])[0].numpy().astype(np.float32)
        ct_gray = pet_tensor_to_gray(batch["ct"])[0].numpy().astype(np.float32)
        if dynamic_suppressor is not None:
            final, dynamic_mask = dynamic_suppressor.suppress(
                final,
                pet_gray,
                static_prior=None if prior is None else prior.prior,
            )
        else:
            dynamic_mask = None
        final, adaptive_mask = apply_adaptive_physio(final, pet_gray, ct_gray, prior, adaptive_suppressor)
        if args.score_mode == "component":
            image_score = lesion_component_score(
                final,
                physio_prior=prior_arr,
                threshold_quantile=args.component_quantile,
                min_area=args.component_min_area,
                prior_penalty=args.prior_penalty,
            )
        elif args.score_mode == "normal_z":
            image_score = normal_calibrated_score(
                final,
                calibration["normal_image_mean"],
                calibration["normal_image_std"],
                physio_prior=prior_arr,
                prior_penalty=args.prior_penalty,
            )
        elif args.score_mode == "lesion_z":
            raw_score = compute_lesion_score(
                final,
                pet_gray,
                ct_gray,
                prior,
                args,
                normal_hotspot_memory=calibration.get("normal_hotspot_memory") if args.use_hotspot_memory else None,
            )
            normal_mean = calibration.get("normal_lesion_ms_mean", calibration.get("normal_lesion_mean", calibration["normal_image_mean"]))
            normal_std = calibration.get("normal_lesion_ms_std", calibration.get("normal_lesion_std", calibration["normal_image_std"]))
            image_score = (raw_score - normal_mean) / (normal_std + 1e-8)
        else:
            image_score = topk_score(final, fraction=0.01)
        # Pred mask is visualization-only and uses a fair, method-agnostic rule:
        # per-image top-quantile threshold plus small-component removal. Metrics
        # are computed from the continuous anomaly map, not from this mask.
        pred_mask = predict_mask(
            final,
            threshold_quantile=args.pred_mask_quantile,
            min_area=args.pred_mask_min_area,
        )
        label = int(batch["label"].item())
        path = batch["path"][0] if isinstance(batch["path"], (list, tuple)) else str(batch["path"])
        labels.append(label)
        masks.append(batch["mask"].numpy())
        paths.append(path)
        final_maps.append(final)
        image_scores.append(image_score)
        records.append({
            "score": image_score,
            "label": label,
            "path": path,
            "pet": batch["pet"][0].cpu(),
            "ct": batch["ct"][0].cpu(),
            "gt": batch["mask"][0].cpu(),
            "pred": pred_mask,
            "residual": residual_np,
            "final": final,
        })
    masks = np.squeeze(np.concatenate(masks, axis=0), axis=1)
    final_maps = np.stack(final_maps, axis=0)
    slice_metrics, pat_metrics = compute_metrics(labels, masks, final_maps, image_scores, paths)
    if vis_dir and args.vis_num > 0:
        os.makedirs(vis_dir, exist_ok=True)
        top = sorted(records, key=lambda r: r["score"], reverse=True)[:args.vis_num]
        for rank, rec in enumerate(top, 1):
            pid = os.path.basename(os.path.dirname(os.path.dirname(str(rec["path"]))))
            name = os.path.splitext(os.path.basename(str(rec["path"])))[0]
            out = os.path.join(vis_dir, f"top{rank:02d}_label{rec['label']}_{pid}_{name}.png")
            save_case_visualization(rec["pet"], rec["ct"], rec["gt"], rec["pred"], rec["residual"], rec["final"], out)

        import csv
        score_csv = os.path.join(os.path.dirname(vis_dir), "slice_scores.csv")
        with open(score_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "label", "score", "path"])
            for rank, rec in enumerate(sorted(records, key=lambda r: r["score"], reverse=True), 1):
                writer.writerow([rank, rec["label"], f"{rec['score']:.8f}", rec["path"]])

        if args.failure_vis_num > 0:
            fp_dir = os.path.join(os.path.dirname(vis_dir), "failure_top_normal")
            fn_dir = os.path.join(os.path.dirname(vis_dir), "failure_low_abnormal")
            os.makedirs(fp_dir, exist_ok=True)
            os.makedirs(fn_dir, exist_ok=True)
            top_normal = [r for r in sorted(records, key=lambda r: r["score"], reverse=True) if r["label"] == 0][:args.failure_vis_num]
            low_abnormal = [r for r in sorted(records, key=lambda r: r["score"]) if r["label"] == 1][:args.failure_vis_num]
            for rank, rec in enumerate(top_normal, 1):
                pid = os.path.basename(os.path.dirname(os.path.dirname(str(rec["path"]))))
                name = os.path.splitext(os.path.basename(str(rec["path"])))[0]
                out = os.path.join(fp_dir, f"fp{rank:02d}_score{rec['score']:.3f}_{pid}_{name}.png")
                save_case_visualization(rec["pet"], rec["ct"], rec["gt"], rec["pred"], rec["residual"], rec["final"], out)
            for rank, rec in enumerate(low_abnormal, 1):
                pid = os.path.basename(os.path.dirname(os.path.dirname(str(rec["path"]))))
                name = os.path.splitext(os.path.basename(str(rec["path"])))[0]
                out = os.path.join(fn_dir, f"fn{rank:02d}_score{rec['score']:.3f}_{pid}_{name}.png")
                save_case_visualization(rec["pet"], rec["ct"], rec["gt"], rec["pred"], rec["residual"], rec["final"], out)
        logger.info(f"Saved visualizations to {vis_dir}")
        logger.info(f"Saved slice score table to {score_csv}")
    return slice_metrics, pat_metrics


def save_full_checkpoint(path, model, memory, prior, calibration, args):
    torch.save({
        "model": model.state_dict(),
        "memory": memory.state_dict(),
        "prior": None if prior is None else prior.state_dict(),
        "calibration": calibration,
        "args": vars(args),
    }, path)


def load_full_checkpoint(path, model):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    memory = ResidualPatchMemory().load_state_dict(state["memory"])
    prior = None
    if state.get("prior") is not None:
        prior = PhysiologicalUptakePrior().load_state_dict(state["prior"])
    return model, memory, prior, state["calibration"]


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    args.data_root = resolve_data_root(args.data_root)
    if args.experiment_name is None:
        data_name = safe_name(os.path.basename(os.path.abspath(os.path.expanduser(args.data_root))))
        args.method_name = f"PhyTwin_PETCT_{data_name}"
    out_dir = os.path.join(args.save_dir, args.method_name)
    ckpt_dir = os.path.join(args.ckpt_dir, args.method_name)
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger(args.method_name, out_dir)
    logger.info(f"PhyTwin-PETCT | data_root={args.data_root}")
    logger.info(f"Experiment: {args.method_name}")
    logger.info(f"Checkpoint directory: {ckpt_dir}")
    logger.info("Labels and masks are used only for evaluation and visualization.")
    logger.info(f"Adaptive PHYSIO: {'enabled' if args.adaptive_physio else 'disabled'}")

    train_loader, memory_loader, test_loader = build_dataloaders(
        args.data_root, image_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = NormalTwinUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        uncertainty=args.uncertainty,
    ).to(device)
    ckpt_twin = os.path.join(ckpt_dir, "BEST_TWIN.pth")
    ckpt_full = os.path.join(ckpt_dir, "BEST_PHYTWIN.pth")

    if args.load_ckpt:
        logger.info(f"Loading checkpoint from {ckpt_full}")
        model, memory, prior, calibration = load_full_checkpoint(ckpt_full, model)
        model = model.to(device)
        if args.score_mode == "lesion_z" and (
            "normal_lesion_ms_mean" not in calibration
            or calibration.get("image_refinement_config") != image_refinement_config(args)
        ):
            logger.info("Checkpoint image-level refinement calibration is missing or outdated; recalibrating on train/normal without retraining.")
            calibration = calibrate_normal_lesion_scores(model, memory, prior, memory_loader, device, args, calibration, logger)
    else:
        train_normal_twin(model, train_loader, device, args, logger, ckpt_twin)
        state = torch.load(ckpt_twin, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        memory, prior, calibration = build_memory_and_prior(model, memory_loader, device, args, logger)
        if prior is not None:
            save_prior_image(prior, os.path.join(out_dir, "physio_prior.png"))
        save_full_checkpoint(ckpt_full, model, memory, prior, calibration, args)
        logger.info(f"Saved full PhyTwin checkpoint to {ckpt_full}")

    dynamic_suppressor = None
    if args.use_dynamic_physio:
        dynamic_suppressor = DynamicPhysioSuppressor(
            alpha=args.dynamic_alpha,
            pet_quantile=args.dynamic_pet_quantile,
            min_area=args.dynamic_min_area,
        )
    logger.info(f"Dynamic physiological suppression: {'enabled' if dynamic_suppressor is not None else 'disabled'}")
    logger.info("--- Final evaluation ---")
    slice_metrics, pat_metrics = evaluate(model, memory, prior, dynamic_suppressor, calibration, test_loader, device, args, logger, vis_dir)
    logger.info(format_metrics(slice_metrics, pat_metrics))


if __name__ == "__main__":
    main()
