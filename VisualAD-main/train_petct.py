"""
VisualAD training script for PET-CT medical anomaly detection.
Supports three modalities: pet / ct / petct (pseudo-RGB).

默认训练 30 个 epoch，只在训练结束后测试一次，并保存最后一轮权重。

python train_petct.py --dataset psma --modality petct --device cuda:0

nohup python test_petct.py --dataset psma --modality petct --epoch 30 --device cuda:0 > psma_npz.log 2>&1 &

nohup python test_petct.py --dataset fdg --modality petct --epoch 30 --device cuda:1 > fdg_npz.log 2>&1 &
"""
import VisualAD_lib
import torch
from torch.cuda.amp import GradScaler, autocast
import argparse
import torch.nn.functional as F
from utils.loss import FocalLoss, BinaryDiceLoss, ContrastiveLoss
from dataset_petct import PETCTDataset
from utils.logger import get_logger
from utils.training_utils import (
    print_training_parameters, validate_training_setup, setup_model_training,
    create_optimizer, setup_feature_transforms, check_for_nan,
    compute_segmentation_loss, validate_gradients, save_checkpoint
)
from utils.anomaly_detection import generate_anomaly_map_from_tokens
from utils.scoring import reduce_anomaly_map, DEFAULT_TOPK_RATIO
from utils.metrics import compute_metrics
from tqdm import tqdm
import numpy as np
import os
import random
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from utils.transforms import get_transform
from utils.petct_config import (
    DEFAULT_DATASET,
    checkpoint_name,
    resolve_petct_paths,
    resolve_save_path,
)

torch.use_deterministic_algorithms(True, warn_only=False)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def compute_classification_loss(anomaly_maps_list, labels, device):
    if not anomaly_maps_list:
        return torch.tensor(0.0, device=device)
    final_anomaly_maps = torch.stack(anomaly_maps_list).sum(dim=0)
    seg_scores = reduce_anomaly_map(
        final_anomaly_maps, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO)
    return F.binary_cross_entropy_with_logits(seg_scores, labels.float().to(device))


def _make_anomaly_map_tokens(anomaly_features, normal_features, patch_tokens, image_size):
    """Thin wrapper kept for training loop (does not import from utils to avoid circular deps)."""
    return generate_anomaly_map_from_tokens(
        anomaly_features, normal_features, patch_tokens, image_size)


@torch.no_grad()
def evaluate_epoch(model, cross_attn, layer_transforms, test_dataloader,
                   features_list, image_size, sigma, device, logger, epoch):
    """
    Run one full pass over the test set.
    Image score  : top-1% pixel mean (DEFAULT_TOPK_RATIO = 0.01)
    Best-model   : Image AP (returned)
    Metrics      : logged via compute_metrics, including 95% CI
    """
    model.eval()
    if cross_attn is not None:
        cross_attn.eval()
    for lt in layer_transforms.values():
        lt.eval()

    obj_list = ['petct']
    results = {obj: {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': [], 'img_paths': []}
               for obj in obj_list}
    all_anomaly_maps = []

    for items in tqdm(test_dataloader, desc=f'  Eval [{epoch+1}]', leave=False):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        results[cls_name]['imgs_masks'].append(gt_mask)
        results[cls_name]['gt_sp'].extend(items['anomaly'].detach().cpu())
        results[cls_name]['img_paths'].append(items['img_path'][0])

        vision_output = model.encode_image(image, features_list)
        anomaly_features = vision_output['anomaly_features']
        normal_features  = vision_output['normal_features']
        patch_tokens     = vision_output['patch_tokens']
        patch_start_idx  = vision_output['patch_start_idx']

        patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
        if cross_attn is not None:
            adapted_list = cross_attn(
                anomaly_features, normal_features, patch_features_list, features_list)
            anomaly_features_list = [a['anomaly'] for a in adapted_list]
            normal_features_list  = [a['normal']  for a in adapted_list]
        else:
            anomaly_features_list = [anomaly_features] * len(patch_tokens)
            normal_features_list  = [normal_features]  * len(patch_tokens)

        anomaly_map_list = []
        for idx, patch_feature in enumerate(patch_tokens):
            af_norm = F.normalize(anomaly_features_list[idx], dim=1, eps=1e-8)
            nf_norm = F.normalize(normal_features_list[idx],  dim=1, eps=1e-8)
            tk = f'layer_{features_list[idx]}'
            if tk in layer_transforms:
                B, N, D = patch_feature.shape
                patch_feature = layer_transforms[tk](
                    patch_feature.view(-1, D)).view(B, N, D)
            am = generate_anomaly_map_from_tokens(
                af_norm, nf_norm,
                patch_feature[:, patch_start_idx:, :], image_size)
            anomaly_map_list.append(am)

        # Sum multi-layer maps, apply gaussian smoothing
        final_am = torch.stack(anomaly_map_list).sum(dim=0).cpu()       # [1, H, W]
        filtered = gaussian_filter(final_am[0].numpy(), sigma=sigma)
        final_am = torch.from_numpy(filtered).unsqueeze(0)               # [1, H, W]

        results[cls_name]['anomaly_maps'].append(final_am)
        all_anomaly_maps.append(final_am)

    # Image score = top-1% pixel mean of smoothed anomaly map
    sample_scores = [
        reduce_anomaly_map(am, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO).item()
        for am in all_anomaly_maps
    ]
    for cls_name in obj_list:
        results[cls_name]['pr_sp'] = np.array(sample_scores, dtype=np.float32)

    # Full metrics table + return dict
    metrics = compute_metrics(results, obj_list, logger)

    # Restore training mode
    model.train()
    if cross_attn is not None:
        cross_attn.train()
    for lt in layer_transforms.values():
        lt.train()

    return metrics


def train(args):
    logger = get_logger(args.save_path)
    device = args.device

    model, _ = VisualAD_lib.load(args.backbone, device=device)
    model.train()
    model.to(device)

    preprocess, target_transform = get_transform(args)

    print_training_parameters(args, logger)
    logger.info(f"Modality: {args.modality}")

    validate_training_setup(args, model, device, logger)

    from utils.spatial_cross_attention import build_layer_adaptive_cross_attention
    cross_attn = build_layer_adaptive_cross_attention(
        layers=args.features_list,
        embed_dim=model.visual.embed_dim,
        num_anchors=4, dropout=0.1, res_scale_init=0.01
    ).to(device)
    cross_attn.train()

    # ---- 训练集 ----
    train_data = PETCTDataset(
        root=args.train_data_path,
        transform=preprocess,
        target_transform=target_transform,
        modality=args.modality,
        split='train',
    )
    logger.info(f"Training samples: {len(train_data)}")
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    feature_dim = model.visual.embed_dim
    layer_transforms = setup_feature_transforms(args.features_list, device, feature_dim)
    setup_model_training(model)
    optimizer = create_optimizer(model, layer_transforms, args, cross_attn=cross_attn)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epoch, eta_min=args.learning_rate * 0.1)

    amp_enabled = False
    scaler = GradScaler(enabled=amp_enabled)

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_token_relation = ContrastiveLoss(temperature=0.1, margin=0.5)

    for epoch in tqdm(range(args.epoch), desc='Epochs'):
        # ============================================================
        # 训练阶段
        # ============================================================
        model.train()
        cross_attn.train()
        loss_list, image_loss_list, token_relation_loss_list = [], [], []

        for items in tqdm(train_dataloader, desc=f'Train [{epoch+1}/{args.epoch}]', leave=False):
            image = items['img'].to(device)
            label = items['anomaly']
            gt = items['img_mask'].squeeze(1).to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            def compute_losses():
                vision_output = model.encode_image(image, args.features_list)
                anomaly_features = vision_output['anomaly_features']
                normal_features = vision_output['normal_features']
                patch_tokens = vision_output['patch_tokens']
                patch_start_idx = vision_output['patch_start_idx']

                patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
                adapted_list = cross_attn(
                    anomaly_features, normal_features,
                    patch_features_list, args.features_list)
                anomaly_features_list = [a['anomaly'] for a in adapted_list]
                normal_features_list = [a['normal'] for a in adapted_list]

                final_af = F.normalize(anomaly_features_list[-1], dim=1, eps=1e-8)
                final_nf = F.normalize(normal_features_list[-1], dim=1, eps=1e-8)

                if (check_for_nan(final_af, "norm anomaly feat", logger, epoch) or
                        check_for_nan(final_nf, "norm normal feat", logger, epoch)):
                    return None

                token_rel = loss_token_relation(final_af, final_nf)
                if check_for_nan(token_rel, "contrastive_loss", logger, epoch):
                    return None

                similarity_map_list, anomaly_maps_list = [], []
                for idx_layer, patch_feature in enumerate(patch_tokens):
                    af_norm = F.normalize(anomaly_features_list[idx_layer], dim=1, eps=1e-8)
                    nf_norm = F.normalize(normal_features_list[idx_layer], dim=1, eps=1e-8)
                    tk = f'layer_{args.features_list[idx_layer]}'
                    if tk in layer_transforms:
                        B, N, D = patch_feature.shape
                        patch_feature = layer_transforms[tk](
                            patch_feature.view(-1, D)).view(B, N, D)
                    anomaly_map = generate_anomaly_map_from_tokens(
                        af_norm, nf_norm,
                        patch_feature[:, patch_start_idx:, :], args.image_size)
                    am_sig = torch.sigmoid(anomaly_map)
                    similarity_map_list.append(torch.stack([1 - am_sig, am_sig], dim=1))
                    anomaly_maps_list.append(anomaly_map)

                image_val = compute_classification_loss(anomaly_maps_list, label, device)
                if check_for_nan(image_val, "image_loss", logger, epoch):
                    return None

                seg_val = torch.tensor(0.0, device=device)
                if similarity_map_list and (
                        anomaly_features.requires_grad or normal_features.requires_grad):
                    seg_val = compute_segmentation_loss(
                        similarity_map_list, gt, loss_focal, loss_dice)

                comps = [c for c in [image_val, token_rel, seg_val] if c.requires_grad]
                if not comps:
                    logger.error("No loss component requires gradients!")
                    return None
                return sum(comps), seg_val, image_val, token_rel

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                result = compute_losses()
            if result is None:
                optimizer.zero_grad(set_to_none=True)
                continue

            total_loss, seg_val, image_val, token_rel_val = result

            if amp_enabled:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
            else:
                total_loss.backward()

            if not validate_gradients(model, logger, epoch):
                optimizer.zero_grad(set_to_none=True)
                continue

            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if (check_for_nan(model.visual.anomaly_token, "anomaly_token", logger, epoch) or
                    check_for_nan(model.visual.normal_token, "normal_token", logger, epoch)):
                break

            loss_list.append(float(seg_val.detach()))
            image_loss_list.append(float(image_val.detach()))
            token_relation_loss_list.append(float(token_rel_val.detach()))

        scheduler.step()

        logger.info(
            f'Epoch [{epoch+1}/{args.epoch}] Train - '
            f'seg: {np.mean(loss_list):.4f}  '
            f'cls: {np.mean(image_loss_list):.4f}  '
            f'contra: {np.mean(token_relation_loss_list):.4f}')

    final_path = os.path.join(args.save_path, checkpoint_name(args.epoch))
    save_checkpoint(model, layer_transforms, args, args.epoch, final_path, cross_attn=cross_attn)
    logger.info(f'Training completed. Final checkpoint: {final_path}')

    test_data = PETCTDataset(
        root=args.test_data_path,
        transform=preprocess,
        target_transform=target_transform,
        modality=args.modality,
        split='test',
    )
    logger.info(f"Test samples:     {len(test_data)}")
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=4)

    metrics = evaluate_epoch(
        model, cross_attn, layer_transforms,
        test_dataloader, args.features_list,
        args.image_size, args.sigma, device, logger, args.epoch - 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VisualAD PET-CT Training")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=["fdg", "psma"],
                        help="PET-CT dataset config to use")
    parser.add_argument("--train_data_path", type=str, default=None,
                        help="Override named dataset train path")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Override named dataset test path")
    parser.add_argument("--save_path", type=str, default='./experiments',
                        help="Experiment root; dataset/modality subdir is added by default")
    parser.add_argument("--no_dataset_subdir", action="store_true",
                        help="Use save_path exactly instead of save_path/<dataset>_<modality>")
    parser.add_argument("--modality", type=str, default='petct',
                        choices=['pet', 'ct', 'petct'],
                        help="Input modality: pet / ct / petct")
    parser.add_argument("--backbone", type=str, default="ViT-L/14@336px",
                        choices=VisualAD_lib.available_models())
    parser.add_argument("--features_list", type=int, nargs="*",
                        default=[6, 12, 18, 24])
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sigma", type=int, default=4,
                        help="Gaussian filter sigma for anomaly map smoothing")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", type=str, default="cuda:0")
    # kept for compatibility with print_training_parameters / validate_training_setup
    parser.add_argument("--train_dataset", type=str, default='petct')

    args = parser.parse_args()
    args.train_data_path, args.test_data_path = resolve_petct_paths(
        args.dataset, args.train_data_path, args.test_data_path)
    args.save_path = resolve_save_path(
        args.save_path, args.dataset, args.modality,
        use_dataset_subdir=not args.no_dataset_subdir)
    args.train_dataset = args.dataset
    os.makedirs(args.save_path, exist_ok=True)
    setup_seed(args.seed)
    args.device = torch.device(args.device)
    train(args)
