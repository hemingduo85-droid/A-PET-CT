"""
VisualAD test script for PET-CT medical anomaly detection.
Evaluates a trained checkpoint on the PET-CT test set.

python test_petct.py --dataset psma --modality petct --epoch 30 --device cuda:5 > psma_ci.log 2>&1 &

python test_petct.py --dataset fdg --modality petct --epoch 30 --device cuda:4 > fdf_ci.log 2>&1 &
"""
import VisualAD_lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataset_petct import PETCTDataset
from utils.logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils.transforms import get_transform
from utils.metrics import compute_metrics
from utils.scoring import reduce_anomaly_map, DEFAULT_TOPK_RATIO
from scipy.ndimage import gaussian_filter
from utils.feature_transform import create_feature_transform
from utils.anomaly_detection import generate_anomaly_map_from_tokens
from utils.petct_config import (
    DEFAULT_DATASET,
    resolve_checkpoint_path,
    resolve_petct_paths,
    select_heatmap_indices,
)
from utils.visualization import visualize_anomaly_results


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def test(args):
    logger = get_logger(args.save_path)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    args.backbone = checkpoint.get("backbone", "ViT-L/14@336px")
    args.image_size = checkpoint.get("image_size", 256)
    args.features_list = checkpoint.get("features_list", [6, 12, 18, 24])

    preprocess, target_transform = get_transform(args)

    model, _ = VisualAD_lib.load(args.backbone, device=device)
    model.eval()
    model.to(device)

    feature_dim = model.visual.embed_dim

    model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
    model.visual.normal_token.data = checkpoint["normal_token"].to(device)
    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None and checkpoint.get("ln_post_weight") is not None:
        ln_post.weight.data = checkpoint["ln_post_weight"].to(device)
        ln_post.bias.data = checkpoint["ln_post_bias"].to(device)

    layer_transforms = nn.ModuleDict()
    if "layer_transforms" in checkpoint:
        for layer_name, state_dict in checkpoint["layer_transforms"].items():
            hidden_dim = state_dict['mlp.0.weight'].shape[0]
            layer_transforms[layer_name] = create_feature_transform(
                transform_type="mlp",
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
                dropout=0.0
            ).to(device)
            layer_transforms[layer_name].load_state_dict(state_dict)
            layer_transforms[layer_name].eval()

    cross_attn = None
    if "cross_attn" in checkpoint:
        from utils.spatial_cross_attention import build_layer_adaptive_cross_attention
        config = checkpoint.get("cross_attn_config", {})
        cross_attn = build_layer_adaptive_cross_attention(
            layers=args.features_list,
            embed_dim=feature_dim,
            num_anchors=config.get("num_anchors", 4),
            dropout=config.get("dropout", 0.1),
            res_scale_init=config.get("res_scale_init", 0.01)
        ).to(device)
        cross_attn.load_state_dict(checkpoint["cross_attn"])
        cross_attn.eval()

    test_data = PETCTDataset(
        root=args.test_data_path,
        transform=preprocess,
        target_transform=target_transform,
        modality=args.modality,
        split='test',
    )
    logger.info(f"Test samples: {len(test_data)} | Modality: {args.modality}")
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=4)

    obj_list = test_data.obj_list
    results = {obj: {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': [], 'img_paths': []}
               for obj in obj_list}

    all_anomaly_maps = []
    all_img_paths = []
    all_original_images = []
    all_gt_masks = []
    all_cls_names = []
    all_anomaly_labels = []

    for items in tqdm(test_dataloader, desc='Testing'):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        results[cls_name]['imgs_masks'].append(gt_mask)
        results[cls_name]['gt_sp'].extend(items['anomaly'].detach().cpu())
        results[cls_name]['img_paths'].append(items['img_path'][0])

        with torch.no_grad():  # noqa: SIM117
            vision_output = model.encode_image(image, args.features_list)
            anomaly_features = vision_output['anomaly_features']
            normal_features = vision_output['normal_features']
            patch_tokens = vision_output['patch_tokens']
            patch_start_idx = vision_output['patch_start_idx']

            patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
            if cross_attn is not None:
                adapted_list = cross_attn(
                    anomaly_features, normal_features,
                    patch_features_list, args.features_list)
                anomaly_features_list = [a['anomaly'] for a in adapted_list]
                normal_features_list = [a['normal'] for a in adapted_list]
            else:
                anomaly_features_list = [anomaly_features] * len(patch_tokens)
                normal_features_list = [normal_features] * len(patch_tokens)

            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_tokens):
                af_norm = F.normalize(anomaly_features_list[idx], dim=1, eps=1e-8)
                nf_norm = F.normalize(normal_features_list[idx], dim=1, eps=1e-8)
                tk = f'layer_{args.features_list[idx]}'
                if tk in layer_transforms:
                    B, N, D = patch_feature.shape
                    patch_feature = layer_transforms[tk](
                        patch_feature.view(-1, D)).view(B, N, D)
                am = generate_anomaly_map_from_tokens(
                    af_norm, nf_norm,
                    patch_feature[:, patch_start_idx:, :],
                    args.image_size)
                anomaly_map_list.append(am)

            final_anomaly_map = torch.stack(anomaly_map_list).sum(dim=0).cpu()
            filtered = gaussian_filter(final_anomaly_map[0].numpy(), sigma=args.sigma)
            final_anomaly_map = torch.from_numpy(filtered).unsqueeze(0)

            results[cls_name]['anomaly_maps'].append(final_anomaly_map)
            all_anomaly_maps.append(final_anomaly_map)
            all_img_paths.append(items['img_path'][0])
            all_original_images.append(items['img'].cpu())
            all_gt_masks.append(gt_mask.cpu())
            all_cls_names.append(cls_name)
            all_anomaly_labels.append(int(items['anomaly'].item()))

    # Image score = top-1% pixel mean (consistent with training evaluate_epoch)
    sample_scores = [
        reduce_anomaly_map(am, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO).item()
        for am in all_anomaly_maps
    ]
    for cls_name in obj_list:
        results[cls_name]['pr_sp'] = np.array(sample_scores, dtype=np.float32)

    metrics = compute_metrics(results, obj_list, logger)

    heatmap_indices = select_heatmap_indices(
        len(all_anomaly_maps), args.heatmap_count, args.save_all_heatmaps)
    if heatmap_indices:
        heatmap_dir = os.path.join(args.save_path, "heatmaps")
        visualize_anomaly_results(
            [all_original_images[i] for i in heatmap_indices],
            [all_anomaly_maps[i] for i in heatmap_indices],
            [all_gt_masks[i] for i in heatmap_indices],
            [sample_scores[i] for i in heatmap_indices],
            [all_cls_names[i] for i in heatmap_indices],
            [all_img_paths[i] for i in heatmap_indices],
            [all_anomaly_labels[i] for i in heatmap_indices],
            f"{args.dataset}_{args.modality}",
            heatmap_dir,
        )
        logger.info(f"Saved heatmaps: {len(heatmap_indices)} -> {heatmap_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VisualAD PET-CT Test")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=["fdg", "psma"],
                        help="PET-CT dataset config to use")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Override named dataset test path")
    parser.add_argument("--save_path", type=str, default='./test_results_petct')
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Optional checkpoint path. If omitted, uses save_root/<dataset>_<modality>/epoch_<epoch>.pth")
    parser.add_argument("--checkpoint_root", type=str, default='./experiments',
                        help="Root used to auto-resolve checkpoint_path")
    parser.add_argument("--epoch", type=int, default=30,
                        help="Checkpoint epoch used when checkpoint_path is omitted")
    parser.add_argument("--modality", type=str, default='petct',
                        choices=['pet', 'ct', 'petct'],
                        help="Input modality: pet / ct / petct")
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--heatmap_count", type=int, default=0,
                        help="Number of heatmaps to save. Default 0 saves none")
    parser.add_argument("--save_all_heatmaps", action="store_true",
                        help="Save heatmaps for the full test set")
    parser.add_argument("--enable_analysis", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # kept for compatibility
    parser.add_argument("--test_dataset", type=str, default='petct')

    args = parser.parse_args()
    _, args.test_data_path = resolve_petct_paths(args.dataset, test_data_path=args.test_data_path)
    args.checkpoint_path = resolve_checkpoint_path(
        args.checkpoint_path, args.checkpoint_root, args.dataset, args.modality, args.epoch)
    os.makedirs(args.save_path, exist_ok=True)
    setup_seed(args.seed)
    test(args)
