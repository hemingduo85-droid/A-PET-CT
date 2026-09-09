import torch
from scipy.ndimage import gaussian_filter
from PIL import Image
import numpy as np
import torch.nn.functional as F
import random
import os
import tqdm

from eval_protocol import compute_metrics


def _log(message):
    print(message, flush=True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def top1pct_score(anomaly_map_2d: np.ndarray) -> float:
    """Image-level score: mean of top-1% pixels in the anomaly map."""
    flat = anomaly_map_2d.flatten()
    k = max(1, int(len(flat) * 0.01))
    return float(np.partition(flat, -k)[-k:].mean())


def _maybe_first_path(sample_path):
    if isinstance(sample_path, (list, tuple)):
        return sample_path[0]
    return sample_path


def _normalise_map_for_saving(anomaly_map):
    p_min, p_max = float(anomaly_map.min()), float(anomaly_map.max())
    return (anomaly_map - p_min) / (p_max - p_min + 1e-8)


def evaluation(
    encoder,
    decoder,
    res,
    dataloader,
    device,
    score_num=None,
    heatmap_count=0,
    save_all_heatmaps=False,
    heatmap_dir='./results',
    dataset_name='dataset',
    modality_tag='pet',
    input_mode='single',
    bootstrap_iters=500,
    ci_pixel_max_samples=0,
    hist_bins=16384,
):
    """
    Unified evaluation protocol (aligned with InversionAD):
      - Pixel anomaly map: multi-layer cosine reconstruction error (+ Gaussian smooth).
      - Image score: mean of top-1% pixels in the anomaly map (raw, no normalisation).
      - Image AUROC / AP / F1: one score per image vs 0/1 label, raw scores.
      - Pixel AUROC / AP: all pixels flattened, raw scores vs binary mask.
      - Pixel F1: pixel maps globally min-max normalised on the test set, then F1-Max.

    Args:
        score_num: kept for backward compatibility, no longer used.
    """
    decoder.eval()
    all_maps = []
    all_masks = []
    all_img_labels = []
    all_paths = []

    saved_heatmaps = 0
    heatmap_save_name = f"{dataset_name}_{modality_tag}_{input_mode}"

    _log(
        f"evaluation start: samples={len(dataloader.dataset)}, "
        f"bootstrap_iters={bootstrap_iters}, ci_pixel_max_samples={ci_pixel_max_samples}, hist_bins={hist_bins}"
    )

    with torch.no_grad():
        iterator = tqdm.tqdm(dataloader, desc="Eval", ncols=80)
        for sample_idx, (img, mask_tensor, label, sample_path) in enumerate(iterator):
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(inputs[3], inputs[0:3], res)
            anomaly_map, a_map_list = cal_anomaly_map(
                inputs[0:3], outputs, img.shape[-1], amap_mode='a'
            )

            for i in range(len(a_map_list)):
                a_map_list[i] = gaussian_filter(a_map_list[i], sigma=4)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            all_maps.append(anomaly_map.astype(np.float32))
            all_img_labels.append(int(label.numpy()[0] if hasattr(label, 'numpy') else label))
            sample_path_value = _maybe_first_path(sample_path)
            all_paths.append(sample_path_value)

            should_save_heatmap = save_all_heatmaps or saved_heatmaps < heatmap_count
            if should_save_heatmap:
                try:
                    from pic import save_heatmap

                    save_heatmap(
                        result={
                            'anomaly_map': _normalise_map_for_saving(anomaly_map).astype(np.float32),
                            'image_path': sample_path_value,
                            'combination_id': f"{modality_tag}_{input_mode}",
                        },
                        save_dir=heatmap_dir,
                        save_name=heatmap_save_name,
                        threshold=None,
                        method='auto',
                        percentile=99,
                    )
                    saved_heatmaps += 1
                except Exception as e:
                    _log(f"Warning: failed to save heatmap for sample {sample_idx}: {e}")

            gt_mask = mask_tensor.squeeze().cpu().numpy()
            gt_bin = (gt_mask > 0).astype(np.uint8)
            if gt_bin.shape != anomaly_map.shape:
                gt_bin = np.array(
                    Image.fromarray(gt_bin, mode='L').resize(
                        (anomaly_map.shape[1], anomaly_map.shape[0]),
                        resample=Image.NEAREST,
                    )
                )
            all_masks.append(gt_bin)

    if not all_maps:
        raise ValueError(
            '测试集没有任何样本，无法计算指标。请检查 --dataset/--data_path、test 目录、'
            'normal/abnormal 目录，以及 ct/pet 模态目录是否存在且包含图片。'
        )

    _log(f"model inference finished: collected {len(all_maps)} maps")
    all_maps_np = np.stack(all_maps, axis=0)
    all_masks_np = np.stack(all_masks, axis=0)
    img_labels = np.array(all_img_labels, dtype=np.int32)

    img_scores = np.array([top1pct_score(m) for m in all_maps_np])
    _log("image scores finished; computing metrics/95CI")
    slice_metrics, pat_metrics = compute_metrics(
        gt_labels=img_labels,
        gt_masks=all_masks_np,
        anomaly_maps=all_maps_np,
        image_scores=img_scores,
        paths=all_paths,
        bootstrap_iters=bootstrap_iters,
        ci_pixel_max_samples=ci_pixel_max_samples,
        hist_bins=hist_bins,
        print_image_first=True,
    )

    npy_dir = './npy'
    try:
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, 'pretreat_results.npy'), {
            'labels': img_labels,
            'scores': img_scores,
            'paths': np.array(all_paths),
            'slice_metrics': slice_metrics,
            'patient_metrics': pat_metrics,
        })
    except Exception as e:
        _log(f"Warning: failed to save npy results — {e}")

    return slice_metrics, pat_metrics
