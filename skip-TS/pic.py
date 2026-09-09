import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional


def print_fn(message):
    """打印函数"""
    print(message)


def save_heatmap(
    result,
    save_dir,
    save_name,
    threshold: Optional[float]=None,      # None 表示自适应阈值
    method: str = 'auto',                 # auto|otsu|percentile|fixed
    percentile: float = 99,             # 当使用分位数或回退时使用
    max_area_ratio: float = 0.5,          # 若前景面积超过脑区比例，则自动提高阈值
    min_region: int = 100                  # 去除很小连通域
):
    """
    保存热力图、叠加图、二值掩码，支持自动/自适应阈值策略。
    """
    try:
        from skimage.filters import threshold_otsu
        from skimage import morphology
    except Exception:
        threshold_otsu = None
        morphology = None

    # ========== 目录与输入信息 ==========
    heatmap_dir = os.path.join(save_dir, save_name, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)

    heatmap = result['anomaly_map']
    original_filename = os.path.basename(result['image_path'])
    filename_no_ext = os.path.splitext(original_filename)[0]
    combination_id = result['combination_id']

    # 读取原图（用于叠加与脑区mask）
    img_path = result['image_path']
    original_img = Image.open(img_path).convert('L')
    original_img = original_img.resize((heatmap.shape[1], heatmap.shape[0]))
    original_img_np = np.array(original_img)

    # ========== 脑区mask ==========
    brain_mask = original_img_np > 0

    # 仅脑区热力图
    heatmap_masked = heatmap.copy().astype(np.float32)
    heatmap_masked[~brain_mask] = 0.0

    # ========== 阈值计算 ==========
    thr = threshold
    if thr is None:
        vals = heatmap_masked[brain_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            vals = heatmap_masked.flatten()

        if method in ('auto', 'otsu') and threshold_otsu is not None:
            try:
                thr = float(threshold_otsu(vals))
            except Exception:
                thr = float(np.percentile(vals, percentile))
        elif method == 'percentile':
            thr = float(np.percentile(vals, percentile))
        elif method == 'fixed' and threshold is not None:
            thr = float(threshold)
        else:
            thr = 0.21  # fallback 默认值

    # ========== 初步二值化 ==========
    binary_map = (heatmap_masked > thr).astype(np.uint8)
    binary_map[~brain_mask] = 0

    # ========== 面积自适应修正 ==========
    brain_area = int(brain_mask.sum())
    fg_area = int(binary_map.sum())
    if brain_area > 0 and fg_area > max_area_ratio * brain_area:
        vals = heatmap_masked[brain_mask]
        thr2 = float(np.percentile(vals, 85))  # 自动提高阈值
        binary_map = (heatmap_masked > thr2).astype(np.uint8)
        binary_map[~brain_mask] = 0
        print_fn(f"⚠️ 前景过大，自动提高阈值: {thr:.3f} → {thr2:.3f}")
        thr = thr2

    # ========== 形态学去噪 ==========
    try:
        if morphology is not None:
            # 先闭合，连接邻近区域，再移除小的孤立区域
            binary_map = morphology.binary_closing(binary_map, morphology.disk(3)).astype(np.uint8)
            binary_map = morphology.remove_small_objects(binary_map.astype(bool), min_size=min_region).astype(np.uint8)
    except Exception:
        pass

    # ========== 保存各类图像 ==========
    # 热力图
    plt.figure(figsize=(2.56, 2.56), facecolor='black')
    plt.imshow(heatmap_masked, cmap='jet', vmin=0, vmax=1)
    plt.axis('off')
    heatmap_path = os.path.join(heatmap_dir, f'{filename_no_ext}_combo{combination_id}_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 叠加图
    plt.figure(figsize=(2.56, 2.56))
    plt.imshow(original_img_np, cmap='gray')
    plt.imshow(heatmap_masked, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    plt.axis('off')
    overlay_path = os.path.join(heatmap_dir, f'{filename_no_ext}_combo{combination_id}_overlay.png')
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 二值mask
    plt.figure(figsize=(2.56, 2.56))
    plt.imshow(binary_map, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    binary_path = os.path.join(heatmap_dir, f'{filename_no_ext}_combo{combination_id}_binary.png')
    plt.savefig(binary_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # ========== 打印日志 ==========
    print_fn(f"组合 [{combination_id}] 阈值: {thr:.4f}")
    print_fn(f"热力图已保存: {heatmap_path}")
    print_fn(f"叠加图已保存: {overlay_path}")
    print_fn(f"二值图已保存: {binary_path}")