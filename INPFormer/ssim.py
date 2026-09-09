import os
import argparse
import cv2
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

"""
修改后的运行命令示例：
python calculate_ssim.py \
  --root_dir /home/wuchangwei/dyx/successful/GatingAno_successful/visualization_results/scans_ct_dice90+ \
  --output_file /data/cyf/codes/TTA/draw_pic/ssim_results_scans_ct.csv \
  --method Ours \
  --dataset scans_ct_dice90+
"""


def calculate_metrics(root_dir, output_file, method="Method", dataset="Dataset"):
    # 遍历根目录下的所有子文件夹
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    results = []

    print(f"Found {len(subfolders)} subfolders in {root_dir}")

    for folder_path in tqdm(subfolders):
        # 获取子文件夹名称（用于标注文件名）
        folder_name = os.path.basename(folder_path)

        # 定义每个子文件夹内的pred和gt文件路径
        pred_path = os.path.join(folder_path, "pred.png")
        gt_path = os.path.join(folder_path, "gt.png")

        # 检查文件是否存在
        if not os.path.exists(pred_path):
            print(f"Warning: pred_mask.png not found in {folder_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"Warning: gt_mask.png not found in {folder_path}")
            continue

        # 读取图像（灰度模式）
        img_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if img_pred is None or img_gt is None:
            print(f"Error reading images in {folder_path}")
            continue

        # 如果尺寸不一致，将pred resize到gt的尺寸
        if img_pred.shape != img_gt.shape:
            img_pred = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 计算SSIM（data_range=255适配uint8图像）
        score, _ = ssim(img_gt, img_pred, full=True, data_range=255)

        # 保存结果：去掉folder_path，列名与截图保持一致
        results.append({
            "filename": folder_name,  # 子文件夹名称作为filename
            "method": method,
            "dataset": dataset,
            "ssim": score
        })

    # 保存结果到CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    # 输出统计信息
    if results:
        avg_ssim = np.mean([item['ssim'] for item in results])
        print(f"Average SSIM across all valid files: {avg_ssim:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate SSIM from subfolders (pred_mask.png & gt_mask.png)")
    # 核心参数改为根目录（替代原来的pred_dir和gt_dir）
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory containing subfolders with pred_mask.png and gt_mask.png")
    parser.add_argument("--output_file", type=str, default="ssim_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--method", type=str, default="Method",
                        help="Method name for labeling data")
    parser.add_argument("--dataset", type=str, default="Dataset",
                        help="Dataset name for labeling data")

    args = parser.parse_args()

    # 自动创建输出文件的目录（如果不存在）
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 调用计算函数（不再需要suffix参数，因为固定读取pred_mask.png/gt_mask.png）
    calculate_metrics(args.root_dir, args.output_file, args.method, args.dataset)