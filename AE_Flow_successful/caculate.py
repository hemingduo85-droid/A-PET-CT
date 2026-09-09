import os
import glob
import numpy as np
import pandas as pd
import cv2
from medpy.metric import binary
import argparse



def calculate_metrics(gt_path, pred_path, dataset_name, method_name, output_csv):
    """
    Calculate PPV, HD95, ASSD for segmentation results.
    """
    if not os.path.exists(os.path.dirname(output_csv)):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Use PNG files
    # GT pattern: *_mask.png or just *.png
    # Pred pattern: *_pred.png
    gt_files = sorted(glob.glob(os.path.join(gt_path, "*_mask.png")))
    if not gt_files:
        gt_files = sorted(glob.glob(os.path.join(gt_path, "*.png")))

    pred_files = sorted(glob.glob(os.path.join(pred_path, "*_pred.png")))

    print(f"Found {len(gt_files)} GT files and {len(pred_files)} Pred files.")

    # Create dictionary mapping base filename to full path
    gt_map = {}
    for f in gt_files:
        basename = os.path.basename(f)
        # Extract base name
        if basename.endswith('_mask.png'):
            key = basename[:-9]
            gt_map[key] = f
        elif basename.endswith('.png'):
            key = basename[:-4]
            gt_map[key] = f

    pred_map = {}
    for f in pred_files:
        basename = os.path.basename(f)
        if basename.endswith('_pred.png'):
            key = basename[:-9]
            pred_map[key] = f

    common_files = sorted(list(set(gt_map.keys()) & set(pred_map.keys())))
    print(f"Processing {len(common_files)} matched files...")

    results = []

    for filename in common_files:
        try:
            # Load images
            gt = cv2.imread(gt_map[filename], cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_map[filename], cv2.IMREAD_GRAYSCALE)

            if gt is None:
                print(f"Failed to read GT: {gt_map[filename]}")
                continue
            if pred is None:
                print(f"Failed to read Pred: {pred_map[filename]}")
                continue

            # Binarize (assuming 0-255 range, threshold at 127)
            gt = (gt > 127).astype(int)
            pred = (pred > 127).astype(int)

            ppv = np.nan
            dsc = np.nan
            hd95 = np.nan
            assd = np.nan

            if np.sum(gt) == 0 and np.sum(pred) == 0:
                ppv = 1.0
                dsc = 1.0
                hd95 = 0.0
                assd = 0.0
            elif np.sum(gt) == 0:
                ppv = 0.0
                dsc = 0.0
                # hd95, assd remain NaN
            elif np.sum(pred) == 0:
                ppv = 0.0
                dsc = 0.0
                # hd95, assd remain NaN
            else:
                ppv = binary.precision(pred, gt)
                dsc = binary.dc(pred, gt)
                hd95 = binary.hd95(pred, gt)
                assd = binary.assd(pred, gt)

            results.append({
                'filename': filename,
                'dataset': dataset_name,
                'method': method_name,
                'PPV': ppv,
                'DSC': dsc,
                'HD95': hd95,
                'ASSD': assd
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append({
                'filename': filename,
                'dataset': dataset_name,
                'method': method_name,
                'PPV': np.nan,
                'DSC': np.nan,
                'HD95': np.nan,
                'ASSD': np.nan
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")


if __name__ == "__main__":
    GT_PATH = "/data/cyf/codes/TTA/2d_data/test_ct_scans/masks"
    PRED_PATH = "/data/cyf/codes/TTA/nnunet/checkpoints_data25_drop0.2_IN_ct_scans/best_epoch_3_results/png"
    OUTPUT_CSV = "/data/cyf/codes/TTA/nnunet/metrics/CT-ICH_ours_metrics.csv"
    DATASET_NAME = "CT-ICH"
    METHOD_NAME = "Ours"

    calculate_metrics(GT_PATH, PRED_PATH, DATASET_NAME, METHOD_NAME, OUTPUT_CSV)
