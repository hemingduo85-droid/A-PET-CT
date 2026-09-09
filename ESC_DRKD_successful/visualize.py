"""Matplotlib visualization for ESC-DRKD anomaly maps."""

import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np


def normalize_img(img):
    img = np.asarray(img, dtype=np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def save_single_image(data, save_path, cmap="gray", vmin=None, vmax=None):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_detection_overlay(pred_bin, gt_bin):
    overlay = np.zeros((*pred_bin.shape, 3), dtype=np.float32)
    tp = np.logical_and(pred_bin, gt_bin)
    fp = np.logical_and(pred_bin, 1 - gt_bin)
    fn = np.logical_and(1 - pred_bin, gt_bin)
    overlay[fp] = [80 / 255, 128 / 255, 209 / 255]
    overlay[fn] = [51 / 255, 186 / 255, 7 / 255]
    overlay[tp] = [224 / 255, 68 / 255, 19 / 255]
    return overlay


def visualize_sample(
    img_id,
    original_img,
    anomaly_map,
    gt_mask,
    pred_threshold,
    out_dir,
    cmap="jet",
    save_single_images=True,
    high_dice_dir=None,
    high_dice_threshold=0.9,
):
    """Save 6-panel overview + optional single images for one test slice."""
    original_img = normalize_img(original_img)
    anomaly_map = np.asarray(anomaly_map).squeeze()
    gt_mask = np.asarray(gt_mask).squeeze()
    pred_mask = (anomaly_map > pred_threshold).astype(np.float32)
    dice = (2 * np.sum(pred_mask * gt_mask)) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-8)

    sample_folder = os.path.join(out_dir, img_id)
    os.makedirs(sample_folder, exist_ok=True)

    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    axes[0].imshow(original_img, cmap="gray")
    axes[0].set_title("Fused CT+PET")
    axes[0].axis("off")

    im = axes[1].imshow(anomaly_map, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Anomaly heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    diff = np.abs(pred_mask - (gt_mask > 0.5).astype(np.float32))
    axes[2].imshow(diff, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Difference")
    axes[2].axis("off")

    axes[3].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("GT mask")
    axes[3].axis("off")

    axes[4].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[4].set_title(f"Pred mask\nDice={dice:.4f}")
    axes[4].axis("off")

    overlay = create_detection_overlay(pred_mask.astype(np.uint8), (gt_mask > 0.5).astype(np.uint8))
    axes[5].imshow(overlay)
    axes[5].set_title("TP red / FP blue / FN green")
    axes[5].axis("off")

    overview_path = os.path.join(sample_folder, f"{img_id}_overview.png")
    plt.suptitle(f"{img_id} | thr={pred_threshold:.4f} | Dice={dice:.4f}")
    plt.savefig(overview_path, bbox_inches="tight", dpi=150)
    plt.close("all")

    if save_single_images:
        save_single_image(original_img, os.path.join(sample_folder, "original.png"), cmap="gray")
        save_single_image(anomaly_map, os.path.join(sample_folder, "heatmap.png"), cmap=cmap, vmin=0, vmax=1)
        save_single_image(gt_mask, os.path.join(sample_folder, "gt_mask.png"), cmap="gray")
        save_single_image(pred_mask, os.path.join(sample_folder, "pred_mask.png"), cmap="gray")
        save_single_image(diff, os.path.join(sample_folder, "difference.png"), cmap="gray")
        save_single_image(overlay, os.path.join(sample_folder, "overlay.png"))

    if high_dice_dir and dice >= high_dice_threshold:
        high_folder = os.path.join(high_dice_dir, img_id)
        os.makedirs(high_folder, exist_ok=True)
        for name in os.listdir(sample_folder):
            src = os.path.join(sample_folder, name)
            dst = os.path.join(high_folder, name)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy(src, dst)

    return sample_folder, float(dice)


def load_display_image(img_path, img_size):
    if img_path and os.path.isfile(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, (img_size, img_size))
    return np.zeros((img_size, img_size), dtype=np.uint8)
