
"""Visualization utilities for PhyTwin."""

import os

import numpy as np
from PIL import Image, ImageDraw

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def _to_gray(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 3:
        x = x * IMAGENET_STD + IMAGENET_MEAN
        x = x[0]
    else:
        x = np.squeeze(x)
    x = x - np.nanmin(x)
    return x / (np.nanmax(x) + 1e-8)


def _heatmap(x):
    x = _to_gray(x)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def _mask_outline(mask, size):
    mask = (_to_gray(mask) > 0.5).astype(np.uint8)
    if mask.max() == 0:
        return Image.new("RGBA", size, (0, 0, 0, 0))
    m = Image.fromarray(mask * 255).resize(size, resample=Image.NEAREST)
    arr = np.asarray(m) > 0
    edge = arr.copy()
    edge[1:-1, 1:-1] = arr[1:-1, 1:-1] & ~(arr[:-2, 1:-1] & arr[2:, 1:-1] & arr[1:-1, :-2] & arr[1:-1, 2:])
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    rgba[edge] = np.array([0, 255, 0, 255], dtype=np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _panel(base, title, overlay=None, gt_mask=None, pred_mask=None, mask_fill=False, pred_fill=False):
    if mask_fill:
        gray = (_to_gray(base) > 0.5).astype(np.float32)
        rgb = np.zeros((*gray.shape, 3), dtype=np.float32)
        rgb[..., 1] = gray
    elif pred_fill:
        gray = (_to_gray(base) > 0.5).astype(np.float32)
        rgb = np.zeros((*gray.shape, 3), dtype=np.float32)
        rgb[..., 0] = gray
        rgb[..., 1] = 0.45 * gray
    else:
        gray = _to_gray(base)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
    if overlay is not None:
        hm = _heatmap(overlay)
        alpha = 0.55 * _to_gray(overlay)[..., None]
        rgb = rgb * (1 - alpha) + hm * alpha
    img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((256, 256))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 255, 20], fill=(0, 0, 0))
    draw.text((6, 4), title, fill=(255, 255, 255))
    if gt_mask is not None and not mask_fill:
        img = Image.alpha_composite(img.convert("RGBA"), _mask_outline(gt_mask, img.size)).convert("RGB")
    if pred_mask is not None and not pred_fill:
        outline = _mask_outline(pred_mask, img.size)
        arr = np.asarray(outline).copy()
        arr[..., 0] = np.maximum(arr[..., 0], arr[..., 1])
        arr[..., 1] = (arr[..., 1] * 0.45).astype(np.uint8)
        outline = Image.fromarray(arr, mode="RGBA")
        img = Image.alpha_composite(img.convert("RGBA"), outline).convert("RGB")
    return img


def save_case_visualization(pet, ct, gt_mask, pred_mask, residual_map, final_map, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    panels = [
        _panel(pet, "PET", gt_mask=gt_mask, pred_mask=pred_mask),
        _panel(ct, "CT"),
        _panel(gt_mask, "GT mask", mask_fill=True),
        _panel(pred_mask, "Pred mask", pred_fill=True),
        _panel(pet, "Residual", overlay=residual_map, gt_mask=gt_mask),
        _panel(pet, "PhyTwin", overlay=final_map, gt_mask=gt_mask, pred_mask=pred_mask),
    ]
    canvas = Image.new("RGB", (256 * len(panels), 256), (255, 255, 255))
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * 256, 0))
    canvas.save(out_path)
