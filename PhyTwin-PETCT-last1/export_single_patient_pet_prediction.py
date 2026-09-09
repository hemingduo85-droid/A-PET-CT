"""Export CT-to-PET predictions for one PSMA patient.

Default behavior is intentionally focused on the training normal cohort because
the Normal Twin is trained to generate a normal PET reference from CT.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image


DEFAULT_DATA_ROOT = "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA"
DEFAULT_CKPT = "./ckpts/PhyTwin_PETCT_PSMA/BEST_PHYTWIN.pth"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
A0_DISPLAY_ZMAX = 30.0


def resolve_data_root(data_root):
    data_root = os.path.abspath(os.path.expanduser(data_root))
    if os.path.isdir(os.path.join(data_root, "train", "normal")):
        return data_root
    psma = os.path.join(data_root, "PSMA")
    if os.path.isdir(os.path.join(psma, "train", "normal")):
        return psma
    candidates = []
    if os.path.isdir(data_root):
        for name in sorted(os.listdir(data_root)):
            child = os.path.join(data_root, name)
            if os.path.isdir(os.path.join(child, "train", "normal")):
                candidates.append(child)
    if len(candidates) == 1:
        return candidates[0]
    return data_root


def find_patient_dir(data_root, split="train", group="normal", patient=None):
    data_root = resolve_data_root(data_root)
    cohort_dir = os.path.join(data_root, split, group)
    if not os.path.isdir(cohort_dir):
        raise FileNotFoundError(f"Missing cohort directory: {cohort_dir}")

    if patient is not None:
        patient_dir = os.path.join(cohort_dir, patient)
        if not os.path.isdir(patient_dir):
            raise FileNotFoundError(f"Missing patient directory: {patient_dir}")
        return patient_dir

    for name in sorted(os.listdir(cohort_dir)):
        patient_dir = os.path.join(cohort_dir, name)
        if _has_pet_ct_dirs(patient_dir):
            return patient_dir
    raise RuntimeError(f"No patient with pet/ and ct/ folders found under {cohort_dir}")


def select_slice_names(names, num_slices):
    names = sorted(names)
    if num_slices is None or num_slices <= 0 or num_slices >= len(names):
        return names
    start = max(0, (len(names) - num_slices) // 2)
    return names[start:start + num_slices]


def compute_residual_statistics(residuals):
    residuals = np.asarray(residuals, dtype=np.float32)
    if residuals.ndim != 3:
        raise ValueError("residuals must have shape [N,H,W]")
    return {
        "residual_mean": float(residuals.mean()),
        "residual_std": float(max(residuals.std(), 1e-6)),
        "residual_mean_map": residuals.mean(axis=0).astype(np.float32),
        "residual_std_map": residuals.std(axis=0).astype(np.float32),
    }


def positive_zscore_map(score_map, mean, std):
    return np.maximum((np.asarray(score_map, dtype=np.float32) - mean) / (float(std) + 1e-8), 0.0).astype(np.float32)


def compute_a_star_map(
    a0,
    pet_gray,
    ct_gray,
    physio_prior,
    alpha=0.35,
    pet_quantile=0.985,
    anomaly_quantile=0.990,
    min_area=24,
    lesion_protect=0.70,
    sigma=2.0,
):
    from scipy.ndimage import binary_dilation, gaussian_filter, label

    anomaly = np.asarray(a0, dtype=np.float32)
    pet = _normalize01(pet_gray)
    ct = _normalize01(ct_gray)
    prior = None if physio_prior is None else np.asarray(physio_prior, dtype=np.float32)
    anomaly_norm = _normalize01(anomaly)

    pet_threshold = float(np.quantile(pet.reshape(-1), pet_quantile))
    anomaly_threshold = float(np.quantile(anomaly_norm.reshape(-1), anomaly_quantile))
    pet_hot = pet >= pet_threshold
    anomaly_hot = anomaly_norm >= anomaly_threshold
    pet_high_gate = pet > np.quantile(pet.reshape(-1), 0.90)
    if prior is not None:
        prior_gate = prior >= max(0.10, float(np.quantile(prior.reshape(-1), 0.70)))
        candidate = pet_hot | (anomaly_hot & (prior_gate | pet_high_gate))
    else:
        candidate = pet_hot | (anomaly_hot & pet_high_gate)

    cc, num = label(candidate)
    h, w = pet.shape
    max_area = int(h * w * 0.12)
    soft = np.zeros_like(pet, dtype=np.float32)
    for idx in range(1, num + 1):
        region = cc == idx
        area = int(region.sum())
        if area < int(min_area) or area > max_area:
            continue

        area_fraction = area / float(pet.size)
        large_score = min(1.0, area_fraction / 0.015)
        prior_score = 0.0 if prior is None else float(prior[region].mean())
        pet_hot_score = float(pet[region].mean())
        anomaly_peak = float(anomaly_norm[region].max())
        ring = binary_dilation(region, iterations=6) & ~binary_dilation(region, iterations=1)
        if ring.any():
            pet_contrast = max(0.0, float(pet[region].mean() - pet[ring].mean()))
            ct_contrast = abs(float(ct[region].mean() - ct[ring].mean()))
        else:
            pet_contrast = float(pet[region].mean())
            ct_contrast = 0.0
        ct_support = min(1.0, 2.0 * ct_contrast)
        small_score = float(np.exp(-area / 80.0))
        mismatch_score = max(0.0, min(1.0, 2.0 * (pet_contrast - 0.5 * ct_contrast)))
        lesion_like = min(
            1.0,
            0.55 * mismatch_score
            + 0.30 * small_score * (1.0 - prior_score)
            + 0.15 * (1.0 - large_score),
        )
        large_weight = 0.35
        prior_weight = 0.40
        ct_weight = 0.25
        pet_weight = 0.20
        anomaly_weight = 0.15
        physio_like = (
            large_weight * large_score
            + prior_weight * prior_score
            + ct_weight * ct_support
            + pet_weight * pet_hot_score
            + anomaly_weight * anomaly_peak
        ) / (large_weight + prior_weight + ct_weight + pet_weight + anomaly_weight)
        strength = max(0.0, min(1.0, physio_like * (1.0 - float(lesion_protect) * lesion_like)))
        if strength > 0:
            soft[region] = np.maximum(soft[region], strength)

    if soft.max() > 0 and sigma > 0:
        soft = gaussian_filter(soft, sigma=sigma)
        soft = soft / (float(soft.max()) + 1e-8)
    factor = np.maximum(1.0 - float(alpha) * soft, 0.30)
    return (anomaly * factor).astype(np.float32), soft.astype(np.float32)


def _has_pet_ct_dirs(patient_dir):
    return (
        os.path.isdir(os.path.join(patient_dir, "pet"))
        and os.path.isdir(os.path.join(patient_dir, "ct"))
    )


def _load_model(ckpt_path, device):
    import torch

    from phytwin_petct.models.normal_twin import NormalTwinUNet

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    base_channels = int(model_state["enc1.block.0.weight"].shape[0])
    uncertainty = any(key.startswith("logvar_head.") for key in model_state)
    model = NormalTwinUNet(base_channels=base_channels, uncertainty=uncertainty)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model, state


def _transform(image_size):
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    return T.Compose([
        T.Resize((image_size, image_size), InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _gray_to_rgb(gray):
    return Image.merge("RGB", [gray, gray, gray])


def _tensor_to_gray01(tensor):
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
        std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
        arr = arr * std + mean
        arr = arr[0]
    else:
        arr = np.squeeze(arr)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _normalize01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - float(np.nanmin(arr))
    return arr / (float(np.nanmax(arr)) + 1e-8)


def _normalize01_fixed(arr, min_value, max_value):
    arr = np.asarray(arr, dtype=np.float32)
    arr = (arr - float(min_value)) / (float(max_value) - float(min_value) + 1e-8)
    return np.clip(arr, 0.0, 1.0)


def _to_uint8(arr):
    return (_normalize01(arr) * 255.0).clip(0, 255).astype(np.uint8)


def _interpolate_palette(values, colors):
    values = np.asarray(values, dtype=np.float32)
    stops = np.linspace(0.0, 1.0, len(colors), dtype=np.float32)
    palette = np.asarray(colors, dtype=np.float32)
    channels = [
        np.interp(values.reshape(-1), stops, palette[:, channel]).reshape(values.shape)
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1).clip(0, 255).astype(np.uint8)


def _heatmap_rgb(arr):
    arr = _normalize01(arr)
    return _interpolate_palette(
        arr,
        colors=[
            (83, 38, 120),    # low residual: reference purple base
            (79, 82, 159),    # low-mid: blue-violet
            (77, 143, 173),   # mid: muted cyan
            (188, 98, 163),   # high: magenta-pink
            (241, 163, 76),   # peak: warm orange
        ],
    )


def _a0_heatmap_rgb(arr):
    arr = _normalize01_fixed(arr, 0.0, A0_DISPLAY_ZMAX)
    return _interpolate_palette(
        arr,
        colors=[
            (83, 38, 120),    # A0=0: no positive residual evidence
            (79, 82, 159),
            (77, 143, 173),
            (188, 98, 163),
            (241, 163, 76),   # A0>=30: very strong z-score evidence
        ],
    )


def _pet_hot_rgb(arr):
    arr = _normalize01(arr)
    return _interpolate_palette(
        arr,
        colors=[
            (0, 0, 0),
            (90, 0, 0),
            (180, 35, 0),
            (255, 140, 0),
            (255, 235, 120),
            (255, 255, 255),
        ],
    )


def _uncertainty_rgb(arr):
    arr = _normalize01(arr)
    return _interpolate_palette(
        arr,
        colors=[
            (68, 1, 84),      # low uncertainty: dark purple
            (59, 82, 139),    # blue
            (33, 145, 140),   # teal
            (94, 201, 98),    # green
            (253, 231, 37),   # high uncertainty: yellow
        ],
    )


def save_heatmap_image(arr, out_path, title, size=512):
    img = Image.fromarray(_heatmap_rgb(arr)).resize((size, size), resample=Image.NEAREST)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def save_a0_heatmap_image(arr, out_path, size=512):
    img = Image.fromarray(_a0_heatmap_rgb(arr)).resize((size, size), resample=Image.NEAREST)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def save_uncertainty_image(arr, out_path, size=512):
    img = Image.fromarray(_uncertainty_rgb(arr)).resize((size, size), resample=Image.NEAREST)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def save_pet_hot_image(arr, out_path, title=None, size=512):
    img = Image.fromarray(_pet_hot_rgb(arr)).resize((size, size), resample=Image.NEAREST)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def save_binary_mask_image(mask, out_path, size=512):
    arr = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8) * 255
    img = Image.fromarray(arr).resize((size, size), resample=Image.NEAREST)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def save_masked_pet_hot_image(pet_gray, mask, out_path, size=512):
    masked = np.asarray(pet_gray, dtype=np.float32) * (np.asarray(mask, dtype=np.float32) > 0.5)
    save_pet_hot_image(masked, out_path, size=size)


def save_high_uptake_mask_image(mask_frequency, out_path, title="High-uptake masks", size=512):
    freq = _normalize01(mask_frequency)
    r = np.clip(3.0 * freq, 0, 1)
    g = np.clip(2.4 * freq - 0.25, 0, 1)
    b = np.clip(1.8 * freq - 0.45, 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize((size, size), resample=Image.NEAREST)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def _panel(arr, title, size=256, heatmap=False, pet_hot=False, uncertainty=False, a0_heatmap=False):
    if a0_heatmap:
        img = Image.fromarray(_a0_heatmap_rgb(arr)).resize((size, size), resample=Image.NEAREST)
    elif heatmap:
        img = Image.fromarray(_heatmap_rgb(arr)).resize((size, size), resample=Image.NEAREST)
    elif pet_hot:
        img = Image.fromarray(_pet_hot_rgb(arr)).resize((size, size), resample=Image.NEAREST)
    elif uncertainty:
        img = Image.fromarray(_uncertainty_rgb(arr)).resize((size, size), resample=Image.NEAREST)
    else:
        img = Image.fromarray(_to_uint8(arr)).convert("RGB").resize((size, size))
    return img


def _save_comparison(ct, pet, pred, residual, a0, a_star, prior, uncertainty, out_path):
    panels = [
        _panel(ct, "CT"),
        _panel(pet, "Real PET", pet_hot=True),
        _panel(pred, "Pred PET from CT", pet_hot=True),
        _panel(residual, "Residual R", heatmap=True),
    ]
    if uncertainty is not None:
        panels.append(_panel(uncertainty, "Uncertainty logvar", uncertainty=True))
    if a0 is not None:
        panels.append(_panel(a0, "A0 = Z+(R)", a0_heatmap=True))
    if a_star is not None:
        panels.append(_panel(a_star, "A*", a0_heatmap=True))
    if prior is not None:
        panels.append(_panel(prior, "Physio prior Pi"))
    canvas = Image.new("RGB", (256 * len(panels), 256), (255, 255, 255))
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (idx * 256, 0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def _save_pred_pet(pred, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(_to_uint8(pred), mode="L").save(out_path)


def _matched_slice_names(patient_dir):
    pet_dir = os.path.join(patient_dir, "pet")
    ct_dir = os.path.join(patient_dir, "ct")
    return [
        name for name in sorted(os.listdir(pet_dir))
        if name.lower().endswith(".png") and os.path.isfile(os.path.join(ct_dir, name))
    ]


def _iter_patient_dirs(data_root, split="train", group="normal"):
    cohort_dir = os.path.join(resolve_data_root(data_root), split, group)
    if not os.path.isdir(cohort_dir):
        raise FileNotFoundError(f"Missing cohort directory: {cohort_dir}")
    for name in sorted(os.listdir(cohort_dir)):
        patient_dir = os.path.join(cohort_dir, name)
        if _has_pet_ct_dirs(patient_dir):
            yield patient_dir


def _residual_from_tensors(pet, pred, logvar):
    import torch

    residual = (pet - pred).abs().mean(dim=1)
    if logvar is not None:
        sigma = torch.exp(0.5 * logvar.squeeze(1)).clamp_min(0.25)
        residual = residual / sigma
    return residual


def logvar_to_uncertainty_map(logvar):
    if logvar is None:
        return None
    if hasattr(logvar, "detach"):
        arr = logvar.detach().float().cpu().numpy()
    else:
        arr = np.asarray(logvar, dtype=np.float32)
    return np.squeeze(arr).astype(np.float32)


def compute_high_uptake_mask(pet_map, uptake_quantile=0.985):
    pet_map = np.asarray(pet_map, dtype=np.float32)
    threshold = np.quantile(pet_map.reshape(-1), uptake_quantile)
    return (pet_map >= threshold).astype(np.float32)


def compute_high_uptake_mask_frequency(pet_maps, uptake_quantile=0.985):
    maps = np.asarray(pet_maps, dtype=np.float32)
    hot = [compute_high_uptake_mask(pet_map, uptake_quantile=uptake_quantile) for pet_map in maps]
    return np.mean(np.stack(hot, axis=0), axis=0).astype(np.float32)


def _fit_physio_prior(pet_maps, uptake_quantile=0.985, sigma=5.0):
    from scipy.ndimage import gaussian_filter

    prior = compute_high_uptake_mask_frequency(pet_maps, uptake_quantile=uptake_quantile)
    if sigma > 0:
        prior = gaussian_filter(prior, sigma=sigma)
    prior = prior - float(prior.min())
    return (prior / (float(prior.max()) + 1e-8)).astype(np.float32)


def _load_png_pair(patient_dir, name, transform):
    pet_img = Image.open(os.path.join(patient_dir, "pet", name)).convert("L")
    ct_img = Image.open(os.path.join(patient_dir, "ct", name)).convert("L")
    return transform(_gray_to_rgb(pet_img)), transform(_gray_to_rgb(ct_img))


def build_training_normal_statistics(model, args, device):
    import torch

    transform = _transform(args.image_size)
    residuals = []
    pet_maps = []
    used = 0
    with torch.no_grad():
        for patient_dir in _iter_patient_dirs(args.data_root, split="train", group="normal"):
            for name in _matched_slice_names(patient_dir):
                if args.max_stat_slices is not None and used >= args.max_stat_slices:
                    break
                pet, ct = _load_png_pair(patient_dir, name, transform)
                pred, logvar = model(ct.unsqueeze(0).to(device))
                residual = _residual_from_tensors(pet.unsqueeze(0).to(device), pred, logvar)
                residuals.append(residual[0].detach().cpu().numpy().astype(np.float32))
                pet_maps.append(_tensor_to_gray01(pet))
                used += 1
            if args.max_stat_slices is not None and used >= args.max_stat_slices:
                break
    if not residuals:
        raise RuntimeError("No train/normal PET/CT slices found for residual statistics.")

    pet_stack = np.stack(pet_maps, axis=0)
    stats = compute_residual_statistics(np.stack(residuals, axis=0))
    stats["high_uptake_mask_frequency"] = compute_high_uptake_mask_frequency(
        pet_stack,
        uptake_quantile=args.physio_quantile,
    )
    stats["physio_prior"] = _fit_physio_prior(
        pet_stack,
        uptake_quantile=args.physio_quantile,
        sigma=args.physio_sigma,
    )
    stats["num_stat_slices"] = int(used)
    return stats


def save_training_statistics(stats, out_dir):
    from phytwin_petct.physio import save_prior_image

    os.makedirs(out_dir, exist_ok=True)
    _save_pred_pet(stats["residual_mean_map"], os.path.join(out_dir, "residual_mean_map.png"))
    _save_pred_pet(stats["residual_std_map"], os.path.join(out_dir, "residual_std_map.png"))
    save_high_uptake_mask_image(
        stats["high_uptake_mask_frequency"],
        os.path.join(out_dir, "high_uptake_masks.png"),
        title="High-uptake masks from train/normal",
    )
    _save_pred_pet(stats["high_uptake_mask_frequency"], os.path.join(out_dir, "high_uptake_masks_gray.png"))
    save_prior_image(stats["physio_prior"], os.path.join(out_dir, "physio_prior_pi.png"))
    _save_pred_pet(stats["physio_prior"], os.path.join(out_dir, "physio_prior_pi_gray.png"))
    np.save(os.path.join(out_dir, "residual_mean_map.npy"), stats["residual_mean_map"])
    np.save(os.path.join(out_dir, "residual_std_map.npy"), stats["residual_std_map"])
    np.save(os.path.join(out_dir, "high_uptake_masks.npy"), stats["high_uptake_mask_frequency"])
    np.save(os.path.join(out_dir, "physio_prior_pi.npy"), stats["physio_prior"])
    payload = {
        "residual_mean": stats["residual_mean"],
        "residual_std": stats["residual_std"],
        "num_stat_slices": stats["num_stat_slices"],
        "note": "residual_mean/residual_std are global scalars; PNG/NPY maps show spatial train-normal distributions.",
    }
    with open(os.path.join(out_dir, "residual_stats.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def export_patient_predictions(args):
    import torch

    device = torch.device(args.device)
    patient_dir = find_patient_dir(args.data_root, split=args.split, group=args.group, patient=args.patient)
    pet_dir = os.path.join(patient_dir, "pet")
    ct_dir = os.path.join(patient_dir, "ct")
    names = _matched_slice_names(patient_dir)
    selected = select_slice_names(names, None if args.all_slices else args.num_slices)
    if not selected:
        raise RuntimeError(f"No matched PET/CT PNG slices found in {patient_dir}")

    model, state = _load_model(args.ckpt, device)
    transform = _transform(args.image_size)
    patient_name = os.path.basename(patient_dir.rstrip(os.sep))
    comparison_dir = os.path.join(args.output_dir, patient_name, "comparison")
    pred_dir = os.path.join(args.output_dir, patient_name, "pred_pet")
    pred_hot_dir = os.path.join(args.output_dir, patient_name, "pred_pet_hot")
    real_pet_hot_dir = os.path.join(args.output_dir, patient_name, "real_pet_hot")
    high_uptake_mask_dir = os.path.join(args.output_dir, patient_name, "high_uptake_mask_per_slice")
    high_uptake_pet_hot_dir = os.path.join(args.output_dir, patient_name, "high_uptake_pet_hot_per_slice")
    a0_dir = os.path.join(args.output_dir, patient_name, "a0_residual_z")
    a0_heatmap_dir = os.path.join(args.output_dir, patient_name, "a0_initial_anomaly")
    a_star_dir = os.path.join(args.output_dir, patient_name, "a_star_final_anomaly_gray")
    a_star_heatmap_dir = os.path.join(args.output_dir, patient_name, "a_star_final_anomaly")
    residual_gray_dir = os.path.join(args.output_dir, patient_name, "residual_gray")
    residual_heatmap_dir = os.path.join(args.output_dir, patient_name, "residual_heatmap")
    uncertainty_dir = os.path.join(args.output_dir, patient_name, "uncertainty_logvar")
    stats_dir = os.path.join(args.output_dir, "training_normal_statistics")

    stats = build_training_normal_statistics(model, args, device)
    if isinstance(state, dict) and isinstance(state.get("calibration"), dict):
        stats["residual_mean"] = float(state["calibration"].get("residual_mean", stats["residual_mean"]))
        stats["residual_std"] = float(state["calibration"].get("residual_std", stats["residual_std"]))
    if isinstance(state, dict) and isinstance(state.get("prior"), dict) and state["prior"].get("prior") is not None:
        stats["physio_prior"] = np.asarray(state["prior"]["prior"], dtype=np.float32)
    save_training_statistics(stats, stats_dir)

    with torch.no_grad():
        for name in selected:
            pet, ct = _load_png_pair(patient_dir, name, transform)
            pred, logvar = model(ct.unsqueeze(0).to(device))
            residual = _residual_from_tensors(pet.unsqueeze(0).to(device), pred, logvar)[0].detach().cpu().numpy().astype(np.float32)
            uncertainty = logvar_to_uncertainty_map(logvar)
            a0 = positive_zscore_map(residual, stats["residual_mean"], stats["residual_std"])
            pred_gray = _tensor_to_gray01(pred[0])
            pet_gray = _tensor_to_gray01(pet)
            ct_gray = _tensor_to_gray01(ct)
            high_uptake_mask = compute_high_uptake_mask(pet_gray, uptake_quantile=args.physio_quantile)
            a_star, _gated_correction = compute_a_star_map(
                a0,
                pet_gray,
                ct_gray,
                stats["physio_prior"],
            )
            _save_comparison(
                ct_gray,
                pet_gray,
                pred_gray,
                residual,
                a0,
                a_star,
                stats["physio_prior"],
                uncertainty,
                os.path.join(comparison_dir, name),
            )
            _save_pred_pet(pred_gray, os.path.join(pred_dir, name))
            save_pet_hot_image(pred_gray, os.path.join(pred_hot_dir, name), title="Pred PET from CT")
            save_pet_hot_image(pet_gray, os.path.join(real_pet_hot_dir, name), title="Real PET")
            save_binary_mask_image(
                high_uptake_mask,
                os.path.join(high_uptake_mask_dir, name),
            )
            save_masked_pet_hot_image(
                pet_gray,
                high_uptake_mask,
                os.path.join(high_uptake_pet_hot_dir, name),
            )
            _save_pred_pet(a0, os.path.join(a0_dir, name))
            save_a0_heatmap_image(
                a0,
                os.path.join(a0_heatmap_dir, name),
            )
            _save_pred_pet(a_star, os.path.join(a_star_dir, name))
            save_a0_heatmap_image(
                a_star,
                os.path.join(a_star_heatmap_dir, name),
            )
            _save_pred_pet(residual, os.path.join(residual_gray_dir, name))
            save_heatmap_image(
                residual,
                os.path.join(residual_heatmap_dir, name),
                "Residual R",
            )
            if uncertainty is not None:
                save_uncertainty_image(
                    uncertainty,
                    os.path.join(uncertainty_dir, name),
                )

    return (
        patient_dir,
        selected,
        comparison_dir,
        pred_dir,
        pred_hot_dir,
        real_pet_hot_dir,
        high_uptake_mask_dir,
        high_uptake_pet_hot_dir,
        residual_gray_dir,
        residual_heatmap_dir,
        a0_dir,
        a0_heatmap_dir,
        a_star_dir,
        a_star_heatmap_dir,
        uncertainty_dir,
        stats_dir,
    )


def parse_args():
    default_device = "cpu"
    try:
        import torch

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        pass

    parser = argparse.ArgumentParser(description="Export one patient's CT-to-PET prediction images.")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--output_dir", type=str, default="./single_patient_pet_outputs")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--group", type=str, default="normal", choices=["normal", "abnormal"])
    parser.add_argument("--patient", type=str, default=None, help="Patient folder name. Defaults to the first patient.")
    parser.add_argument("--num_slices", type=int, default=12, help="Centered slices to export. Use --all_slices for all.")
    parser.add_argument("--all_slices", action="store_true", default=False)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--physio_quantile", type=float, default=0.985)
    parser.add_argument("--physio_sigma", type=float, default=5.0)
    parser.add_argument("--max_stat_slices", type=int, default=None,
                        help="Limit train/normal slices used for residual/Pi statistics. Default uses all.")
    parser.add_argument("--device", type=str, default=default_device)
    return parser.parse_args()


def main():
    args = parse_args()
    (
        patient_dir,
        selected,
        comparison_dir,
        pred_dir,
        pred_hot_dir,
        real_pet_hot_dir,
        high_uptake_mask_dir,
        high_uptake_pet_hot_dir,
        residual_gray_dir,
        residual_heatmap_dir,
        a0_dir,
        a0_heatmap_dir,
        a_star_dir,
        a_star_heatmap_dir,
        uncertainty_dir,
        stats_dir,
    ) = export_patient_predictions(args)
    print(f"Patient: {patient_dir}")
    print(f"Exported slices: {len(selected)}")
    print(f"Comparison PNGs: {comparison_dir}")
    print(f"Predicted PET gray PNGs: {pred_dir}")
    print(f"Predicted PET hot PNGs: {pred_hot_dir}")
    print(f"Real PET hot PNGs: {real_pet_hot_dir}")
    print(f"Per-slice high-uptake binary masks: {high_uptake_mask_dir}")
    print(f"Per-slice high-uptake PET hot PNGs: {high_uptake_pet_hot_dir}")
    print(f"Residual gray PNGs: {residual_gray_dir}")
    print(f"Residual heatmaps: {residual_heatmap_dir}")
    print(f"A0 gray PNGs: {a0_dir}")
    print(f"A0 initial anomaly heatmaps: {a0_heatmap_dir}")
    print(f"A* gray PNGs: {a_star_dir}")
    print(f"A* final anomaly heatmaps: {a_star_heatmap_dir}")
    print(f"Uncertainty logvar PNGs: {uncertainty_dir}")
    print(f"Training-normal statistics: {stats_dir}")


if __name__ == "__main__":
    main()
