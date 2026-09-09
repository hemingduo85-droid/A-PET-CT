
"""Physiological uptake prior suppression.

The prior is learned exclusively from train/normal PET images. It captures
locations that are frequently among the hottest normal uptake regions and
suppresses anomaly scores there to reduce PSMA physiological false positives.
"""

import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter, label

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pet_tensor_to_gray(pet_tensor):
    if not torch.is_tensor(pet_tensor):
        pet_tensor = torch.as_tensor(pet_tensor)
    pet = pet_tensor.detach().float().cpu()
    if pet.ndim == 3:
        pet = pet.unsqueeze(0)
    mean = IMAGENET_MEAN.to(pet.dtype)
    std = IMAGENET_STD.to(pet.dtype)
    pet = pet * std + mean
    return pet[:, 0].clamp(0.0, 1.0)


class PhysiologicalUptakePrior:
    def __init__(self, alpha=0.65, uptake_quantile=0.985, sigma=5.0, min_factor=0.25):
        self.alpha = float(alpha)
        self.uptake_quantile = float(uptake_quantile)
        self.sigma = float(sigma)
        self.min_factor = float(min_factor)
        self.prior = None

    def fit_from_pet_maps(self, pet_maps):
        if torch.is_tensor(pet_maps):
            maps = pet_maps.detach().float().cpu()
            if maps.ndim == 4:
                maps = pet_tensor_to_gray(maps)
            elif maps.ndim != 3:
                raise ValueError("pet_maps must be [N,H,W] or [N,3,H,W]")
            maps = maps.numpy()
        else:
            maps = np.asarray(pet_maps, dtype=np.float32)
            if maps.ndim != 3:
                raise ValueError("pet_maps must be [N,H,W]")
        hot = []
        for m in maps:
            threshold = np.quantile(m.reshape(-1), self.uptake_quantile)
            hot.append((m >= threshold).astype(np.float32))
        prior = np.mean(np.stack(hot, axis=0), axis=0)
        if self.sigma > 0:
            prior = gaussian_filter(prior, sigma=self.sigma)
        prior = prior - prior.min()
        prior = prior / (prior.max() + 1e-8)
        self.prior = prior.astype(np.float32)
        return self

    def suppress(self, anomaly_map):
        score = np.asarray(anomaly_map, dtype=np.float32)
        if self.prior is None:
            return score
        if self.prior.shape != score.shape:
            raise ValueError(f"prior shape {self.prior.shape} != score shape {score.shape}")
        factor = np.maximum(1.0 - self.alpha * self.prior, self.min_factor)
        return (score * factor).astype(np.float32)

    def state_dict(self):
        return {
            "prior": self.prior,
            "alpha": self.alpha,
            "uptake_quantile": self.uptake_quantile,
            "sigma": self.sigma,
            "min_factor": self.min_factor,
        }

    def load_state_dict(self, state):
        self.alpha = float(state.get("alpha", self.alpha))
        self.uptake_quantile = float(state.get("uptake_quantile", self.uptake_quantile))
        self.sigma = float(state.get("sigma", self.sigma))
        self.min_factor = float(state.get("min_factor", self.min_factor))
        self.prior = None if state.get("prior") is None else np.asarray(state["prior"], dtype=np.float32)
        return self


def _normalize01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - float(arr.min())
    return arr / (float(arr.max()) + 1e-8)


class AdaptivePhysioSuppressor:
    """Lesion-preserving physiological uptake suppression.

    Static PHYSIO handles population-level normal hot locations. This suppressor
    adds a per-slice physiological estimate for large current PET hotspots, but
    gates it with simple lesion-preserving cues so compact PET/CT-mismatched
    lesions are less likely to be suppressed.
    """

    def __init__(
        self,
        alpha=0.35,
        pet_quantile=0.985,
        min_area=24,
        max_area_fraction=0.12,
        prior_weight=0.40,
        ct_weight=0.25,
        large_weight=0.35,
        lesion_protect=0.70,
        sigma=2.0,
        min_factor=0.30,
    ):
        self.alpha = float(alpha)
        self.pet_quantile = float(pet_quantile)
        self.min_area = int(min_area)
        self.max_area_fraction = float(max_area_fraction)
        self.prior_weight = float(prior_weight)
        self.ct_weight = float(ct_weight)
        self.large_weight = float(large_weight)
        self.lesion_protect = float(lesion_protect)
        self.sigma = float(sigma)
        self.min_factor = float(min_factor)

    def _component_mask(self, anomaly_map, pet_map, ct_map=None, static_prior=None):
        anomaly = _normalize01(anomaly_map)
        pet = _normalize01(pet_map)
        ct = None if ct_map is None else _normalize01(ct_map)
        prior = None if static_prior is None else np.asarray(static_prior, dtype=np.float32)
        threshold = float(np.quantile(pet.reshape(-1), self.pet_quantile))
        cc, num = label(pet >= threshold)
        h, w = pet.shape
        max_area = int(h * w * self.max_area_fraction)
        soft = np.zeros_like(pet, dtype=np.float32)

        for i in range(1, num + 1):
            region = cc == i
            area = int(region.sum())
            if area < self.min_area or area > max_area:
                continue

            area_fraction = area / float(pet.size)
            large_score = min(1.0, area_fraction / 0.015)
            prior_score = 0.0 if prior is None else float(prior[region].mean())

            ring = binary_dilation(region, iterations=6) & ~binary_dilation(region, iterations=1)
            if ring.any():
                pet_contrast = max(0.0, float(pet[region].mean() - pet[ring].mean()))
                ct_contrast = 0.0 if ct is None else abs(float(ct[region].mean() - ct[ring].mean()))
            else:
                pet_contrast = float(pet[region].mean())
                ct_contrast = 0.0
            ct_support = min(1.0, 2.0 * ct_contrast)

            # Small, compact, high-anomaly PET/CT-mismatched components are more
            # likely to be lesions, so suppress them less.
            small_score = float(np.exp(-area / 80.0))
            anomaly_peak = float(anomaly[region].max())
            mismatch_score = max(0.0, min(1.0, 2.0 * (pet_contrast - 0.5 * ct_contrast)))
            lesion_like = min(1.0, 0.45 * small_score + 0.35 * mismatch_score + 0.20 * anomaly_peak)

            physio_like = (
                self.large_weight * large_score
                + self.prior_weight * prior_score
                + self.ct_weight * ct_support
            )
            strength = max(0.0, min(1.0, physio_like * (1.0 - self.lesion_protect * lesion_like)))
            if strength > 0:
                soft[region] = np.maximum(soft[region], strength)

        if soft.max() > 0 and self.sigma > 0:
            soft = gaussian_filter(soft, sigma=self.sigma)
            soft = soft / (float(soft.max()) + 1e-8)
        return soft.astype(np.float32)

    def suppress(self, anomaly_map, pet_map, ct_map=None, static_prior=None):
        anomaly = np.asarray(anomaly_map, dtype=np.float32)
        soft = self._component_mask(anomaly, pet_map, ct_map=ct_map, static_prior=static_prior)
        if soft.max() == 0:
            return anomaly, soft
        factor = np.maximum(1.0 - self.alpha * soft, self.min_factor)
        return (anomaly * factor).astype(np.float32), soft.astype(np.float32)


def save_prior_image(prior, out_path):
    arr = prior.prior if isinstance(prior, PhysiologicalUptakePrior) else prior
    if arr is None:
        return
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    r = np.clip(1.5 - np.abs(4 * arr - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * arr - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * arr - 1), 0, 1)
    img = Image.fromarray((np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)).resize((512, 512))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 511, 24], fill=(0, 0, 0))
    draw.text((8, 6), "Physiological uptake prior from train/normal", fill=(255, 255, 255))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
