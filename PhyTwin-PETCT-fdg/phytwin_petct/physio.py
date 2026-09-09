
"""Physiological uptake prior suppression.

The prior is learned exclusively from train/normal PET images. It captures
locations that are frequently among the hottest normal uptake regions and
suppresses anomaly scores there to reduce PSMA physiological false positives.
"""

import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

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
