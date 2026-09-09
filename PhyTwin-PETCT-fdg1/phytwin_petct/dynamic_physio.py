
"""Dynamic physiological hotspot suppression.

Static PHYSIO learns where normal subjects are often hot. This module inspects
each test PET slice and suppresses large, very hot blobs that look like current
physiological uptake, especially when they overlap the static prior or lie in
lower-body urinary/bowel regions.
"""

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label


class DynamicPhysioSuppressor:
    def __init__(
        self,
        alpha=0.45,
        pet_quantile=0.985,
        min_area=24,
        max_area_fraction=0.20,
        prior_overlap_threshold=0.15,
        lower_body_start=0.45,
        dilation_iters=2,
        sigma=2.0,
        min_factor=0.20,
    ):
        self.alpha = float(alpha)
        self.pet_quantile = float(pet_quantile)
        self.min_area = int(min_area)
        self.max_area_fraction = float(max_area_fraction)
        self.prior_overlap_threshold = float(prior_overlap_threshold)
        self.lower_body_start = float(lower_body_start)
        self.dilation_iters = int(dilation_iters)
        self.sigma = float(sigma)
        self.min_factor = float(min_factor)

    def _candidate_mask(self, pet_map, static_prior=None):
        pet = np.asarray(pet_map, dtype=np.float32)
        pet = pet - pet.min()
        pet = pet / (pet.max() + 1e-8)
        threshold = float(np.quantile(pet.reshape(-1), self.pet_quantile))
        hot = pet >= threshold
        cc, num = label(hot)
        out = np.zeros_like(pet, dtype=bool)
        h, w = pet.shape
        max_area = int(h * w * self.max_area_fraction)
        prior = None if static_prior is None else np.asarray(static_prior, dtype=np.float32)
        yy = np.arange(h)[:, None] / max(1, h - 1)
        lower_mask = np.broadcast_to(yy >= self.lower_body_start, (h, w))
        for i in range(1, num + 1):
            region = cc == i
            area = int(region.sum())
            if area < self.min_area or area > max_area:
                continue
            prior_overlap = 0.0 if prior is None else float(prior[region].mean())
            lower_overlap = float(lower_mask[region].mean())
            if prior_overlap >= self.prior_overlap_threshold or lower_overlap >= 0.50:
                out[region] = True
        if self.dilation_iters > 0 and out.any():
            out = binary_dilation(out, iterations=self.dilation_iters)
        return out.astype(np.float32)

    def suppress(self, anomaly_map, pet_map, static_prior=None):
        anomaly = np.asarray(anomaly_map, dtype=np.float32)
        mask = self._candidate_mask(pet_map, static_prior=static_prior)
        if mask.max() == 0:
            return anomaly, mask
        if self.sigma > 0:
            soft = gaussian_filter(mask, sigma=self.sigma)
            soft = soft / (soft.max() + 1e-8)
        else:
            soft = mask
        factor = np.maximum(1.0 - self.alpha * soft, self.min_factor)
        return (anomaly * factor).astype(np.float32), soft.astype(np.float32)

    def state_dict(self):
        return {
            "alpha": self.alpha,
            "pet_quantile": self.pet_quantile,
            "min_area": self.min_area,
            "max_area_fraction": self.max_area_fraction,
            "prior_overlap_threshold": self.prior_overlap_threshold,
            "lower_body_start": self.lower_body_start,
            "dilation_iters": self.dilation_iters,
            "sigma": self.sigma,
            "min_factor": self.min_factor,
        }
