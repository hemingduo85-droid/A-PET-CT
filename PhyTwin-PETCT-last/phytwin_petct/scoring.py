
"""Anomaly-map fusion and image-level scoring."""

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_fill_holes, gaussian_filter, label


def positive_zscore_map(score_map, mean, std):
    return np.maximum((np.asarray(score_map, dtype=np.float32) - mean) / (std + 1e-8), 0.0).astype(np.float32)


def fuse_maps(residual_z, memory_z, residual_weight=0.45, memory_weight=0.55, physio_prior=None, sigma=3.0):
    score = residual_weight * residual_z + memory_weight * memory_z
    if physio_prior is not None:
        score = physio_prior.suppress(score)
    if sigma > 0:
        score = gaussian_filter(score, sigma=sigma).astype(np.float32)
    return score.astype(np.float32)


def topk_score(score_map, fraction=0.01):
    flat = np.asarray(score_map, dtype=np.float32).reshape(-1)
    k = max(1, int(len(flat) * fraction))
    return float(np.partition(flat, -k)[-k:].mean())


def lesion_component_score(score_map, physio_prior=None, threshold_quantile=0.995, min_area=4, prior_penalty=0.65):
    score = np.asarray(score_map, dtype=np.float32)
    threshold = float(np.quantile(score.reshape(-1), threshold_quantile))
    binary = score >= threshold
    cc, num = label(binary)
    if num == 0:
        return topk_score(score)
    best = 0.0
    prior = None if physio_prior is None else np.asarray(physio_prior, dtype=np.float32)
    for i in range(1, num + 1):
        region = cc == i
        area = int(region.sum())
        if area < min_area:
            continue
        local = float(score[region].mean()) * np.log1p(area)
        if prior is not None:
            overlap = float(prior[region].mean())
            local *= max(0.05, 1.0 - prior_penalty * overlap)
        best = max(best, local)
    return best if best > 0 else topk_score(score)



def _normalize01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - float(arr.min())
    return arr / (float(arr.max()) + 1e-8)


def _body_and_surface(score, pet, ct):
    body_source = np.zeros_like(score, dtype=np.float32)
    if pet is not None:
        body_source = np.maximum(body_source, gaussian_filter(pet, sigma=2.0))
    if ct is not None:
        body_source = np.maximum(body_source, gaussian_filter(ct, sigma=2.0))
    body = body_source > max(0.03, float(np.quantile(body_source.reshape(-1), 0.20)))
    if not body.any():
        return None, None
    h, w = score.shape
    surface = body & ~binary_erosion(body, iterations=max(2, int(round(min(h, w) * 0.025))))
    return body, surface


def _component_feature(region, score, pet, ct, prior_overlap, boundary_overlap):
    h, w = score.shape
    yy, xx = np.nonzero(region)
    cy = float(yy.mean() / max(1, h - 1))
    cx = float(xx.mean() / max(1, w - 1))
    area_fraction = float(region.sum() / score.size)
    log_area = float(np.log1p(region.sum()) / np.log1p(score.size))
    pet_mean = 0.0 if pet is None else float(pet[region].mean())
    ct_mean = 0.0 if ct is None else float(ct[region].mean())
    peak = float(score[region].max() / (float(score.max()) + 1e-8))
    return np.asarray([cy, cx, log_area, prior_overlap, boundary_overlap, pet_mean, ct_mean, peak], dtype=np.float32)


def _hotspot_memory_factor(feature, memory, penalty=0.35, sigma=0.35):
    if memory is None:
        return 1.0
    mem = np.asarray(memory, dtype=np.float32)
    if mem.size == 0:
        return 1.0
    if mem.ndim != 2:
        return 1.0
    dist = np.linalg.norm(mem - feature.reshape(1, -1), axis=1)
    nearest = float(dist.min())
    similarity = np.exp(-(nearest * nearest) / (2.0 * float(sigma) * float(sigma) + 1e-8))
    return max(0.15, 1.0 - float(penalty) * similarity)


def _lesion_component_values(
    score,
    pet=None,
    ct=None,
    prior=None,
    threshold_quantile=0.992,
    min_area=3,
    prior_penalty=0.85,
    edge_penalty=0.55,
    large_area_penalty=0.70,
    fov_penalty=0.65,
    fov_band_fraction=0.06,
    organ_prior_threshold=0.45,
    organ_area_fraction=0.0035,
    normal_hotspot_memory=None,
    hotspot_penalty=0.35,
    hotspot_sigma=0.35,
):
    threshold = float(np.quantile(score.reshape(-1), threshold_quantile))
    binary = score >= threshold
    cc, num = label(binary)
    if num == 0:
        return []

    _body, surface = _body_and_surface(score, pet, ct) if (pet is not None or ct is not None) else (None, None)
    h, w = score.shape
    by = max(1, int(round(h * float(fov_band_fraction))))
    bx = max(1, int(round(w * float(fov_band_fraction))))
    boundary = np.zeros_like(score, dtype=bool)
    boundary[:by, :] = True
    boundary[-by:, :] = True
    boundary[:, :bx] = True
    boundary[:, -bx:] = True

    values = []
    for i in range(1, num + 1):
        region = cc == i
        area = int(region.sum())
        if area < int(min_area):
            continue
        area_fraction = area / float(score.size)
        mean_score = float(score[region].mean())
        peak_score = float(score[region].max())
        intensity = 0.65 * mean_score + 0.35 * peak_score

        compact_bonus = 1.0 + 0.35 * np.exp(-area / 80.0)
        size_gain = np.log1p(min(area, 180)) / np.log1p(180)

        mismatch_bonus = 1.0
        if pet is not None and ct is not None:
            ring = binary_dilation(region, iterations=6) & ~binary_dilation(region, iterations=1)
            if ring.any():
                pet_contrast = float(pet[region].mean() - pet[ring].mean())
                ct_contrast = abs(float(ct[region].mean() - ct[ring].mean()))
                mismatch = max(0.0, pet_contrast - 0.5 * ct_contrast)
                mismatch_bonus += min(0.45, 1.8 * mismatch)

        penalty = 1.0
        prior_overlap = 0.0
        if prior is not None:
            prior_overlap = float(prior[region].mean())
            penalty *= max(0.08, 1.0 - float(prior_penalty) * prior_overlap)
        if surface is not None and surface.any():
            surface_overlap = float(surface[region].mean())
            penalty *= max(0.20, 1.0 - float(edge_penalty) * surface_overlap)
        if area_fraction > 0.006:
            excess = min(1.0, (area_fraction - 0.006) / 0.05)
            penalty *= max(0.20, 1.0 - float(large_area_penalty) * excess)

        boundary_overlap = float(boundary[region].mean())
        if boundary_overlap > 0.0:
            penalty *= max(0.12, 1.0 - float(fov_penalty) * boundary_overlap)

        # Stronger organ-like suppression for broad hotspots that overlap the
        # normal physiological prior. This mainly targets kidney/bladder/bowel
        # blocks that can be bright but are poor image-level disease evidence.
        if prior_overlap >= float(organ_prior_threshold) and area_fraction >= float(organ_area_fraction):
            organ_excess = min(1.0, area_fraction / max(float(organ_area_fraction), 1e-8))
            penalty *= max(0.08, 1.0 - 0.55 * organ_excess)

        feature = _component_feature(region, score, pet, ct, prior_overlap, boundary_overlap)
        penalty *= _hotspot_memory_factor(
            feature,
            normal_hotspot_memory,
            penalty=hotspot_penalty,
            sigma=hotspot_sigma,
        )

        local = intensity * compact_bonus * (0.55 + 0.45 * size_gain) * mismatch_bonus * penalty
        values.append(float(local))
    return values


def lesion_likeness_score(
    score_map,
    pet_map=None,
    ct_map=None,
    physio_prior=None,
    threshold_quantile=0.992,
    threshold_quantiles=None,
    min_area=3,
    max_components=6,
    prior_penalty=0.85,
    edge_penalty=0.55,
    large_area_penalty=0.70,
    fov_penalty=0.65,
    fov_band_fraction=0.06,
    organ_prior_threshold=0.45,
    organ_area_fraction=0.0035,
    normal_hotspot_memory=None,
    hotspot_penalty=0.35,
    hotspot_sigma=0.35,
):
    """Image-level score that favors lesion-like hotspots over physiologic blobs.

    Multi-scale mode evaluates high-response connected components at several
    quantiles. Lower quantiles catch weak/multifocal lesions; higher quantiles
    preserve strong focal evidence. The continuous anomaly map is unchanged.
    """
    score = np.asarray(score_map, dtype=np.float32)
    if score.size == 0:
        return 0.0
    pet = None if pet_map is None else _normalize01(pet_map)
    ct = None if ct_map is None else _normalize01(ct_map)
    prior = None if physio_prior is None else np.asarray(physio_prior, dtype=np.float32)
    quantiles = threshold_quantiles if threshold_quantiles is not None else [threshold_quantile]
    quantiles = [float(q) for q in quantiles]

    per_scale = []
    all_components = []
    for q in quantiles:
        values = _lesion_component_values(
            score,
            pet=pet,
            ct=ct,
            prior=prior,
            threshold_quantile=q,
            min_area=min_area,
            prior_penalty=prior_penalty,
            edge_penalty=edge_penalty,
            large_area_penalty=large_area_penalty,
            fov_penalty=fov_penalty,
            fov_band_fraction=fov_band_fraction,
            organ_prior_threshold=organ_prior_threshold,
            organ_area_fraction=organ_area_fraction,
            normal_hotspot_memory=normal_hotspot_memory,
            hotspot_penalty=hotspot_penalty,
            hotspot_sigma=hotspot_sigma,
        )
        if not values:
            continue
        values = sorted(values, reverse=True)[:int(max_components)]
        primary = values[0]
        multifocal = 0.0 if len(values) == 1 else float(np.mean(values[1:]))
        per_scale.append(primary + 0.35 * multifocal)
        all_components.extend(values)

    if not per_scale:
        return topk_score(score)
    # Balanced multi-scale fusion: the maximum keeps strong lesions, the mean
    # rewards consistent weak/multifocal evidence across thresholds.
    return float(0.65 * max(per_scale) + 0.35 * np.mean(per_scale))


def extract_normal_hotspot_features(
    score_map,
    pet_map=None,
    ct_map=None,
    physio_prior=None,
    threshold_quantiles=None,
    min_area=3,
    max_components=6,
    fov_band_fraction=0.06,
):
    score = np.asarray(score_map, dtype=np.float32)
    pet = None if pet_map is None else _normalize01(pet_map)
    ct = None if ct_map is None else _normalize01(ct_map)
    prior = None if physio_prior is None else np.asarray(physio_prior, dtype=np.float32)
    quantiles = [0.985, 0.992, 0.997] if threshold_quantiles is None else [float(q) for q in threshold_quantiles]
    h, w = score.shape
    by = max(1, int(round(h * float(fov_band_fraction))))
    bx = max(1, int(round(w * float(fov_band_fraction))))
    boundary = np.zeros_like(score, dtype=bool)
    boundary[:by, :] = True
    boundary[-by:, :] = True
    boundary[:, :bx] = True
    boundary[:, -bx:] = True
    features = []
    for q in quantiles:
        threshold = float(np.quantile(score.reshape(-1), q))
        cc, num = label(score >= threshold)
        candidates = []
        for i in range(1, num + 1):
            region = cc == i
            area = int(region.sum())
            if area < int(min_area):
                continue
            prior_overlap = 0.0 if prior is None else float(prior[region].mean())
            boundary_overlap = float(boundary[region].mean())
            feature = _component_feature(region, score, pet, ct, prior_overlap, boundary_overlap)
            strength = float(score[region].mean())
            candidates.append((strength, feature))
        candidates.sort(key=lambda x: x[0], reverse=True)
        features.extend([f for _strength, f in candidates[:int(max_components)]])
    return features

def predict_mask(score_map, threshold_quantile=0.995, min_area=4):
    score = np.asarray(score_map, dtype=np.float32)
    threshold = float(np.quantile(score.reshape(-1), threshold_quantile))
    binary = score >= threshold
    cc, num = label(binary)
    out = np.zeros_like(binary, dtype=np.float32)
    for i in range(1, num + 1):
        region = cc == i
        if int(region.sum()) >= min_area:
            out[region] = 1.0
    return out


def predict_mask_from_threshold(score_map, threshold, min_area=16, close_iters=1):
    score = np.asarray(score_map, dtype=np.float32)
    binary = score >= float(threshold)
    if close_iters > 0:
        binary = binary_closing(binary, iterations=int(close_iters))
        binary = binary_fill_holes(binary)
    cc, num = label(binary)
    out = np.zeros_like(binary, dtype=np.float32)
    for i in range(1, num + 1):
        region = cc == i
        if int(region.sum()) >= int(min_area):
            out[region] = 1.0
    return out


def normal_calibrated_score(score_map, normal_mean, normal_std, top_fraction=0.01, max_fraction=0.001, physio_prior=None, prior_penalty=0.25):
    top = topk_score(score_map, fraction=top_fraction)
    peak = topk_score(score_map, fraction=max_fraction)
    score = 0.7 * top + 0.3 * peak
    z = (score - float(normal_mean)) / (float(normal_std) + 1e-8)
    if physio_prior is not None:
        high = np.asarray(score_map) >= np.quantile(np.asarray(score_map).reshape(-1), 0.99)
        if high.any():
            z *= max(0.05, 1.0 - prior_penalty * float(np.asarray(physio_prior)[high].mean()))
    return float(z)


def predict_mask_components(
    score_map,
    threshold_quantile=0.97,
    min_area=8,
    max_area_fraction=0.08,
    max_components=3,
    keep_ratio=0.35,
    physio_prior=None,
    prior_penalty=0.75,
):
    """Create a visualization mask by selecting lesion-like connected components.

    This is intentionally separate from metric computation. A raw threshold often
    keeps physiologic hot blobs. We therefore rank connected components by their
    score and penalize components overlapping the normal physiologic prior.
    """
    score = np.asarray(score_map, dtype=np.float32)
    threshold = float(np.quantile(score.reshape(-1), threshold_quantile))
    binary = score >= threshold
    cc, num = label(binary)
    if num == 0:
        return np.zeros_like(score, dtype=np.float32)

    prior = None if physio_prior is None else np.asarray(physio_prior, dtype=np.float32)
    max_area = int(score.size * float(max_area_fraction))
    candidates = []
    for i in range(1, num + 1):
        region = cc == i
        area = int(region.sum())
        if area < int(min_area) or area > max_area:
            continue
        mean_score = float(score[region].mean())
        peak_score = float(score[region].max())
        compact_score = (0.7 * mean_score + 0.3 * peak_score) * np.log1p(area)
        overlap = 0.0 if prior is None else float(prior[region].mean())
        ranked_score = compact_score * max(0.05, 1.0 - float(prior_penalty) * overlap)
        candidates.append((ranked_score, i))

    out = np.zeros_like(score, dtype=np.float32)
    if not candidates:
        return out
    candidates.sort(reverse=True)
    best = candidates[0][0]
    kept = 0
    for ranked_score, idx in candidates:
        if kept >= int(max_components):
            break
        if ranked_score < best * float(keep_ratio):
            break
        out[cc == idx] = 1.0
        kept += 1
    return out
