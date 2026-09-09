import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
VALID_MODALITIES = {"pet", "ct"}


def parse_modalities(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def modality_output_dir(modalities):
    modalities = sorted(m.lower() for m in modalities)
    if modalities == ["ct"]:
        return "CT"
    if modalities == ["pet"]:
        return "PET"
    if modalities == ["ct", "pet"]:
        return "PETCT"
    raise ValueError(f"Unsupported modality combination: {modalities}")


def resolve_path(path, base=PROJECT_ROOT):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    for candidate in (base / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (base / path).resolve()


def top_percent_mean(score_maps, percent=1.0):
    score_maps = np.asarray(score_maps, dtype=np.float32)
    if score_maps.ndim == 2:
        score_maps = score_maps[None]
    flat = score_maps.reshape(score_maps.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * percent / 100.0)))
    return np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)


def cache_output_paths(method, tracer, modalities):
    run_tag = f"{tracer.upper()}_{'_'.join(modalities)}"
    cache_dir = PROJECT_ROOT / "paper_figures" / method / "plot_cache" / run_tag
    out_dir = PROJECT_ROOT / "paper_figures" / method / "full_evaluation" / run_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "test_cache.npz", cache_dir / "metrics.json", out_dir


def save_cache(cache_file, method, tracer, modalities, filenames, labels, scores, maps, masks, topk_percent):
    np.savez_compressed(
        cache_file,
        schema_version=2,
        method=method,
        tracer=tracer.upper(),
        modalities=",".join(modalities),
        score_source="topk_from_map",
        topk_percent=float(topk_percent),
        filenames=np.asarray(filenames, dtype=object),
        slice_labels=np.asarray(labels, dtype=np.int64),
        slice_scores=np.asarray(scores, dtype=np.float32),
        pixel_maps=np.asarray(maps, dtype=np.float32),
        pixel_masks=np.asarray(masks, dtype=np.uint8),
    )


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items() if not str(key).startswith("_")}
    return value


def json_ready_metrics(metrics):
    return _json_ready(metrics)


def patient_id_from_path(path):
    path_str = str(path)
    if "__" in path_str:
        parts = os.path.basename(path_str).split("__")
        if len(parts) >= 2:
            return parts[1]
    return os.path.basename(os.path.dirname(os.path.dirname(path_str)))


def _clip_ci(lo, hi):
    return (float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0)))


def _safe_auroc(y_true, y_score):
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _bootstrap_ci(y_true, y_score, metric_fn, bootstrap_iters=500, seed=42):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    point = float(metric_fn(y_true, y_score)) if y_true.size else 0.0
    if y_true.size == 0 or len(np.unique(y_true)) < 2 or bootstrap_iters <= 0:
        return (point, point)

    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(int(bootstrap_iters)):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(float(metric_fn(y_true[idx], y_score[idx])))
    if not values:
        return (point, point)
    return _clip_ci(*np.percentile(values, [2.5, 97.5]))


def _metrics_from_hist(pos_hist, neg_hist):
    tp = pos_hist[::-1].astype(np.float64, copy=False)
    fp = neg_hist[::-1].astype(np.float64, copy=False)
    total_pos = float(tp.sum())
    total_neg = float(fp.sum())
    if total_pos <= 0.0 or total_neg <= 0.0:
        return 0.0, 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    tpr = cum_tp / total_pos
    fpr = cum_fp / total_neg
    integrate = getattr(np, "trapezoid", np.trapz)
    auroc = float(integrate(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    denom = cum_tp + cum_fp
    precision = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)
    aupr = float(np.sum(precision * (tp / total_pos)))
    return auroc, aupr


def _pixel_slice_bootstrap_metrics(
    gt_labels,
    gt_masks,
    maps,
    bootstrap_iters=500,
    seed=42,
    bins=16384,
    exact_auroc=None,
    exact_aupr=None,
    progress_callback=None,
):
    keep = np.asarray(gt_labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    masks = np.asarray(gt_masks)
    scores = np.asarray(maps)
    if len(abn_idx) == 0:
        return (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0)), (0.0, 0.0)

    abn_scores = scores[keep]
    score_min = float(np.min(abn_scores))
    score_max = float(np.max(abn_scores))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    scale = (bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        slice_scores = scores[idx].reshape(-1)
        slice_mask = masks[idx].reshape(-1).astype(bool)
        bin_idx = np.floor((slice_scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[slice_mask], minlength=bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~slice_mask], minlength=bins).astype(np.uint32)

    binned_auroc, binned_aupr = _metrics_from_hist(pos_hists.sum(axis=0), neg_hists.sum(axis=0))
    auroc_value = binned_auroc if exact_auroc is None else float(exact_auroc)
    aupr_value = binned_aupr if exact_aupr is None else float(exact_aupr)
    if bootstrap_iters <= 0:
        return (auroc_value, (auroc_value, auroc_value)), (aupr_value, (aupr_value, aupr_value)), (binned_auroc, binned_aupr)

    rng = np.random.default_rng(seed)
    aurocs = []
    auprs = []
    n = len(abn_idx)
    for _ in range(int(bootstrap_iters)):
        step = len(aurocs) + 1
        idx = rng.integers(0, n, size=n)
        weights = np.bincount(idx, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)
        if progress_callback is not None and (step % 50 == 0 or step == int(bootstrap_iters)):
            progress_callback(f"Pixel histogram slice bootstrap: {step}/{int(bootstrap_iters)}")

    auroc_ci = _clip_ci(*np.percentile(aurocs, [2.5, 97.5])) if aurocs else (auroc_value, auroc_value)
    aupr_ci = _clip_ci(*np.percentile(auprs, [2.5, 97.5])) if auprs else (aupr_value, aupr_value)
    return (auroc_value, auroc_ci), (aupr_value, aupr_ci), (binned_auroc, binned_aupr)


def _format_metric(name, value, ci, scale):
    lo, hi = ci
    return f"{name}={value*scale:.2f}% (95% CI {lo*scale:.2f}-{hi*scale:.2f}%)"


def eval_protocol_format_image_metrics(slice_metrics, as_percent=True):
    s = 100.0 if as_percent else 1.0
    m = slice_metrics
    return "[Slice-Img]  " + "  ".join([
        _format_metric("AUROC", m["img_auroc"], m["img_auroc_ci"], s),
        _format_metric("AUPR", m["img_ap"], m["img_ap_ci"], s),
        _format_metric("F1", m["img_f1"], m["img_f1_ci"], s),
    ])


def eval_protocol_compute_metrics(
    gt_labels,
    gt_masks,
    anomaly_maps,
    image_scores,
    paths,
    bootstrap_iters=500,
    ci_seed=42,
    ci_pixel_max_samples=200000,
    progress_callback=None,
):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    def f1_score_max(y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if len(np.unique(y_true)) < 2:
            return 0.0
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        f1 = f1[:-1]
        return float(f1.max()) if len(f1) else 0.0

    def safe_ap(y_true, y_score):
        y_true = np.asarray(y_true)
        if y_true.size == 0 or len(np.unique(y_true)) < 2:
            return 0.0
        return float(average_precision_score(y_true, y_score))

    gt_labels = np.asarray(gt_labels, dtype=np.int32).reshape(-1)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    image_scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    if progress_callback is not None:
        progress_callback("Computing image-level metrics and 95CI...")

    slice_metrics = {
        "img_auroc": _safe_auroc(gt_labels, image_scores),
        "img_auroc_ci": _bootstrap_ci(gt_labels, image_scores, _safe_auroc, bootstrap_iters, ci_seed + 10),
        "img_ap": safe_ap(gt_labels, image_scores),
        "img_ap_ci": _bootstrap_ci(gt_labels, image_scores, safe_ap, bootstrap_iters, ci_seed + 11),
        "img_f1": f1_score_max(gt_labels, image_scores),
        "img_f1_ci": _bootstrap_ci(gt_labels, image_scores, f1_score_max, bootstrap_iters, ci_seed + 12),
        "_pr_sp": image_scores,
        "_gt_sp": gt_labels,
    }
    if progress_callback is not None:
        progress_callback(eval_protocol_format_image_metrics(slice_metrics))

    if progress_callback is not None:
        progress_callback("Computing pixel-level Slice-Px(abn) metrics and 95CI...")
        progress_callback("Computing exact full-pixel point estimates with sklearn...")

    gt_px_abn = gt_masks[gt_labels == 1].reshape(-1).astype(np.uint8)
    pr_px_abn = maps[gt_labels == 1].reshape(-1).astype(np.float64)
    px_auroc_abn = _safe_auroc(gt_px_abn, pr_px_abn)
    px_aupr_abn = safe_ap(gt_px_abn, pr_px_abn)
    px_auroc_abn_ci, px_aupr_abn_ci, binned_pixel_check = _pixel_slice_bootstrap_metrics(
        gt_labels,
        gt_masks,
        maps,
        bootstrap_iters,
        ci_seed + 21,
        exact_auroc=px_auroc_abn,
        exact_aupr=px_aupr_abn,
        progress_callback=progress_callback,
    )
    slice_metrics.update({
        "px_auroc_abn": px_auroc_abn_ci[0],
        "px_auroc_abn_ci": px_auroc_abn_ci[1],
        "px_aupr_abn": px_aupr_abn_ci[0],
        "px_aupr_abn_ci": px_aupr_abn_ci[1],
        "px_binned_auroc_abn": binned_pixel_check[0],
        "px_binned_aupr_abn": binned_pixel_check[1],
    })

    if progress_callback is not None:
        progress_callback("Computing patient-level metrics and 95CI...")

    patient_scores = defaultdict(list)
    patient_labels = defaultdict(list)
    for path, score, label_value in zip(paths, image_scores, gt_labels):
        pid = patient_id_from_path(path)
        patient_scores[pid].append(float(score))
        patient_labels[pid].append(int(label_value))

    pat_score, pat_label = [], []
    for pid in patient_scores:
        pat_score.append(float(np.max(patient_scores[pid])))
        pat_label.append(int(np.max(patient_labels[pid])))
    pat_score = np.asarray(pat_score, dtype=np.float64)
    pat_label = np.asarray(pat_label, dtype=np.int32)
    pat_metrics = {
        "pat_auroc": _safe_auroc(pat_label, pat_score),
        "pat_auroc_ci": _bootstrap_ci(pat_label, pat_score, _safe_auroc, bootstrap_iters, ci_seed + 30),
        "pat_ap": safe_ap(pat_label, pat_score),
        "pat_ap_ci": _bootstrap_ci(pat_label, pat_score, safe_ap, bootstrap_iters, ci_seed + 31),
        "pat_f1": f1_score_max(pat_label, pat_score),
        "pat_f1_ci": _bootstrap_ci(pat_label, pat_score, f1_score_max, bootstrap_iters, ci_seed + 32),
        "n_patients": len(pat_label),
        "n_abn_patients": int(pat_label.sum()),
    }
    return slice_metrics, pat_metrics


def eval_protocol_format_metrics(slice_metrics, pat_metrics, as_percent=True):
    s = 100.0 if as_percent else 1.0
    m = slice_metrics
    lines = [
        eval_protocol_format_image_metrics(slice_metrics, as_percent=as_percent),
        "[Slice-Px(abn)]  " + "  ".join([
            _format_metric("AUROC", m["px_auroc_abn"], m["px_auroc_abn_ci"], s),
            _format_metric("AUPR", m["px_aupr_abn"], m["px_aupr_abn_ci"], s),
        ]),
    ]
    p = pat_metrics
    lines.append(
        f"[Patient({p['n_abn_patients']}/{p['n_patients']}abn)]  " + "  ".join([
            _format_metric("AUROC", p["pat_auroc"], p["pat_auroc_ci"], s),
            _format_metric("AUPR", p["pat_ap"], p["pat_ap_ci"], s),
            _format_metric("F1", p["pat_f1"], p["pat_f1_ci"], s),
        ])
    )
    return "\n".join(lines)


def selected_csv_path(tracer):
    return PROJECT_ROOT / "paper_figures" / "samples_data" / tracer.upper() / "selected_samples.csv"


def hot_output_root(method, tracer, modalities):
    return PROJECT_ROOT / "paper_figures" / method / "sample_hot_figures" / tracer.upper() / modality_output_dir(modalities)


def read_gray(path, image_size=256, resample=Image.BILINEAR):
    image = Image.open(path).convert("L")
    if image_size:
        image = image.resize((image_size, image_size), resample=resample)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path, image_size=256):
    return (read_gray(path, image_size, Image.NEAREST) > 0).astype(np.uint8)


def load_selected_samples(csv_path, modalities, image_size=256):
    csv_path = Path(csv_path)
    samples = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case", "patient", "slice", "mask_path"} | {f"{m}_path" for m in modalities}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            images = {m: read_gray(resolve_path(row[f"{m}_path"], csv_path.parent), image_size) for m in modalities}
            mask = load_mask(resolve_path(row["mask_path"], csv_path.parent), image_size)
            samples.append({
                "case": row["case"],
                "patient": row["patient"],
                "slice": row["slice"],
                "case_name": f"{row['case']}_{row['patient']}__{row['slice']}",
                "label": int(str(row["case"]).lower() == "abnormal"),
                "images": images,
                "mask": mask,
            })
    return samples


def normalize01(image):
    image = image.astype(np.float32)
    lo = float(np.nanmin(image))
    hi = float(np.nanmax(image))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return (image - lo) / (hi - lo)


def percentile_normalize(image, roi=None, q_low=0.01, q_high=0.995):
    image = image.astype(np.float32)
    values = image[roi.astype(bool)] if roi is not None and np.count_nonzero(roi) > 10 else image.ravel()
    lo = float(np.quantile(values, q_low))
    hi = float(np.quantile(values, q_high))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def display_score_map(score_map, roi=None, mode="percentile", q_low=0.01, q_high=0.995):
    if mode == "none":
        out = score_map.astype(np.float32)
    elif mode == "minmax":
        out = normalize01(score_map)
    elif mode == "percentile":
        out = percentile_normalize(score_map, roi=roi, q_low=q_low, q_high=q_high)
    else:
        raise ValueError(f"Unknown display normalization: {mode}")
    if roi is not None:
        out = out.copy()
        out[roi == 0] = 0
    return out


def best_f1_threshold(gt_mask, score_map):
    from sklearn.metrics import precision_recall_curve

    gt = gt_mask.astype(np.uint8).ravel()
    score = score_map.astype(np.float32).ravel()
    precision, recall, thresholds = precision_recall_curve(gt, score)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    f1 = f1[:-1]
    if len(f1) == 0:
        return 0.5, 0.0
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def connected_components(mask):
    mask = (mask > 0).astype(np.uint8)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    components = []
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
            components.append(coords)
    return components


def remove_small_components(mask, min_pixels):
    if min_pixels <= 0:
        return mask.astype(np.uint8)
    output = np.zeros_like(mask, dtype=np.uint8)
    for component in connected_components(mask):
        if len(component) >= min_pixels:
            for y, x in component:
                output[y, x] = 1
    return output


def save_image(data, output, cmap="gray", vmin=None, vmax=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 4))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_overlay(base, pred, gt, alpha=0.72):
    rgb = np.repeat(base[..., None], 3, axis=2)
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = pred & gt
    fn = (~pred) & gt
    fp = pred & (~gt)
    rgb[tp] = rgb[tp] * (1 - alpha) + np.array([0.0, 0.9, 0.0]) * alpha
    rgb[fn] = rgb[fn] * (1 - alpha) + np.array([1.0, 0.05, 0.05]) * alpha
    rgb[fp] = rgb[fp] * (1 - alpha) + np.array([0.0, 0.45, 1.0]) * alpha
    return rgb


def save_case_outputs(
    case_out,
    images,
    gt,
    raw_map,
    display_map,
    threshold,
    score=None,
    case_name=None,
    roi=None,
    min_component_pixels=0,
):
    import matplotlib.pyplot as plt

    case_out.mkdir(parents=True, exist_ok=True)
    pred = (raw_map >= threshold).astype(np.uint8)
    if roi is not None:
        pred = (pred & (roi > 0)).astype(np.uint8)
    pred = remove_small_components(pred, min_component_pixels)
    if "pet" in images:
        save_image(images["pet"], case_out / "pet_original.png", "gray_r", 0, 1)
    if "ct" in images:
        save_image(images["ct"], case_out / "ct_original.png", "gray", 0, 1)
    save_image(gt, case_out / "gt_mask.png", "gray", 0, 1)
    save_image(pred, case_out / "pred_mask.png", "gray", 0, 1)
    save_image(display_map, case_out / "pred_heatmap.png", "jet")
    base = images.get("pet", images.get("ct", next(iter(images.values()))))
    overlay = make_overlay(base, pred, gt)
    plt.imsave(case_out / "threshold_map.png", overlay)
    if score is None:
        score = float(top_percent_mean(raw_map, 1.0)[0])
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.4))
    for ax, title, data, cmap in [
        (axes[0], "Input", base, "gray"),
        (axes[1], "GT", gt, "gray"),
        (axes[2], "Heatmap", display_map, "jet"),
        (axes[3], "Threshold", overlay, None),
    ]:
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{case_name or case_out.name} | score={float(score):.4f}")
    fig.tight_layout()
    fig.savefig(case_out / "overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pred
