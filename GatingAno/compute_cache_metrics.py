import argparse
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _max_f1(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    f1 = f1[:-1]
    return float(f1.max()) if len(f1) else 0.0


def _bootstrap_ci(y_true, y_score, metric_fn, iters, seed):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    value = metric_fn(y_true, y_score)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_fn(y_true[idx], y_score[idx]))
    if not values:
        return value, (value, value)
    low, high = np.percentile(values, [2.5, 97.5])
    return value, (float(low), float(high))


def _patient_id_from_path(path):
    return os.path.basename(os.path.dirname(os.path.dirname(str(path))))


def _patient_arrays(paths, image_scores, labels):
    patient_scores = defaultdict(list)
    patient_labels = defaultdict(list)
    for path, score, label in zip(paths, image_scores, labels):
        pid = _patient_id_from_path(path)
        patient_scores[pid].append(float(score))
        patient_labels[pid].append(int(label))
    scores, labels_out = [], []
    for pid in patient_scores:
        scores.append(float(np.max(patient_scores[pid])))
        labels_out.append(int(np.max(patient_labels[pid])))
    return np.asarray(labels_out, dtype=np.int32), np.asarray(scores, dtype=np.float64)


def _metrics_from_hist(pos_hist, neg_hist):
    tp = pos_hist[::-1].astype(np.float64, copy=False)
    fp = neg_hist[::-1].astype(np.float64, copy=False)
    total_pos = float(tp.sum())
    total_neg = float(fp.sum())
    if total_pos <= 0 or total_neg <= 0:
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


def _pixel_slice_bootstrap(labels, masks, maps, iters, seed, bins, exact_auroc=None, exact_aupr=None):
    keep = np.asarray(labels, dtype=np.int32).reshape(-1) == 1
    abn_idx = np.flatnonzero(keep)
    if len(abn_idx) == 0:
        return (0.0, (0.0, 0.0)), (0.0, (0.0, 0.0)), (0.0, 0.0)

    score_min = float(np.min(maps[keep]))
    score_max = float(np.max(maps[keep]))
    if score_max <= score_min:
        score_max = score_min + 1e-8
    scale = (bins - 1) / (score_max - score_min)

    pos_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    neg_hists = np.zeros((len(abn_idx), bins), dtype=np.uint32)
    for row, idx in enumerate(abn_idx):
        scores = maps[idx].reshape(-1)
        mask = masks[idx].reshape(-1).astype(bool)
        bin_idx = np.floor((scores - score_min) * scale).astype(np.int32)
        np.clip(bin_idx, 0, bins - 1, out=bin_idx)
        pos_hists[row] = np.bincount(bin_idx[mask], minlength=bins).astype(np.uint32)
        neg_hists[row] = np.bincount(bin_idx[~mask], minlength=bins).astype(np.uint32)

    binned_auroc, binned_aupr = _metrics_from_hist(pos_hists.sum(axis=0), neg_hists.sum(axis=0))
    auroc_value = binned_auroc if exact_auroc is None else float(exact_auroc)
    aupr_value = binned_aupr if exact_aupr is None else float(exact_aupr)

    rng = np.random.default_rng(seed)
    aurocs, auprs = [], []
    n = len(abn_idx)
    for _ in range(iters):
        sample = rng.integers(0, n, size=n)
        weights = np.bincount(sample, minlength=n).astype(np.uint32)
        pos = np.einsum("i,ij->j", weights, pos_hists, optimize=True)
        neg = np.einsum("i,ij->j", weights, neg_hists, optimize=True)
        auroc, aupr = _metrics_from_hist(pos, neg)
        aurocs.append(auroc)
        auprs.append(aupr)

    auroc_ci = tuple(float(v) for v in np.percentile(aurocs, [2.5, 97.5]))
    aupr_ci = tuple(float(v) for v in np.percentile(auprs, [2.5, 97.5]))
    return (auroc_value, auroc_ci), (aupr_value, aupr_ci), (binned_auroc, binned_aupr)


def _fmt(name, value_ci):
    value, ci = value_ci
    return f"{name}={value * 100:.2f}% (95% CI {ci[0] * 100:.2f}-{ci[1] * 100:.2f}%)"


def main():
    parser = argparse.ArgumentParser(description="Compute bootstrap metrics from a GatingAno eval cache.")
    parser.add_argument("--cache", required=True, help=".npz file with labels, masks, maps, image_scores, paths")
    parser.add_argument("--output", default=None, help="optional txt output path")
    parser.add_argument("--bootstrap_iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--hist_bins", type=int, default=16384)
    parser.add_argument("--exact_pixel_auroc", type=float, default=None, help="optional exact full pixel AUROC point estimate")
    parser.add_argument("--exact_pixel_aupr", type=float, default=None, help="optional exact full pixel AUPR point estimate")
    args = parser.parse_args()

    cache = np.load(args.cache, allow_pickle=True)
    labels = cache["labels"].astype(np.int32)
    image_scores = cache["image_scores"].astype(np.float64)
    paths = cache["paths"]
    masks = cache["masks"]
    maps = cache["maps"]
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)

    img_auroc = _bootstrap_ci(labels, image_scores, _safe_auroc, args.bootstrap_iters, args.seed)
    img_ap = _bootstrap_ci(labels, image_scores, _safe_ap, args.bootstrap_iters, args.seed + 1)
    img_f1 = _bootstrap_ci(labels, image_scores, _max_f1, args.bootstrap_iters, args.seed + 2)

    pat_labels, pat_scores = _patient_arrays(paths, image_scores, labels)
    pat_auroc = _bootstrap_ci(pat_labels, pat_scores, _safe_auroc, args.bootstrap_iters, args.seed + 5)
    pat_ap = _bootstrap_ci(pat_labels, pat_scores, _safe_ap, args.bootstrap_iters, args.seed + 6)
    pat_f1 = _bootstrap_ci(pat_labels, pat_scores, _max_f1, args.bootstrap_iters, args.seed + 7)

    keep = labels == 1
    px_true = masks[keep].reshape(-1).astype(np.int32)
    px_score = maps[keep].reshape(-1).astype(np.float64)
    lesion_prevalence = float(px_true.mean()) if px_true.size else 0.0
    print(
        f"Pixel AUPR random baseline (lesion-pixel prevalence): "
        f"{lesion_prevalence * 100:.4f}%"
    )
    exact_auroc = _safe_auroc(px_true, px_score)
    exact_aupr = _safe_ap(px_true, px_score)

    px_auroc, px_aupr, binned = _pixel_slice_bootstrap(
        labels,
        masks,
        maps,
        args.bootstrap_iters,
        args.seed + 10,
        args.hist_bins,
        exact_auroc,
        exact_aupr,
    )

    lines = [
        "[Slice-Img]  " + "  ".join([_fmt("AUROC", img_auroc), _fmt("AUPR", img_ap), _fmt("F1", img_f1)]),
        "[Slice-Px(abn)]  " + "  ".join([_fmt("AUROC", px_auroc), _fmt("AUPR", px_aupr)]),
        f"[Patient({int(pat_labels.sum())}/{len(pat_labels)}abn)]  "
        + "  ".join([_fmt("AUROC", pat_auroc), _fmt("AUPR", pat_ap), _fmt("F1", pat_f1)]),
        "",
        "CI protocol:",
        "- All CIs use bootstrap resampling with replacement.",
        "- Image-level CI: resample slices.",
        "- Pixel-level CI: resample abnormal slices, then compute pixel metrics on pixels from sampled slices.",
        "- Patient-level CI: resample patients.",
        f"- Pixel bootstrap uses {args.hist_bins} score bins for efficient weighted recomputation.",
        f"- Binned full pixel check: AUROC={binned[0] * 100:.4f}% AUPR={binned[1] * 100:.4f}%",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
