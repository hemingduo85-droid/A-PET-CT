import argparse
import os

import numpy as np

from petct_metrics import compute_petct_metrics, format_petct_metrics


def main():
    parser = argparse.ArgumentParser(description="Compute PET-CT metrics with 95CI from a Dinomaly eval cache.")
    parser.add_argument("--cache", required=True, help=".npz file exported by dinomaly_petct.py --export_only")
    parser.add_argument("--output", default=None, help="Optional text output path")
    parser.add_argument("--bootstrap_iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hist_pixel_points",
        action="store_true",
        help="Use histogram pixel point estimates for speed. Default uses exact sklearn full-pixel point estimates.",
    )
    parser.add_argument("--exact_pixel_auroc", type=float, default=None,
                        help="Optional exact full-pixel AUROC point estimate, e.g. 0.9518.")
    parser.add_argument("--exact_pixel_aupr", type=float, default=None,
                        help="Optional exact full-pixel AUPR point estimate, e.g. 0.2436.")
    args = parser.parse_args()

    def progress(message):
        print(message, flush=True)

    progress(f"Loading eval cache from {args.cache}")
    cache = np.load(args.cache, allow_pickle=True)
    progress(
        f"Cache loaded: labels={cache['labels'].shape}, masks={cache['masks'].shape}, "
        f"maps={cache['maps'].shape}, bootstrap_iters={args.bootstrap_iters}"
    )
    slice_metrics, patient_metrics = compute_petct_metrics(
        cache["labels"].astype(np.int32),
        cache["masks"].astype(np.float32),
        cache["maps"].astype(np.float32),
        cache["image_scores"].astype(np.float64),
        cache["paths"],
        n_bootstrap=args.bootstrap_iters,
        seed=args.seed,
        exact_pixel_points=not args.hist_pixel_points,
        exact_pixel_auroc=args.exact_pixel_auroc,
        exact_pixel_aupr=args.exact_pixel_aupr,
        progress_fn=progress,
    )
    text = format_petct_metrics(slice_metrics, patient_metrics) + "\n"
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
