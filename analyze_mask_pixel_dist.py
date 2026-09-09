#!/usr/bin/env python3
"""Analyze abnormal mask foreground pixel counts in 2d_equal PET-CT test sets."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOTS = {
    "FDG": Path("/data/cyf/shared_data/PET-CT/AutoPET/2d_equal/FDG/test"),
    "PSMA": Path("/data/cyf/shared_data/PET-CT/AutoPET/2d_equal/PSMA/test"),
}
OUT_DIR = Path("/data/cyf/codes/A-PET-CT/mask_pixel_analysis")


def count_mask_pixels(mask_path: Path) -> int:
    arr = np.array(Image.open(mask_path))
    return int((arr > 0).sum())


def collect_pixels(root: Path) -> dict:
    abn_dir = root / "abnormal"
    records = []
    for patient in sorted(os.listdir(abn_dir)):
        label_dir = abn_dir / patient / "label"
        if not label_dir.is_dir():
            continue
        for fname in sorted(os.listdir(label_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue
            px = count_mask_pixels(label_dir / fname)
            records.append(
                {
                    "patient": patient,
                    "slice": fname,
                    "pixels": px,
                    "path": str(label_dir / fname),
                }
            )
    pixels = np.array([r["pixels"] for r in records], dtype=np.int64)
    return {"records": records, "pixels": pixels}


def summarize(pixels: np.ndarray) -> dict:
    if len(pixels) == 0:
        return {}
    p = pixels.astype(float)
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    out = {
        "n_masks": int(len(pixels)),
        "min": int(pixels.min()),
        "max": int(pixels.max()),
        "mean": float(p.mean()),
        "median": float(np.median(p)),
        "std": float(p.std()),
        "quantiles": {f"p{q}": float(np.percentile(p, q)) for q in qs},
    }
    for th in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
        out[f"count_lt_{th}"] = int((pixels < th).sum())
        out[f"pct_lt_{th}"] = float(100 * (pixels < th).sum() / len(pixels))
    return out


def plot_histograms(data: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Abnormal mask foreground pixel count (2d_equal test)", fontsize=14)

    for i, (name, d) in enumerate(data.items()):
        pixels = d["pixels"]
        ax_full = axes[i, 0]
        ax_zoom = axes[i, 1]
        ax_full.hist(pixels, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
        ax_full.axvline(5, color="red", ls="--", lw=1.5, label="threshold=5")
        ax_full.axvline(10, color="orange", ls="--", lw=1.5, label="threshold=10")
        ax_full.set_title(f"{name}: full range (n={len(pixels)})")
        ax_full.set_xlabel("foreground pixels")
        ax_full.set_ylabel("count")
        ax_full.legend()
        ax_full.set_yscale("log")

        zoom = pixels[pixels <= 200]
        ax_zoom.hist(zoom, bins=np.arange(0, 201, 1) - 0.5, color="coral", edgecolor="white", alpha=0.9)
        ax_zoom.axvline(5, color="red", ls="--", lw=1.5, label="threshold=5")
        ax_zoom.axvline(10, color="orange", ls="--", lw=1.5, label="threshold=10")
        ax_zoom.set_title(f"{name}: zoom 0-200 px")
        ax_zoom.set_xlabel("foreground pixels")
        ax_zoom.set_ylabel("count")
        ax_zoom.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "hist_full_and_zoom.png", dpi=150)
    plt.close()

    # Combined + cumulative
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"FDG": "steelblue", "PSMA": "seagreen"}
    all_px = []
    for name, d in data.items():
        pixels = d["pixels"]
        all_px.append(pixels)
        axes[0].hist(
            pixels[pixels <= 100],
            bins=np.arange(0, 101, 1) - 0.5,
            alpha=0.6,
            label=name,
            color=colors[name],
        )
        sorted_px = np.sort(pixels)
        cdf = np.arange(1, len(sorted_px) + 1) / len(sorted_px)
        axes[1].plot(sorted_px, cdf, label=name, color=colors[name], lw=2)
    axes[0].axvline(5, color="red", ls="--", label="5")
    axes[0].axvline(10, color="orange", ls="--", label="10")
    axes[0].set_xlabel("foreground pixels (<=100)")
    axes[0].set_ylabel("count")
    axes[0].set_title("0-100 px zoom (both tracers)")
    axes[0].legend()
    axes[1].axvline(5, color="red", ls="--")
    axes[1].axvline(10, color="orange", ls="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("foreground pixels (log scale)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("CDF (log x)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    combined = np.concatenate(all_px)
    axes[2].hist(
        combined[combined <= 50],
        bins=np.arange(0, 51, 1) - 0.5,
        color="purple",
        alpha=0.8,
        edgecolor="white",
    )
    for th, c in [(5, "red"), (10, "orange"), (20, "green")]:
        axes[2].axvline(th, color=c, ls="--", lw=1.5, label=f"<{th}: {(combined<th).sum()} ({100*(combined<th).mean():.1f}%)")
    axes[2].set_title(f"FDG+PSMA combined n={len(combined)}")
    axes[2].set_xlabel("foreground pixels")
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "hist_cdf_combined.png", dpi=150)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, root in ROOTS.items():
        print(f"Scanning {name} ...")
        d = collect_pixels(root)
        d["summary"] = summarize(d["pixels"])
        results[name] = {k: v for k, v in d.items() if k != "records"}
        results[name]["records_count"] = len(d["records"])
        # save tiny examples
        tiny = sorted(d["records"], key=lambda x: x["pixels"])[:15]
        results[name]["smallest_15"] = tiny
        data_for_plot = {name: d for name, d in [(name, d)]}
        # rebuild full dict below
    # re-collect for plotting
    plot_data = {}
    full = {}
    for name, root in ROOTS.items():
        d = collect_pixels(root)
        plot_data[name] = d
        full[name] = d
    plot_histograms(plot_data, OUT_DIR)

    combined_pixels = np.concatenate([full[n]["pixels"] for n in ROOTS])
    results["COMBINED"] = summarize(combined_pixels)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(
            {
                name: {
                    "summary": summarize(full[name]["pixels"]),
                    "smallest_15": sorted(full[name]["records"], key=lambda x: x["pixels"])[:15],
                }
                for name in ROOTS
            }
            | {"COMBINED": results["COMBINED"]},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
