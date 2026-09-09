#!/usr/bin/env python3
"""
Run:
cd /data/cyf/codes/A-PET-CT/PhyTwin-PETCT-last1
python export_fdg_large_two_cases.py \
  --output_dir saved_results_fdg_eval/PhyTwin_PETCT_FDG/figure_scores/fdg_large_two_cases \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG \
  --experiment_name PhyTwin_PETCT_FDG \
  --without_ucr_experiment_name PhyTwin_PETCT_FDG_no_uncertainty_residual \
  --adaptive_alpha 0.50 \
  --adaptive_lesion_protect 0.70 \
  --image_size 256 \
  --gpu 4


"""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from pathlib import Path


FIXED_LARGE_CASES = (
    (
        "fdg_ad7cd4a9d2_10-02-2003-NA-Unspecified CT ABDOMEN-67897",
        "0020",
    ),
    (
        "fdg_510fb36781_02-06-2003-NA-PET-CT Ganzkoerper  primaer mit KM-07563",
        "0017",
    ),
)


def write_group_csv(group_root: Path) -> Path:
    csv_path = group_root / "large" / "slices.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "case_id",
        "slice_id",
        "true_label",
        "lesion_size_group",
        "ours_score",
        "lesion_pixels",
        "lesion_area_ratio",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for order, (case_id, slice_id) in enumerate(FIXED_LARGE_CASES):
            writer.writerow(
                {
                    "case_id": case_id,
                    "slice_id": slice_id,
                    "true_label": "1",
                    "lesion_size_group": "large",
                    "ours_score": str(len(FIXED_LARGE_CASES) - order),
                    "lesion_pixels": "",
                    "lesion_area_ratio": "",
                }
            )
    return csv_path


def copy_png_outputs(temp_output: Path, final_output: Path) -> None:
    source_cases = temp_output / "individual_top_cases" / "large"
    source_plate = temp_output / "ablation_large_top_cases.png"
    if not source_cases.is_dir():
        raise FileNotFoundError(f"Missing rendered large-case directory: {source_cases}")
    if not source_plate.is_file():
        raise FileNotFoundError(f"Missing rendered large-case plate: {source_plate}")

    final_output.mkdir(parents=True, exist_ok=True)
    final_cases = final_output / "large"
    for source_case in sorted(path for path in source_cases.iterdir() if path.is_dir()):
        destination = final_cases / source_case.name
        destination.mkdir(parents=True, exist_ok=True)
        for source_png in sorted(source_case.glob("*.png")):
            shutil.copy2(source_png, destination / source_png.name)
    shutil.copy2(source_plate, final_output / "fdg_large_two_cases.png")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--data_root",
        default="/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG",
    )
    parser.add_argument("--experiment_name", default="PhyTwin_PETCT_FDG")
    parser.add_argument(
        "--without_ucr_experiment_name",
        default="PhyTwin_PETCT_FDG_no_uncertainty_residual",
    )
    parser.add_argument("--adaptive_alpha", type=float, default=0.50)
    parser.add_argument("--adaptive_lesion_protect", type=float, default=0.70)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--overwrite", action="store_true")
    args, extra = parser.parse_known_args(argv)
    return args, extra


def main(argv=None):
    args, extra = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. "
                "Use a new directory or pass --overwrite."
            )
        shutil.rmtree(args.output_dir)

    import export_ablation_lesion_case_heatmaps as exporter

    with tempfile.TemporaryDirectory(prefix="fdg_large_two_cases_") as tmp:
        temp_root = Path(tmp)
        group_root = temp_root / "groups"
        temp_output = temp_root / "export"
        write_group_csv(group_root)
        export_argv = [
            "--group_root",
            str(group_root),
            "--groups",
            "large",
            "--top_k",
            "2",
            "--individual_top_k",
            "2",
            "--output_dir",
            str(temp_output),
            "--without_ucr_experiment_name",
            args.without_ucr_experiment_name,
            "--data_root",
            args.data_root,
            "--load_ckpt",
            "--experiment_name",
            args.experiment_name,
            "--score_mode",
            "lesion_z",
            "--adaptive_physio",
            "--adaptive_alpha",
            str(args.adaptive_alpha),
            "--adaptive_lesion_protect",
            str(args.adaptive_lesion_protect),
            "--image_size",
            str(args.image_size),
            "--gpu",
            str(args.gpu),
        ]
        exporter.main(export_argv + list(extra))
        copy_png_outputs(temp_output, args.output_dir)

    print(f"Saved two fixed FDG large-lesion cases to {args.output_dir}")


if __name__ == "__main__":
    main()
