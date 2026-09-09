#!/usr/bin/env python3
"""Create selected-sample CSVs for the fixed lesion-size paper cases.

cd /data/cyf/codes/lyh/DAE_successful

python make_fixed_lesion_case_csvs.py \
  --data_root /data/cyf/shared_data/A-PETCT/2d_equal_mask50 \
  --output_dir fixed_lesion_case_csvs \
  --image_size 256
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent

FIXED_CASES = {
    "PSMA": [
        ("small", "psma_1ca3af29b7df127d_2020-06-27", "0009"),
        ("medium", "psma_ec45934c2fa23c76_2019-08-05", "0027"),
        ("large", "psma_18eba3b35ee1ddac_2020-09-05", "0036"),
    ],
    "FDG": [
        ("small", "fdg_791ec15924_04-16-2001-NA-PET-CT Ganzkoerper  primaer mit KM-50308", "0053"),
        ("medium", "fdg_19b68a666b_04-14-2005-NA-PET-CT Ganzkoerper  primaer mit KM-01313", "0036"),
        ("large", "fdg_ea0fd89f0f_10-25-2003-NA-PET-CT Ganzkoerper  primaer mit KM-32502", "0038"),
    ],
}


def resolve_data_root(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    for candidate in (PROJECT_DIR / p, PROJECT_DIR.parent / p, Path.cwd() / p):
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_DIR / p).resolve()


def mask_pixels(path: Path, image_size: int | None) -> int:
    image = Image.open(path).convert("L")
    if image_size:
        image = image.resize((image_size, image_size), resample=Image.NEAREST)
    return int((np.asarray(image) > 0).sum())


def write_case_csv(data_root: Path, tracer: str, rows, output_csv: Path, image_size: int, strict: bool) -> None:
    tracer_root = data_root if (data_root / "test").exists() else data_root / tracer
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    missing = []
    for size_group, patient, slice_id in rows:
        base = tracer_root / "test" / "abnormal" / patient
        pet_path = base / "pet" / f"{slice_id}.png"
        ct_path = base / "ct" / f"{slice_id}.png"
        mask_path = base / "label" / f"{slice_id}.png"
        paths = [pet_path, ct_path, mask_path]
        absent = [str(path) for path in paths if not path.exists()]
        if absent:
            missing.extend(absent)
            if strict:
                continue
        pixels = mask_pixels(mask_path, image_size) if mask_path.exists() else ""
        out_rows.append(
            {
                "case": size_group,
                "patient": patient,
                "slice": slice_id,
                "mask_pixels": pixels,
                "pet_path": pet_path,
                "ct_path": ct_path,
                "mask_path": mask_path,
            }
        )
    if missing and strict:
        raise FileNotFoundError("Missing selected case files:\n" + "\n".join(missing))
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "patient", "slice", "mask_pixels", "pet_path", "ct_path", "mask_path"],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved {len(out_rows)} rows: {output_csv}")
    if missing and not strict:
        print("Warning: missing files:")
        for path in missing:
            print(f"  {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="A_data/2d_equal_mask50")
    parser.add_argument("--output_dir", default="fixed_lesion_case_csvs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--no_strict", action="store_true", help="Write CSV even if some files are missing.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    for tracer, rows in FIXED_CASES.items():
        write_case_csv(
            data_root,
            tracer,
            rows,
            output_dir / f"{tracer.lower()}_fixed_lesion_cases.csv",
            image_size=args.image_size,
            strict=not args.no_strict,
        )


if __name__ == "__main__":
    main()
