#!/usr/bin/env python3
"""
Build balanced PET-CT dataset with mask foreground >= min_pixels.

Reads ONLY from 2d_equal and 2d_data. Writes to AutoPET/2d_equal_mask{N}.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

AUTOPET = Path("/data/cyf/shared_data/PET-CT/AutoPET")
SRC_EQUAL = AUTOPET / "2d_equal"
SRC_DATA = AUTOPET / "2d_data"
MODALITIES = ["FDG", "PSMA"]

MIN_PIXELS: int
DST: Path


def count_foreground_pixels(mask_path: Path) -> int:
    arr = np.array(Image.open(mask_path))
    return int((arr > 0).sum())


@dataclass
class PatientInfo:
    name: str
    slices: list[str]


def scan_abnormal_patient(patient_dir: Path) -> PatientInfo | None:
    label_dir = patient_dir / "label"
    if not label_dir.is_dir():
        return None
    valid = []
    for fname in sorted(os.listdir(label_dir)):
        if not fname.lower().endswith(".png"):
            continue
        lp = label_dir / fname
        if count_foreground_pixels(lp) >= MIN_PIXELS:
            valid.append(fname)
    if not valid:
        return None
    return PatientInfo(patient_dir.name, valid)


def copy_patient_slices(
    src_patient: Path,
    dst_patient: Path,
    slice_names: list[str],
    is_abnormal: bool,
):
    dst_patient.mkdir(parents=True, exist_ok=True)
    for sub in ("pet", "ct", "label") if is_abnormal else ("pet", "ct"):
        (dst_patient / sub).mkdir(exist_ok=True)
    for fname in slice_names:
        for sub in ("pet", "ct"):
            shutil.copy2(src_patient / sub / fname, dst_patient / sub / fname)
        if is_abnormal:
            shutil.copy2(src_patient / "label" / fname, dst_patient / "label" / fname)


def select_abnormal_patients(
    pool: list[PatientInfo],
    k_patients: int,
    target_slices: int,
) -> list[PatientInfo]:
    if len(pool) <= k_patients:
        return pool

    pool = sorted(pool, key=lambda p: len(p.slices), reverse=True)
    top_k = pool[:k_patients]
    best = top_k
    best_diff = abs(sum(len(p.slices) for p in top_k) - target_slices)

    selected = list(top_k)
    selected_set = {p.name for p in selected}
    outside = [p for p in pool if p.name not in selected_set]

    for _ in range(500):
        cur_sum = sum(len(p.slices) for p in selected)
        diff = abs(cur_sum - target_slices)
        if diff < best_diff:
            best_diff = diff
            best = list(selected)
        if cur_sum >= target_slices * 0.98 and cur_sum <= target_slices * 1.15:
            break
        if not outside:
            break
        selected.sort(key=lambda p: len(p.slices))
        outside.sort(key=lambda p: len(p.slices), reverse=True)
        worst = selected[0]
        for cand in outside:
            new_sum = cur_sum - len(worst.slices) + len(cand.slices)
            if abs(new_sum - target_slices) < diff:
                outside.remove(cand)
                outside.append(worst)
                selected_set.discard(worst.name)
                selected_set.add(cand.name)
                selected[0] = cand
                break
        else:
            break

    return best


def count_split(mod_root: Path, split: str) -> dict:
    out = {}
    for cls in ("abnormal", "normal"):
        d = mod_root / split / cls
        if not d.is_dir():
            out[cls] = {"patients": 0, "slices": 0}
            continue
        patients = 0
        slices = 0
        for patient in os.listdir(d):
            pet = d / patient / "pet"
            if pet.is_dir():
                patients += 1
                slices += len([f for f in os.listdir(pet) if f.endswith(".png")])
        out[cls] = {"patients": patients, "slices": slices}
    return out


def rebuild_test_abnormal(tracer: str):
    raw_test = SRC_DATA / tracer / "test"
    eq_test = SRC_EQUAL / tracer / "test"
    dst_test = DST / tracer / "test"

    normal_patients = sorted(os.listdir(eq_test / "normal"))
    target_slices = 0
    for p in normal_patients:
        pet = dst_test / "normal" / p / "pet"
        if pet.is_dir():
            target_slices += len([f for f in os.listdir(pet) if f.endswith(".png")])

    print(f"  [{tracer}] normal patients={len(normal_patients)}, target abn slices≈{target_slices}")

    pool: list[PatientInfo] = []
    abn_raw = raw_test / "abnormal"
    for patient in tqdm(sorted(os.listdir(abn_raw)), desc=f"  scan {tracer} pool"):
        info = scan_abnormal_patient(abn_raw / patient)
        if info:
            pool.append(info)

    print(f"  [{tracer}] pool patients with >={MIN_PIXELS}px slices: {len(pool)}")
    selected = select_abnormal_patients(pool, len(normal_patients), target_slices)
    abn_sum = sum(len(p.slices) for p in selected)

    dst_abn = dst_test / "abnormal"
    if dst_abn.exists():
        shutil.rmtree(dst_abn)
    dst_abn.mkdir(parents=True)

    for info in tqdm(selected, desc=f"  copy {tracer} abn"):
        copy_patient_slices(
            abn_raw / info.name,
            dst_abn / info.name,
            info.slices,
            is_abnormal=True,
        )

    print(
        f"  [{tracer}] selected abn patients={len(selected)}, "
        f"slices={abn_sum}, ratio abn/norm={abn_sum/target_slices:.3f}"
    )
    return {
        "normal": {"patients": len(normal_patients), "slices": target_slices},
        "abnormal": {"patients": len(selected), "slices": abn_sum},
        "min_pixels": MIN_PIXELS,
    }


def verify_no_small_masks(tracer: str) -> int:
    bad = 0
    abn = DST / tracer / "test" / "abnormal"
    for patient in os.listdir(abn):
        ld = abn / patient / "label"
        for fname in os.listdir(ld):
            if not fname.endswith(".png"):
                continue
            n = count_foreground_pixels(ld / fname)
            if n < MIN_PIXELS:
                bad += 1
    return bad


def main():
    global MIN_PIXELS, DST
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pixels", type=int, default=50)
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Output dir name under AutoPET (default: 2d_equal_mask{min_pixels})",
    )
    parser.add_argument("--force", action="store_true", help="Remove existing dst first")
    args = parser.parse_args()

    MIN_PIXELS = args.min_pixels
    dst_name = args.dst or f"2d_equal_mask{MIN_PIXELS}"
    DST = AUTOPET / dst_name

    if DST.exists():
        if not args.force:
            print(f"Destination exists: {DST}  (use --force to rebuild)")
            sys.exit(1)
        shutil.rmtree(DST)

    print(f"min_pixels={MIN_PIXELS}, output={DST}")
    print(f"Copy {SRC_EQUAL} -> {DST} (rsync)...")
    subprocess.run(["rsync", "-a", f"{SRC_EQUAL}/", f"{DST}/"], check=True)

    summary = {
        "source": str(SRC_EQUAL),
        "raw_source": str(SRC_DATA),
        "min_pixels": MIN_PIXELS,
    }
    for tracer in MODALITIES:
        print(f"\nRebuild {tracer}/test/abnormal ...")
        summary[tracer] = {"test": rebuild_test_abnormal(tracer)}
        summary[tracer]["train"] = count_split(DST / tracer, "train")
        bad = verify_no_small_masks(tracer)
        print(f"  [{tracer}] verify masks < {MIN_PIXELS}: {bad}")

    readme = DST / "README.md"
    readme.write_text(
        f"# {dst_name}\n\n"
        f"- 前景像素 **≥ {MIN_PIXELS}** 的异常切片才保留\n"
        f"- `train/`、`test/normal/` 来自 `2d_equal`\n"
        f"- `test/abnormal/` 从 `2d_data` 重选，患者数与正常相同，切片约 1:1\n\n"
        f"构建: `python3 build_2d_equal_filtered.py --min-pixels {MIN_PIXELS} --force`\n",
        encoding="utf-8",
    )
    out_json = DST / "dataset_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
