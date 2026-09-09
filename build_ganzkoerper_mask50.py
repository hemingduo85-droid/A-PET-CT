#!/usr/bin/env python3
"""
Build FDG dataset with Ganzkoerper-only whole-body scans.

Combines:
- rebuild_normal_only.py: PET.max() >= min_pet_max for train/test normal from 2d_data
- build_2d_equal_filtered.py: test/abnormal mask foreground >= min_pixels, balanced selection
- Ganzkoerper filter: drop Teilkoerper / ABDOMEN / Other scan types

Patient lists enumerated from 2d_data (then filtered to Ganzkoerper).
train/normal and test/normal patient IDs must not overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

AUTOPET = Path("/data/cyf/shared_data/PET-CT/AutoPET")
SRC_DATA = AUTOPET / "2d_data"
TRACER = "FDG"

MIN_PET_MAX: int
MIN_PIXELS: int
DST: Path


def is_ganzkoerper(name: str) -> bool:
    return "Ganzkoerper" in name


def pet_max_value(pet_path: Path) -> int:
    return int(np.array(Image.open(pet_path)).max())


def count_foreground_pixels(mask_path: Path) -> int:
    return int((np.array(Image.open(mask_path)) > 0).sum())


def valid_normal_slices(patient_dir: Path) -> list[str]:
    valid = []
    pet_dir = patient_dir / "pet"
    for fname in sorted(os.listdir(pet_dir)):
        if not fname.lower().endswith(".png"):
            continue
        if pet_max_value(pet_dir / fname) >= MIN_PET_MAX:
            valid.append(fname)
    return valid


def copy_normal_patient(src: Path, dst: Path, slice_names: list[str]) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for sub in ("pet", "ct"):
        (dst / sub).mkdir()
    for fname in slice_names:
        for sub in ("pet", "ct"):
            shutil.copy2(src / sub / fname, dst / sub / fname)
    return len(slice_names)


def copy_abnormal_patient(src: Path, dst: Path, slice_names: list[str]) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for sub in ("pet", "ct", "label"):
        (dst / sub).mkdir()
    for fname in slice_names:
        for sub in ("pet", "ct", "label"):
            shutil.copy2(src / sub / fname, dst / sub / fname)
    return len(slice_names)


def resolve_src_normal(split: str, name: str) -> Path:
    if split == "train":
        src_p = SRC_DATA / TRACER / "train" / name
    else:
        src_p = SRC_DATA / TRACER / "test" / "normal" / name
        if not src_p.is_dir():
            src_p = SRC_DATA / TRACER / "train" / name
    if not src_p.is_dir():
        raise FileNotFoundError(f"Missing source patient: {src_p}")
    return src_p


@dataclass
class PatientInfo:
    name: str
    slices: list[str]


def scan_abnormal_patient(patient_dir: Path) -> PatientInfo | None:
    if not is_ganzkoerper(patient_dir.name):
        return None
    label_dir = patient_dir / "label"
    if not label_dir.is_dir():
        return None
    valid = []
    for fname in sorted(os.listdir(label_dir)):
        if not fname.lower().endswith(".png"):
            continue
        if count_foreground_pixels(label_dir / fname) >= MIN_PIXELS:
            valid.append(fname)
    if not valid:
        return None
    return PatientInfo(patient_dir.name, valid)


def select_abnormal_patients(
    pool: list[PatientInfo],
    k_patients: int,
    target_slices: int,
) -> list[PatientInfo]:
    if len(pool) < k_patients:
        raise RuntimeError(
            f"Abnormal pool too small: {len(pool)} < {k_patients} required"
        )
    if len(pool) == k_patients:
        return pool

    pool = sorted(pool, key=lambda p: len(p.slices), reverse=True)
    top_k = pool[:k_patients]
    best = top_k
    best_diff = abs(sum(len(p.slices) for p in top_k) - target_slices)

    selected = list(top_k)
    outside = [p for p in pool if p.name not in {s.name for s in selected}]

    for _ in range(500):
        cur_sum = sum(len(p.slices) for p in selected)
        diff = abs(cur_sum - target_slices)
        if diff < best_diff:
            best_diff = diff
            best = list(selected)
        if cur_sum >= target_slices * 0.98 and cur_sum <= target_slices * 1.05:
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
                selected[0] = cand
                break
        else:
            break

    return best


def list_patients(root: Path) -> list[str]:
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "pet").is_dir()
    )


def filter_patient_lists() -> tuple[list[str], list[str], dict]:
    cur_train = list_patients(SRC_DATA / TRACER / "train")
    cur_test = list_patients(SRC_DATA / TRACER / "test" / "normal")

    removed_train = [p for p in cur_train if not is_ganzkoerper(p)]
    removed_test = [p for p in cur_test if not is_ganzkoerper(p)]

    train = [p for p in cur_train if is_ganzkoerper(p)]
    test = [p for p in cur_test if is_ganzkoerper(p)]

    overlap = set(train) & set(test)
    if overlap:
        raise RuntimeError(f"train/test overlap after Ganzkoerper filter: {overlap}")

    log = {
        "removed_non_ganz_train": removed_train,
        "removed_non_ganz_test": removed_test,
        "removed_train_count": len(removed_train),
        "removed_test_count": len(removed_test),
    }
    return sorted(train), sorted(test), log


def rebuild_normal_split(split: str, patients: list[str]) -> dict:
    dst_root = DST / TRACER / split / "normal"
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)

    removed_slices = 0
    per_patient = {}

    for name in tqdm(patients, desc=f"  {split}/normal"):
        src_p = resolve_src_normal(split, name)
        all_pet = sorted(f for f in os.listdir(src_p / "pet") if f.endswith(".png"))
        valid = valid_normal_slices(src_p)
        removed_slices += len(all_pet) - len(valid)
        if not valid:
            raise RuntimeError(f"patient {name} has 0 valid PET slices")
        n = copy_normal_patient(src_p, dst_root / name, valid)
        per_patient[name] = {"slices": n, "removed": len(all_pet) - n}

    return {
        "patients": len(patients),
        "slices": sum(v["slices"] for v in per_patient.values()),
        "removed_low_signal_pet_slices": removed_slices,
        "per_patient": per_patient,
    }


def rebuild_test_abnormal(test_normal_slices: int, k_patients: int) -> dict:
    raw_abn = SRC_DATA / TRACER / "test" / "abnormal"
    dst_abn = DST / TRACER / "test" / "abnormal"

    pool: list[PatientInfo] = []
    for patient in tqdm(sorted(os.listdir(raw_abn)), desc="  scan abnormal pool"):
        info = scan_abnormal_patient(raw_abn / patient)
        if info:
            pool.append(info)

    selected = select_abnormal_patients(pool, k_patients, test_normal_slices)
    abn_sum = sum(len(p.slices) for p in selected)

    if dst_abn.exists():
        shutil.rmtree(dst_abn)
    dst_abn.mkdir(parents=True)

    for info in tqdm(selected, desc="  copy abnormal"):
        copy_abnormal_patient(
            raw_abn / info.name,
            dst_abn / info.name,
            info.slices,
        )

    return {
        "patients": len(selected),
        "slices": abn_sum,
        "pool_patients": len(pool),
        "pool_slices": sum(len(p.slices) for p in pool),
        "slice_ratio_abn_over_norm": abn_sum / test_normal_slices if test_normal_slices else 0,
        "selected_patients": [p.name for p in selected],
    }


def verify() -> None:
    tr = set(os.listdir(DST / TRACER / "train" / "normal"))
    te = set(os.listdir(DST / TRACER / "test" / "normal"))
    if tr & te:
        raise RuntimeError(f"Overlap train/test normal: {tr & te}")

    for split in ("train", "test"):
        root = DST / TRACER / (split if split == "train" else "test") / "normal"
        for patient in os.listdir(root):
            pdir = root / patient
            if not is_ganzkoerper(patient):
                raise RuntimeError(f"Non-Ganzkoerper in {split}/normal: {patient}")
            for fname in os.listdir(pdir / "pet"):
                if pet_max_value(pdir / "pet" / fname) < MIN_PET_MAX:
                    raise RuntimeError(f"Low-signal PET: {pdir / 'pet' / fname}")

    abn = DST / TRACER / "test" / "abnormal"
    for patient in os.listdir(abn):
        if not is_ganzkoerper(patient):
            raise RuntimeError(f"Non-Ganzkoerper in test/abnormal: {patient}")
        ld = abn / patient / "label"
        for fname in os.listdir(ld):
            if count_foreground_pixels(ld / fname) < MIN_PIXELS:
                raise RuntimeError(f"Small mask: {ld / fname}")


def main():
    global MIN_PET_MAX, MIN_PIXELS, DST
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pet-max", type=int, default=50)
    parser.add_argument("--min-pixels", type=int, default=50)
    parser.add_argument(
        "--dst",
        type=str,
        default="2d_equal_mask50_ganzkoerper",
        help="Output dir name under AutoPET",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    MIN_PET_MAX = args.min_pet_max
    MIN_PIXELS = args.min_pixels
    DST = AUTOPET / args.dst

    if DST.exists():
        if not args.force:
            raise SystemExit(f"Destination exists: {DST} (use --force)")
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    print(f"Building {DST}")
    print(f"  Ganzkoerper only | PET.max()>={MIN_PET_MAX} | mask>={MIN_PIXELS}px")

    train_ids, test_ids, ganz_log = filter_patient_lists()
    print(f"  train normal: {len(train_ids)}, test normal: {len(test_ids)}")
    print(f"  removed non-Ganzkoerper: train={ganz_log['removed_train_count']}, test={ganz_log['removed_test_count']}")

    train_info = rebuild_normal_split("train", train_ids)
    test_info = rebuild_normal_split("test", test_ids)
    abn_info = rebuild_test_abnormal(test_info["slices"], len(test_ids))

    verify()

    report = {
        "source_data": str(SRC_DATA),
        "patient_list_from": {
            "train_normal": str(SRC_DATA / TRACER / "train"),
            "test_normal": str(SRC_DATA / TRACER / "test" / "normal"),
            "test_abnormal_pool": str(SRC_DATA / TRACER / "test" / "abnormal"),
        },
        "tracer": TRACER,
        "scan_filter": "Ganzkoerper only (exclude Teilkoerper/ABDOMEN/Other)",
        "min_pet_max": MIN_PET_MAX,
        "min_pixels": MIN_PIXELS,
        "ganzkoerper_filter": ganz_log,
        "train": {k: v for k, v in train_info.items() if k != "per_patient"},
        "test_normal": {k: v for k, v in test_info.items() if k != "per_patient"},
        "test_abnormal": {k: v for k, v in abn_info.items() if k != "selected_patients"},
        "test_abnormal_selected": abn_info["selected_patients"],
    }

    summary_path = DST / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    readme = DST / "README.md"
    readme.write_text(
        f"# {args.dst}\n\n"
        f"- 仅保留 **Ganzkoerper** 全身 PET/CT（剔除 Teilkoerper / ABDOMEN / Other）\n"
        f"- `train/normal`、`test/normal` 从 `2d_data` 重建，切片条件 **PET.max() >= {MIN_PET_MAX}**\n"
        f"- `test/abnormal` 从 `2d_data` 重选，mask 前景 **>= {MIN_PIXELS}px**\n"
        f"- test normal/abnormal 患者数相等，切片数尽量 1:1\n"
        f"- train/test normal 病人 ID 不重复\n\n"
        f"构建: `python3 build_ganzkoerper_mask50.py --force`\n",
        encoding="utf-8",
    )

    print("\n=== Result ===")
    print(f"train/normal:  {train_info['patients']} patients, {train_info['slices']} slices")
    print(
        f"test/normal:   {test_info['patients']} patients, {test_info['slices']} slices "
        f"(removed {test_info['removed_low_signal_pet_slices']} low-signal)"
    )
    print(
        f"test/abnormal: {abn_info['patients']} patients, {abn_info['slices']} slices "
        f"(ratio={abn_info['slice_ratio_abn_over_norm']:.4f})"
    )
    print(f"\nDone -> {summary_path}")


if __name__ == "__main__":
    main()
