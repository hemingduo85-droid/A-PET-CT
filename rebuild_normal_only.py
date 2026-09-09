#!/usr/bin/env python3
"""
Rebuild train/normal and test/normal only in 2d_equal_mask50 (or similar).

- Source: 2d_data/{FDG,PSMA}
- Filter: drop PET slices with max <= min_pet_max; remove paired CT
- PSMA test: remove psma_709da54824f16bc1_2016-07-02, swap in a train patient
- Ensure train/normal and test/normal patient IDs do not overlap
- Does NOT modify test/abnormal
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

AUTOPET = Path("/data/cyf/shared_data/PET-CT/AutoPET")
SRC_DATA = AUTOPET / "2d_data"
SRC_EQUAL = AUTOPET / "2d_equal"
MODALITIES = ["FDG", "PSMA"]

PSMA_REMOVE_TEST = ["psma_709da54824f16bc1_2016-07-02"]
PSMA_SWAP_TRAIN_TO_TEST = "psma_b3930f515d30fd6e_2018-03-31"

MIN_PET_MAX: int
DST: Path


def pet_max_value(pet_path: Path) -> int:
    arr = np.array(Image.open(pet_path))
    return int(arr.max())


def is_valid_pet_slice(pet_path: Path) -> bool:
    return pet_max_value(pet_path) >= MIN_PET_MAX


def valid_slice_names(patient_dir: Path) -> list[str]:
    valid = []
    pet_dir = patient_dir / "pet"
    for fname in sorted(os.listdir(pet_dir)):
        if not fname.lower().endswith(".png"):
            continue
        if is_valid_pet_slice(pet_dir / fname):
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


def resolve_patient_lists(tracer: str, cur_train: list[str], cur_test: list[str]) -> tuple[list[str], list[str], dict]:
    log: dict = {"removed_test": [], "removed_train": [], "added_test": [], "added_train": []}
    train = list(cur_train)
    test = list(cur_test)

    if tracer == "PSMA":
        for p in PSMA_REMOVE_TEST:
            if p in test:
                test.remove(p)
                log["removed_test"].append(p)

        swap = PSMA_SWAP_TRAIN_TO_TEST
        if swap in train and swap not in test:
            train.remove(swap)
            log["removed_train"].append(swap)
            test.append(swap)
            log["added_test"].append(swap)

    overlap = set(train) & set(test)
    if overlap:
        raise RuntimeError(f"[{tracer}] train/test overlap after resolve: {overlap}")

    return sorted(train), sorted(test), log


def resolve_src_patient(tracer: str, split: str, name: str) -> Path:
    if split == "train":
        src_p = SRC_DATA / tracer / "train" / name
    else:
        src_p = SRC_DATA / tracer / "test" / "normal" / name
        if not src_p.is_dir():
            src_p = SRC_DATA / tracer / "train" / name
    if not src_p.is_dir():
        raise FileNotFoundError(f"Missing source patient: {src_p}")
    return src_p


def rebuild_split(tracer: str, split: str, patients: list[str]) -> dict:
    dst_root = DST / tracer / split / "normal"

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)

    removed_slices = 0
    per_patient = {}

    for name in tqdm(patients, desc=f"  {tracer}/{split}/normal"):
        src_p = resolve_src_patient(tracer, split, name)

        all_pet = sorted(f for f in os.listdir(src_p / "pet") if f.endswith(".png"))
        valid = valid_slice_names(src_p)
        removed_slices += len(all_pet) - len(valid)

        if not valid:
            raise RuntimeError(f"[{tracer}] patient {name} has 0 valid PET slices")

        n = copy_normal_patient(src_p, dst_root / name, valid)
        per_patient[name] = {"slices": n, "removed": len(all_pet) - n}

    return {
        "patients": len(patients),
        "slices": sum(v["slices"] for v in per_patient.values()),
        "removed_low_signal_pet_slices": removed_slices,
        "per_patient": per_patient,
    }


def verify(dst_tracer: Path) -> None:
    tr = set(os.listdir(dst_tracer / "train" / "normal"))
    te = set(os.listdir(dst_tracer / "test" / "normal"))
    if tr & te:
        raise RuntimeError(f"Overlap train/test: {tr & te}")

    for split in ("train", "test"):
        for patient in os.listdir(dst_tracer / split / "normal"):
            pdir = dst_tracer / split / "normal" / patient
            for fname in os.listdir(pdir / "pet"):
                pet_path = pdir / "pet" / fname
                if pet_max_value(pet_path) < MIN_PET_MAX:
                    raise RuntimeError(f"Low-signal PET slice remains: {pet_path}")


def main():
    global MIN_PET_MAX, DST
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pet-max", type=int, default=50, help="Keep slices with PET.max() >= this value")
    parser.add_argument("--dst", type=str, default="2d_equal_mask50")
    args = parser.parse_args()

    MIN_PET_MAX = args.min_pet_max
    DST = AUTOPET / args.dst

    if not DST.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DST}")

    summary_path = DST / "dataset_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    report = {"min_pet_max": MIN_PET_MAX, "filter_rule": f"PET.max() >= {MIN_PET_MAX}", "source": str(SRC_DATA), "changes": {}}

    for tracer in MODALITIES:
        print(f"\n=== {tracer} ===")
        # baseline patient IDs from 2d_equal (stable reference, not current dst)
        cur_train = sorted(os.listdir(SRC_EQUAL / tracer / "train" / "normal"))
        cur_test = sorted(os.listdir(SRC_EQUAL / tracer / "test" / "normal"))

        train_ids, test_ids, log = resolve_patient_lists(tracer, cur_train, cur_test)
        print(f"  train: {len(train_ids)}, test: {len(test_ids)}")
        if any(log.values()):
            print(f"  changes: {log}")

        train_info = rebuild_split(tracer, "train", train_ids)
        test_info = rebuild_split(tracer, "test", test_ids)

        verify(DST / tracer)

        print(
            f"  train: {train_info['patients']} patients, {train_info['slices']} slices, "
            f"removed {train_info['removed_low_signal_pet_slices']} low-signal PET"
        )
        print(
            f"  test:  {test_info['patients']} patients, {test_info['slices']} slices, "
            f"removed {test_info['removed_low_signal_pet_slices']} low-signal PET"
        )

        report[tracer] = {
            "changes": log,
            "train": {k: v for k, v in train_info.items() if k != "per_patient"},
            "test_normal": {k: v for k, v in test_info.items() if k != "per_patient"},
        }

        if tracer in summary:
            summary[tracer]["train"] = {
                "normal": {
                    "patients": train_info["patients"],
                    "slices": train_info["slices"],
                }
            }
            if "test" not in summary[tracer]:
                summary[tracer]["test"] = {}
            summary[tracer]["test"]["normal"] = {
                "patients": test_info["patients"],
                "slices": test_info["slices"],
            }

    summary["normal_rebuild"] = report
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    readme = DST / "README.md"
    readme.write_text(
        f"# {DST.name}\n\n"
        f"- normal 切片保留条件: **PET.max() >= {MIN_PET_MAX}**（同步移除对应 CT）\n"
        f"- `train/normal`、`test/normal` 从 `2d_data` 重建\n"
        f"- `test/abnormal` 未改动（mask 前景仍要求 ≥ 50）\n"
        f"- train/test normal 病人 ID 不重复\n\n"
        f"重建 normal: `python3 rebuild_normal_only.py --min-pet-max {MIN_PET_MAX}`\n",
        encoding="utf-8",
    )

    print(f"\nDone. Summary -> {summary_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
