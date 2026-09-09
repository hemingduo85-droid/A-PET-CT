#!/usr/bin/env python3
"""Count patient and slice totals for PET/CT datasets.

Supports the PhyTwin layout:
  train/normal/<patient>/pet/*.png
  train/normal/<patient>/ct/*.png
  test/normal/<patient>/pet/*.png
  test/normal/<patient>/ct/*.png
  test/abnormal/<patient>/pet/*.png
  test/abnormal/<patient>/ct/*.png
  test/abnormal/<patient>/label/*.png

Also supports flat slice-level folders such as:
  test/normal/pet/*.png
  test/abnormal/pet/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PARENT_ROOT = Path("/data/cyf/shared_data/A-PETCT/2d_equal_mask50")
DEFAULT_DATASETS = ("PSMA", "FDG")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SECTIONS = (("train", "normal"), ("test", "normal"), ("test", "abnormal"))


def image_names(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def is_patient_dir(path: Path) -> bool:
    return path.is_dir() and (path / "pet").is_dir() and (path / "ct").is_dir()


def count_patient_layout(section_root: Path, require_label: bool) -> dict:
    patient_rows = []
    for patient_dir in sorted(section_root.iterdir()) if section_root.is_dir() else []:
        if not is_patient_dir(patient_dir):
            continue
        pet = image_names(patient_dir / "pet")
        ct = image_names(patient_dir / "ct")
        labels = image_names(patient_dir / "label") if require_label else set()
        paired = pet & ct
        label_paired = paired & labels if require_label else set()
        patient_rows.append(
            {
                "patient": patient_dir.name,
                "pet_slices": len(pet),
                "ct_slices": len(ct),
                "paired_slices": len(paired),
                "label_slices": len(labels),
                "paired_label_slices": len(label_paired),
                "missing_ct": len(pet - ct),
                "missing_label": len(paired - labels) if require_label else 0,
            }
        )

    return {
        "layout": "patient",
        "patients": len(patient_rows),
        "patient_ids": [row["patient"] for row in patient_rows],
        "pet_slices": sum(row["pet_slices"] for row in patient_rows),
        "ct_slices": sum(row["ct_slices"] for row in patient_rows),
        "paired_slices": sum(row["paired_slices"] for row in patient_rows),
        "label_slices": sum(row["label_slices"] for row in patient_rows),
        "paired_label_slices": sum(row["paired_label_slices"] for row in patient_rows),
        "missing_ct": sum(row["missing_ct"] for row in patient_rows),
        "missing_label": sum(row["missing_label"] for row in patient_rows),
    }


def count_flat_layout(section_root: Path, require_label: bool) -> dict:
    pet = image_names(section_root / "pet")
    ct = image_names(section_root / "ct")
    labels = image_names(section_root / "label") if require_label else set()
    paired = pet & ct
    label_paired = paired & labels if require_label else set()
    return {
        "layout": "flat",
        "patients": 0,
        "patient_ids": [],
        "pet_slices": len(pet),
        "ct_slices": len(ct),
        "paired_slices": len(paired),
        "label_slices": len(labels),
        "paired_label_slices": len(label_paired),
        "missing_ct": len(pet - ct),
        "missing_label": len(paired - labels) if require_label else 0,
    }


def count_section(root: Path, split: str, group: str) -> dict:
    section_root = root / split / group
    require_label = group == "abnormal"
    if not section_root.is_dir():
        return {
            "layout": "missing",
            "patients": 0,
            "patient_ids": [],
            "pet_slices": 0,
            "ct_slices": 0,
            "paired_slices": 0,
            "label_slices": 0,
            "paired_label_slices": 0,
            "missing_ct": 0,
            "missing_label": 0,
        }

    patient_stats = count_patient_layout(section_root, require_label=require_label)
    if patient_stats["patients"] > 0:
        return patient_stats
    return count_flat_layout(section_root, require_label=require_label)


def summarize_dataset(name: str, root: str | Path) -> dict:
    root = Path(root).expanduser().resolve()
    sections = {}
    patient_ids = set()
    for split, group in SECTIONS:
        stats = count_section(root, split, group)
        sections[(split, group)] = stats
        patient_ids.update(stats["patient_ids"])

    return {
        "name": name,
        "root": str(root),
        "exists": root.is_dir(),
        "unique_patients": len(patient_ids),
        "total_pet_slices": sum(stats["pet_slices"] for stats in sections.values()),
        "total_ct_slices": sum(stats["ct_slices"] for stats in sections.values()),
        "total_paired_slices": sum(stats["paired_slices"] for stats in sections.values()),
        "total_label_slices": sum(stats["label_slices"] for stats in sections.values()),
        "total_missing_ct": sum(stats["missing_ct"] for stats in sections.values()),
        "total_missing_label": sum(stats["missing_label"] for stats in sections.values()),
        "sections": sections,
    }


def default_dataset_roots(parent_root: Path) -> list[tuple[str, Path]]:
    parent_root = parent_root.expanduser()
    if parent_root.name in DEFAULT_DATASETS:
        sibling_parent = parent_root.parent
        if any((sibling_parent / name).is_dir() for name in DEFAULT_DATASETS):
            parent_root = sibling_parent
    return [(name, parent_root / name) for name in DEFAULT_DATASETS]


def named_roots(paths: list[str]) -> list[tuple[str, Path]]:
    roots = []
    for raw in paths:
        path = Path(raw).expanduser()
        roots.append((path.name or str(path), path))
    return roots


def printable_section_name(split: str, group: str) -> str:
    return f"{split}/{group}"


def print_text_report(summaries: list[dict]) -> None:
    header = (
        f"{'Dataset':<10} {'Section':<15} {'Layout':<8} {'Patients':>8} "
        f"{'PET':>8} {'CT':>8} {'Paired':>8} {'Label':>8} {'MissCT':>8} {'MissLabel':>10}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        for split, group in SECTIONS:
            stats = summary["sections"][(split, group)]
            print(
                f"{summary['name']:<10} {printable_section_name(split, group):<15} "
                f"{stats['layout']:<8} {stats['patients']:>8} {stats['pet_slices']:>8} "
                f"{stats['ct_slices']:>8} {stats['paired_slices']:>8} {stats['label_slices']:>8} "
                f"{stats['missing_ct']:>8} {stats['missing_label']:>10}"
            )
        status = "ok" if summary["exists"] else "missing root"
        print(
            f"[{summary['name']}] {status} | patients={summary['unique_patients']} "
            f"paired_slices={summary['total_paired_slices']} pet_slices={summary['total_pet_slices']} "
            f"ct_slices={summary['total_ct_slices']} labels={summary['total_label_slices']} "
            f"root={summary['root']}"
        )
        print()


def json_ready(summary: dict) -> dict:
    converted = dict(summary)
    converted["sections"] = {
        printable_section_name(split, group): stats
        for (split, group), stats in summary["sections"].items()
    }
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count patients and paired PET/CT slices for PSMA/FDG-style datasets."
    )
    parser.add_argument(
        "data_roots",
        nargs="*",
        help="Dataset roots to count. Defaults to PSMA and FDG under --parent_root.",
    )
    parser.add_argument(
        "--parent_root",
        default=str(DEFAULT_PARENT_ROOT),
        help="Parent folder containing PSMA and FDG when data_roots are not provided.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = named_roots(args.data_roots) if args.data_roots else default_dataset_roots(Path(args.parent_root))
    summaries = [summarize_dataset(name, root) for name, root in roots]
    if args.json:
        print(json.dumps([json_ready(summary) for summary in summaries], indent=2, ensure_ascii=False))
    else:
        print_text_report(summaries)


if __name__ == "__main__":
    main()
