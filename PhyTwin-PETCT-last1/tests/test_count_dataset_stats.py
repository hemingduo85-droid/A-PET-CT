import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestCountDatasetStats(unittest.TestCase):
    def touch(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    def test_patient_layout_counts_unique_patients_and_paired_slices(self):
        from count_dataset_stats import summarize_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "PSMA"
            self.touch(root / "train" / "normal" / "p001" / "pet" / "001.png")
            self.touch(root / "train" / "normal" / "p001" / "ct" / "001.png")
            self.touch(root / "test" / "normal" / "p002" / "pet" / "001.png")
            self.touch(root / "test" / "normal" / "p002" / "ct" / "001.png")
            self.touch(root / "test" / "abnormal" / "p003" / "pet" / "001.png")
            self.touch(root / "test" / "abnormal" / "p003" / "ct" / "001.png")
            self.touch(root / "test" / "abnormal" / "p003" / "label" / "001.png")
            self.touch(root / "test" / "abnormal" / "p003" / "pet" / "002.png")

            summary = summarize_dataset("PSMA", root)

        self.assertEqual(summary["unique_patients"], 3)
        self.assertEqual(summary["total_paired_slices"], 3)
        self.assertEqual(summary["total_pet_slices"], 4)
        self.assertEqual(summary["total_missing_ct"], 1)
        self.assertEqual(summary["sections"][("test", "abnormal")]["patients"], 1)
        self.assertEqual(summary["sections"][("test", "abnormal")]["paired_slices"], 1)
        self.assertEqual(summary["sections"][("test", "abnormal")]["label_slices"], 1)

    def test_flat_layout_counts_slices_without_patient_directories(self):
        from count_dataset_stats import summarize_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "deeppsma"
            self.touch(root / "test" / "normal" / "pet" / "001.png")
            self.touch(root / "test" / "normal" / "ct" / "001.png")
            self.touch(root / "test" / "abnormal" / "pet" / "002.png")
            self.touch(root / "test" / "abnormal" / "ct" / "002.png")
            self.touch(root / "test" / "abnormal" / "label" / "002.png")

            summary = summarize_dataset("deeppsma", root)

        self.assertEqual(summary["unique_patients"], 0)
        self.assertEqual(summary["total_paired_slices"], 2)
        self.assertEqual(summary["sections"][("test", "normal")]["layout"], "flat")
        self.assertEqual(summary["sections"][("test", "abnormal")]["label_slices"], 1)

    def test_parent_root_may_point_at_one_dataset_folder(self):
        from count_dataset_stats import default_dataset_roots

        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "2d_equal_mask50"
            (parent / "PSMA" / "train" / "normal").mkdir(parents=True)
            (parent / "FDG" / "train" / "normal").mkdir(parents=True)

            roots = default_dataset_roots(parent / "PSMA")

        self.assertEqual(roots, [("PSMA", parent / "PSMA"), ("FDG", parent / "FDG")])


if __name__ == "__main__":
    unittest.main()
