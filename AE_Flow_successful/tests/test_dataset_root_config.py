from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DatasetRootConfigTest(unittest.TestCase):
    def test_server_dataset_roots_are_configured(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn('/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA/FDG', source)
        self.assertIn('/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG', source)
        self.assertIn('/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA"', source)
        self.assertIn('parser.add_argument("--data_root", default=None', source)
        self.assertIn("DEFAULT_TRACER_ROOTS", source)
        self.assertIn("No dataset split root found", source)


if __name__ == "__main__":
    unittest.main()
