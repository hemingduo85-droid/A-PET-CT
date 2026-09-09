from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NoTorchmetricsImportTest(unittest.TestCase):
    def test_ae_flow_does_not_import_torchmetrics(self):
        source = (ROOT / "model" / "ae_flow.py").read_text(encoding="utf-8")
        self.assertNotIn("torchmetrics", source)
        self.assertNotIn("structural_similarity_index_measure", source)


if __name__ == "__main__":
    unittest.main()
