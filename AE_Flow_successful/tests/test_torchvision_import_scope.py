from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TorchvisionImportScopeTest(unittest.TestCase):
    def test_only_encoder_imports_torchvision(self):
        files = {
            path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in (ROOT / "dataloader.py", ROOT / "model" / "decoder.py", ROOT / "model" / "encoder.py")
        }

        self.assertNotIn("torchvision", files["dataloader.py"])
        self.assertNotIn("torchvision", files["model/decoder.py"])
        self.assertIn("torchvision.models", files["model/encoder.py"])


if __name__ == "__main__":
    unittest.main()
