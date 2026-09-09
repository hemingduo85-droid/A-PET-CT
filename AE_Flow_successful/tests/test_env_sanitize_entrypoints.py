from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EnvSanitizeEntrypointsTest(unittest.TestCase):
    def test_train_and_infer_disable_user_site_before_model_import(self):
        for name in ("train.py", "infer.py"):
            source = (ROOT / name).read_text(encoding="utf-8")
            call_index = source.index("disable_user_site_packages()")
            model_index = source.index("from model import ae_flow")
            self.assertLess(call_index, model_index, name)


if __name__ == "__main__":
    unittest.main()
