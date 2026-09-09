import os
import sys
import unittest
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestPetCTRunConfig(unittest.TestCase):
    def test_default_checkpoint_path_uses_final_epoch(self):
        from petct_run_config import resolve_checkpoint_path

        args = SimpleNamespace(
            ckpt_path=None,
            save_dir="./saved_results",
            save_name="petct_fdg",
            num_epochs=30,
        )

        self.assertEqual(
            resolve_checkpoint_path(args),
            os.path.join("./saved_results", "petct_fdg", "epoch_30_model.pth"),
        )

    def test_explicit_checkpoint_path_wins(self):
        from petct_run_config import resolve_checkpoint_path

        args = SimpleNamespace(
            ckpt_path="/tmp/model.pth",
            save_dir="./saved_results",
            save_name="petct_fdg",
            num_epochs=30,
        )

        self.assertEqual(resolve_checkpoint_path(args), "/tmp/model.pth")

    def test_default_anomaly_dir_lives_under_run_dir(self):
        from petct_run_config import resolve_anomaly_dir

        args = SimpleNamespace(
            anomaly_dir=None,
            save_dir="./saved_results",
            save_name="petct_fdg",
        )

        self.assertEqual(
            resolve_anomaly_dir(args),
            os.path.join("./saved_results", "petct_fdg", "anomaly_maps"),
        )


if __name__ == "__main__":
    unittest.main()
