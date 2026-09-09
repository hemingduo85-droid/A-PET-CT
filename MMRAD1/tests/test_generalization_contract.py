from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GeneralizationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval_source = (ROOT / "evaluate_deeppsma_generalization.py").read_text(
            encoding="utf-8-sig"
        )
        cls.train_source = (ROOT / "train.py").read_text(encoding="utf-8-sig")

    def test_deeppsma_reports_distribution_and_skipped_samples(self):
        self.assertIn("skipped_unpaired_or_unmasked", self.eval_source)
        self.assertIn("DeepPSMA distribution:", self.eval_source)

    def test_checkpoint_contains_and_requires_architecture_version(self):
        self.assertIn('"architecture_version": MMRAD_ARCHITECTURE_VERSION', self.train_source)
        self.assertIn("Incompatible MMRAD checkpoint", self.train_source)

    def test_dead_image_score_path_is_removed(self):
        self.assertNotIn("image_score = compute_image_scores(", self.train_source)
        self.assertNotIn("--image_score_mode", self.train_source)
        self.assertNotIn("--image_score_topk_ratio", self.train_source)


if __name__ == "__main__":
    unittest.main()
