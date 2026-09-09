from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GatingAnoAuditContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_source = (ROOT / "train.py").read_text(encoding="utf-8-sig")
        cls.loader_source = (ROOT / "dataloader.py").read_text(encoding="utf-8-sig")
        cls.eval_source = (ROOT / "eval.py").read_text(encoding="utf-8-sig")
        cls.generalization_source = (
            ROOT / "evaluate_deeppsma_generalization.py"
        ).read_text(encoding="utf-8-sig")
        cls.cache_source = (ROOT / "compute_cache_metrics.py").read_text(
            encoding="utf-8-sig"
        )

    def test_training_metrics_use_exact_full_abnormal_pixels(self):
        self.assertNotIn("def _subsample_binary_scores", self.train_source)
        self.assertIn("_safe_auroc(gt_px_abn, pr_px_abn)", self.train_source)
        self.assertIn("_safe_ap(gt_px_abn, pr_px_abn)", self.train_source)

    def test_training_metrics_use_histogram_slice_bootstrap(self):
        self.assertIn("def bootstrap_pixel_hist_ci", self.train_source)
        self.assertIn("Pixel histogram slice bootstrap:", self.train_source)
        self.assertIn("ci_hist_bins", self.train_source)

    def test_offline_cache_computes_exact_pixel_point_estimates(self):
        self.assertIn("exact_auroc = _safe_auroc(px_true, px_score)", self.cache_source)
        self.assertIn("exact_aupr = _safe_ap(px_true, px_score)", self.cache_source)

    def test_masks_are_resized_with_nearest_neighbor(self):
        self.assertIn(
            "mask.resize((self.image_size, self.image_size), Image.NEAREST)",
            self.loader_source,
        )

    def test_checkpoint_metadata_is_validated_for_all_evaluators(self):
        self.assertIn("def validate_checkpoint_metadata", self.train_source)
        self.assertIn("validate_checkpoint_metadata(", self.eval_source)
        self.assertIn("validate_checkpoint_metadata(", self.generalization_source)

    def test_deeppsma_reports_skipped_samples(self):
        self.assertIn("skipped_unpaired_or_unmasked", self.generalization_source)


if __name__ == "__main__":
    unittest.main()
