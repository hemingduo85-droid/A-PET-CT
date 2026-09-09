import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestEvalProtocol(unittest.TestCase):
    def test_histogram_metrics_support_numpy_without_trapezoid(self):
        from phytwin_petct.eval_protocol import _metrics_from_hist

        pos_hist = np.asarray([0, 1, 1], dtype=np.uint32)
        neg_hist = np.asarray([1, 1, 0], dtype=np.uint32)
        with mock.patch.object(np, "trapezoid", None, create=True):
            auroc, aupr = _metrics_from_hist(pos_hist, neg_hist)

        self.assertAlmostEqual(auroc, 0.875)
        self.assertAlmostEqual(aupr, 5.0 / 6.0)

    def test_metrics_include_bootstrap_ci_for_each_reported_metric(self):
        from phytwin_petct.eval_protocol import compute_metrics, format_metrics

        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32)
        masks = np.zeros((6, 8, 8), dtype=np.float32)
        maps = np.zeros((6, 8, 8), dtype=np.float32)
        for i, label in enumerate(labels):
            if label:
                masks[i, 2:5, 2:5] = 1.0
                maps[i, 2:5, 2:5] = 0.9
                maps[i, 0:2, 0:2] = 0.2
            else:
                maps[i, :, :] = 0.1
        scores = np.asarray([0.1, 0.8, 0.2, 0.9, 0.3, 0.7], dtype=np.float32)
        paths = [f"/data/patient_{i}/study/slice_{i}.png" for i in range(len(labels))]

        slice_metrics, pat_metrics = compute_metrics(
            labels, masks, maps, scores, paths, bootstrap_iters=25, bootstrap_seed=7
        )

        for name in ("img_auroc", "img_ap", "img_f1", "px_auroc_abn", "px_aupr_abn"):
            self.assertIn(f"{name}_ci", slice_metrics)
            low, high = slice_metrics[f"{name}_ci"]
            self.assertLessEqual(low, slice_metrics[name])
            self.assertGreaterEqual(high, slice_metrics[name])

        self.assertNotIn("px_ap_abn", slice_metrics)
        self.assertNotIn("px_f1_abn", slice_metrics)
        self.assertNotIn("px_pro_abn", slice_metrics)

        for name in ("pat_auroc", "pat_ap", "pat_f1"):
            self.assertIn(f"{name}_ci", pat_metrics)
            low, high = pat_metrics[f"{name}_ci"]
            self.assertLessEqual(low, pat_metrics[name])
            self.assertGreaterEqual(high, pat_metrics[name])

        self.assertNotIn("pat_sens90", pat_metrics)
        self.assertNotIn("pat_sens95", pat_metrics)

        formatted = format_metrics(slice_metrics, pat_metrics)
        self.assertIn("95% CI", formatted)
        self.assertIn("AUROC=", formatted)
        self.assertIn("AUPR=", formatted)
        self.assertNotIn("PRO=", formatted)
        self.assertNotIn("Sens@90Spec", formatted)
        self.assertNotIn("Sens@95Spec", formatted)

    def test_slice_ap_ci_uses_slice_bootstrap(self):
        from phytwin_petct.eval_protocol import compute_metrics
        from sklearn.metrics import average_precision_score

        labels = np.asarray([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.int32)
        scores = np.asarray([0.05, 0.95, 0.20, 0.80, 0.15, 0.70, 0.40, 0.60], dtype=np.float32)
        masks = np.zeros((8, 4, 4), dtype=np.float32)
        maps = np.zeros((8, 4, 4), dtype=np.float32)
        paths = [
            "/data/patient_a/study/slice_0.png",
            "/data/patient_a/study/slice_1.png",
            "/data/patient_a/study/slice_2.png",
            "/data/patient_b/study/slice_0.png",
            "/data/patient_b/study/slice_1.png",
            "/data/patient_b/study/slice_2.png",
            "/data/patient_c/study/slice_0.png",
            "/data/patient_d/study/slice_0.png",
        ]

        slice_metrics, _ = compute_metrics(
            labels, masks, maps, scores, paths, bootstrap_iters=50, bootstrap_seed=11
        )

        rng = np.random.default_rng(11)
        values = []
        for _ in range(50):
            idx = rng.integers(0, len(labels), size=len(labels))
            if len(np.unique(labels[idx])) < 2:
                continue
            values.append(float(average_precision_score(labels[idx], scores[idx])))
        expected = np.percentile(values, [2.5, 97.5])
        expected[0] = min(float(expected[0]), slice_metrics["img_ap"])
        expected[1] = max(float(expected[1]), slice_metrics["img_ap"])

        np.testing.assert_allclose(slice_metrics["img_ap_ci"], expected, rtol=1e-12, atol=1e-12)

    def test_auroc_ci_uses_bootstrap_seed(self):
        from phytwin_petct.eval_protocol import compute_metrics

        labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int32)
        scores = np.asarray([0.2, 0.4, 0.3, 0.8, 0.1, 0.7], dtype=np.float32)
        masks = np.zeros((6, 4, 4), dtype=np.float32)
        maps = np.zeros((6, 4, 4), dtype=np.float32)
        masks[labels == 1, 1:3, 1:3] = 1.0
        maps[labels == 1, 1:3, 1:3] = 0.9
        maps[labels == 0] = 0.1
        paths = [f"/data/patient_{i}/study/slice_{i}.png" for i in range(len(labels))]

        slice_a, pat_a = compute_metrics(labels, masks, maps, scores, paths, bootstrap_iters=10, bootstrap_seed=1)
        slice_b, pat_b = compute_metrics(labels, masks, maps, scores, paths, bootstrap_iters=10, bootstrap_seed=99)

        self.assertNotEqual(slice_a["img_auroc_ci"], slice_b["img_auroc_ci"])
        self.assertNotEqual(pat_a["pat_auroc_ci"], pat_b["pat_auroc_ci"])

    def test_progress_callback_reports_slice_image_metrics_first(self):
        from phytwin_petct.eval_protocol import compute_metrics

        labels = np.asarray([0, 1, 0, 1], dtype=np.int32)
        masks = np.zeros((4, 4, 4), dtype=np.float32)
        maps = np.zeros((4, 4, 4), dtype=np.float32)
        masks[labels == 1, 1:3, 1:3] = 1.0
        maps[labels == 1, 1:3, 1:3] = 0.9
        maps[labels == 0] = 0.1
        scores = np.asarray([0.1, 0.8, 0.2, 0.7], dtype=np.float32)
        paths = [f"/data/patient_{i}/study/slice_{i}.png" for i in range(len(labels))]
        messages = []

        compute_metrics(
            labels,
            masks,
            maps,
            scores,
            paths,
            bootstrap_iters=5,
            bootstrap_seed=3,
            progress_callback=messages.append,
        )

        self.assertGreaterEqual(len(messages), 5)
        self.assertIn("Computing image-level metrics and 95CI...", messages[0])
        self.assertIn("[Slice-Img]", messages[1])
        self.assertIn("95% CI", messages[1])
        self.assertIn("Computing pixel-level Slice-Px(abn) metrics and 95CI...", messages[2])
        self.assertIn("Computing exact full-pixel point estimates with sklearn...", messages[3])
        self.assertTrue(any("Pixel histogram slice bootstrap:" in msg for msg in messages))
        self.assertTrue(any(msg.startswith("[Slice-Px(abn)]") for msg in messages))


if __name__ == "__main__":
    unittest.main()
