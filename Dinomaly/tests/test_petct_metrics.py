import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestPetCTMetrics(unittest.TestCase):
    @staticmethod
    def _binary_auroc(y_true, y_score):
        y_true = np.asarray(y_true).astype(bool)
        y_score = np.asarray(y_score, dtype=np.float64)
        pos = y_score[y_true]
        neg = y_score[~y_true]
        total = 0.0
        for p in pos:
            total += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        return total / (len(pos) * len(neg))

    def test_metrics_match_phytwin_patient_and_abnormal_pixel_protocol(self):
        from petct_metrics import compute_petct_metrics

        labels = [0, 1, 1]
        masks = np.array([
            [[1, 1], [1, 1]],
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
        ], dtype=np.float32)
        maps = np.array([
            [[0.9, 0.9], [0.9, 0.9]],
            [[0.1, 0.8], [0.2, 0.7]],
            [[0.6, 0.3], [0.5, 0.4]],
        ], dtype=np.float32)
        image_scores = [0.4, 0.8, 0.6]
        paths = [
            "/data/test/normal/patient_n/ct/001.png",
            "/data/test/abnormal/patient_a/ct/001.png",
            "/data/test/abnormal/patient_a/ct/002.png",
        ]

        slice_metrics, patient_metrics = compute_petct_metrics(
            labels,
            masks,
            maps,
            image_scores,
            paths,
            n_bootstrap=5,
            pixel_bootstrap_max_samples=20,
        )

        expected_px_gt = masks[np.array(labels) == 1].reshape(-1).astype(bool)
        expected_px_pr = maps[np.array(labels) == 1].reshape(-1)
        self.assertAlmostEqual(slice_metrics["px_auroc_abn"]["value"], self._binary_auroc(expected_px_gt, expected_px_pr))

        self.assertAlmostEqual(slice_metrics["img_auroc"]["value"], self._binary_auroc(labels, image_scores))
        self.assertIn("ci", slice_metrics["img_auroc"])
        self.assertIn("px_aupr_abn", slice_metrics)
        self.assertNotIn("px_ap_abn", slice_metrics)
        self.assertNotIn("px_f1_abn", slice_metrics)
        self.assertEqual(patient_metrics["n_patients"], 2)
        self.assertEqual(patient_metrics["n_abn_patients"], 1)
        self.assertAlmostEqual(patient_metrics["pat_auroc"]["value"], 1.0)
        self.assertIn("pat_aupr", patient_metrics)
        self.assertNotIn("pat_sens90", patient_metrics)
        self.assertNotIn("pat_sens95", patient_metrics)

    def test_single_class_metrics_return_zero(self):
        from petct_metrics import compute_petct_metrics

        labels = [0, 0]
        masks = np.zeros((2, 2, 2), dtype=np.float32)
        maps = np.ones((2, 2, 2), dtype=np.float32)
        scores = [0.2, 0.3]
        paths = [
            "/data/test/normal/patient_1/ct/001.png",
            "/data/test/normal/patient_2/ct/001.png",
        ]

        slice_metrics, patient_metrics = compute_petct_metrics(labels, masks, maps, scores, paths, n_bootstrap=5)

        self.assertEqual(slice_metrics["img_auroc"]["value"], 0.0)
        self.assertEqual(slice_metrics["px_auroc_abn"]["value"], 0.0)
        self.assertEqual(patient_metrics["pat_aupr"]["value"], 0.0)


if __name__ == "__main__":
    unittest.main()
