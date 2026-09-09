import unittest

import numpy as np


class MetricsProtocolTests(unittest.TestCase):
    def test_compute_protocol_metrics_uses_eval_protocol_keys(self):
        import metrics

        labels = np.array([0, 1, 1])
        masks = np.array([
            [[0, 0], [0, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]],
        ])
        maps = np.array([
            [[0.1, 0.2], [0.1, 0.2]],
            [[0.1, 0.9], [0.2, 0.3]],
            [[0.1, 0.2], [0.8, 0.3]],
        ])
        scores = np.array([0.2, 0.9, 0.8])
        paths = [
            "/root/test/normal/p0/pet/001.png",
            "/root/test/abnormal/p1/pet/001.png",
            "/root/test/abnormal/p1/pet/002.png",
        ]

        ev = metrics.compute_protocol_metrics(labels, masks, maps, scores, paths)

        self.assertIn("image_auroc", ev)
        self.assertIn("image_auroc_ci_low", ev)
        self.assertIn("image_ap_ci_high", ev)
        self.assertIn("pixel_auroc_abn", ev)
        self.assertIn("pixel_aupr_abn", ev)
        self.assertIn("pixel_aupr_abn_ci_low", ev)
        self.assertNotIn("pixel_ap_abn", ev)
        self.assertNotIn("pixel_f1_abn", ev)
        self.assertNotIn("patient_sens90", ev)
        self.assertNotIn("patient_sens95", ev)
        self.assertIn("patient_f1_ci_high", ev)
        self.assertNotIn("pixel_auroc", ev)
        self.assertEqual(ev["n_patients"], 2)
        self.assertEqual(ev["n_abn_patients"], 1)

    def test_pixel_aupr_ci_uses_slice_bootstrap_not_flat_pixel_bootstrap(self):
        import metrics

        labels = np.array([0, 1, 1])
        masks = np.array([
            [[0, 0], [0, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]],
        ])
        maps = np.array([
            [[0.1, 0.2], [0.1, 0.2]],
            [[0.1, 0.9], [0.2, 0.3]],
            [[0.1, 0.2], [0.8, 0.3]],
        ])
        scores = np.array([0.2, 0.9, 0.8])
        paths = [
            "/root/test/normal/p0/pet/001.png",
            "/root/test/abnormal/p1/pet/001.png",
            "/root/test/abnormal/p1/pet/002.png",
        ]

        original_bootstrap = metrics._bootstrap_ci

        def fail_on_flat_pixel_bootstrap(*args, **kwargs):
            labels_arg = np.asarray(args[0])
            if labels_arg.size == masks[labels == 1].size:
                raise AssertionError("pixel AUPR CI used flat pixel bootstrap")
            return original_bootstrap(*args, **kwargs)

        metrics._bootstrap_ci = fail_on_flat_pixel_bootstrap
        try:
            ev = metrics.compute_protocol_metrics(labels, masks, maps, scores, paths)
        finally:
            metrics._bootstrap_ci = original_bootstrap

        self.assertIn("pixel_aupr_abn_ci_low", ev)


if __name__ == "__main__":
    unittest.main()
