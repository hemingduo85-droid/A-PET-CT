import unittest

import numpy as np

from paper_eval_utils import eval_protocol_compute_metrics, eval_protocol_format_metrics


class EvalProtocolTest(unittest.TestCase):
    def test_requested_metric_set_and_ci_output(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        scores = np.array([0.05, 0.20, 0.70, 0.95], dtype=np.float64)
        masks = np.zeros((4, 4, 4), dtype=np.uint8)
        maps = np.zeros((4, 4, 4), dtype=np.float32)
        masks[2, 1:3, 1:3] = 1
        masks[3, 0:2, 0:2] = 1
        maps[2] = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
        maps[3] = np.flipud(maps[2])
        paths = [
            "normal__p0__s0",
            "normal__p1__s0",
            "abnormal__p2__s0",
            "abnormal__p3__s0",
        ]

        progress_messages = []
        slice_metrics, patient_metrics = eval_protocol_compute_metrics(
            labels,
            masks,
            maps,
            scores,
            paths,
            bootstrap_iters=20,
            ci_pixel_max_samples=0,
            progress_callback=progress_messages.append,
        )
        text = eval_protocol_format_metrics(slice_metrics, patient_metrics)

        for removed in ("Sens@90Spec", "Sens@95Spec", "px_ap_abn", "px_f1_abn", "AUPRO"):
            self.assertNotIn(removed, text)
            self.assertNotIn(removed, slice_metrics)
            self.assertNotIn(removed, patient_metrics)

        for key in ("img_auroc", "img_ap", "img_f1", "px_auroc_abn", "px_aupr_abn"):
            self.assertIn(key, slice_metrics)
            self.assertIn(f"{key}_ci", slice_metrics)

        for key in ("pat_auroc", "pat_ap", "pat_f1"):
            self.assertIn(key, patient_metrics)
            self.assertIn(f"{key}_ci", patient_metrics)

        self.assertIn("AUPR=", text)
        self.assertEqual(text.count("95% CI"), 8)
        self.assertEqual(progress_messages[0], "Computing image-level metrics and 95CI...")
        self.assertTrue(progress_messages[1].startswith("[Slice-Img]"))
        self.assertEqual(progress_messages[2], "Computing pixel-level Slice-Px(abn) metrics and 95CI...")
        self.assertEqual(progress_messages[3], "Computing exact full-pixel point estimates with sklearn...")
        self.assertTrue(progress_messages[-2].startswith("[Slice-Px(abn)]"))
        self.assertEqual(progress_messages[-1], "Computing patient-level metrics and 95CI...")
        self.assertIn("AUPR=", progress_messages[1])


if __name__ == "__main__":
    unittest.main()
