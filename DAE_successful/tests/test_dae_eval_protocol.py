import unittest

import numpy as np

from dae_eval_protocol import eval_protocol_compute_metrics, eval_protocol_format_metrics


class DaeEvalProtocolTests(unittest.TestCase):
    def test_format_includes_ci_and_requested_metric_set(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        masks = np.zeros((4, 4, 4), dtype=np.uint8)
        masks[2, 1:3, 1:3] = 1
        masks[3, 0:2, 0:2] = 1
        maps = np.random.default_rng(1).random((4, 4, 4)).astype(np.float32)
        maps[2, 1:3, 1:3] += 1.0
        maps[3, 0:2, 0:2] += 1.0
        paths = [
            "normal/patient_n1/slice_1.png",
            "normal/patient_n2/slice_1.png",
            "abnormal/patient_a1/slice_1.png",
            "abnormal/patient_a2/slice_1.png",
        ]

        slice_metrics, patient_metrics = eval_protocol_compute_metrics(
            labels,
            masks,
            maps,
            scores,
            paths,
            bootstrap_iters=20,
            hist_bins=64,
            random_state=7,
        )
        text = eval_protocol_format_metrics(slice_metrics, patient_metrics)

        self.assertIn("95% CI", text)
        self.assertIn("[Slice-Px(abn)]", text)
        self.assertIn("AUPR=", text)
        self.assertNotIn("Sens@90Spec", text)
        self.assertNotIn("Sens@95Spec", text)
        pixel_line = next(line for line in text.splitlines() if line.startswith("[Slice-Px"))
        self.assertNotIn(" AP=", pixel_line)
        self.assertNotIn(" F1=", pixel_line)

    def test_progress_messages_match_gatingano_protocol(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.3, 0.7, 0.9])
        masks = np.zeros((4, 3, 3), dtype=np.uint8)
        masks[2, 1, 1] = 1
        masks[3, 0, 0] = 1
        maps = np.random.default_rng(2).random((4, 3, 3)).astype(np.float32)
        paths = [
            "normal/patient_n1/slice_1.png",
            "normal/patient_n2/slice_1.png",
            "abnormal/patient_a1/slice_1.png",
            "abnormal/patient_a2/slice_1.png",
        ]
        messages = []

        eval_protocol_compute_metrics(
            labels,
            masks,
            maps,
            scores,
            paths,
            bootstrap_iters=50,
            hist_bins=32,
            random_state=9,
            progress_callback=messages.append,
        )

        self.assertIn("Computing image-level metrics and 95CI...", messages)
        self.assertIn("Computing pixel-level Slice-Px(abn) metrics and 95CI...", messages)
        self.assertIn("Computing exact full-pixel point estimates with sklearn...", messages)
        self.assertIn("Pixel histogram slice bootstrap: 50/50", messages)
        self.assertIn("Computing patient-level metrics and 95CI...", messages)


if __name__ == "__main__":
    unittest.main()
