import json
import unittest

import numpy as np

from paper_eval_utils import (
    eval_protocol_compute_metrics,
    eval_protocol_format_image_metrics,
    eval_protocol_format_metrics,
    json_ready_metrics,
)


class EvalProtocolCITest(unittest.TestCase):
    def test_ci_metric_contract_and_image_progress_callback(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        masks = np.array(
            [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[1, 0], [0, 0]],
                [[0, 1], [1, 0]],
            ],
            dtype=np.uint8,
        )
        maps = np.array(
            [
                [[0.1, 0.2], [0.1, 0.0]],
                [[0.2, 0.1], [0.1, 0.0]],
                [[0.9, 0.2], [0.3, 0.1]],
                [[0.2, 0.8], [0.7, 0.1]],
            ],
            dtype=np.float32,
        )
        image_scores = np.array([0.10, 0.20, 0.80, 0.90], dtype=np.float32)
        paths = [
            "normal__p001__001",
            "normal__p002__001",
            "abnormal__p003__001",
            "abnormal__p004__001",
        ]
        progress = []

        slice_metrics, pat_metrics = eval_protocol_compute_metrics(
            labels,
            masks,
            maps,
            image_scores,
            paths,
            bootstrap_iters=10,
            ci_pixel_max_samples=8,
            progress_callback=progress.append,
        )
        image_line = eval_protocol_format_image_metrics(slice_metrics)
        full_output = eval_protocol_format_metrics(slice_metrics, pat_metrics)

        self.assertEqual(
            progress,
            [
                "Computing image-level metrics and 95CI...",
                image_line,
                "Computing pixel-level Slice-Px(abn) metrics and 95CI...",
                "Computing exact full-pixel point estimates with sklearn...",
                "Pixel histogram slice bootstrap: 10/10",
                "Computing patient-level metrics and 95CI...",
            ],
        )
        self.assertEqual(full_output.count("95% CI"), 8)
        self.assertNotIn("95%CI", full_output)
        self.assertIn("[Slice-Img]", full_output)
        self.assertIn("[Slice-Px(abn)]", full_output)
        self.assertIn("AUPR=", full_output)
        self.assertNotIn("Sens@90Spec", full_output)
        self.assertNotIn("Sens@95Spec", full_output)
        pixel_line = next(line for line in full_output.splitlines() if line.startswith("[Slice-Px"))
        self.assertNotIn("AP=", pixel_line)
        self.assertNotIn("F1=", pixel_line)

    def test_metrics_payload_is_json_serializable_without_internal_arrays(self):
        metrics = {
            "img_auroc": 0.9,
            "img_auroc_ci": (0.8, 1.0),
            "_pr_sp": np.array([0.1, 0.9], dtype=np.float32),
            "_gt_sp": np.array([0, 1], dtype=np.int64),
        }

        public_metrics = json_ready_metrics(metrics)

        json.dumps(public_metrics)
        self.assertEqual(public_metrics["img_auroc"], 0.9)
        self.assertEqual(public_metrics["img_auroc_ci"], [0.8, 1.0])
        self.assertNotIn("_pr_sp", public_metrics)
        self.assertNotIn("_gt_sp", public_metrics)


if __name__ == "__main__":
    unittest.main()
