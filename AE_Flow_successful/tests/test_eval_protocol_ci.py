import unittest

import numpy as np

from paper_eval_utils import (
    eval_protocol_compute_metrics,
    eval_protocol_format_image_metrics,
    eval_protocol_format_metrics,
)


class EvalProtocolCITest(unittest.TestCase):
    def test_metrics_include_ci_and_requested_metric_set(self):
        labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
        scores = np.array([0.05, 0.20, 0.72, 0.83, 0.35, 0.91], dtype=np.float64)
        masks = np.zeros((6, 4, 4), dtype=np.uint8)
        maps = np.full((6, 4, 4), 0.1, dtype=np.float32)
        for idx in np.where(labels == 1)[0]:
            masks[idx, 1:3, 1:3] = 1
            maps[idx, 1:3, 1:3] = 0.9
            maps[idx, masks[idx] == 0] = 0.2
        paths = [
            "normal__p0__000.png",
            "normal__p1__000.png",
            "abnormal__p2__000.png",
            "abnormal__p3__000.png",
            "normal__p4__000.png",
            "abnormal__p5__000.png",
        ]

        slice_metrics, pat_metrics = eval_protocol_compute_metrics(
            labels,
            masks,
            maps,
            scores,
            paths,
            bootstrap_iters=25,
            ci_seed=7,
        )

        for key in ("img_auroc", "img_ap", "img_f1", "px_auroc_abn", "px_aupr_abn"):
            self.assertIn(key, slice_metrics)
            self.assertIn(f"{key}_ci", slice_metrics)
            self.assertEqual(len(slice_metrics[f"{key}_ci"]), 2)

        for key in ("px_ap_abn", "px_f1_abn", "px_aupro_abn"):
            self.assertNotIn(key, slice_metrics)

        for key in ("pat_auroc", "pat_ap", "pat_f1"):
            self.assertIn(key, pat_metrics)
            self.assertIn(f"{key}_ci", pat_metrics)

        for key in ("pat_sens90", "pat_sens95"):
            self.assertNotIn(key, pat_metrics)

        image_text = eval_protocol_format_image_metrics(slice_metrics)
        self.assertIn("[Slice-Img]", image_text)
        self.assertIn("95% CI", image_text)
        self.assertIn("AUPR", image_text)
        self.assertNotIn(" AP=", image_text)
        self.assertNotIn("[Slice-Px", image_text)
        self.assertNotIn("[Patient", image_text)

        full_text = eval_protocol_format_metrics(slice_metrics, pat_metrics)
        self.assertIn("[Slice-Img]", full_text)
        self.assertIn("[Slice-Px(abn)]", full_text)
        self.assertIn("AUPR", full_text)
        self.assertNotIn("AUPRO", full_text)
        self.assertNotIn("Sens@90Spec", full_text)
        self.assertNotIn("Sens@95Spec", full_text)

    def test_protocol_uses_gatingano_bootstrap_contract(self):
        with open("paper_eval_utils.py", encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("_delong_auc_ci", source)
        self.assertNotIn("NormalDist", source)
        self.assertIn("_pixel_slice_bootstrap", source)
        self.assertIn("Computing exact full-pixel point estimates with sklearn", source)
        self.assertIn("Pixel histogram slice bootstrap", source)


if __name__ == "__main__":
    unittest.main()
