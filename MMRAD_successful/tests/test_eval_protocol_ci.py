import unittest

import numpy as np

from paper_eval_utils import eval_protocol_compute_metrics, eval_protocol_format_metrics


class EvalProtocolCITest(unittest.TestCase):
    def _sample_inputs(self):
        labels = np.array([0, 1, 0, 1], dtype=np.int32)
        masks = np.array(
            [
                [[0, 0], [0, 0]],
                [[1, 0], [0, 1]],
                [[0, 0], [0, 0]],
                [[0, 1], [1, 0]],
            ],
            dtype=np.uint8,
        )
        maps = np.array(
            [
                [[0.1, 0.2], [0.1, 0.2]],
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.2, 0.1], [0.2, 0.1]],
                [[0.1, 0.7], [0.8, 0.1]],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.2, 0.9, 0.1, 0.8], dtype=np.float64)
        paths = [
            "normal__n001__000.png",
            "abnormal__a001__000.png",
            "normal__n002__000.png",
            "abnormal__a002__000.png",
        ]
        return labels, masks, maps, scores, paths

    def test_retained_metrics_have_ci_and_removed_metrics_are_absent(self):
        slice_metrics, patient_metrics = eval_protocol_compute_metrics(
            *self._sample_inputs(), bootstrap_iters=20, ci_seed=7
        )

        expected_slice = {
            "img_auroc",
            "img_ap",
            "img_f1",
            "px_auroc_abn",
            "px_aupr_abn",
        }
        expected_patient = {"pat_auroc", "pat_ap", "pat_f1"}
        for key in expected_slice:
            self.assertIn(key, slice_metrics)
            self.assertIn(f"{key}_ci", slice_metrics)
            self.assertEqual(len(slice_metrics[f"{key}_ci"]), 2)
        for key in expected_patient:
            self.assertIn(key, patient_metrics)
            self.assertIn(f"{key}_ci", patient_metrics)
            self.assertEqual(len(patient_metrics[f"{key}_ci"]), 2)

        for removed in ("px_ap_abn", "px_f1_abn", "px_aupro_abn"):
            self.assertNotIn(removed, slice_metrics)
            self.assertNotIn(f"{removed}_ci", slice_metrics)
        for removed in ("pat_sens90", "pat_sens95"):
            self.assertNotIn(removed, patient_metrics)

    def test_formatted_output_contains_ci_and_image_progress_can_print_first(self):
        progress = []
        slice_metrics, patient_metrics = eval_protocol_compute_metrics(
            *self._sample_inputs(),
            bootstrap_iters=20,
            ci_seed=7,
            progress_callback=progress.append,
        )
        final_text = eval_protocol_format_metrics(slice_metrics, patient_metrics)

        self.assertGreaterEqual(len(progress), 1)
        self.assertIn("[Slice-Img]", progress[0])
        self.assertIn("95%CI", progress[0])
        self.assertTrue(any("[Slice-Px(abn)] starting pixel metrics" in item for item in progress))
        self.assertTrue(any("AUROC DeLong CI" in item for item in progress))
        self.assertIn("[Slice-Img]", final_text)
        self.assertIn("[Slice-Px(abn)]", final_text)
        self.assertIn("AUPR", final_text)
        self.assertIn("95%CI", final_text)
        self.assertNotIn("Sens@90Spec", final_text)
        self.assertNotIn("Sens@95Spec", final_text)
        self.assertNotIn(" AUPRO=", final_text)
        self.assertNotIn(" AP=", final_text.split("[Slice-Px(abn)]", 1)[1].splitlines()[0])
        self.assertNotIn(" F1=", final_text.split("[Slice-Px(abn)]", 1)[1].splitlines()[0])


if __name__ == "__main__":
    unittest.main()
