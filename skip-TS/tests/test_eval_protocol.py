import unittest

import numpy as np

from eval_protocol import compute_metrics, format_metrics, patient_id_from_path


class EvalProtocolTest(unittest.TestCase):
    def test_patient_id_from_pet_or_ct_slice_path(self):
        path = "/data/root/test/abnormal/case001/ct/slice001.png"
        self.assertEqual(patient_id_from_path(path), "case001")

    def test_compute_metrics_returns_slice_and_patient_metrics(self):
        labels = np.array([0, 1, 1])
        masks = np.zeros((3, 4, 4), dtype=np.float32)
        masks[1, 1:3, 1:3] = 1
        maps = np.zeros((3, 4, 4), dtype=np.float32)
        maps[0] = 0.1
        maps[1, 1:3, 1:3] = 0.9
        maps[2] = 0.2
        scores = np.array([0.1, 0.9, 0.2])
        paths = [
            "/data/root/test/normal/case000/pet/slice001.png",
            "/data/root/test/abnormal/case001/pet/slice001.png",
            "/data/root/test/abnormal/case001/pet/slice002.png",
        ]

        slice_metrics, pat_metrics = compute_metrics(
            labels,
            masks,
            maps,
            scores,
            paths,
            bootstrap_iters=20,
        )

        self.assertEqual(slice_metrics["img_aupr"], 1.0)
        self.assertIn("img_aupr_ci", slice_metrics)
        self.assertIn("img_auroc_ci", slice_metrics)
        self.assertEqual(slice_metrics["px_aupr_abn"], 1.0)
        self.assertIn("px_aupr_abn_ci", slice_metrics)
        self.assertNotIn("px_ap_abn", slice_metrics)
        self.assertNotIn("px_f1_abn", slice_metrics)
        self.assertEqual(pat_metrics["n_patients"], 2)
        self.assertEqual(pat_metrics["n_abn_patients"], 1)
        self.assertEqual(pat_metrics["pat_aupr"], 1.0)
        self.assertIn("pat_aupr_ci", pat_metrics)
        self.assertNotIn("pat_sens90", pat_metrics)
        self.assertNotIn("pat_sens95", pat_metrics)

        output = format_metrics(slice_metrics, pat_metrics)
        self.assertIn("[Slice-Px(abn)]", output)
        self.assertIn("AUPR=", output)
        self.assertNotIn(" AP=", output)
        self.assertNotIn("Sens@90Spec", output)
        self.assertNotIn("Sens@95Spec", output)
        self.assertNotIn("px_ap", slice_metrics)
        self.assertNotIn("px_f1", slice_metrics)


if __name__ == "__main__":
    unittest.main()
