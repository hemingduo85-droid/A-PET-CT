import unittest

import numpy as np

from utils import compute_petct_metrics, format_petct_metrics


class PetCTEvalProtocolTest(unittest.TestCase):
    def test_matches_phytwin_slice_pixel_and_patient_protocol(self):
        labels = [0, 1, 1, 0]
        masks = np.array(
            [
                [[0, 0], [0, 0]],
                [[1, 0], [0, 0]],
                [[0, 1], [1, 0]],
                [[0, 0], [0, 0]],
            ],
            dtype=np.float32,
        )
        maps = np.array(
            [
                [[0.1, 0.2], [0.2, 0.3]],
                [[0.9, 0.1], [0.2, 0.4]],
                [[0.1, 0.8], [0.7, 0.2]],
                [[0.2, 0.1], [0.2, 0.1]],
            ],
            dtype=np.float32,
        )
        scores = [0.3, 0.9, 0.8, 0.2]
        paths = [
            "/data/test/normal/patient_a/ct/000.png",
            "/data/test/abnormal/patient_b/ct/000.png",
            "/data/test/abnormal/patient_b/ct/001.png",
            "/data/test/normal/patient_c/ct/000.png",
        ]

        slice_metrics, patient_metrics = compute_petct_metrics(labels, masks, maps, scores, paths)

        self.assertEqual(slice_metrics["_gt_sp"].tolist(), labels)
        self.assertEqual(patient_metrics["n_patients"], 3)
        self.assertEqual(patient_metrics["n_abn_patients"], 1)
        self.assertAlmostEqual(slice_metrics["img_auroc"], 1.0)
        self.assertAlmostEqual(slice_metrics["img_ap"], 1.0)
        self.assertAlmostEqual(slice_metrics["px_auroc_abn"], 1.0)
        self.assertAlmostEqual(slice_metrics["px_ap_abn"], 1.0)
        self.assertAlmostEqual(patient_metrics["pat_auroc"], 1.0)
        self.assertAlmostEqual(patient_metrics["pat_ap"], 1.0)

        formatted = format_petct_metrics(slice_metrics, patient_metrics)
        self.assertIn("[Slice-Img]", formatted)
        self.assertIn("[Slice-Px(abn)]", formatted)
        self.assertIn("[Patient(1/3abn)]", formatted)


if __name__ == "__main__":
    unittest.main()
