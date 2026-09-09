import logging
import unittest

import numpy as np

from utils.metrics import compute_metrics


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def squeeze(self):
        return FakeTensor(np.squeeze(self.value))

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class PETCTMetricsTest(unittest.TestCase):
    def test_pixel_metrics_use_abnormal_slices_only_and_patient_metrics_are_logged(self):
        results = {
            "petct": {
                "gt_sp": [0, 1],
                "pr_sp": [0.1, 0.9],
                "imgs_masks": [
                    FakeTensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
                    FakeTensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
                ],
                "anomaly_maps": [
                    FakeTensor([[[0.9, 0.1], [0.1, 0.1]]]),
                    FakeTensor([[[0.9, 0.1], [0.1, 0.1]]]),
                ],
                "img_paths": [
                    "/root/test/normal/patient_a/pet/0000.png",
                    "/root/test/abnormal/patient_b/pet/0000.png",
                ],
            }
        }

        metrics = compute_metrics(results, ["petct"], logging.getLogger("test"))

        self.assertAlmostEqual(metrics["px_auroc"], 1.0)
        self.assertAlmostEqual(metrics["px_aupr"], 1.0)
        self.assertNotIn("px_ap", metrics)
        self.assertNotIn("px_f1", metrics)
        self.assertNotIn("pat_sens90", metrics)
        self.assertNotIn("pat_sens95", metrics)
        self.assertAlmostEqual(metrics["pat_auroc"], 1.0)
        self.assertEqual(metrics["n_patients"], 2)
        self.assertEqual(metrics["n_abn_patients"], 1)


if __name__ == "__main__":
    unittest.main()
