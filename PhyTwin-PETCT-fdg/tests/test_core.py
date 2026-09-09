
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestPhyTwinCore(unittest.TestCase):
    def test_normal_twin_forward_loss_and_residual(self):
        from phytwin_petct.models.normal_twin import NormalTwinUNet, normal_twin_loss, residual_map

        model = NormalTwinUNet(in_channels=3, out_channels=3, base_channels=8, uncertainty=False)
        ct = torch.randn(2, 3, 64, 64)
        pet = torch.randn(2, 3, 64, 64)
        pred, logvar = model(ct)
        loss, parts = normal_twin_loss(pred, logvar, pet, nll_weight=0.0)
        residual = residual_map(pet, pred, logvar=None)

        self.assertEqual(tuple(pred.shape), tuple(pet.shape))
        self.assertEqual(tuple(residual.shape), (2, 64, 64))
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        self.assertIn("l1", parts)

    def test_residual_memory_scores_far_patch_higher(self):
        from phytwin_petct.memory import ResidualPatchMemory

        memory = ResidualPatchMemory(patch_size=4, stride=4, max_patches=128, k=1)
        memory.fit(torch.zeros(4, 16, 16))
        near_map, near_score = memory.score_map(torch.zeros(1, 16, 16))
        far = torch.zeros(1, 16, 16)
        far[:, 4:12, 4:12] = 5.0
        far_map, far_score = memory.score_map(far)

        self.assertEqual(tuple(near_map.shape), (1, 16, 16))
        self.assertGreater(float(far_score[0]), float(near_score[0]))
        self.assertGreater(float(far_map.max()), float(near_map.max()))

    def test_physio_prior_suppresses_normal_hot_regions(self):
        from phytwin_petct.physio import PhysiologicalUptakePrior

        prior = PhysiologicalUptakePrior(alpha=0.8, sigma=0.0)
        maps = np.zeros((4, 16, 16), dtype=np.float32)
        maps[:, 4:8, 4:8] = 1.0
        prior.fit_from_pet_maps(maps)
        suppressed = prior.suppress(np.ones((16, 16), dtype=np.float32))

        self.assertLess(float(suppressed[5, 5]), float(suppressed[12, 12]))

    def test_component_score_penalizes_prior_overlap(self):
        from phytwin_petct.scoring import lesion_component_score

        score_map = np.zeros((32, 32), dtype=np.float32)
        score_map[4:10, 4:10] = 2.0
        score_map[20:26, 20:26] = 2.0
        prior = np.zeros((32, 32), dtype=np.float32)
        prior[4:10, 4:10] = 1.0

        score_without_prior = lesion_component_score(score_map, physio_prior=None)
        score_with_prior = lesion_component_score(score_map, physio_prior=prior, prior_penalty=0.8)

        self.assertLess(score_with_prior, score_without_prior)
        self.assertGreater(score_with_prior, 0.0)

    def test_visualization_writes_png_with_gt_panel(self):
        from phytwin_petct.visualization import save_case_visualization

        out = "/private/tmp/phytwin_vis_test.png"
        pet = torch.zeros(3, 32, 32)
        ct = torch.zeros(3, 32, 32)
        mask = torch.zeros(1, 32, 32)
        mask[:, 10:18, 12:20] = 1
        arr = np.zeros((32, 32), dtype=np.float32)
        arr[10:18, 12:20] = 1
        save_case_visualization(pet, ct, mask, arr, arr, arr, out)

        self.assertTrue(os.path.isfile(out))
        self.assertGreater(os.path.getsize(out), 0)

    def test_predict_mask_from_normal_threshold_keeps_extended_region(self):
        from phytwin_petct.scoring import predict_mask_from_threshold

        score = np.zeros((32, 32), dtype=np.float32)
        score[8:20, 10:24] = 1.0
        score[12:16, 14:18] = 2.0
        pred = predict_mask_from_threshold(score, threshold=0.7, min_area=8, close_iters=1)

        self.assertGreater(int(pred.sum()), 100)
        self.assertEqual(pred[13, 15], 1.0)


    def test_dynamic_hotspot_suppression_reduces_large_physio_blob(self):
        from phytwin_petct.dynamic_physio import DynamicPhysioSuppressor

        pet = np.zeros((32, 32), dtype=np.float32)
        pet[20:29, 10:22] = 1.0
        anomaly = np.ones((32, 32), dtype=np.float32)
        prior = np.zeros((32, 32), dtype=np.float32)
        prior[18:31, 8:24] = 1.0
        suppressor = DynamicPhysioSuppressor(alpha=0.7, pet_quantile=0.95, min_area=16)

        out, mask = suppressor.suppress(anomaly, pet, prior)

        self.assertGreater(float(mask.sum()), 0.0)
        self.assertLess(float(out[24, 16]), float(out[8, 8]))



if __name__ == "__main__":
    unittest.main()
