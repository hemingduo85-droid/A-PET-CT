import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestExportSinglePatientPetPrediction(unittest.TestCase):
    def test_default_patient_dir_uses_train_normal(self):
        from export_single_patient_pet_prediction import find_patient_dir

        with tempfile.TemporaryDirectory() as tmp:
            patient_dir = os.path.join(tmp, "train", "normal", "patient_a")
            os.makedirs(os.path.join(patient_dir, "pet"))
            os.makedirs(os.path.join(patient_dir, "ct"))

            found = find_patient_dir(tmp, split="train", group="normal", patient=None)

        self.assertEqual(found, patient_dir)

    def test_center_slice_selection_limits_outputs(self):
        from export_single_patient_pet_prediction import select_slice_names

        names = [f"{i:03d}.png" for i in range(10)]

        self.assertEqual(select_slice_names(names, num_slices=4), ["003.png", "004.png", "005.png", "006.png"])

    def test_residual_statistics_include_scalar_and_maps(self):
        import numpy as np

        from export_single_patient_pet_prediction import compute_residual_statistics

        residuals = np.asarray([
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ], dtype=np.float32)

        stats = compute_residual_statistics(residuals)

        self.assertAlmostEqual(stats["residual_mean"], 3.75)
        self.assertGreater(stats["residual_std"], 0.0)
        np.testing.assert_allclose(stats["residual_mean_map"], [[1.5, 3.0], [4.5, 6.0]])
        np.testing.assert_allclose(stats["residual_std_map"], [[0.5, 1.0], [1.5, 2.0]])

    def test_heatmap_writer_saves_rgb_png_with_purple_palette(self):
        import numpy as np
        from PIL import Image

        from export_single_patient_pet_prediction import save_heatmap_image

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "residual_heatmap.png")
            arr = np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
            save_heatmap_image(arr, out, "Residual R")
            with Image.open(out) as img:
                mode = img.mode
                size = img.size
                top_left = img.getpixel((0, 0))
                low_pixel = img.getpixel((0, 30))

        self.assertEqual(mode, "RGB")
        self.assertEqual(size, (512, 512))
        self.assertEqual(top_left, (83, 38, 120))
        self.assertEqual(low_pixel, (83, 38, 120))

    def test_a0_heatmap_uses_fixed_zscore_display_scale(self):
        import numpy as np

        from export_single_patient_pet_prediction import _a0_heatmap_rgb

        weak = _a0_heatmap_rgb(np.asarray([[0.0, 1.0]], dtype=np.float32))
        strong = _a0_heatmap_rgb(np.asarray([[0.0, 5.0]], dtype=np.float32))

        self.assertEqual(tuple(weak[0, 0]), (83, 38, 120))
        self.assertNotEqual(tuple(weak[0, 1]), tuple(strong[0, 1]))
        self.assertNotEqual(tuple(strong[0, 1]), (241, 163, 76))

    def test_pet_hot_colormap_keeps_black_low_and_warm_high(self):
        import numpy as np

        from export_single_patient_pet_prediction import _pet_hot_rgb

        rgb = _pet_hot_rgb(np.asarray([[0.0, 0.75, 1.0]], dtype=np.float32))

        self.assertEqual(tuple(rgb[0, 0]), (0, 0, 0))
        self.assertGreater(int(rgb[0, 1, 0]), int(rgb[0, 1, 2]))
        self.assertEqual(tuple(rgb[0, 2]), (255, 255, 255))

    def test_high_uptake_mask_frequency_comes_before_smoothing(self):
        import numpy as np

        from export_single_patient_pet_prediction import compute_high_uptake_mask_frequency

        pet_maps = np.asarray([
            [[0.0, 1.0], [0.2, 0.3]],
            [[0.9, 0.1], [0.2, 0.3]],
        ], dtype=np.float32)

        freq = compute_high_uptake_mask_frequency(pet_maps, uptake_quantile=0.75)

        np.testing.assert_allclose(freq, [[0.5, 0.5], [0.0, 0.0]])

    def test_single_high_uptake_mask_is_per_slice(self):
        import numpy as np

        from export_single_patient_pet_prediction import compute_high_uptake_mask

        pet_map = np.asarray([[0.0, 1.0], [0.2, 0.3]], dtype=np.float32)
        mask = compute_high_uptake_mask(pet_map, uptake_quantile=0.75)

        np.testing.assert_allclose(mask, [[0.0, 1.0], [0.0, 0.0]])

    def test_a_star_suppresses_physio_hotspot_more_than_small_focus(self):
        import numpy as np

        from export_single_patient_pet_prediction import compute_a_star_map

        a0 = np.ones((64, 64), dtype=np.float32)
        pet = np.zeros((64, 64), dtype=np.float32)
        ct = np.zeros((64, 64), dtype=np.float32)
        prior = np.zeros((64, 64), dtype=np.float32)
        pet[20:36, 18:38] = 1.0
        ct[20:36, 18:38] = 0.7
        prior[20:36, 18:38] = 1.0
        pet[48:51, 48:51] = 1.0
        a0[48:51, 48:51] = 3.0

        a_star, correction = compute_a_star_map(a0, pet, ct, prior, sigma=0.0, min_area=6)

        self.assertGreater(float(correction[25, 25]), float(correction[49, 49]))
        self.assertLess(float(a_star[25, 25]), float(a0[25, 25]))
        self.assertGreater(float(a_star[49, 49] / a0[49, 49]), float(a_star[25, 25] / a0[25, 25]))

    def test_binary_mask_writer_saves_black_and_white(self):
        import numpy as np
        from PIL import Image

        from export_single_patient_pet_prediction import save_binary_mask_image

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "mask.png")
            save_binary_mask_image(np.asarray([[0.0, 1.0]], dtype=np.float32), out)
            with Image.open(out) as img:
                low = img.getpixel((0, 0))
                high = img.getpixel((511, 0))

        self.assertEqual(low, 0)
        self.assertEqual(high, 255)

    def test_logvar_to_uncertainty_map_returns_2d_or_none(self):
        import numpy as np

        from export_single_patient_pet_prediction import logvar_to_uncertainty_map

        logvar = np.asarray([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=np.float32)
        uncertainty = logvar_to_uncertainty_map(logvar)

        self.assertEqual(tuple(uncertainty.shape), (2, 2))
        self.assertIsNone(logvar_to_uncertainty_map(None))

    def test_uncertainty_colormap_is_not_grayscale(self):
        import numpy as np

        from export_single_patient_pet_prediction import _uncertainty_rgb

        rgb = _uncertainty_rgb(np.asarray([[0.0, 1.0]], dtype=np.float32))

        self.assertNotEqual(tuple(rgb[0, 0]), (0, 0, 0))
        self.assertNotEqual(int(rgb[0, 1, 0]), int(rgb[0, 1, 1]))


if __name__ == "__main__":
    unittest.main()
