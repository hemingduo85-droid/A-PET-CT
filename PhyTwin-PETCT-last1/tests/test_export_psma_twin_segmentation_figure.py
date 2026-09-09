import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class MaskMorphologyTests(unittest.TestCase):
    def test_extract_mask_features_distinguishes_compact_and_disseminated_masks(self):
        from export_psma_twin_segmentation_figure import extract_mask_features

        compact = np.zeros((100, 80), dtype=np.uint8)
        compact[12:22, 32:44] = 1
        disseminated = np.zeros_like(compact)
        disseminated[8:14, 10:18] = 1
        disseminated[45:52, 35:43] = 1
        disseminated[82:91, 60:70] = 1

        compact_features = extract_mask_features(compact, min_component_area=3)
        disseminated_features = extract_mask_features(disseminated, min_component_area=3)

        self.assertEqual(compact_features["n_components"], 1)
        self.assertGreater(compact_features["dominant_component_ratio"], 0.99)
        self.assertEqual(disseminated_features["n_components"], 3)
        self.assertGreater(disseminated_features["vertical_span"], 0.75)
        self.assertGreaterEqual(disseminated_features["occupied_bands"], 3)

    def test_scene_fit_prefers_matching_anatomical_distribution(self):
        from export_psma_twin_segmentation_figure import extract_mask_features, scene_fit_score

        upper = np.zeros((100, 80), dtype=np.uint8)
        upper[10:24, 30:45] = 1
        lower = np.zeros_like(upper)
        lower[74:90, 30:45] = 1
        spread = np.zeros_like(upper)
        spread[8:14, 15:23] = 1
        spread[43:51, 34:42] = 1
        spread[82:90, 55:64] = 1

        uf = extract_mask_features(upper)
        lf = extract_mask_features(lower)
        sf = extract_mask_features(spread)

        self.assertGreater(
            scene_fit_score(uf, "upper_compact"),
            scene_fit_score(lf, "upper_compact"),
        )
        self.assertGreater(
            scene_fit_score(lf, "lower_compact"),
            scene_fit_score(uf, "lower_compact"),
        )
        self.assertGreater(
            scene_fit_score(sf, "whole_body_disseminated"),
            scene_fit_score(uf, "whole_body_disseminated"),
        )

    def test_scene_crop_is_bounded_and_preserves_full_body_row(self):
        from export_psma_twin_segmentation_figure import crop_box_for_scene

        shape = (100, 80)
        features = {"y_min": 0.08, "y_max": 0.24, "centroid_y": 0.16}
        self.assertEqual(crop_box_for_scene("whole_body_disseminated", shape, features), (0, 100, 0, 80))
        y0, y1, x0, x1 = crop_box_for_scene("upper_compact", shape, features)
        self.assertEqual((x0, x1), (0, 80))
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(y1, 100)
        self.assertLess(y0, y1)
        self.assertLessEqual(y1, 58)


class OverlayTests(unittest.TestCase):
    def test_prediction_overlay_uses_exact_tp_fn_fp_colors(self):
        from export_psma_twin_segmentation_figure import prediction_overlay

        base = np.full((2, 2), 0.5, dtype=np.float32)
        gt = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)

        rgb = prediction_overlay(base, gt, pred, alpha=1.0)

        np.testing.assert_array_equal(rgb[0, 0], np.array([0, 190, 0], dtype=np.uint8))
        np.testing.assert_array_equal(rgb[0, 1], np.array([230, 35, 35], dtype=np.uint8))
        np.testing.assert_array_equal(rgb[1, 0], np.array([30, 145, 235], dtype=np.uint8))
        np.testing.assert_array_equal(rgb[1, 1], np.array([128, 128, 128], dtype=np.uint8))


class FinalSelectionTests(unittest.TestCase):
    def test_choose_final_cases_prefers_unique_patients(self):
        from export_psma_twin_segmentation_figure import choose_final_cases

        candidates = {
            "upper_compact": [
                {"case_id": "P1", "slice_id": "0001", "final_selection_score": 0.95},
                {"case_id": "P2", "slice_id": "0001", "final_selection_score": 0.80},
            ],
            "whole_body_disseminated": [
                {"case_id": "P1", "slice_id": "0010", "final_selection_score": 0.99},
                {"case_id": "P3", "slice_id": "0010", "final_selection_score": 0.85},
            ],
            "lower_compact": [
                {"case_id": "P4", "slice_id": "0020", "final_selection_score": 0.82},
            ],
            "upper_multifocal": [
                {"case_id": "P5", "slice_id": "0030", "final_selection_score": 0.81},
            ],
        }

        selected = choose_final_cases(candidates)

        self.assertEqual(selected["upper_compact"]["case_id"], "P1")
        self.assertEqual(selected["whole_body_disseminated"]["case_id"], "P3")
        self.assertEqual(len({row["case_id"] for row in selected.values()}), 4)

    def test_choose_final_cases_has_deterministic_duplicate_fallback(self):
        from export_psma_twin_segmentation_figure import choose_final_cases

        candidates = {
            scene: [{"case_id": "only", "slice_id": f"{index:04d}", "final_selection_score": 0.8}]
            for index, scene in enumerate(
                ("upper_compact", "whole_body_disseminated", "lower_compact", "upper_multifocal"),
                start=1,
            )
        }

        selected = choose_final_cases(candidates)

        self.assertEqual(list(selected), [
            "upper_compact",
            "whole_body_disseminated",
            "lower_compact",
            "upper_multifocal",
        ])
        self.assertTrue(all(row["case_id"] == "only" for row in selected.values()))


class RenderingTests(unittest.TestCase):
    def test_render_publication_figure_exports_all_formats(self):
        from PIL import Image
        from export_psma_twin_segmentation_figure import (
            SCENES,
            extract_mask_features,
            render_publication_figure,
        )

        selected = {}
        for index, scene in enumerate(SCENES):
            base = np.linspace(0, 1, 64 * 48, dtype=np.float32).reshape(64, 48)
            gt = np.zeros_like(base, dtype=np.uint8)
            y0 = min(52, 4 + index * 12)
            gt[y0:y0 + 7, 18:26] = 1
            pred = gt.copy()
            anomaly = np.zeros_like(base)
            anomaly[gt > 0] = 1.0
            selected[scene] = {
                "case_id": f"P{index + 1}",
                "slice_id": f"{index + 1:04d}",
                **extract_mask_features(gt),
                "record": {
                    "ct": base,
                    "normal_twin_pet": np.clip(base * 0.8, 0, 1),
                    "pet": base,
                    "gt": gt,
                    "pred": pred,
                    "anomaly": anomaly,
                },
            }

        with TemporaryDirectory() as tmp:
            output_prefix = Path(tmp) / "figure"
            outputs = render_publication_figure(selected, output_prefix, dpi=100)

            self.assertEqual(set(outputs), {"png", "tiff", "pdf", "svg"})
            for path in outputs.values():
                self.assertTrue(Path(path).is_file(), path)
            with Image.open(outputs["png"]) as preview:
                self.assertGreater(preview.width, preview.height)


if __name__ == "__main__":
    unittest.main()
