import unittest


class PETCTConfigTests(unittest.TestCase):
    def test_dataset_preset_paths(self):
        from petct_config import resolve_dataset

        self.assertEqual(
            resolve_dataset("fdg")["data_path"],
            "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG",
        )
        self.assertEqual(
            resolve_dataset("psma")["data_path"],
            "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA",
        )

    def test_default_save_dir_distinguishes_dataset_and_modality(self):
        from petct_config import default_save_dir

        self.assertEqual(
            default_save_dir("checkpoints", "psma", "petct"),
            "checkpoints/psma_petct",
        )
        self.assertEqual(
            default_save_dir("checkpoints", "fdg", "pet"),
            "checkpoints/fdg_pet",
        )

    def test_final_checkpoint_name_uses_one_based_epoch(self):
        from petct_config import final_checkpoint_name

        self.assertEqual(final_checkpoint_name(30), "ckpt_epoch_30.pth")

    def test_heatmap_count_parser(self):
        from petct_config import normalize_heatmap_count

        self.assertEqual(normalize_heatmap_count(0), 0)
        self.assertEqual(normalize_heatmap_count(5), 5)
        self.assertEqual(normalize_heatmap_count(-1), -1)
        with self.assertRaises(ValueError):
            normalize_heatmap_count(-2)


if __name__ == "__main__":
    unittest.main()
