import unittest

from utils.petct_config import (
    DEFAULT_DATASET,
    build_experiment_dir,
    resolve_checkpoint_path,
    resolve_petct_paths,
    select_heatmap_indices,
)


class PETCTConfigTest(unittest.TestCase):
    def test_resolves_named_dataset_paths(self):
        train_path, test_path = resolve_petct_paths("psma")

        self.assertEqual(
            train_path,
            "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA/train/normal",
        )
        self.assertEqual(
            test_path,
            "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA/test",
        )

    def test_resolves_fdg_dataset_paths(self):
        train_path, test_path = resolve_petct_paths("fdg")

        self.assertEqual(
            train_path,
            "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG/train/normal",
        )
        self.assertEqual(
            test_path,
            "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG/test",
        )

    def test_builds_dataset_modality_experiment_dir(self):
        self.assertEqual(
            build_experiment_dir("./experiments", "fdg", "petct"),
            "./experiments/fdg_petct",
        )

    def test_resolves_default_checkpoint_without_manual_pth_name(self):
        self.assertEqual(
            resolve_checkpoint_path(
                checkpoint_path=None,
                save_path="./experiments",
                dataset="psma",
                modality="ct",
                epoch=30,
            ),
            "./experiments/psma_ct/epoch_30.pth",
        )

    def test_heatmap_indices_default_none_count_and_all(self):
        self.assertEqual(select_heatmap_indices(total=5, count=0, save_all=False), [])
        self.assertEqual(select_heatmap_indices(total=5, count=2, save_all=False), [0, 1])
        self.assertEqual(select_heatmap_indices(total=3, count=99, save_all=False), [0, 1, 2])
        self.assertEqual(select_heatmap_indices(total=3, count=0, save_all=True), [0, 1, 2])

    def test_default_dataset_is_psma(self):
        self.assertEqual(DEFAULT_DATASET, "psma")


if __name__ == "__main__":
    unittest.main()
