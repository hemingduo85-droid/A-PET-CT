import os
import sys
import types
import unittest


def _install_train_dependency_stubs():
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.device = lambda name: name
    torch_stub.no_grad = lambda: types.SimpleNamespace(
        __enter__=lambda self: None,
        __exit__=lambda self, exc_type, exc, tb: False,
    )
    torch_stub.nn = types.SimpleNamespace(L1Loss=object)
    torch_stub.optim = types.ModuleType("torch.optim")

    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = object

    sklearn_stub = types.ModuleType("sklearn")
    sklearn_metrics_stub = types.ModuleType("sklearn.metrics")
    sklearn_metrics_stub.roc_auc_score = lambda *args, **kwargs: 0.0
    sklearn_metrics_stub.average_precision_score = lambda *args, **kwargs: 0.0
    sklearn_metrics_stub.precision_recall_curve = lambda *args, **kwargs: ([], [], [])
    sklearn_metrics_stub.roc_curve = lambda *args, **kwargs: ([], [], [])

    torchvision_stub = types.ModuleType("torchvision")
    torchvision_transforms_stub = types.ModuleType("torchvision.transforms")

    dataloader_stub = types.ModuleType("dataloader")
    dataloader_stub.PETCTAnomalyDataset = object

    models_stub = types.ModuleType("models")
    models_stub.GatingAno = object
    models_stub.Discriminator = object
    models_stub.AdversarialLoss = object

    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, **kwargs: iterable

    sys.modules.setdefault("torch", torch_stub)
    sys.modules.setdefault("torch.optim", torch_stub.optim)
    sys.modules.setdefault("torch.utils", torch_utils)
    sys.modules.setdefault("torch.utils.data", torch_utils_data)
    sys.modules.setdefault("sklearn", sklearn_stub)
    sys.modules.setdefault("sklearn.metrics", sklearn_metrics_stub)
    sys.modules.setdefault("torchvision", torchvision_stub)
    sys.modules.setdefault("torchvision.transforms", torchvision_transforms_stub)
    sys.modules.setdefault("dataloader", dataloader_stub)
    sys.modules.setdefault("models", models_stub)
    sys.modules.setdefault("tqdm", tqdm_stub)


_install_train_dependency_stubs()

import numpy as np

from train import (
    Config,
    bootstrap_pixel_hist_ci,
    format_metrics,
    should_evaluate_epoch,
    validate_checkpoint_metadata,
)


class TrainConfigTest(unittest.TestCase):
    def test_dataset_config_builds_dataset_specific_outputs(self):
        config = Config(modality="pet", dataset="FDG", gpu="0")

        self.assertEqual(config.dataset_name, "FDG")
        self.assertEqual(config.input_channels, 3)
        self.assertEqual(config.output_channels, 3)
        self.assertEqual(config.bootstrap_iters, 500)
        self.assertEqual(config.ci_pixel_max_samples, 200000)
        self.assertEqual(config.ci_hist_bins, 16384)
        self.assertTrue(config.data_root.endswith(os.path.join("2d_equal_mask50", "FDG")))
        self.assertEqual(config.final_epoch, 30)
        self.assertEqual(config.eval_epochs, [30])
        self.assertTrue(config.final_ckpt_path.endswith("FDG_pet_epoch30.pth"))
        self.assertTrue(config.final_metrics_path.endswith("FDG_pet_epoch30_metrics.txt"))
        self.assertTrue(config.eval_metrics_path.endswith("FDG_pet_eval_metrics.txt"))
        self.assertTrue(config.heatmap_dir.endswith(os.path.join("results", "FDG_pet", "heatmaps_epoch30")))

    def test_metric_ci_config_can_use_full_pixel_ci_when_requested(self):
        config = Config(
            modality="pet",
            dataset="FDG",
            gpu="0",
            bootstrap_iters=20,
            ci_pixel_max_samples=0,
        )

        self.assertEqual(config.bootstrap_iters, 20)
        self.assertEqual(config.ci_pixel_max_samples, 0)

    def test_pixel_histogram_ci_preserves_exact_point_estimates(self):
        masks = np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
            ],
            dtype=np.float32,
        )
        maps = np.array(
            [
                [[0.9, 0.1], [0.2, 0.3]],
                [[0.2, 0.8], [0.1, 0.4]],
            ],
            dtype=np.float32,
        )

        auroc, aupr = bootstrap_pixel_hist_ci(
            masks,
            maps,
            exact_auroc=0.812345,
            exact_aupr=0.456789,
            n_boot=10,
            seed=7,
            bins=32,
        )

        self.assertEqual(auroc["value"], 0.812345)
        self.assertEqual(aupr["value"], 0.456789)

    def test_checkpoint_metadata_rejects_wrong_modality(self):
        config = Config(modality="petct", dataset="PSMA", gpu="0")
        checkpoint = {
            "state_dict": {},
            "dataset": "PSMA",
            "modality": "ct",
            "input_channels": 3,
            "output_channels": 3,
            "dual_input_mode": "pseudo_rgb",
        }

        with self.assertRaisesRegex(RuntimeError, "modality"):
            validate_checkpoint_metadata(checkpoint, "PSMA", "petct", config)

    def test_dual_modality_config_uses_pseudo_rgb_three_channels(self):
        config = Config(modality="petct", dataset="FDG", gpu="0")

        self.assertEqual(config.modality, "petct")
        self.assertEqual(config.dual_input_mode, "pseudo_rgb")
        self.assertEqual(config.input_channels, 3)
        self.assertEqual(config.output_channels, 3)
        self.assertTrue(config.final_ckpt_path.endswith("FDG_petct_epoch30.pth"))
        self.assertTrue(config.heatmap_dir.endswith(os.path.join("results", "FDG_petct", "heatmaps_epoch30")))

    def test_custom_data_root_infers_dataset_name(self):
        config = Config(modality="ct", data_root="/tmp/my_dataset", gpu="0")

        self.assertEqual(config.dataset_name, "my_dataset")
        self.assertTrue(config.final_ckpt_path.endswith("my_dataset_ct_epoch30.pth"))

    def test_custom_checkpoint_directory_controls_output_paths(self):
        checkpoint_dir = os.path.join("/tmp", "gatingano_run")
        config = Config(
            modality="petct",
            dataset="PSMA",
            gpu="0",
            checkpoint_dir=checkpoint_dir,
        )

        self.assertEqual(config.checkpoint_dir, checkpoint_dir)
        self.assertEqual(
            config.final_ckpt_path,
            os.path.join(checkpoint_dir, "PSMA_petct_epoch30.pth"),
        )
        self.assertEqual(
            config.final_metrics_path,
            os.path.join(checkpoint_dir, "PSMA_petct_epoch30_metrics.txt"),
        )
        self.assertEqual(
            config.eval_metrics_path,
            os.path.join(checkpoint_dir, "PSMA_petct_eval_metrics.txt"),
        )

    def test_only_final_epoch_is_evaluated(self):
        config = Config(modality="pet", dataset="PSMA", gpu="0")

        self.assertFalse(should_evaluate_epoch(10, config))
        self.assertFalse(should_evaluate_epoch(29, config))
        self.assertTrue(should_evaluate_epoch(30, config))

    def test_metric_format_uses_ci_and_requested_fields(self):
        slice_metrics = {
            "img_auroc": {"value": 0.8, "ci": (0.7, 0.9)},
            "img_ap": {"value": 0.75, "ci": (0.6, 0.85)},
            "img_f1": {"value": 0.7, "ci": (0.55, 0.8)},
            "px_auroc_abn": {"value": 0.9, "ci": (0.8, 0.95)},
            "px_aupr_abn": {"value": 0.4, "ci": (0.2, 0.6)},
        }
        pat_metrics = {
            "pat_auroc": {"value": 0.85, "ci": (0.75, 0.95)},
            "pat_ap": {"value": 0.82, "ci": (0.7, 0.9)},
            "pat_f1": {"value": 0.78, "ci": (0.65, 0.88)},
            "n_patients": 36,
            "n_abn_patients": 18,
        }

        text = format_metrics(slice_metrics, pat_metrics, as_percent=True)

        self.assertIn("95% CI", text)
        self.assertIn("AUPR", text)
        self.assertNotIn("Sens@90Spec", text)
        self.assertNotIn("Sens@95Spec", text)
        self.assertNotIn("AP=", text.split("[Slice-Px(abn)]", 1)[1].splitlines()[0])
        self.assertNotIn("F1=", text.split("[Slice-Px(abn)]", 1)[1].splitlines()[0])


if __name__ == "__main__":
    unittest.main()
