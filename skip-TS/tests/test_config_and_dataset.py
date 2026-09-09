import os
import ast
import tempfile
import unittest
from importlib import import_module
from unittest import SkipTest

import numpy as np
from PIL import Image

from main import build_run_config, resolve_data_path, resolve_device_name, resolve_save_path, validate_data_path


class ConfigTest(unittest.TestCase):
    def test_dataset_name_resolves_default_paths(self):
        self.assertEqual(
            resolve_data_path("fdg", None),
            "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG",
        )
        self.assertEqual(
            resolve_data_path("psma", None),
            "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA",
        )

    def test_data_path_rejects_wrong_dataset_leaf(self):
        with self.assertRaises(ValueError):
            validate_data_path("psma", "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG")

    def test_data_path_rejects_nested_dataset_names(self):
        with self.assertRaises(ValueError):
            validate_data_path("psma", "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA/FDG")

    def test_save_path_includes_dataset_and_input_mode(self):
        save_path = resolve_save_path(
            checkpoint_root="./checkpoints",
            dataset="psma",
            modalities=["ct", "pet"],
            input_mode="pseudo_rgb",
        )
        self.assertEqual(
            save_path,
            os.path.join("./checkpoints", "psma", "ct+pet_pseudo_rgb"),
        )

    def test_config_defaults_to_single_final_eval_and_no_heatmaps(self):
        args = build_run_config(["--dataset", "fdg", "--cuda", "2"])
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.eval_interval, 0)
        self.assertEqual(args.heatmap_count, 0)
        self.assertFalse(args.save_all_heatmaps)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.cuda, "2")
        self.assertEqual(args.ci_pixel_max_samples, 0)
        self.assertEqual(args.num_workers, 4)

    def test_ci_pixel_max_samples_can_be_configured(self):
        args = build_run_config(["--dataset", "fdg", "--ci_pixel_max_samples", "0"])
        self.assertEqual(args.ci_pixel_max_samples, 0)

    def test_cuda_argument_resolves_to_explicit_device_index(self):
        args = build_run_config(["--dataset", "fdg", "--cuda", "6"])
        self.assertEqual(resolve_device_name(args.device, args.cuda), "cuda:6")

    def test_explicit_device_overrides_cuda_argument(self):
        args = build_run_config(["--dataset", "fdg", "--device", "cuda:3", "--cuda", "6"])
        self.assertEqual(resolve_device_name(args.device, args.cuda), "cuda:3")

    def test_bootstrap_iters_is_not_passed_to_test_dataloader(self):
        source_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train_and_test.py")
        with open(source_path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())

        bad_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "_build_test_dataloader":
                if any(keyword.arg == "bootstrap_iters" for keyword in node.keywords):
                    bad_calls.append(node.lineno)

        self.assertEqual(bad_calls, [])


class PseudoRgbDatasetTest(unittest.TestCase):
    def test_ct_pet_pseudo_rgb_uses_ct_pet_pet_channels(self):
        try:
            dataset_module = import_module("dataset")
        except ModuleNotFoundError as exc:
            if exc.name in {"torch", "torchvision"}:
                raise SkipTest(f"{exc.name} is not installed")
            raise

        with tempfile.TemporaryDirectory() as tmp:
            patient = os.path.join(tmp, "train", "normal", "case001")
            os.makedirs(os.path.join(patient, "ct"))
            os.makedirs(os.path.join(patient, "pet"))

            Image.fromarray(np.full((8, 8), 20, dtype=np.uint8)).save(
                os.path.join(patient, "ct", "slice001.png")
            )
            Image.fromarray(np.full((8, 8), 200, dtype=np.uint8)).save(
                os.path.join(patient, "pet", "slice001.png")
            )

            transform, _, mean, std = dataset_module.get_data_transforms(
                8,
                8,
                num_modalities=2,
                replicate_channels=1,
                input_mode="pseudo_rgb",
            )
            dataset = dataset_module.MRIDataset(
                root=tmp,
                transform=transform,
                phase="train",
                modalities=["ct", "pet"],
                mean=mean,
                std=std,
                input_mode="pseudo_rgb",
            )

            image, _, label, sample_path = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 8, 8))
            self.assertEqual(label, 0)
            self.assertTrue(sample_path.endswith("slice001.png"))
            self.assertTrue(np.allclose(image[0].numpy(), image[0, 0, 0].item()))
            self.assertTrue(np.allclose(image[1].numpy(), image[2].numpy()))
            self.assertGreater(float(image[1, 0, 0]), float(image[0, 0, 0]))

    def test_test_loader_uses_nonempty_lowercase_dir_when_uppercase_exists_empty(self):
        try:
            dataset_module = import_module("dataset")
        except ModuleNotFoundError as exc:
            if exc.name in {"torch", "torchvision"}:
                raise SkipTest(f"{exc.name} is not installed")
            raise

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "test", "NORMAL"))
            patient = os.path.join(tmp, "test", "normal", "case001")
            os.makedirs(os.path.join(patient, "CT"))
            os.makedirs(os.path.join(patient, "PET"))
            Image.fromarray(np.full((8, 8), 20, dtype=np.uint8)).save(
                os.path.join(patient, "CT", "slice001.PNG")
            )
            Image.fromarray(np.full((8, 8), 200, dtype=np.uint8)).save(
                os.path.join(patient, "PET", "slice001.PNG")
            )
            abnormal = os.path.join(tmp, "test", "abnormal", "case002")
            os.makedirs(os.path.join(abnormal, "CT"))
            os.makedirs(os.path.join(abnormal, "PET"))
            os.makedirs(os.path.join(abnormal, "label"))
            Image.fromarray(np.full((8, 8), 30, dtype=np.uint8)).save(
                os.path.join(abnormal, "CT", "slice001.PNG")
            )
            Image.fromarray(np.full((8, 8), 210, dtype=np.uint8)).save(
                os.path.join(abnormal, "PET", "slice001.PNG")
            )
            Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 255).save(
                os.path.join(abnormal, "label", "slice001.PNG")
            )

            transform, _, mean, std = dataset_module.get_data_transforms(
                8,
                8,
                num_modalities=2,
                input_mode="pseudo_rgb",
            )
            dataset = dataset_module.MRIDataset(
                root=tmp,
                transform=transform,
                phase="test",
                modalities=["ct", "pet"],
                mean=mean,
                std=std,
                input_mode="pseudo_rgb",
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.labels, [0, 1])

    def test_test_loader_supports_label_dir_with_modalities_directly(self):
        try:
            dataset_module = import_module("dataset")
        except ModuleNotFoundError as exc:
            if exc.name in {"torch", "torchvision"}:
                raise SkipTest(f"{exc.name} is not installed")
            raise

        with tempfile.TemporaryDirectory() as tmp:
            normal = os.path.join(tmp, "test", "normal")
            os.makedirs(os.path.join(normal, "CT"))
            os.makedirs(os.path.join(normal, "PET"))
            Image.fromarray(np.full((8, 8), 20, dtype=np.uint8)).save(
                os.path.join(normal, "CT", "slice001.PNG")
            )
            Image.fromarray(np.full((8, 8), 200, dtype=np.uint8)).save(
                os.path.join(normal, "PET", "slice001.PNG")
            )

            abnormal = os.path.join(tmp, "test", "abnormal")
            os.makedirs(os.path.join(abnormal, "CT"))
            os.makedirs(os.path.join(abnormal, "PET"))
            Image.fromarray(np.full((8, 8), 30, dtype=np.uint8)).save(
                os.path.join(abnormal, "CT", "slice001.PNG")
            )
            Image.fromarray(np.full((8, 8), 210, dtype=np.uint8)).save(
                os.path.join(abnormal, "PET", "slice001.PNG")
            )

            transform, _, mean, std = dataset_module.get_data_transforms(
                8,
                8,
                num_modalities=2,
                input_mode="pseudo_rgb",
            )
            dataset = dataset_module.MRIDataset(
                root=tmp,
                transform=transform,
                phase="test",
                modalities=["ct", "pet"],
                mean=mean,
                std=std,
                input_mode="pseudo_rgb",
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.labels, [0, 1])


if __name__ == "__main__":
    unittest.main()
