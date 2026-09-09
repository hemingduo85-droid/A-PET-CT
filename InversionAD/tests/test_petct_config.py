import unittest

from src.config import PETCT_DATASETS, resolve_petct_config


class PETCTConfigTests(unittest.TestCase):
    def test_resolves_dataset_root_category_and_save_dir(self):
        config = {
            "data": {
                "dataset_name": "petct",
                "petct_dataset": "fdg",
                "input_mode": "dual",
            },
            "logging": {
                "save_root": "results/exp_dit_petct",
            },
        }

        resolved = resolve_petct_config(config)

        self.assertEqual(resolved["data"]["category"], "fdg")
        self.assertEqual(resolved["data"]["data_root"], PETCT_DATASETS["fdg"])
        self.assertEqual(resolved["logging"]["save_dir"], "results/exp_dit_petct/fdg/dual")
        self.assertNotIn("data_root", config["data"])

    def test_rejects_unknown_petct_dataset(self):
        config = {
            "data": {
                "dataset_name": "petct",
                "petct_dataset": "unknown",
                "input_mode": "dual",
            },
        }

        with self.assertRaises(ValueError):
            resolve_petct_config(config)


if __name__ == "__main__":
    unittest.main()
