import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import get_detection_dataset


class PetCtPseudoRgbDatasetTest(unittest.TestCase):
    def test_petct_modality_returns_ct_pet_pet_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            patient_dir = Path(tmp) / "train" / "normal" / "patient001"
            pet_dir = patient_dir / "pet"
            ct_dir = patient_dir / "ct"
            pet_dir.mkdir(parents=True)
            ct_dir.mkdir(parents=True)

            pet = np.full((8, 8), 40, dtype=np.uint8)
            ct = np.full((8, 8), 160, dtype=np.uint8)
            Image.fromarray(pet).save(pet_dir / "slice001.png")
            Image.fromarray(ct).save(ct_dir / "slice001.png")

            ds = get_detection_dataset(
                str(tmp),
                "train",
                modality="petct",
                replicate_channels=3,
                normalize=False,
                image_size=8,
                crop_size=8,
            )

            img, gt, label, path = ds[0]

            self.assertEqual(tuple(img.shape), (3, 8, 8))
            self.assertTrue(torch_allclose_value(img[0], 160 / 255.0))
            self.assertTrue(torch_allclose_value(img[1], 40 / 255.0))
            self.assertTrue(torch_allclose_value(img[2], 40 / 255.0))
            self.assertEqual(tuple(gt.shape), (1, 8, 8))
            self.assertEqual(label, 0)
            self.assertTrue(path.endswith("slice001.png"))


def torch_allclose_value(tensor, value):
    return bool((tensor - value).abs().max().item() < 1e-6)


if __name__ == "__main__":
    unittest.main()
