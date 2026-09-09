import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PETCTDataset(Dataset):
    """Dataset for PET-CT anomaly detection.

    Supports three input modes:
      - 'ct':   grayscale CT repeated 3x → pseudo-RGB
      - 'pet':  grayscale PET repeated 3x → pseudo-RGB
      - 'dual': [CT, PET, PET] channels → pseudo-RGB
    """

    def __init__(
        self,
        data_root: str,
        input_res: int,
        split: str,
        transform: Optional[transforms.Compose] = None,
        input_mode: str = "dual",
        category: str = "psma",
        is_mask: bool = True,
        anom_only: bool = False,
        normal_only: bool = False,
        **kwargs,
    ):
        self.data_root = data_root
        self.input_res = input_res
        self.split = split
        self.custom_transforms = transform
        self.input_mode = input_mode
        self.is_mask = is_mask
        self.anom_only = anom_only
        self.normal_only = normal_only
        self.category = category

        assert split in ("train", "test")
        assert input_mode in ("ct", "pet", "dual")

        self.samples = []  # list of (ct_path, pet_path, label, mask_path_or_None)
        self._load_samples()

        if self.split == "test":
            self.normal_indices = [i for i, s in enumerate(self.samples) if s[2] == 0]
            self.anom_indices = [i for i, s in enumerate(self.samples) if s[2] == 1]

            def mask_to_tensor(img):
                return torch.from_numpy(np.array(img, dtype=np.uint8)).long()

            self.mask_transform = transforms.Compose([
                transforms.Resize((input_res, input_res)),
                transforms.Lambda(mask_to_tensor),
            ])

    def _load_samples(self):
        root = Path(self.data_root)

        if self.split == "train":
            normal_dir = root / "train" / "normal"
            self._add_patient_dir(normal_dir, label=0)
        else:
            normal_dir = root / "test" / "normal"
            abnormal_dir = root / "test" / "abnormal"
            self._add_patient_dir(normal_dir, label=0)
            self._add_patient_dir(abnormal_dir, label=1)

    def _add_patient_dir(self, base_dir: Path, label: int):
        if not base_dir.exists():
            return
        for patient_dir in sorted(base_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            ct_dir = patient_dir / "ct"
            pet_dir = patient_dir / "pet"
            label_dir = patient_dir / "label"
            if not ct_dir.exists() or not pet_dir.exists():
                continue

            ct_files = sorted(ct_dir.glob("*.png"))
            for ct_path in ct_files:
                fname = ct_path.name
                pet_path = pet_dir / fname
                if not pet_path.exists():
                    continue
                mask_path = None
                if label == 1 and label_dir.exists():
                    candidate = label_dir / fname
                    if candidate.exists():
                        mask_path = candidate
                self.samples.append((ct_path, pet_path, label, mask_path))

    def __len__(self):
        if self.anom_only:
            return len(self.anom_indices)
        elif self.normal_only:
            return len(self.normal_indices)
        return len(self.samples)

    def __getitem__(self, index):
        if self.anom_only:
            index = self.anom_indices[index]
        elif self.normal_only:
            index = self.normal_indices[index]

        ct_path, pet_path, label, mask_path = self.samples[index]

        ct_img = Image.open(ct_path).convert("L")
        pet_img = Image.open(pet_path).convert("L")

        if self.input_mode == "ct":
            img = Image.merge("RGB", (ct_img, ct_img, ct_img))
        elif self.input_mode == "pet":
            img = Image.merge("RGB", (pet_img, pet_img, pet_img))
        else:  # dual
            img = Image.merge("RGB", (ct_img, pet_img, pet_img))

        sample = self.custom_transforms(img)

        inputs = {
            "samples": sample,
            "clsnames": self.category,
            "clslabels": 0,
            "filenames": str(ct_path),
        }

        if self.split == "test":
            inputs["labels"] = label
            if label == 0:
                inputs["anom_type"] = "good"
            else:
                inputs["anom_type"] = "anomaly"

            if self.is_mask:
                if mask_path is not None:
                    mask = Image.open(mask_path).convert("L")
                else:
                    mask = Image.new("L", (self.input_res, self.input_res), 0)
                mask = self.mask_transform(mask)
                inputs["masks"] = mask

        return inputs
