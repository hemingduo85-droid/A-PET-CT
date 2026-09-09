
"""PET/CT one-class dataset utilities for PhyTwin.

Expected layout:
  train/normal/<patient>/pet/*.png
  train/normal/<patient>/ct/*.png
  test/normal/<patient>/pet/*.png
  test/normal/<patient>/ct/*.png
  test/abnormal/<patient>/pet/*.png
  test/abnormal/<patient>/ct/*.png
  test/abnormal/<patient>/label/*.png
"""

import glob
import os

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_data_root(data_root):
    data_root = os.path.abspath(os.path.expanduser(data_root))
    if os.path.isdir(os.path.join(data_root, "train", "normal")):
        return data_root
    psma = os.path.join(data_root, "PSMA")
    if os.path.isdir(os.path.join(psma, "train", "normal")):
        return psma
    candidates = []
    if os.path.isdir(data_root):
        for name in sorted(os.listdir(data_root)):
            child = os.path.join(data_root, name)
            if os.path.isdir(os.path.join(child, "train", "normal")):
                candidates.append(child)
    if len(candidates) == 1:
        return candidates[0]
    return data_root


class PETCTDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, phase, image_size=256):
        assert phase in ("train", "test")
        self.data_root = resolve_data_root(data_root)
        self.phase = phase
        self.image_size = image_size
        self.img_transform = T.Compose([
            T.Resize((image_size, image_size), InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), InterpolationMode.NEAREST),
            T.ToTensor(),
        ])
        self.samples = []
        self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No PET/CT samples found for phase={phase}. Resolved data root: {self.data_root}. "
                "Expected train/normal and test/normal,test/abnormal folders."
            )

    def _build_samples(self):
        if self.phase == "train":
            self._collect(os.path.join(self.data_root, "train", "normal"), label=0, has_mask=False)
        else:
            self._collect(os.path.join(self.data_root, "test", "normal"), label=0, has_mask=False)
            self._collect(os.path.join(self.data_root, "test", "abnormal"), label=1, has_mask=True)
        print(f"[PETCTDataset] phase={self.phase}, root={self.data_root}, samples={len(self.samples)}")

    def _collect(self, root_dir, label, has_mask):
        if not os.path.isdir(root_dir):
            print(f"[PETCTDataset] warning: missing directory -> {root_dir}")
            return
        for patient in sorted(os.listdir(root_dir)):
            patient_dir = os.path.join(root_dir, patient)
            pet_dir = os.path.join(patient_dir, "pet")
            ct_dir = os.path.join(patient_dir, "ct")
            label_dir = os.path.join(patient_dir, "label")
            if not os.path.isdir(pet_dir) or not os.path.isdir(ct_dir):
                continue
            for pet_path in sorted(glob.glob(os.path.join(pet_dir, "*.png"))):
                name = os.path.basename(pet_path)
                ct_path = os.path.join(ct_dir, name)
                if not os.path.isfile(ct_path):
                    continue
                mask_path = None
                if has_mask:
                    candidate = os.path.join(label_dir, name)
                    if os.path.isfile(candidate):
                        mask_path = candidate
                self.samples.append((pet_path, ct_path, int(label), mask_path))

    @staticmethod
    def _rgb(gray):
        return Image.merge("RGB", [gray, gray, gray])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pet_path, ct_path, label, mask_path = self.samples[index]
        pet_img = Image.open(pet_path).convert("L")
        ct_img = Image.open(ct_path).convert("L")
        pet = self.img_transform(self._rgb(pet_img))
        ct = self.img_transform(self._rgb(ct_img))
        if mask_path is None:
            mask = torch.zeros(1, self.image_size, self.image_size)
        else:
            mask = Image.open(mask_path).convert("L")
            mask = (self.mask_transform(mask) > 0.5).float()
        return {"pet": pet, "ct": ct, "label": label, "mask": mask, "path": pet_path}


def build_dataloaders(data_root, image_size=256, batch_size=8, num_workers=4):
    train_set = PETCTDataset(data_root, phase="train", image_size=image_size)
    test_set = PETCTDataset(data_root, phase="test", image_size=image_size)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )
    memory_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )
    return train_loader, memory_loader, test_loader
