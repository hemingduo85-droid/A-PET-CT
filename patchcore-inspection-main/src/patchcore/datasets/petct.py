"""PET-CT dual-modal dataset for anomaly detection.

Dataset structure:
  {source}/
    train/
      normal/
        {patient_id}/
          pet/  {0000.png, 0001.png, ...}
          ct/   {0000.png, 0001.png, ...}
    test/
      normal/
        {patient_id}/
          pet/  {0000.png, ...}
          ct/   {0000.png, ...}
      abnormal/
        {patient_id}/
          pet/   {0000.png, ...}
          ct/    {0000.png, ...}
          label/ {0000.png, ...}
"""
import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class PetCTDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for dual-modal PET-CT anomaly detection.

    Each sample returns a dict containing:
        image     : PET image tensor  [3, H, W]  (ImageNet-normalised)
        image_ct  : CT  image tensor  [3, H, W]  (ImageNet-normalised)
        mask      : binary mask tensor [1, H, W]  (zero for normal slices)
        classname : str
        anomaly   : "good" | "abnormal"
        is_anomaly: int  (0 or 1)
        image_name: str
        image_path: str (path to PET image)
    """

    def __init__(
        self,
        source,
        classname=None,      # not used; kept for API compatibility
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        modality="pet",
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.train_val_split = train_val_split
        self.modality = modality.lower()
        if self.modality not in ("pet", "ct"):
            raise ValueError(f"modality must be 'pet' or 'ct', got {modality!r}")

        self.data_to_iterate = self._build_data_list()

        self.transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

        # imagesize is exposed so run_patchcore.py can read it
        self.imagesize = (3, imagesize, imagesize)

        # Attributes expected by run_patchcore.py visualisation helper
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_patient_slices(self, root, anomaly_label, has_label=False):
        """Iterate over patient folders under `root` and collect slice paths.

        Returns a list of 5-tuples:
            (classname, anomaly, pet_path, mask_path_or_None, ct_path)
        """
        entries = []
        if not os.path.isdir(root):
            return entries
        for patient in sorted(os.listdir(root)):
            patient_dir = os.path.join(root, patient)
            if not os.path.isdir(patient_dir):
                continue
            pet_dir = os.path.join(patient_dir, "pet")
            ct_dir = os.path.join(patient_dir, "ct")
            label_dir = os.path.join(patient_dir, "label") if has_label else None

            if not os.path.isdir(pet_dir) or not os.path.isdir(ct_dir):
                continue

            for fname in sorted(os.listdir(pet_dir)):
                pet_path = os.path.join(pet_dir, fname)
                ct_path = os.path.join(ct_dir, fname)
                mask_path = os.path.join(label_dir, fname) if label_dir else None
                entries.append(("psma", anomaly_label, pet_path, mask_path, ct_path))
        return entries

    def _build_data_list(self):
        """Build the master list of data tuples depending on the split."""
        if self.split in (DatasetSplit.TRAIN, DatasetSplit.VAL):
            all_entries = self._collect_patient_slices(
                os.path.join(self.source, "train", "normal"),
                anomaly_label="good",
                has_label=False,
            )
            if self.train_val_split < 1.0:
                split_idx = int(len(all_entries) * self.train_val_split)
                if self.split == DatasetSplit.TRAIN:
                    return all_entries[:split_idx]
                else:
                    return all_entries[split_idx:]
            return all_entries  # val split == 1.0 → all data goes to TRAIN

        # TEST split
        normal_entries = self._collect_patient_slices(
            os.path.join(self.source, "test", "normal"),
            anomaly_label="good",
            has_label=False,
        )
        abnormal_entries = self._collect_patient_slices(
            os.path.join(self.source, "test", "abnormal"),
            anomaly_label="abnormal",
            has_label=True,
        )
        return normal_entries + abnormal_entries

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, idx):
        classname, anomaly, pet_path, mask_path, ct_path = self.data_to_iterate[idx]

        pet_img = PIL.Image.open(pet_path).convert("RGB")
        ct_img = PIL.Image.open(ct_path).convert("RGB")

        pet_tensor = self.transform_img(pet_img)
        ct_tensor = self.transform_img(ct_img)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            # Binarise: pixel > 0 → 1
            mask = (mask > 0).float()
        else:
            mask = torch.zeros([1, *pet_tensor.shape[1:]])

        if self.modality == "ct":
            primary_tensor, primary_path = ct_tensor, ct_path
        else:
            primary_tensor, primary_path = pet_tensor, pet_path

        return {
            "image": primary_tensor,      # primary image for single-modal mode
            "image_ct": ct_tensor,        # second modality (dual-modal only)
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(primary_path.split("/")[-4:]),
            "image_path": primary_path,
        }
