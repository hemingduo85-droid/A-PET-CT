import os
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
import numpy as np


class PETCTAnomalyDataset(VisionDataset):
    """
    PET-CT 单模态/双模态异常检测（pet、ct 或 petct）。

    PSMA 目录:
        {root}/train/normal/{patient}/{modality}/*.png
        {root}/test/normal|abnormal/{patient}/{modality}/*.png
        abnormal 另有 label/*.png

    单模态灰度图复制为 3 通道 (R=G=B) 送入网络。
    双模态读取同名 pet/ct 切片，支持 [CT, PET, PET] 和 [CT, CT, PET]。
    """

    def __init__(self, root, mode='train', modality='pet', transform=None,
                 image_size=256, debug_ratio=1.0, dual_input_mode='pseudo_rgb',
                 return_path=False):
        super().__init__(root, transform=transform)
        self.root = root
        self.mode = mode.lower()
        self.modality = modality.lower()
        if self.modality not in ('pet', 'ct', 'petct'):
            raise ValueError("modality must be 'pet', 'ct' or 'petct'")
        self.dual_input_mode = dual_input_mode
        if self.modality == 'petct' and self.dual_input_mode not in ('pseudo_rgb', 'ctctpet'):
            raise ValueError("petct dual_input_mode must be 'pseudo_rgb' or 'ctctpet'")
        self.modalities = ('pet', 'ct') if self.modality == 'petct' else (self.modality,)
        self.transform = transform
        self.image_size = image_size
        self.return_path = return_path
        self.samples = []

        if self.mode == 'train':
            self._load_train()
        elif self.mode == 'test':
            self._load_test()
        else:
            raise ValueError("mode must be 'train' or 'test'")

        if debug_ratio < 1.0:
            total = len(self.samples)
            keep = max(1, int(total * debug_ratio))
            self.samples = self.samples[:keep]
            print(f"DEBUG MODE: keeping {keep}/{total} samples")

    def _is_image(self, fname):
        return fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))

    def _resolve_split_root(self, split_name):
        split_root = self.root if self.root.endswith(split_name) else os.path.join(self.root, split_name)
        if split_name == 'train':
            normal_sub = os.path.join(split_root, 'normal')
            if os.path.isdir(normal_sub):
                split_root = normal_sub
        return split_root

    def _is_valid_patient_dir(self, patient_dir):
        for modality in self.modalities:
            mod_dir = os.path.join(patient_dir, modality)
            if not os.path.isdir(mod_dir):
                return False
            if not any(self._is_image(f) for f in os.listdir(mod_dir)):
                return False
        return True

    def _collect_patient_slices(self, patient_dir, label, label_dir=None):
        primary_modality = self.modalities[0]
        primary_dir = os.path.join(patient_dir, primary_modality)
        for fname in sorted(os.listdir(primary_dir)):
            if not self._is_image(fname):
                continue

            image_paths = tuple(os.path.join(patient_dir, modality, fname) for modality in self.modalities)
            if not all(os.path.exists(path) for path in image_paths):
                continue

            img_path = image_paths[0] if len(image_paths) == 1 else image_paths
            mask_path = None
            if label == 1 and label_dir is not None:
                mask_path = os.path.join(label_dir, fname)
                if not os.path.exists(mask_path):
                    continue
            self.samples.append((img_path, label, mask_path))

    def _load_train(self):
        train_root = self._resolve_split_root('train')
        if not os.path.exists(train_root):
            raise FileNotFoundError(f"Train root not found: {train_root}")

        for entry in sorted(os.listdir(train_root)):
            patient_dir = os.path.join(train_root, entry)
            if not os.path.isdir(patient_dir) or not self._is_valid_patient_dir(patient_dir):
                continue
            self._collect_patient_slices(patient_dir, label=0)

        print(f"Train [{self.modality.upper()}]: {len(self.samples)} samples")
        if len(self.samples) == 0:
            raise ValueError("No training samples loaded!")

    def _load_test(self):
        test_root = self._resolve_split_root('test')
        if not os.path.exists(test_root):
            raise FileNotFoundError(f"Test root not found: {test_root}")

        normal_count = 0
        abnormal_count = 0

        normal_dir = os.path.join(test_root, 'normal')
        if os.path.isdir(normal_dir):
            for entry in sorted(os.listdir(normal_dir)):
                patient_dir = os.path.join(normal_dir, entry)
                if not os.path.isdir(patient_dir) or not self._is_valid_patient_dir(patient_dir):
                    continue
                n_before = len(self.samples)
                self._collect_patient_slices(patient_dir, label=0)
                normal_count += len(self.samples) - n_before

        abnormal_dir = os.path.join(test_root, 'abnormal')
        if os.path.isdir(abnormal_dir):
            for entry in sorted(os.listdir(abnormal_dir)):
                patient_dir = os.path.join(abnormal_dir, entry)
                if not os.path.isdir(patient_dir) or not self._is_valid_patient_dir(patient_dir):
                    continue
                label_dir = os.path.join(patient_dir, 'label')
                if not os.path.isdir(label_dir):
                    continue
                n_before = len(self.samples)
                self._collect_patient_slices(patient_dir, label=1, label_dir=label_dir)
                abnormal_count += len(self.samples) - n_before

        print(f"Test [{self.modality.upper()}]: {normal_count} normal + {abnormal_count} abnormal = {len(self.samples)} total")
        if len(self.samples) == 0:
            raise ValueError("No test samples loaded!")

    def _gray_to_rgb_pil(self, img_path):
        gray = Image.open(img_path).convert('L')
        return gray.convert('RGB')

    def _petct_to_pseudo_rgb_pil(self, image_paths):
        path_by_modality = dict(zip(self.modalities, image_paths))
        ct = Image.open(path_by_modality['ct']).convert('L')
        pet = Image.open(path_by_modality['pet']).convert('L')
        if self.dual_input_mode == 'ctctpet':
            return Image.merge('RGB', (ct, ct, pet))
        return Image.merge('RGB', (ct, pet, pet))

    def _load_image_tensor(self, img_path):
        if isinstance(img_path, tuple):
            img = self._petct_to_pseudo_rgb_pil(img_path)
        else:
            img = self._gray_to_rgb_pil(img_path)

        if self.transform:
            return self.transform(img)

        img = img.resize((self.image_size, self.image_size))
        return torch.from_numpy(
            np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label, mask_path = self.samples[index]
        img_tensor = self._load_image_tensor(img_path)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        if self.mode == 'train':
            return img_tensor, label_tensor

        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_np = (np.array(mask, dtype=np.float32) > 127.5).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        else:
            mask_tensor = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)

        if self.return_path:
            if isinstance(img_path, tuple):
                return img_tensor, label_tensor, mask_tensor, img_path[0]
            return img_tensor, label_tensor, mask_tensor, img_path

        return img_tensor, label_tensor, mask_tensor
