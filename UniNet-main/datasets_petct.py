"""
PET-CT 数据集适配模块

数据结构:
  train/normal/
      <patient_id>/
          pet/   *.png
          ct/    *.png
  test/
      normal/
          <patient_id>/
              pet/   *.png
              ct/    *.png
      abnormal/
          <patient_id>/
              pet/    *.png
              ct/     *.png
              label/  *.png  (异常掩膜, 同名文件)

modality 参数:
  'pet'   -> [PET, PET, PET]  伪 RGB (三通道复制)
  'ct'    -> [CT,  CT,  CT]   伪 RGB (三通道复制)
  'petct' -> [CT,  PET, PET]  伪 RGB (CT 作为 R 通道, PET 作为 G/B 通道)
"""

import os
import glob

import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


# ImageNet 归一化参数 (与预训练 Wide ResNet-50-2 匹配)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class PETCTDataset(torch.utils.data.Dataset):
    """
    PET-CT 无监督异常检测数据集。

    参数:
        data_root (str): 数据集根目录, 下含 train/ 和 test/ 子目录
        phase     (str): 'train' 或 'test'
        modality  (str): 'pet', 'ct', 'petct'
        image_size(int): 输入图像边长 (正方形)
    """

    def __init__(self, data_root, phase, modality='petct', image_size=256):
        assert phase in ('train', 'test'), f"phase 必须是 'train' 或 'test', 收到 {phase}"
        assert modality in ('pet', 'ct', 'petct'), \
            f"modality 必须是 'pet', 'ct' 或 'petct', 收到 {modality}"

        self.data_root  = data_root
        self.phase      = phase
        self.modality   = modality
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

        # samples: list of (pet_path, ct_path, label, mask_path_or_None)
        self.samples = []
        self._build_sample_list()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_sample_list(self):
        if self.phase == 'train':
            normal_dir = os.path.join(self.data_root, 'train', 'normal')
            self._collect_patients(normal_dir, label=0, has_mask=False)
        else:
            normal_dir   = os.path.join(self.data_root, 'test', 'normal')
            abnormal_dir = os.path.join(self.data_root, 'test', 'abnormal')
            self._collect_patients(normal_dir,   label=0, has_mask=False)
            self._collect_patients(abnormal_dir, label=1, has_mask=True)

        print(f"[PETCTDataset] phase={self.phase}, modality={self.modality}, "
              f"samples={len(self.samples)}")

    def _collect_patients(self, root_dir, label, has_mask):
        if not os.path.isdir(root_dir):
            print(f"[PETCTDataset] 警告: 目录不存在 -> {root_dir}")
            return

        for patient in sorted(os.listdir(root_dir)):
            patient_dir = os.path.join(root_dir, patient)
            if not os.path.isdir(patient_dir):
                continue

            pet_dir   = os.path.join(patient_dir, 'pet')
            ct_dir    = os.path.join(patient_dir, 'ct')
            label_dir = os.path.join(patient_dir, 'label')

            if not os.path.isdir(pet_dir) or not os.path.isdir(ct_dir):
                print(f"[PETCTDataset] 警告: 缺少 pet 或 ct 子目录 -> {patient_dir}")
                continue

            pet_files = sorted(glob.glob(os.path.join(pet_dir, '*.png')))
            if len(pet_files) == 0:
                print(f"[PETCTDataset] 警告: 无 PNG 文件 -> {pet_dir}")
                continue

            for pet_path in pet_files:
                fname   = os.path.basename(pet_path)
                ct_path = os.path.join(ct_dir, fname)
                if not os.path.isfile(ct_path):
                    continue  # 跳过没有对应 CT 切片的 PET

                mask_path = None
                if has_mask and os.path.isdir(label_dir):
                    candidate = os.path.join(label_dir, fname)
                    if os.path.isfile(candidate):
                        mask_path = candidate
                    # 若 abnormal 样本没有 mask 文件, mask_path 保持 None -> 零 mask

                self.samples.append((pet_path, ct_path, label, mask_path))

    def _build_rgb(self, pet_img_L, ct_img_L):
        """将单通道 PIL Image 按 modality 合并为 RGB PIL Image."""
        if self.modality == 'pet':
            return Image.merge('RGB', [pet_img_L, pet_img_L, pet_img_L])
        elif self.modality == 'ct':
            return Image.merge('RGB', [ct_img_L, ct_img_L, ct_img_L])
        else:  # petct: R=CT, G=PET, B=PET
            return Image.merge('RGB', [ct_img_L, pet_img_L, pet_img_L])

    # ------------------------------------------------------------------
    # 标准接口
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pet_path, ct_path, label, mask_path = self.samples[idx]

        pet_img = Image.open(pet_path).convert('L')
        ct_img  = Image.open(ct_path).convert('L')

        rgb_img = self._build_rgb(pet_img, ct_img)
        img     = self.img_transform(rgb_img)

        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        return img, label, mask, pet_path


def get_petct_dataloaders(data_root, modality='petct', image_size=256,
                          batch_size=8, num_workers=4):
    """
    返回 (train_dataloader, test_dataloader)。

    train_dataloader: shuffle=True,  batch_size=batch_size
    test_dataloader : shuffle=False, batch_size=1
    """
    train_dataset = PETCTDataset(data_root, phase='train',
                                 modality=modality, image_size=image_size)
    test_dataset  = PETCTDataset(data_root, phase='test',
                                 modality=modality, image_size=image_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
