import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image


IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


def _is_image_file(fname: str) -> bool:
    lower = fname.lower()
    return any(lower.endswith(ext) for ext in IMG_EXTS)


class DetectionDataset(Dataset):
    """
    PET/CT 异常检测。

    目录结构 (2d_equal_mask50/{FDG|PSMA}):
      train:  /train/normal/<patient>/{pet|ct}/*.png  (无 mask)
      test:
        normal:   /test/normal/<patient>/{pet|ct}/*.png (无 mask)
        abnormal: /test/abnormal/<patient>/{pet|ct,label}/*.png (label 是 mask)

    输出:
      img:   [C,224,224]  pet/ct 为灰度复制，petct 为 [CT,PET,PET] 伪 RGB
      gt:    [1,224,224]  异常为 mask，正常/训练为 0
      label: 0/1
      name:  image path
    """

    def __init__(self, root_dir, transform=None, gt_transform=None, label=0, phase='train',
                 modality='pet', replicate_channels=3):
        self.root_dir = root_dir
        self.transform = transform
        self.gt_transform = gt_transform
        self.label = label
        self.phase = phase
        self.modality = modality.lower()
        self.replicate_channels = replicate_channels

        if self.modality not in ('pet', 'ct', 'petct'):
            raise ValueError("modality must be 'pet', 'ct', or 'petct'")

        self.img_paths, self.gt_paths, self.labels, self.names = self._find_images()

    def _resolve_train_root(self):
        normal_sub = os.path.join(self.root_dir, 'normal')
        if os.path.isdir(normal_sub):
            return normal_sub
        return self.root_dir

    def _is_valid_patient_dir(self, patient_dir):
        if self.modality == 'petct':
            pet_dir = os.path.join(patient_dir, 'pet')
            ct_dir = os.path.join(patient_dir, 'ct')
            if not (os.path.isdir(pet_dir) and os.path.isdir(ct_dir)):
                return False
            return any(_is_image_file(f) and os.path.exists(os.path.join(ct_dir, f)) for f in os.listdir(pet_dir))

        mod_dir = os.path.join(patient_dir, self.modality)
        if not os.path.isdir(mod_dir):
            return False
        return any(_is_image_file(f) for f in os.listdir(mod_dir))

    def _collect_patient(self, patient_dir, img_paths, gt_paths, labels, names, label_dir=None):
        mod_dir = os.path.join(patient_dir, 'pet' if self.modality == 'petct' else self.modality)
        for fname in sorted(os.listdir(mod_dir)):
            if not _is_image_file(fname):
                continue
            if self.modality == 'petct' and not os.path.exists(os.path.join(patient_dir, 'ct', fname)):
                continue
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(mod_dir, fname)
            img_paths.append(img_path)
            labels.append(self.label)
            names.append(base)

            if self.label == 1 and self.phase == 'test' and label_dir is not None:
                gt_path = os.path.join(label_dir, fname)
                gt_paths.append(gt_path if os.path.exists(gt_path) else None)
            else:
                gt_paths.append(None)

    def _find_images(self):
        img_paths = []
        gt_paths = []
        labels = []
        names = []

        if self.phase == 'train':
            root = self._resolve_train_root()
            patient_ids = [
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]
            for pid in sorted(patient_ids):
                patient_path = os.path.join(root, pid)
                if not self._is_valid_patient_dir(patient_path):
                    continue
                self._collect_patient(patient_path, img_paths, gt_paths, labels, names)
        else:
            # test root_dir: /.../test/normal 或 /.../test/abnormal
            patient_ids = [
                d for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ]
            for pid in sorted(patient_ids):
                patient_path = os.path.join(self.root_dir, pid)
                if not self._is_valid_patient_dir(patient_path):
                    continue
                label_dir = os.path.join(patient_path, 'label') if self.label == 1 else None
                self._collect_patient(
                    patient_path, img_paths, gt_paths, labels, names, label_dir=label_dir
                )

        return (
            np.array(img_paths, dtype=object),
            np.array(gt_paths, dtype=object),
            np.array(labels, dtype=np.int64),
            np.array(names, dtype=object),
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt_path = self.gt_paths[idx]
        label = int(self.labels[idx])
        base_name = self.names[idx]

        gray = Image.open(img_path).convert('L')
        if self.modality == 'petct':
            ct_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'ct', os.path.basename(img_path))
            ct_gray = Image.open(ct_path).convert('L')
            img_pil = Image.merge('RGB', (ct_gray, gray, gray))
        elif self.replicate_channels == 3:
            img_pil = gray.convert('RGB')
        elif self.replicate_channels == 1:
            img_pil = gray
        else:
            img_pil = gray.convert('RGB')

        if self.transform:
            img = self.transform(img_pil)
        else:
            img = transforms.ToTensor()(img_pil)

        if self.replicate_channels not in (1, 3):
            if img.shape[0] == 1:
                img = img.repeat(self.replicate_channels, 1, 1)

        h, w = img.shape[-2], img.shape[-1]
        if (label == 0) or (self.phase == 'train') or (gt_path is None) or (not os.path.exists(gt_path)):
            gt = torch.zeros((1, h, w), dtype=torch.float32)
        else:
            gt_img = Image.open(gt_path).convert('L')
            if self.gt_transform:
                gt = self.gt_transform(gt_img)
            else:
                gt = transforms.ToTensor()(gt_img)
            if gt.dim() == 3 and gt.shape[0] > 1:
                gt = gt[:1, ...]

        return img, gt, label, str(img_path)


def get_data_transforms(size=256, isize=224, num_channels=3, normalize=True):
    transform_steps = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
    ]
    if normalize:
        mean_train = [0.485] * num_channels
        std_train = [0.229] * num_channels
        transform_steps.append(transforms.Normalize(mean=mean_train, std=std_train))

    data_transform = transforms.Compose(transform_steps)

    gt_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor(),
    ])

    return data_transform, gt_transform


def get_detection_dataset(data_dir, file_name='train', modality='pet', replicate_channels=3,
                          normalize=True, image_size=256, crop_size=224):
    """
    data_dir: /data/.../{FDG|PSMA}
    train: data_dir/train/normal/<patient>/{pet|ct}/*.png
    test : data_dir/test/normal/<patient>/{pet|ct}/*.png
           data_dir/test/abnormal/<patient>/{pet|ct,label}/*.png
    """
    data_transform, gt_transform = get_data_transforms(
        size=image_size, isize=crop_size, num_channels=replicate_channels, normalize=normalize
    )

    if file_name == 'train':
        return DetectionDataset(
            root_dir=os.path.join(data_dir, 'train'),
            transform=data_transform,
            gt_transform=gt_transform,
            label=0,
            phase='train',
            modality=modality,
            replicate_channels=replicate_channels,
        )
    elif file_name == 'test':
        normal_test = DetectionDataset(
            root_dir=os.path.join(data_dir, 'test', 'normal'),
            transform=data_transform,
            gt_transform=gt_transform,
            label=0,
            phase='test',
            modality=modality,
            replicate_channels=replicate_channels,
        )
        abnormal_test = DetectionDataset(
            root_dir=os.path.join(data_dir, 'test', 'abnormal'),
            transform=data_transform,
            gt_transform=gt_transform,
            label=1,
            phase='test',
            modality=modality,
            replicate_channels=replicate_channels,
        )
        return ConcatDataset([normal_test, abnormal_test])
    else:
        raise ValueError("file_name must be 'train' or 'test'")


if __name__ == "__main__":
    data_dir = "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA"
    for mod in ('pet', 'ct', 'petct'):
        ds = get_detection_dataset(data_dir, 'train', modality=mod)
        print(f"train [{mod}] len:", len(ds))
        x, gt, y, name = ds[0]
        print(f"  img: {x.shape}, gt: {gt.shape}, label: {y}, name: {name}")
