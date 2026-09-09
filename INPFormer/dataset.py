from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


torch.multiprocessing.set_sharing_strategy("file_system")


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_train, std=std_train),
        ]
    )
    gt_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.CenterCrop(isize),
            transforms.ToTensor(),
        ]
    )
    return data_transforms, gt_transforms


def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomResizedCrop((isize, isize), scale=(0.6, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train),
        ]
    )


class MVTecDataset(Dataset):
    """PET/CT adapter for INPFormer.

    Expected root is A_data/2d_equal/<tracer>:
        train/normal/<patient>/<ct|pet>/<slice>.png
        test/normal/<patient>/<ct|pet>/<slice>.png
        test/abnormal/<patient>/<ct|pet>/<slice>.png
        test/abnormal/<patient>/label/<slice>.png

    INPFormer expects 3-channel input. Single modality is repeated to RGB.
    With ct,pet, the default PET/CT adapter is [CT, PET, (CT + PET) / 2].
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root,
        transform,
        gt_transform,
        phase,
        modalities=("ct", "pet"),
        label_mode="folder",
        channel_fill="mean_modalities",
    ):
        self.root = self._resolve_root(root)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase
        self.modalities = [m.lower() for m in modalities]
        self.label_mode = label_mode
        self.channel_fill = channel_fill
        invalid = sorted(set(self.modalities) - {"ct", "pet"})
        if invalid:
            raise ValueError(f"Invalid modalities: {invalid}. Use ct, pet, or ct,pet.")
        if self.label_mode not in {"folder", "mask"}:
            raise ValueError("label_mode must be 'folder' or 'mask'.")
        if self.channel_fill not in {"zero", "imagenet_mean", "mean_modalities"}:
            raise ValueError("channel_fill must be 'zero', 'imagenet_mean', or 'mean_modalities'.")
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
        self.cls_idx = 0

    @staticmethod
    def _resolve_root(root):
        path = Path(root).expanduser()
        if not path.is_absolute():
            module_dir = Path(__file__).resolve().parent
            for candidate in (path, module_dir / path, module_dir.parent / path):
                if candidate.exists():
                    return candidate.resolve()
        return path.resolve()

    def load_dataset(self):
        split = "train" if self.phase in {"train", "new_train"} else "test"
        classes = ("normal",) if split == "train" else ("normal", "abnormal")
        img_paths, gt_paths, labels, types = [], [], [], []

        for class_name in classes:
            class_dir = self.root / split / class_name
            if not class_dir.exists():
                if split == "train" or class_name == "normal":
                    raise FileNotFoundError(f"Required class directory not found: {class_dir}")
                continue

            for patient_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                base_dir = patient_dir / self.modalities[0]
                if not base_dir.exists():
                    continue
                for base_path in self._list_images(base_dir):
                    paths = [patient_dir / mod / base_path.name for mod in self.modalities]
                    if not all(p.is_file() for p in paths):
                        continue
                    mask_path = None
                    if split == "test" and class_name == "abnormal":
                        candidate = patient_dir / "label" / base_path.name
                        mask_path = candidate if candidate.is_file() and candidate.stat().st_size > 0 else None
                    img_paths.append([str(p) for p in paths])
                    gt_paths.append(str(mask_path) if mask_path else None)
                    labels.append(1 if class_name == "abnormal" else 0)
                    types.append(class_name)

        print("\n===== Data Statistics =====")
        print(f"Root: {self.root}")
        print(f"Phase: {self.phase}")
        print(f"Modalities: {self.modalities}")
        print(f"Label mode: {self.label_mode}")
        print(f"Channel fill: {self.channel_fill}")
        print(f"Total samples: {len(img_paths)}")
        print(f"Total masks: {len([x for x in gt_paths if x is not None])}")
        print("==========================\n")
        return img_paths, gt_paths, labels, types

    @classmethod
    def _list_images(cls, folder):
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in cls.IMG_EXTS)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        modal_images = []
        for mod_path in self.img_paths[idx]:
            img = Image.open(mod_path).convert("L")
            modal_images.append(np.asarray(img, dtype=np.float32) / 255.0)

        if len(modal_images) == 1:
            modal_images = [modal_images[0], modal_images[0], modal_images[0]]
        while len(modal_images) < 3:
            modal_images.append(self._make_fill_channel(modal_images))
        fused = np.stack(modal_images[:3], axis=-1)
        img_pil = Image.fromarray((fused * 255).astype(np.uint8))
        img_tensor = self.transform(img_pil)

        if self.gt_paths[idx] is not None:
            try:
                gt = Image.open(self.gt_paths[idx]).convert("L")
                gt = (self.gt_transform(gt) > 0.5).float()
            except OSError:
                gt = torch.zeros((1, *img_tensor.shape[1:]))
        else:
            gt = torch.zeros((1, *img_tensor.shape[1:]))

        label = self.labels[idx]
        if self.label_mode == "mask" and self.phase not in {"train", "new_train"}:
            label = int(gt.sum().item() > 0)

        return img_tensor, gt, label, self.img_paths[idx][0]

    def _make_fill_channel(self, modal_images):
        if self.channel_fill == "zero":
            return np.zeros_like(modal_images[0], dtype=np.float32)
        if self.channel_fill == "mean_modalities":
            return np.mean(np.stack(modal_images, axis=0), axis=0).astype(np.float32)
        return np.full_like(modal_images[0], 0.406, dtype=np.float32)
