import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ResizeToTensorNormalize:
    def __init__(self, image_size):
        self.image_size = image_size
        self.mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image):
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
        return (tensor - self.mean) / self.std


class BrainTumorAnomalyDataset(torch.utils.data.Dataset):
    """PET/CT anomaly dataset for A_data/2d_equal/{FDG,PSMA}.

    Expected layout:
        root/
          train/normal/<patient>/<ct|pet>/<slice>.png
          test/normal/<patient>/<ct|pet>/<slice>.png
          test/abnormal/<patient>/<ct|pet>/<slice>.png
          test/abnormal/<patient>/label/<slice>.png

    Training samples are loaded only from train/normal. During testing, normal
    samples receive empty masks. Abnormal samples load label masks when present.
    Image-level labels are derived from the normal/abnormal directory name.
    """

    def __init__(
        self,
        root,
        mode="train",
        modalities=("ct", "pet"),
        transform=None,
        image_size=256,
        debug_ratio=1.0,
        allow_empty_masks=True,
    ):
        self.root = Path(root)
        self.mode = mode
        self.modalities = [mod.lower() for mod in modalities]
        self.image_size = image_size
        self.allow_empty_masks = allow_empty_masks
        self.transform = transform or self.get_default_transform(image_size)
        self.samples = []
        self.image_files = []

        valid_modalities = {"ct", "pet"}
        invalid = sorted(set(self.modalities) - valid_modalities)
        if invalid:
            raise ValueError(f"Invalid modalities: {invalid}. Valid options are {sorted(valid_modalities)}")

        if self.mode not in {"train", "test"}:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'test'.")

        self._load_patient_slice_data()

        if debug_ratio < 1.0:
            total = len(self.samples)
            keep = max(1, int(total * debug_ratio))
            self.samples = self.samples[:keep]
            self.image_files = self.image_files[:keep]
            print(f"Debug mode: using {keep}/{total} samples ({debug_ratio:.0%}).")

        if not self.samples:
            raise FileNotFoundError(
                f"No valid {self.mode} samples found under {self.root}. "
                "Check the dataset path, modalities, and whether PNG files are non-empty."
            )

        print(
            f"Loaded {len(self.samples)} {self.mode} samples from {self.root} "
            f"with modalities={self.modalities}"
        )

    def _load_patient_slice_data(self):
        split_dir = self.root / self.mode
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        class_names = ("normal",) if self.mode == "train" else ("normal", "abnormal")
        skipped = 0

        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                if self.mode == "train" or class_name == "normal":
                    raise FileNotFoundError(f"Required class directory not found: {class_dir}")
                continue

            patient_dirs = sorted(p for p in class_dir.iterdir() if p.is_dir())
            for patient_dir in patient_dirs:
                base_mod_dir = patient_dir / self.modalities[0]
                if not base_mod_dir.exists():
                    skipped += 1
                    continue

                for base_path in self._list_images(base_mod_dir):
                    if not self._is_usable_image(base_path):
                        skipped += 1
                        continue

                    img_paths = {self.modalities[0]: str(base_path)}
                    valid = True
                    for mod in self.modalities[1:]:
                        mod_path = patient_dir / mod / base_path.name
                        if not self._is_usable_image(mod_path):
                            valid = False
                            skipped += 1
                            break
                        img_paths[mod] = str(mod_path)

                    if not valid:
                        continue

                    mask_path = None
                    if self.mode == "test" and class_name == "abnormal":
                        candidate = patient_dir / "label" / base_path.name
                        if self._is_usable_image(candidate):
                            mask_path = str(candidate)
                        elif not self.allow_empty_masks and candidate.exists():
                            skipped += 1
                            continue

                    self.samples.append(
                        {
                            "image_paths": img_paths,
                            "mask_path": mask_path,
                            "class_name": class_name,
                            "patient": patient_dir.name,
                            "slice": base_path.stem,
                        }
                    )
                    self.image_files.append(str(base_path))

        if skipped:
            print(f"Skipped {skipped} missing or unusable files/directories while loading {self.mode}.")

    @staticmethod
    def _list_images(folder):
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)

    @staticmethod
    def _is_usable_image(path):
        return path.exists() and path.is_file() and path.stat().st_size > 0 and path.suffix.lower() in IMG_EXTS

    @staticmethod
    def _read_gray_image(path):
        try:
            with Image.open(path) as img:
                return np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Failed to read image: {path}") from exc

    def get_default_transform(self, image_size):
        return ResizeToTensorNormalize(image_size)

    def _load_and_fuse_modalities(self, img_paths):
        modality_images = [self._read_gray_image(img_paths[mod]) for mod in self.modalities]

        if len(modality_images) == 1:
            modality_images = modality_images * 3
        elif len(modality_images) == 2:
            # AE-FLOW uses an ImageNet-pretrained 3-channel encoder. For PET/CT,
            # use [CT, PET, CT] and repeat CT as the structural modality.
            # modality_images.append(modality_images[0])
             # 核心修改：CT + PET 取平均，替代原有的重复CT
            fused_channel = (modality_images[0] + modality_images[1]) / 2.0
            modality_images.append(fused_channel)
        else:
        
            modality_images = modality_images[:3]

        fused_image = np.stack(modality_images, axis=-1)
        return Image.fromarray((fused_image * 255).astype(np.uint8))

    def _load_mask(self, mask_path, spatial_shape):
        if mask_path is None:
            return torch.zeros(1, *spatial_shape, dtype=torch.float32)

        try:
            with Image.open(mask_path) as mask_img:
                mask_img = mask_img.convert("L").resize(spatial_shape[::-1], resample=Image.NEAREST)
                mask = torch.from_numpy(np.asarray(mask_img, dtype=np.float32) / 255.0).unsqueeze(0)
        except (OSError, UnidentifiedImageError):
            return torch.zeros(1, *spatial_shape, dtype=torch.float32)
        return (mask > 0.5).float()

    def __getitem__(self, index):
        sample = self.samples[index]
        fused_img = self._load_and_fuse_modalities(sample["image_paths"])
        img_tensor = self.transform(fused_img)

        if self.mode == "train":
            return img_tensor, torch.tensor(0.0, dtype=torch.float32)

        mask = self._load_mask(sample["mask_path"], img_tensor.shape[-2:])
        label = torch.tensor(float(sample["class_name"] == "abnormal"), dtype=torch.float32)
        filename = f"{sample['class_name']}__{sample['patient']}__{sample['slice']}"
        return img_tensor, label, mask, filename

    def __len__(self):
        return len(self.samples)

    def get_class_distribution(self):
        if self.mode == "train":
            return {"normal": len(self.samples)}

        normal = sum(1 for sample in self.samples if sample["class_name"] == "normal")
        abnormal = sum(1 for sample in self.samples if sample["class_name"] == "abnormal")
        return {"normal": normal, "abnormal": abnormal}
