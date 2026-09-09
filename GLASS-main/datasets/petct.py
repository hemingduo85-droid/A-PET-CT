from torchvision import transforms
from perlin import perlin_mask
from enum import Enum

import numpy as np
import PIL
import torch
import os
import glob
import logging

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class PETCTDataset(torch.utils.data.Dataset):
    """
    PET-CT medical image dataset for anomaly detection.

    Supports three modality modes via `classname`:
      - 'pet':   single PET modality  (grayscale -> pseudo-RGB by repeating)
      - 'ct':    single CT modality   (grayscale -> pseudo-RGB by repeating)
      - 'petct': dual modality merged as pseudo-RGB  [CT, PET, PET]

    Expected directory layout:
      {source}/train/normal/{patient}/pet/*.png
      {source}/train/normal/{patient}/ct/*.png
      {source}/test/normal/{patient}/pet/*.png
      {source}/test/normal/{patient}/ct/*.png
      {source}/test/abnormal/{patient}/pet/*.png
      {source}/test/abnormal/{patient}/ct/*.png
      {source}/test/abnormal/{patient}/label/*.png
    """

    def __init__(
            self,
            source,
            anomaly_source_path='',
            dataset_name='petct',
            classname='pet',
            resize=256,
            imagesize=256,
            split=DatasetSplit.TRAIN,
            rotate_degrees=0,
            translate=0,
            brightness_factor=0,
            contrast_factor=0,
            saturation_factor=0,
            gray_p=0,
            h_flip_p=0,
            v_flip_p=0,
            distribution=0,
            mean=0.5,
            std=0.1,
            fg=0,
            rand_aug=1,
            downsampling=8,
            scale=0,
            batch_size=8,
            **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.batch_size = batch_size
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.fg = 0
        self.class_fg = 0
        self.rand_aug = rand_aug
        self.downsampling = downsampling
        self.resize = resize if self.distribution != 1 else [resize, resize]
        self.imgsize = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)
        self.classname = classname
        self.dataset_name = dataset_name
        self.modality = classname  # 'pet', 'ct', or 'petct'

        self.imgpaths_per_class, self.data_to_iterate = self._collect_paths()
        if len(self.data_to_iterate) == 0:
            split_name = self.split.value
            found_dirs = self._summarize_dirs(self.source)
            expected = (
                f"{self.source}/{split_name}/normal/<patient>/pet/*.png and ct/*.png"
                if self.split == DatasetSplit.TRAIN
                else f"{self.source}/test/normal/<patient>/pet/*.png and ct/*.png; "
                     f"{self.source}/test/abnormal/<patient>/pet/*.png, ct/*.png, label/*.png"
            )
            raise RuntimeError(
                f"No PET-CT samples found for source='{self.source}', split='{split_name}', "
                f"modality='{self.modality}'. Expected layout: {expected}. "
                f"Found directories: {found_dirs}"
            )

        dtd_files = sorted(glob.glob(anomaly_source_path + "/*/*.jpg")) if anomaly_source_path else []
        if len(dtd_files) > 0:
            self.anomaly_source_paths = dtd_files
        else:
            self.anomaly_source_paths = self._fallback_anomaly_sources()
            if self.split == DatasetSplit.TRAIN:
                LOGGER.warning("DTD textures not found – using training images as anomaly source (less ideal)")

        self.transform_img = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(
                rotate_degrees,
                translate=(translate, translate) if translate > 0 else None,
                scale=(1.0 - scale, 1.0 + scale) if scale > 0 else None,
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
        ])

    def _fallback_anomaly_sources(self):
        """Use CT training images as anomaly texture source when DTD is unavailable."""
        train_dir = os.path.join(self.source, "train", "normal")
        paths = []
        if os.path.exists(train_dir):
            for patient in sorted(os.listdir(train_dir)):
                ct_dir = os.path.join(train_dir, patient, "ct")
                if os.path.isdir(ct_dir):
                    for f in sorted(os.listdir(ct_dir)):
                        if f.endswith('.png'):
                            paths.append(os.path.join(ct_dir, f))
        return paths if paths else ["__none__"]

    @staticmethod
    def _normalize_gray(arr):
        """Per-image min-max normalization to [0, 255]."""
        vmin, vmax = arr.min(), arr.max()
        if vmax > vmin:
            return ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    def _load_image(self, pet_path=None, ct_path=None):
        """Load and assemble a 3-channel PIL Image depending on modality."""
        if self.modality == 'pet':
            gray = np.array(PIL.Image.open(pet_path).convert('L'), dtype=np.float32)
            gray = self._normalize_gray(gray)
            return PIL.Image.fromarray(gray).convert('RGB')

        if self.modality == 'ct':
            gray = np.array(PIL.Image.open(ct_path).convert('L'), dtype=np.float32)
            gray = self._normalize_gray(gray)
            return PIL.Image.fromarray(gray).convert('RGB')

        # petct  ->  [CT, PET, PET] pseudo-RGB
        pet = self._normalize_gray(np.array(PIL.Image.open(pet_path).convert('L'), dtype=np.float32))
        ct  = self._normalize_gray(np.array(PIL.Image.open(ct_path).convert('L'), dtype=np.float32))
        merged = np.stack([ct, pet, pet], axis=-1)
        return PIL.Image.fromarray(merged, mode='RGB')

    def rand_augmenter(self):
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)
        return transforms.Compose([
            transforms.Resize(self.resize),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __getitem__(self, idx):
        classname, anomaly, image_ref, mask_path = self.data_to_iterate[idx]

        if self.modality == 'petct':
            pet_path, ct_path = image_ref
            image = self._load_image(pet_path=pet_path, ct_path=ct_path)
            display_path = pet_path
        elif self.modality == 'pet':
            image = self._load_image(pet_path=image_ref)
            display_path = image_ref
        else:
            image = self._load_image(ct_path=image_ref)
            display_path = image_ref

        image = self.transform_img(image)

        mask_fg = mask_s = aug_image = torch.tensor([1])
        if self.split == DatasetSplit.TRAIN:
            aug_src = np.random.choice(self.anomaly_source_paths)
            if aug_src != "__none__":
                aug = PIL.Image.open(aug_src).convert("RGB")
            else:
                aug = PIL.Image.fromarray(
                    np.random.randint(0, 256, (self.imgsize, self.imgsize, 3), dtype=np.uint8))

            if self.rand_aug:
                aug = self.rand_augmenter()(aug)
            else:
                aug = self.transform_img(aug)

            mask_all = perlin_mask(image.shape, self.imgsize // self.downsampling, 0, 6, mask_fg, 1)
            mask_s = torch.from_numpy(mask_all[0])
            mask_l = torch.from_numpy(mask_all[1])

            beta = np.clip(np.random.normal(loc=self.mean, scale=self.std), .2, .8)
            aug_image = image * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * image * mask_l

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask_gt = PIL.Image.open(mask_path).convert('L')
            mask_gt = self.transform_mask(mask_gt)
        else:
            mask_gt = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "aug": aug_image,
            "mask_s": mask_s,
            "mask_gt": mask_gt,
            "is_anomaly": int(anomaly != "good"),
            "image_path": display_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def _collect_paths(self):
        """Walk the PET-CT directory tree and build the iteration list."""
        imgpaths = {self.classname: {}}
        data = []

        if self.split == DatasetSplit.TRAIN:
            train_dir = self._find_child_dir(self.source, ["train", "training"])
            normal_dir = self._find_child_dir(train_dir, ["normal", "good", "negative"]) if train_dir else None
            paths = self._scan_patients(normal_dir, label_subdir=None)
            imgpaths[self.classname]["good"] = [p[0] for p in paths]
            for img_ref, _ in paths:
                data.append([self.classname, "good", img_ref, None])

        else:  # TEST
            test_dir = self._find_child_dir(self.source, ["test", "testing", "val", "validation"])
            normal_dir = self._find_child_dir(test_dir, ["normal", "good", "negative"]) if test_dir else None
            if normal_dir and os.path.isdir(normal_dir):
                paths = self._scan_patients(normal_dir, label_subdir=None)
                imgpaths[self.classname]["good"] = [p[0] for p in paths]
                for img_ref, _ in paths:
                    data.append([self.classname, "good", img_ref, None])

            abnormal_dir = self._find_child_dir(test_dir, ["abnormal", "anom", "anomaly", "positive", "bad"]) if test_dir else None
            if abnormal_dir and os.path.isdir(abnormal_dir):
                paths = self._scan_patients(abnormal_dir, label_subdir=["label", "labels", "mask", "masks", "gt", "ground_truth"])
                imgpaths[self.classname]["abnormal"] = [p[0] for p in paths]
                for img_ref, mask_path in paths:
                    data.append([self.classname, "abnormal", img_ref, mask_path])

        return imgpaths, data

    @staticmethod
    def _find_child_dir(parent, aliases):
        if not parent or not os.path.isdir(parent):
            return None
        alias_map = {alias.lower(): alias for alias in aliases}
        for child in sorted(os.listdir(parent)):
            path = os.path.join(parent, child)
            if os.path.isdir(path) and child.lower() in alias_map:
                return path
        return None

    @staticmethod
    def _image_files(path):
        if not path or not os.path.isdir(path):
            return {}
        return {
            os.path.splitext(f)[0].lower(): os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        }

    @staticmethod
    def _summarize_dirs(root, max_dirs=40):
        if not os.path.isdir(root):
            return "<source path does not exist>"
        found = []
        for current, dirs, _ in os.walk(root):
            rel = os.path.relpath(current, root)
            found.append("." if rel == "." else rel)
            dirs[:] = sorted(dirs)
            if len(found) >= max_dirs:
                found.append("...")
                break
        return ", ".join(found)

    def _scan_patients(self, root_dir, label_subdir=None):
        """Scan patient directories and return (image_ref, mask_path) pairs."""
        results = []
        if not os.path.isdir(root_dir):
            return results
        for patient in sorted(os.listdir(root_dir)):
            pdir = os.path.join(root_dir, patient)
            if not os.path.isdir(pdir):
                continue
            pet_dir = self._find_child_dir(pdir, ["pet", "suv", "fdg", "psma"])
            ct_dir = self._find_child_dir(pdir, ["ct"])
            if not os.path.isdir(pet_dir) or not os.path.isdir(ct_dir):
                continue

            pet_files = self._image_files(pet_dir)
            ct_files = self._image_files(ct_dir)
            common = sorted(set(pet_files) & set(ct_files))
            if not common and len(pet_files) == len(ct_files):
                common = list(range(len(pet_files)))
                pet_values = [pet_files[k] for k in sorted(pet_files)]
                ct_values = [ct_files[k] for k in sorted(ct_files)]
            else:
                pet_values = None
                ct_values = None

            lbl_dir = self._find_child_dir(pdir, label_subdir) if label_subdir else None
            lbl_files = self._image_files(lbl_dir)

            for idx, key in enumerate(common):
                if pet_values is None:
                    pet_p = pet_files[key]
                    ct_p = ct_files[key]
                    mask = lbl_files.get(key)
                else:
                    pet_p = pet_values[idx]
                    ct_p = ct_values[idx]
                    mask = lbl_files.get(os.path.splitext(os.path.basename(pet_p))[0].lower())

                if self.modality == 'pet':
                    ref = pet_p
                elif self.modality == 'ct':
                    ref = ct_p
                else:
                    ref = (pet_p, ct_p)

                results.append((ref, mask))
        return results
