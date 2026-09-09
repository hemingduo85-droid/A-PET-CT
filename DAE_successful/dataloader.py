from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from skimage.transform import resize
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_tracer_root(data_root, tracer):
    data_root = Path(data_root).expanduser()
    if not data_root.is_absolute():
        module_dir = Path(__file__).resolve().parent
        for candidate in (data_root, module_dir / data_root, module_dir.parent / data_root):
            if candidate.exists():
                data_root = candidate
                break
    data_root = data_root.resolve()
    if (data_root / "train").exists() and (data_root / "test").exists():
        return data_root
    return data_root / tracer.upper()


class PETCTSliceDataset(Dataset):
    """Dataset for A_data/2d_equal/{FDG,PSMA}.

    Expected layout:
        root/
          train/normal/<patient>/<ct|pet>/<slice>.png
          test/normal/<patient>/<ct|pet>/<slice>.png
          test/abnormal/<patient>/<ct|pet>/<slice>.png
          test/abnormal/<patient>/label/<slice>.png
    """

    def __init__(
        self,
        root,
        modalities=("ct", "pet"),
        mode="train",
        target_size=(256, 256),
        debug_ratio=1.0,
        allow_empty_masks=True,
    ):
        self.root = Path(root)
        self.modalities = [m.lower() for m in modalities]
        self.mode = mode
        self.target_size = tuple(target_size)
        self.allow_empty_masks = allow_empty_masks
        self.samples = []

        invalid = sorted(set(self.modalities) - {"ct", "pet"})
        if invalid:
            raise ValueError(f"Invalid modalities: {invalid}. Valid options are ['ct', 'pet'].")
        if self.mode not in {"train", "test"}:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'test'.")

        self._load_samples()

        if debug_ratio < 1.0:
            keep = max(1, int(len(self.samples) * debug_ratio))
            self.samples = self.samples[:keep]

        if not self.samples:
            raise FileNotFoundError(
                f"No valid {self.mode} samples found under {self.root} with modalities={self.modalities}."
            )

        print(f"Loaded {len(self.samples)} {self.mode} samples from {self.root} with modalities={self.modalities}")

    def _load_samples(self):
        split_dir = self.root / self.mode
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        skipped = 0
        class_names = ("normal",) if self.mode == "train" else ("normal", "abnormal")
        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                if self.mode == "train" or class_name == "normal":
                    raise FileNotFoundError(f"Required class directory not found: {class_dir}")
                continue

            for patient_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                base_dir = patient_dir / self.modalities[0]
                if not base_dir.exists():
                    skipped += 1
                    continue

                for base_path in self._list_images(base_dir):
                    if not self._is_usable_image(base_path):
                        skipped += 1
                        continue

                    modality_paths = {self.modalities[0]: base_path}
                    valid = True
                    for mod in self.modalities[1:]:
                        mod_path = patient_dir / mod / base_path.name
                        if not self._is_usable_image(mod_path):
                            valid = False
                            skipped += 1
                            break
                        modality_paths[mod] = mod_path
                    if not valid:
                        continue

                    mask_path = None
                    if self.mode == "test" and class_name == "abnormal":
                        candidate = patient_dir / "label" / base_path.name
                        if self._is_usable_image(candidate):
                            mask_path = candidate
                        elif not self.allow_empty_masks and candidate.exists():
                            skipped += 1
                            continue

                    self.samples.append(
                        {
                            "id": f"{class_name}__{patient_dir.name}__{base_path.stem}",
                            "class_name": class_name,
                            "patient": patient_dir.name,
                            "slice": base_path.stem,
                            "modalities": modality_paths,
                            "mask_path": mask_path,
                        }
                    )

        if skipped:
            print(f"Skipped {skipped} missing or unusable files/directories while loading {self.mode}.")

    @staticmethod
    def _list_images(folder):
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)

    @staticmethod
    def _is_usable_image(path):
        return path.exists() and path.is_file() and path.stat().st_size > 0 and path.suffix.lower() in IMG_EXTS

    def _read_gray(self, path, is_mask=False):
        try:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Failed to read image: {path}") from exc

        if arr.shape != self.target_size:
            arr = resize(
                arr,
                self.target_size,
                anti_aliasing=not is_mask,
                preserve_range=True,
            ).astype(np.float32)
        if is_mask:
            arr = (arr > 0.5).astype(np.float32)
        return arr

    def __getitem__(self, index):
        sample = self.samples[index]
        data = {}
        for mod in self.modalities:
            arr = self._read_gray(sample["modalities"][mod], is_mask=False)
            data[mod] = torch.from_numpy(arr).float().unsqueeze(0)

        if self.mode == "test" and sample["mask_path"] is not None:
            try:
                mask_np = self._read_gray(sample["mask_path"], is_mask=True)
            except RuntimeError:
                mask_np = np.zeros(self.target_size, dtype=np.float32)
        else:
            mask_np = np.zeros(self.target_size, dtype=np.float32)

        mask = torch.from_numpy(mask_np).float().unsqueeze(0)
        label = torch.tensor(float(sample["class_name"] == "abnormal"), dtype=torch.float32)

        return {
            "modalities": data,
            "label": label,
            "id": sample["id"],
            "mask": mask,
        }

    def __len__(self):
        return len(self.samples)

    def get_class_distribution(self):
        if self.mode == "train":
            return {"normal": len(self.samples)}

        normal = sum(1 for sample in self.samples if sample["class_name"] == "normal")
        abnormal = sum(1 for sample in self.samples if sample["class_name"] == "abnormal")
        return {"normal": normal, "abnormal": abnormal}
