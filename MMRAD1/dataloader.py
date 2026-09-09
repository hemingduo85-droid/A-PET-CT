from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class MultimodalGrayDataset(Dataset):
    """PET/CT slice dataset for A_data/2d_equal/<tracer>.

    Layout:
        train/normal/<patient>/<ct|pet>/<slice>.png
        test/normal/<patient>/<ct|pet>/<slice>.png
        test/abnormal/<patient>/<ct|pet>/<slice>.png
        test/abnormal/<patient>/label/<slice>.png
    """

    def __init__(self, data_path, modalities=("ct", "pet"), mode="train", image_size=256):
        self.data_path = self._resolve_data_path(data_path)
        self.modalities = [m.lower() for m in modalities]
        self.mode = mode
        self.image_size = image_size
        self.samples = []

        if self.mode not in {"train", "test"}:
            raise ValueError(f"Invalid mode: {mode}. Expected 'train' or 'test'.")
        invalid = sorted(set(self.modalities) - {"ct", "pet"})
        if invalid:
            raise ValueError(f"Invalid modalities: {invalid}. Expected ct/pet.")

        self._load_samples()
        if not self.samples:
            raise FileNotFoundError(f"No samples found under {self.data_path} for mode={self.mode}.")

    @staticmethod
    def _resolve_data_path(data_path):
        path = Path(data_path).expanduser()
        if not path.is_absolute():
            module_dir = Path(__file__).resolve().parent
            for candidate in (path, module_dir / path, module_dir.parent / path):
                if candidate.exists():
                    return candidate.resolve()
        return path.resolve()

    def _load_samples(self):
        class_names = ("normal",) if self.mode == "train" else ("normal", "abnormal")
        for class_name in class_names:
            class_dir = self.data_path / self.mode / class_name
            if not class_dir.exists():
                if self.mode == "train" or class_name == "normal":
                    raise FileNotFoundError(f"Required class directory not found: {class_dir}")
                continue

            for patient_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                base_dir = patient_dir / self.modalities[0]
                if not base_dir.exists():
                    continue
                for base_path in self._list_images(base_dir):
                    if not self._has_all_modalities(patient_dir, base_path.name):
                        continue
                    mask_path = None
                    if self.mode == "test" and class_name == "abnormal":
                        candidate = patient_dir / "label" / base_path.name
                        mask_path = candidate if candidate.is_file() and candidate.stat().st_size > 0 else None

                    self.samples.append(
                        {
                            "id": f"{class_name}__{patient_dir.name}__{base_path.stem}",
                            "label": 1 if class_name == "abnormal" else 0,
                            "modalities": {mod: patient_dir / mod / base_path.name for mod in self.modalities},
                            "mask": mask_path,
                        }
                    )

    @staticmethod
    def _list_images(folder):
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)

    def _has_all_modalities(self, patient_dir, filename):
        return all((patient_dir / mod / filename).is_file() for mod in self.modalities)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        modalities_data = []

        for mod in self.modalities:
            img = Image.open(sample["modalities"][mod]).convert("L")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img = np.asarray(img, dtype=np.float32) / 255.0
            modalities_data.append(img)

        if sample["mask"] is not None:
            try:
                mask = Image.open(sample["mask"]).convert("L")
                mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
                mask = (np.asarray(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
            except OSError:
                mask = np.zeros_like(modalities_data[0], dtype=np.float32)
        else:
            mask = np.zeros_like(modalities_data[0], dtype=np.float32)

        return {
            "image": torch.FloatTensor(np.stack(modalities_data, axis=0)),
            "label": torch.tensor(float(sample["label"]), dtype=torch.float32),
            "id": sample["id"],
            "mask": torch.FloatTensor(mask),
        }

    def class_distribution(self):
        if self.mode == "train":
            return {"normal": len(self.samples)}
        normal = sum(1 for sample in self.samples if sample["label"] == 0)
        abnormal = sum(1 for sample in self.samples if sample["label"] == 1)
        return {"normal": normal, "abnormal": abnormal}
