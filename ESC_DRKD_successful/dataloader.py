"""PET/CT dataloader for A_data/2d_equal/{FDG,PSMA}."""

from pathlib import Path

import cv2
import numpy as np
import torch

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROJECT_DIR = Path(__file__).resolve().parent


def resolve_tracer_root(data_root, tracer):
    data_root = Path(data_root).expanduser()
    if not data_root.is_absolute():
        for candidate in (data_root, PROJECT_DIR / data_root, PROJECT_DIR.parent / data_root):
            if candidate.exists():
                data_root = candidate
                break
    data_root = data_root.resolve()
    if (data_root / "train").exists() and (data_root / "test").exists():
        return data_root
    return data_root / tracer.upper()


def modality_tag(modalities):
    return "_".join(modalities)


class MultimodalDataset(torch.utils.data.Dataset):
  """Layout: {tracer}/train/normal|test/{normal,abnormal}/<patient>/{ct,pet,label}/"""

  def __init__(
      self,
      data_path,
      modalities=("ct", "pet"),
      mode="train",
      img_size=256,
      target_sample_ids=None,
  ):
    self.data_path = Path(data_path)
    self.modalities = [m.lower() for m in modalities]
    self.mode = mode
    self.img_size = img_size
    self.target_sample_ids = set(target_sample_ids) if target_sample_ids else None
    self.samples = []
    self._load_samples()

  def _list_images(self, folder):
    if not folder.is_dir():
      return []
    return sorted(
        p.name
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

  def _load_samples(self):
    if self.mode == "train":
      class_dir = self.data_path / "train" / "normal"
      if not class_dir.is_dir():
        raise FileNotFoundError(f"Missing train directory: {class_dir}")
      self._load_class_dir(class_dir, class_name="normal", is_train=True)
    else:
      for class_name in ("normal", "abnormal"):
        class_dir = self.data_path / "test" / class_name
        if class_dir.is_dir():
          self._load_class_dir(class_dir, class_name=class_name, is_train=False)

    if self.mode != "train" and self.target_sample_ids is not None:
      self.samples = [s for s in self.samples if s["id"] in self.target_sample_ids]

    print(f"Loaded {len(self.samples)} samples [{self.mode}] from {self.data_path}")

  def _load_class_dir(self, class_dir, class_name, is_train):
    for patient_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
      base_dir = patient_dir / self.modalities[0]
      if not base_dir.is_dir():
        continue
      for img_name in self._list_images(base_dir):
        if not all((patient_dir / mod / img_name).is_file() for mod in self.modalities):
          continue
        slice_stem = Path(img_name).stem
        sample_id = f"{class_name}__{patient_dir.name}__{slice_stem}"
        if self.target_sample_ids is not None and sample_id not in self.target_sample_ids:
          continue
        entry = {"id": sample_id, "class_name": class_name}
        for mod in self.modalities:
          entry[mod] = str(patient_dir / mod / img_name)
        if not is_train:
          entry["label"] = 0 if class_name == "normal" else 1
          mask_path = patient_dir / "label" / img_name
          entry["mask"] = (
              str(mask_path)
              if class_name == "abnormal" and mask_path.is_file() and mask_path.stat().st_size > 0
              else None
          )
          entry["img_path"] = entry[self.modalities[0]]
        self.samples.append(entry)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    sample = self.samples[idx]
    data = {}
    for mod in self.modalities:
      img = cv2.imread(sample[mod], cv2.IMREAD_GRAYSCALE)
      img = cv2.resize(img, (self.img_size, self.img_size))
      data[mod] = torch.from_numpy(img / 255.0).float().unsqueeze(0)

    if self.mode == "train":
      return {"modalities": data}

    mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
    if sample.get("mask"):
      loaded = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE)
      if loaded is not None:
        loaded = cv2.resize(loaded, (self.img_size, self.img_size))
        mask = (loaded > 127).astype(np.float32)

    return {
        "modalities": data,
        "label": sample["label"],
        "mask": torch.from_numpy(mask).float(),
        "id": sample["id"],
        "img_path": sample.get("img_path"),
    }


def collate_fn(batch, modalities):
  if "label" in batch[0]:
    out = {
        "modalities": {
            mod: torch.stack([item["modalities"][mod] for item in batch], dim=0)
            for mod in modalities
        },
        "label": torch.tensor([item["label"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "id": [item["id"] for item in batch],
    }
    if batch[0].get("img_path"):
      out["img_path"] = [item["img_path"] for item in batch]
    return out
  return {
      "modalities": {
          mod: torch.stack([item["modalities"][mod] for item in batch], dim=0)
          for mod in modalities
      }
  }
