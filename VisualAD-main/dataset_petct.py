import torch.utils.data as data
from PIL import Image
import numpy as np
import os


class PETCTDataset(data.Dataset):
    """
    Dataset for PET-CT medical image anomaly detection.

    Supports three input modalities:
      - 'pet':   pseudo-RGB = [PET, PET, PET]
      - 'ct':    pseudo-RGB = [CT, CT, CT]
      - 'petct': pseudo-RGB = [CT, PET, PET]

    Folder layout expected:
      train_root/  (e.g. .../train/normal)
        <patient_id>/
          pet/  0000.png, 0001.png, ...
          ct/   0000.png, 0001.png, ...

      test_root/  (e.g. .../test)
        normal/
          <patient_id>/
            pet/ ...
            ct/  ...
        abnormal/
          <patient_id>/
            pet/   ...
            ct/    ...
            label/ ...   (binary masks, 0/255)
    """

    def __init__(self, root, transform, target_transform,
                 modality='petct', split='train'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.modality = modality
        self.split = split
        self.samples = []

        if split == 'train':
            self._scan_train(root)
        else:
            self._scan_test(root)

        self.obj_list = ['petct']
        self.class_name_map_class_id = {'petct': 0}

    def _scan_train(self, root):
        for patient in sorted(os.listdir(root)):
            patient_dir = os.path.join(root, patient)
            if not os.path.isdir(patient_dir):
                continue
            pet_dir = os.path.join(patient_dir, 'pet')
            ct_dir = os.path.join(patient_dir, 'ct')
            if not os.path.isdir(pet_dir) or not os.path.isdir(ct_dir):
                continue
            for fname in sorted(os.listdir(pet_dir)):
                if not fname.endswith('.png'):
                    continue
                ct_path = os.path.join(ct_dir, fname)
                if not os.path.exists(ct_path):
                    continue
                self.samples.append({
                    'pet_path': os.path.join(pet_dir, fname),
                    'ct_path': ct_path,
                    'label_path': None,
                    'anomaly': 0,
                    'patient': patient,
                })

    def _scan_test(self, root):
        for category in ['normal', 'abnormal']:
            cat_dir = os.path.join(root, category)
            if not os.path.isdir(cat_dir):
                continue
            is_anomaly = (category == 'abnormal')
            for patient in sorted(os.listdir(cat_dir)):
                patient_dir = os.path.join(cat_dir, patient)
                if not os.path.isdir(patient_dir):
                    continue
                pet_dir = os.path.join(patient_dir, 'pet')
                ct_dir = os.path.join(patient_dir, 'ct')
                label_dir = os.path.join(patient_dir, 'label')
                if not os.path.isdir(pet_dir) or not os.path.isdir(ct_dir):
                    continue
                for fname in sorted(os.listdir(pet_dir)):
                    if not fname.endswith('.png'):
                        continue
                    ct_path = os.path.join(ct_dir, fname)
                    if not os.path.exists(ct_path):
                        continue
                    label_path = None
                    if is_anomaly:
                        lp = os.path.join(label_dir, fname)
                        if os.path.exists(lp):
                            label_path = lp
                    self.samples.append({
                        'pet_path': os.path.join(pet_dir, fname),
                        'ct_path': ct_path,
                        'label_path': label_path,
                        'anomaly': 1 if is_anomaly else 0,
                        'patient': patient,
                    })

    def _build_image(self, pet_img, ct_img):
        if self.modality == 'pet':
            return Image.merge('RGB', [pet_img, pet_img, pet_img])
        elif self.modality == 'ct':
            return Image.merge('RGB', [ct_img, ct_img, ct_img])
        else:  # petct
            return Image.merge('RGB', [ct_img, pet_img, pet_img])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]

        pet = Image.open(s['pet_path']).convert('L')
        ct = Image.open(s['ct_path']).convert('L')

        if pet.size != ct.size:
            ct = ct.resize(pet.size, Image.BILINEAR)

        img = self._build_image(pet, ct)

        if s['label_path'] is not None:
            mask_arr = np.array(Image.open(s['label_path']).convert('L')) > 0
            img_mask = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode='L')
        else:
            img_mask = Image.fromarray(
                np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode='L'
            )

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)

        return {
            'img': img,
            'img_mask': img_mask,
            'anomaly': s['anomaly'],
            'cls_name': 'petct',
            'img_path': s['pet_path'],
            'cls_id': 0,
        }
