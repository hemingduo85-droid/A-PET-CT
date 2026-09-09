from torchvision import transforms
from PIL import Image
import os
import torch
import glob

def gray_to_rgb_tensor(img_l, transform):
    """灰度图复制为 3 通道（R=G=B），以适配 ImageNet 预训练的 3 通道 ResNet。"""
    img_rgb = img_l.convert('RGB')
    return transform(img_rgb)


def gray_to_tensor(img_l, size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(size),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])(img_l)


def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

# 新增：通用取图函数，支持多扩展名
def _list_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, modality='t1', modalities=None,
                 input_mode='rgb', image_size=256):
        # 兼容：若传入 modalities 列表则使用之，否则退回单模态
        if modalities is None:
            modalities = [modality]
        self.modalities = modalities
        self.input_mode = input_mode
        self.image_size = image_size
        self.phase = phase
        # 根路径组织保持不变
        if phase == 'train':
            # 训练根: root/<mod> 但我们只用第一个模态列目录做基准
            self.base_train_root = root
        else:
            self.base_test_root = root
        self.transform = transform
        self.gt_transform = gt_transform
        self.gray_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        img_tot_paths = []   # 若多模态: 每个元素是 [mod1_path, mod2_path, ...]
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        # ---------- Test ----------
        if self.phase != 'train':
            # Normal: root/normal/<patient>/<modality>/*.png
            base_normal_dir = os.path.join(self.base_test_root, 'normal')
            if os.path.exists(base_normal_dir):
                for patient_dir in os.listdir(base_normal_dir):
                    patient_path = os.path.join(base_normal_dir, patient_dir)
                    if not os.path.isdir(patient_path):
                        continue
                    # 获取第一个模态的图片作为基准
                    base_mod_dir = os.path.join(patient_path, self.modalities[0])
                    if not os.path.exists(base_mod_dir):
                        continue
                    base_imgs = _list_images(base_mod_dir)
                    if not base_imgs:
                        continue
                    for p in base_imgs:
                        fname = os.path.basename(p)
                        per_modal_paths = []
                        miss = False
                        for m in self.modalities:
                            mp = os.path.join(patient_path, m, fname)
                            if not os.path.exists(mp):
                                miss = True
                                break
                            per_modal_paths.append(mp)
                        if miss:
                            continue
                        img_tot_paths.append(per_modal_paths)
                        gt_tot_paths.append(0)
                        tot_labels.append(0)
                        tot_types.append('normal')

            # Anomaly: root/abnormal/<patient>/<modality>/*.png, mask在label文件夹
            base_anomaly_dir = os.path.join(self.base_test_root, 'abnormal')
            if os.path.exists(base_anomaly_dir):
                for patient_dir in os.listdir(base_anomaly_dir):
                    patient_path = os.path.join(base_anomaly_dir, patient_dir)
                    if not os.path.isdir(patient_path):
                        continue
                    # 获取第一个模态的图片作为基准
                    base_mod_dir = os.path.join(patient_path, self.modalities[0])
                    if not os.path.exists(base_mod_dir):
                        continue
                    base_imgs = _list_images(base_mod_dir)
                    if not base_imgs:
                        continue
                    # 获取mask路径
                    label_dir = os.path.join(patient_path, 'label')
                    label_imgs = _list_images(label_dir) if os.path.exists(label_dir) else []
                    
                    for p in base_imgs:
                        fname = os.path.basename(p)
                        per_modal_paths = []
                        miss = False
                        for m in self.modalities:
                            mp = os.path.join(patient_path, m, fname)
                            if not os.path.exists(mp):
                                miss = True
                                break
                            per_modal_paths.append(mp)
                        if miss:
                            continue
                        # 查找对应的mask
                        mask_path = 0
                        if label_imgs:
                            for lp in label_imgs:
                                if os.path.basename(lp) == fname:
                                    mask_path = lp
                                    break
                        img_tot_paths.append(per_modal_paths)
                        gt_tot_paths.append(mask_path)
                        tot_labels.append(1)
                        tot_types.append('anomaly')
        # ---------- Train ----------
        else:
            # Train: root/normal/<patient>/<modality>/*.png
            base_train_dir = os.path.join(self.base_train_root, 'normal')
            if not os.path.exists(base_train_dir):
                raise ValueError(f"Train directory not found: {base_train_dir}")
            for patient_dir in os.listdir(base_train_dir):
                patient_path = os.path.join(base_train_dir, patient_dir)
                if not os.path.isdir(patient_path):
                    continue
                # 获取第一个模态的图片作为基准
                base_mod_dir = os.path.join(patient_path, self.modalities[0])
                if not os.path.exists(base_mod_dir):
                    continue
                base_imgs = _list_images(base_mod_dir)
                if not base_imgs:
                    continue
                for p in base_imgs:
                    fname = os.path.basename(p)
                    per_modal_paths = []
                    miss = False
                    for m in self.modalities:
                        mp = os.path.join(patient_path, m, fname)
                        if not os.path.exists(mp):
                            miss = True
                            break
                        per_modal_paths.append(mp)
                    if miss:
                        continue
                    img_tot_paths.append(per_modal_paths)
                    gt_tot_paths.append(0)
                    tot_labels.append(0)
                    tot_types.append('train')

        assert len(img_tot_paths) == len(gt_tot_paths), "Mismatch between images and ground truth!"
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_paths, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        path_by_modality = dict(zip(self.modalities, img_paths))
        if self.input_mode == 'ct_pet_pet':
            ct_path = path_by_modality.get('ct')
            pet_path = path_by_modality.get('pet')
            if ct_path is None or pet_path is None:
                raise ValueError("input_mode='ct_pet_pet' requires both ct and pet modalities")
            ct = self.gray_transform(Image.open(ct_path).convert('L'))
            pet = self.gray_transform(Image.open(pet_path).convert('L'))
            img = torch.cat([ct, pet, pet], dim=0)
            sample_path = ct_path
        elif self.input_mode == 'concat':
            imgs = []
            for p in img_paths:
                im = Image.open(p).convert('L')
                imgs.append(gray_to_rgb_tensor(im, self.transform))
            img = torch.cat(imgs, dim=0)
            sample_path = img_paths[0]
        else:
            im = Image.open(img_paths[0]).convert('L')
            img = gray_to_rgb_tensor(im, self.transform)
            sample_path = img_paths[0]
        if gt == 0:
            gt_tensor = torch.zeros([1, img.size(-2), img.size(-1)])
        else:
            g = Image.open(gt)
            gt_tensor = self.gt_transform(g)
        assert img.size()[-2:] == gt_tensor.size()[-2:], "image.size != gt.size !!!"
        return img, gt_tensor, label, sample_path
