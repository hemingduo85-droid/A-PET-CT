from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _case_insensitive_child(parent, name):
    direct = os.path.join(parent, name)
    if os.path.exists(direct):
        return direct
    if not os.path.isdir(parent):
        return direct
    wanted = name.lower()
    for child in os.listdir(parent):
        child_path = os.path.join(parent, child)
        if os.path.isdir(child_path) and child.lower() == wanted:
            return child_path
    return direct


def _image_files(directory):
    if not os.path.isdir(directory):
        return []
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        files.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))
    return sorted(set(files))


def _patient_dirs(label_root):
    if not os.path.isdir(label_root):
        return []
    return sorted(
        p for p in glob.glob(os.path.join(label_root, '*'))
        if os.path.isdir(p)
    )


def _sample_dirs(label_root, modalities):
    patients = _patient_dirs(label_root)
    if any(os.path.basename(p).lower() in {m.lower() for m in modalities} for p in patients):
        if all(os.path.isdir(_case_insensitive_child(label_root, m)) for m in modalities):
            return [label_root]
    return patients


def _choose_label_dir(test_root, candidates):
    choices = [os.path.join(test_root, candidate) for candidate in candidates]
    existing = [path for path in choices if os.path.isdir(path)]
    with_patients = [path for path in existing if _patient_dirs(path)]
    if with_patients:
        return with_patients[0]
    if existing:
        return existing[0]
    return choices[0]


def _matching_slice_path(modality_dir, slice_name):
    direct = os.path.join(modality_dir, slice_name)
    if os.path.exists(direct):
        return direct
    stem, _ = os.path.splitext(slice_name)
    for candidate in _image_files(modality_dir):
        if os.path.splitext(os.path.basename(candidate))[0] == stem:
            return candidate
    return direct


def _dataset_empty_error(root, phase, modalities):
    test_root = os.path.join(root, phase)
    details = []
    if os.path.isdir(test_root):
        for child in sorted(os.listdir(test_root)):
            child_path = os.path.join(test_root, child)
            if os.path.isdir(child_path):
                details.append(f'{child}: {len(_patient_dirs(child_path))} child dirs')
    else:
        details.append(f'missing phase directory: {test_root}')
    return (
        f'数据集为空: root={root}, phase={phase}, modalities={modalities}. '
        f'期望结构: {phase}/normal|NORMAL/{{patient}}/{{ct,pet}}/*.png 和 '
        f'{phase}/abnormal|ABNORMAL/{{patient}}/{{ct,pet}}/*.png. '
        f'当前目录概况: {details}'
    )


def get_data_transforms(size, isize, num_modalities=1, replicate_channels=1, input_mode='single'):
    # 基础预处理：Resize -> Grayscale(1) -> ToTensor -> CenterCrop
    # 归一化挪到 MRIDataset 内在多通道拼接后统一做
    if input_mode == 'pseudo_rgb' and num_modalities > 1:
        num_channels = 3
    elif num_modalities > 1:
        num_channels = num_modalities
    else:
        num_channels = replicate_channels
    mean_train = [0.485] * num_channels
    std_train = [0.229] * num_channels
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(isize),
        transforms.ToTensor(),
    ])
    # gt 仍保持单通道（若后续需要与 mask 结合可扩展）
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms, mean_train, std_train


class MRIDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase, modalities, mean=None, std=None,
                 replicate_channels=1, input_mode='single'):
        # modalities: 可以是 str 或 list[str]，单模态时例如 ['pet'] 或 ['ct']
        if isinstance(modalities, str):
            modalities = [modalities]
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.input_mode = input_mode
        if self.input_mode == 'pseudo_rgb' and self.num_modalities > 1:
            lowered = {m.lower() for m in self.modalities}
            if not {'ct', 'pet'}.issubset(lowered):
                raise ValueError("pseudo_rgb 输入模式需要同时包含 ct 和 pet")
            self.replicate_channels = 1
            self.num_output_channels = 3
        else:
            self.replicate_channels = replicate_channels if self.num_modalities == 1 else 1
            self.num_output_channels = (
                self.num_modalities if self.num_modalities > 1 else self.replicate_channels
            )
        self.root = root
        self.phase = phase
        self.transform = transform
        self.mean = mean if mean is not None else [0.485] * self.num_output_channels
        self.std = std if std is not None else [0.229] * self.num_output_channels
        # 载入对齐后的多模态路径
        self.img_paths_per_modality, self.labels, self.types, self.mask_paths = self.load_dataset()

    def load_dataset(self):
        # 返回:
        #   img_paths_per_modality: List[List[path_mod0, path_mod1, ...]]
        img_tot_paths = []
        tot_labels = []
        tot_types = []
        mask_tot_paths = []

        if self.phase == 'train':
            # train/normal/{patient}/{modality}/*.png
            train_dir = _choose_label_dir(os.path.join(self.root, 'train'), ['normal', 'NORMAL'])
            patient_dirs = _sample_dirs(train_dir, self.modalities)
            for patient_dir in patient_dirs:
                patient_name = os.path.basename(patient_dir)
                # 获取第一个模态的所有切片
                first_modality_dir = _case_insensitive_child(patient_dir, self.modalities[0])
                if not os.path.exists(first_modality_dir):
                    raise FileNotFoundError(f'未找到模态目录: {first_modality_dir}')
                first_modality_files = _image_files(first_modality_dir)
                if len(first_modality_files) == 0:
                    raise FileNotFoundError(f'模态目录中没有图片: {first_modality_dir}')
                
                # 为每个切片创建一个样本
                for slice_file in first_modality_files:
                    slice_name = os.path.basename(slice_file)
                    sample_paths = []
                    # 对每个模态，找到对应的切片
                    for m in self.modalities:
                        modality_dir = _case_insensitive_child(patient_dir, m)
                        if not os.path.exists(modality_dir):
                            raise FileNotFoundError(f'未找到模态目录: {modality_dir}')
                        # 找到对应切片名的文件
                        slice_path = _matching_slice_path(modality_dir, slice_name)
                        if not os.path.exists(slice_path):
                            raise FileNotFoundError(f'未找到匹配切片: {slice_path}')
                        sample_paths.append(slice_path)
                    img_tot_paths.append(sample_paths)
                    tot_labels.append(0)
                    tot_types.append('normal')
                    mask_tot_paths.append(None)
        else:
            # test/{normal|abnormal}/{patient}/{modality}/*.png
            for label_name in ['normal', 'abnormal']:
                test_subdir = _choose_label_dir(
                    os.path.join(self.root, 'test'),
                    [label_name, label_name.upper()],
                )
                patient_dirs = _sample_dirs(test_subdir, self.modalities)
                for patient_dir in patient_dirs:
                    patient_name = os.path.basename(patient_dir)
                    # 获取第一个模态的所有切片
                    first_modality_dir = _case_insensitive_child(patient_dir, self.modalities[0])
                    if not os.path.exists(first_modality_dir):
                        raise FileNotFoundError(f'未找到模态目录: {first_modality_dir}')
                    first_modality_files = _image_files(first_modality_dir)
                    if len(first_modality_files) == 0:
                        raise FileNotFoundError(f'模态目录中没有图片: {first_modality_dir}')
                    
                    # 为每个切片创建一个样本
                    for slice_file in first_modality_files:
                        slice_name = os.path.basename(slice_file)
                        sample_paths = []
                        # 对每个模态，找到对应的切片
                        for m in self.modalities:
                            modality_dir = _case_insensitive_child(patient_dir, m)
                            if not os.path.exists(modality_dir):
                                raise FileNotFoundError(f'未找到模态目录: {modality_dir}')
                            # 找到对应切片名的文件
                            slice_path = _matching_slice_path(modality_dir, slice_name)
                            if not os.path.exists(slice_path):
                                raise FileNotFoundError(f'未找到匹配切片: {slice_path}')
                            sample_paths.append(slice_path)
                        img_tot_paths.append(sample_paths)
                        tot_labels.append(0 if label_name == 'normal' else 1)
                        tot_types.append(label_name)
                        # abnormal 样本需要 mask
                        if label_name == 'abnormal':
                            mask_dir = _case_insensitive_child(patient_dir, 'label')
                            if not os.path.exists(mask_dir):
                                mask_dir = _case_insensitive_child(patient_dir, 'mask')
                            if os.path.exists(mask_dir):
                                mask_path = _matching_slice_path(mask_dir, slice_name)
                                if os.path.exists(mask_path):
                                    mask_tot_paths.append(mask_path)
                                else:
                                    mask_tot_paths.append(None)
                            else:
                                mask_tot_paths.append(None)
                        else:
                            mask_tot_paths.append(None)

        if not img_tot_paths:
            raise ValueError(_dataset_empty_error(self.root, self.phase, self.modalities))

        return img_tot_paths, tot_labels, tot_types, mask_tot_paths

    def __len__(self):
        return len(self.img_paths_per_modality)

    def __getitem__(self, idx):
        paths_per_mod = self.img_paths_per_modality[idx]
        label = self.labels[idx]
        img_type = self.types[idx]
        # 逐模态读取并 transform -> [1,H,W]
        modality_tensors = {}
        for p in paths_per_mod:
            modality = os.path.basename(os.path.dirname(p)).lower()
            img = Image.open(p).convert('L')
            t = self.transform(img)  # [1,H,W]
            modality_tensors[modality] = t

        if self.input_mode == 'pseudo_rgb' and self.num_modalities > 1:
            img_tensor = torch.cat(
                [modality_tensors['ct'], modality_tensors['pet'], modality_tensors['pet']],
                dim=0,
            )
        else:
            img_tensor = torch.cat(
                [modality_tensors[m.lower()] for m in self.modalities],
                dim=0,
            )  # [C,H,W]

        # 单模态时可将灰度图复制为多通道（如 3 通道以匹配 ImageNet 预训练）
        if self.num_modalities == 1 and self.replicate_channels > 1:
            img_tensor = img_tensor.repeat(self.replicate_channels, 1, 1)
        # 通道归一化
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # mask 仅在测试阶段有效，否则返回占位零张量
        mask_tensor = None
        if self.phase == 'test':
            mask_path = self.mask_paths[idx]
            if mask_path is not None and os.path.exists(mask_path):
                m = Image.open(mask_path).convert('L')
                mask_tensor = self.transform(m)  # [1,H,W]，与图像同尺寸
                mask_tensor = (mask_tensor >= 0.5).float()  # 二值化
            else:
                mask_tensor = torch.zeros(1, img_tensor.shape[-2], img_tensor.shape[-1])
        else:
            mask_tensor = torch.zeros(1, img_tensor.shape[-2], img_tensor.shape[-1])

        return img_tensor, mask_tensor, label, paths_per_mod[0]
