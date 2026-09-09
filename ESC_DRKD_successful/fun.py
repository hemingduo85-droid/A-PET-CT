import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import torch.nn as nn


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def denormalization1(x):
    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, feature1, feature2):
        cos = nn.functional.cosine_similarity(feature1, feature2, dim=1)
        ano_map = torch.ones_like(cos) - cos
        loss = (ano_map.view(ano_map.shape[0], -1).mean(-1)).mean()
        return loss

class loss_fucntion(nn.Module):
    def __init__(self):
        super(loss_fucntion, self).__init__()

    def forward(self, a, b):
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(a)):
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))

        loss = loss / (len(a))
        return loss

def cut(img, t, b):
    # h, w, c = img.shape
    x = np.random.randint(0, img.shape[1] - t)
    y = np.random.randint(0, img.shape[0] - b)
    if (x - t) % 2 == 1:
        t -= 1
    if (y - b) % 2 == 1:
        b -= 1

    roi = img[y:y + b, x:x + t]
    return roi

def paste_patch(img, patch):
    imgh, imgw, imgc = img.shape
    patchh, patchw, patchc = patch.shape

    patch_h_position = random.randrange(1, round(imgh) - round(patchh) - 1)
    patch_w_position = random.randrange(1, round(imgw) - round(patchw) - 1)
    pasteimg = np.copy(img)
    pasteimg[patch_h_position:patch_h_position + patchh, patch_w_position:patch_w_position + patchw, :] = patch + 0.2 * img[patch_h_position:patch_h_position + patchh,
    patch_w_position:patch_w_position + patchw, :]

    return pasteimg

class Normalize(object):
    """
    Only normalize images
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0,1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image


def noise_generate(image_np, t):
    transform = transforms.Compose([
        Normalize(),
        ToTensor(),
    ])
    rotated_list = []
    for i in range(image_np.size(0)):
        np_img = image_np[i]
        patch_img = cut(np_img, t, t)
        patch_img = paste_patch(np_img, patch_img)
        img_noise = transform(patch_img)
        img_noise = torch.unsqueeze(img_noise, dim=0)
        rotated_list.append(img_noise)
    img_noise = torch.cat(rotated_list, dim=0)
    return img_noise


def get_anomap(output, Dn, data, device):
    """
    计算异常分数图
    参数:
        output: 教师网络特征列表 [layer1, layer2, layer3], 形状应为:
               [B,C,64,64], [B,C,32,32], [B,C,16,16]
        Dn: 学生网络特征列表 [output1, output2, output3], 形状应与output对应
        data: 输入数据 [B,C,H,W], 用于确定最终输出尺寸
        device: 计算设备
    """
    n, c, h, w = data.shape

    # 预定义各层期望的空间尺寸
    target_sizes = [64, 32, 16]

    anomaly_maps = []
    for i in range(3):
        t_feat = output[i]  # 教师特征
        s_feat = Dn[i]  # 学生特征

        # 确保学生特征与教师特征尺寸匹配
        if s_feat.shape[2:] != t_feat.shape[2:]:
            s_feat = F.interpolate(s_feat,
                                   size=t_feat.shape[2:],
                                   mode='bilinear',
                                   align_corners=True)

        # 计算余弦相似度，确保输出是 [B, 1, H, W] 维度
        cos_sim = F.cosine_similarity(t_feat, s_feat, dim=1, keepdim=True)  # [B, 1, H, W]
        anomaly_map = 1 - cos_sim  # 得到异常分数图

        # 上采样到输入尺寸
        anomaly_map = F.interpolate(anomaly_map,
                                    size=(h, w),
                                    mode='bilinear',
                                    align_corners=True)
        anomaly_maps.append(anomaly_map)

    # 融合多尺度异常图
    final_anomaly_map = torch.mean(torch.cat(anomaly_maps, dim=1), dim=1, keepdim=True)

    return final_anomaly_map




