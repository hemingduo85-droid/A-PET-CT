import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional, Dict

__all__ = ['ESC_DRKD_Multimodal', 'MultimodalResNet', 'resnet50', 'de_resnet50']

# Pretrained model URLs
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=1, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class DeconvBottleneck(nn.Module):
    """反向Bottleneck块（用于解码器）"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1,
                 base_width=1, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)

        if stride == 2:
            self.conv2 = nn.ConvTranspose2d(width, width, kernel_size=3,
                                            stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(width, width, kernel_size=3,
                                   stride=1, padding=1, bias=False)

        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x, skip=None):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        if skip is not None:
            out += skip
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group

        # 修改初始卷积层以适配灰度图像输入
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        print("第一次卷积：",x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print("第一次池化：",x.shape)


        x1 = self.layer1(x)  # [B, 256, 64, 64]
        print("layer1",x1.shape)
        x2 = self.layer2(x1)  # [B, 512, 32, 32]
        print("layer2",x2.shape)
        x3 = self.layer3(x2)  # [B, 1024, 16, 16]
        print("layer3",x3.shape)
        x4 = self.layer4(x3)  # [B, 2048, 8, 8]
        print("layer4",x4.shape)

        return [x1, x2, x3, x4]



class DeResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, norm_layer=None):
        super(DeResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # 初始输入通道数设为编码器最后一层的输出通道数
        self.inplanes = 2048
        self.groups = 1
        self.base_width = 64

        # 跳跃连接通道调整层（不改变空间尺寸）
        self.layer1 = self._make_layer(block, 512, layers[3], stride=2)  # 2048->1024
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)  # 1024->512
        self.layer3 = self._make_layer(block, 128, layers[1], stride=2)  # 512->256

        # 确保跳跃连接调整层与解码器匹配
        self.skip_conv2 = nn.Sequential(
            conv1x1(512, 256),  # 匹配layer2输出
            norm_layer(256)
        )
        self.skip_conv3 = nn.Sequential(
            conv1x1(1024, 512),  # 匹配layer1输出
            norm_layer(512)
        )
        # 最终上采样层
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1,
                              output_padding=1, bias=False),  # 64x64 -> 128x128
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                              output_padding=1, bias=False),  # 128x128 -> 256x256
            norm_layer(32),
            nn.Conv2d(32, num_classes, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        upsample = None

        expansion = block.expansion
        if stride != 1 or self.inplanes != planes * expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * expansion,
                                   kernel_size=1, stride=stride,
                                   output_padding=stride - 1, bias=False),
                norm_layer(planes * expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, norm_layer=norm_layer))

        # 关键修正：正确更新inplanes
        self.inplanes = planes * expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, skip_features=None):
        # skip_features应为[layer2_output, layer3_output]
        if skip_features is None:
            skip_features = [None, None]

        skip2, skip3 = skip_features  # skip2:[8,512,32,32], skip3:[8,1024,16,16]

        print(f"\n[输入] x shape: {x.shape}")
        print(f"[Skip2] input shape: {skip2.shape if skip2 is not None else 'None'}")
        print(f"[Skip3] input shape: {skip3.shape if skip3 is not None else 'None'}")

        # 调整通道数（保持空间尺寸不变）
        if skip2 is not None:
            skip2 = self.skip_conv2(skip2)  # [8,512,32,32]->[8,512,32,32]
            print(f"[Skip2 adjusted] shape: {skip2.shape}")

        if skip3 is not None:
            skip3 = self.skip_conv3(skip3)  # [8,1024,16,16]->[8,1024,16,16]
            print(f"[Skip3 adjusted] shape: {skip3.shape}")

        # 第一层解码（无跳跃连接）
        print("\n[Layer1] 解码:")
        d1 = self.layer1(x)  # [8,2048,8,8]->[8,1024,16,16]
        print(f"Layer1输出: {d1.shape}")

        # 第二层解码（连接layer3特征）
        print("\n[Layer2] 解码:")
        d2 = self.layer2(d1)  # [8,1024,16,16]->[8,512,32,32]
        if skip3 is not None:
            # 需要将skip3从16x16上采样到32x32
            skip3_resized = F.interpolate(skip3, scale_factor=2, mode='bilinear')  # [8,1024,32,32]
            print(f"将skip3从{skip3.shape}上采样到{skip3_resized.shape}")

            # 通道数调整（1024->512）
            skip3_adjusted = conv1x1(1024, 512).to(skip3_resized.device)(
                skip3_resized)  # Ensure weights are on the same device
            print(f"通道调整后: {skip3_adjusted.shape}")

            d2 = d2 + skip3_adjusted
        print(f"Layer2输出: {d2.shape}")

        # 第三层解码（连接layer2特征）
        print("\n[Layer3] 解码:")
        d3 = self.layer3(d2)  # [8,512,32,32]->[8,256,64,64]
        if skip2 is not None:
            # 需要将skip2从32x32上采样到64x64
            skip2_resized = F.interpolate(skip2, scale_factor=2, mode='bilinear')  # [8,512,64,64]
            print(f"将skip2从{skip2.shape}上采样到{skip2_resized.shape}")

            # 通道数调整（512->256）
            skip2_adjusted = conv1x1(512, 256).to(skip2_resized.device)(
                skip2_resized)  # Ensure weights are on the same device
            print(f"通道调整后: {skip2_adjusted.shape}")

            d3 = d3 + skip2_adjusted
        print(f"Layer3输出: {d3.shape}")

        # 最终输出
        print("\n[Layer4] 上采样:")
        output = self.layer4(d3)  # [8,256,64,64]->[8,1,256,256]
        print(f"最终输出: {output.shape}")

        return output


class MultimodalResNet(nn.Module):
    def __init__(self, modalities: List[str], pretrained: bool = True):
        super().__init__()
        self.modalities = modalities
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3])

        if pretrained:
            # 加载预训练权重，跳过第一层（因为输入通道不同）
            state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('conv1')}
            self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x_dict: Dict[str, torch.Tensor], available_mods: List[str]):
        # 简单平均融合多模态输入
        x = torch.stack([x_dict[mod] for mod in available_mods], dim=0).mean(dim=0)
        features = self.backbone(x)
        return features


class ESC_DRKD_Multimodal(nn.Module):
    def __init__(self, modalities: List[str]):
        super().__init__()
        self.modalities = modalities
        self.teacher = MultimodalResNet(modalities, pretrained=True)
        self.student = de_resnet50()

        # 冻结教师网络
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, x_dict, available_mods, mode='train'):
        # 提取教师特征
        t_features = self.teacher(x_dict, available_mods)  # [x1, x2, x3, x4]

        if mode == 'train':
            # 学生网络重建
            reconstruction = self.student(t_features[-1], t_features[:-1])  # 使用x4作为输入，x1-x3作为skip

            # 计算各层特征对齐损失
            with torch.no_grad():
                # 教师网络重建（用于特征对齐）
                t_recon = self.student(t_features[-1], t_features[:-1])
                t_recon_features = self.teacher.backbone(t_recon)

            return {
                'reconstruction': reconstruction,
                't_features': t_features,
                't_recon_features': t_recon_features
            }
        else:
            # 推理时直接使用学生网络重建
            return self.student(t_features[-1], t_features[:-1])

    def compute_loss(self, outputs, targets):
        # 重建损失
        rec_loss = self.l1_loss(outputs['reconstruction'], targets)

        # 特征对齐损失
        t_features = outputs['t_features']
        t_recon_features = outputs['t_recon_features']

        # 对齐教师原始特征和教师重建特征
        align_loss = 0
        for t_feat, t_recon_feat in zip(t_features, t_recon_features):
            align_loss += self.cosine_loss(
                t_feat.flatten(1),
                t_recon_feat.flatten(1),
                torch.ones(t_feat.size(0)).to(t_feat.device)
            )

        total_loss = rec_loss + 0.1 * align_loss
        return total_loss


def resnet50(modalities: List[str], pretrained: bool = True) -> MultimodalResNet:
    """构造多模态ResNet-50"""
    return MultimodalResNet(modalities, pretrained)


def de_resnet50(norm_layer=None):
    model = DeResNet(
        DeconvBottleneck,
        [3, 4, 6, 3],  # layers配置
        num_classes=1,   # 这里只传num_classes
        norm_layer=norm_layer
    )
    return model