import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional, Dict

__all__ = ['ResNet', 'MultimodalResNet', 'wide_resnet50_2']

# Pretrained model URLs
model_urls = {
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'
}


class MultimodalAttentionFusion(nn.Module):
    """处理多模态输入（支持模态缺失）"""

    def __init__(self, modalities: List[str], base_width: int = 64):
        super().__init__()
        self.modalities = modalities
        # 各模态独立预处理层
        self.mod_proj = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Conv2d(1, base_width, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ) for mod in modalities
        })
        # 跨模态注意力
        self.attn = nn.MultiheadAttention(embed_dim=base_width, num_heads=4)
        # 缺失模态标记
        self.mask_token = nn.Parameter(torch.randn(1, base_width, 1, 1))

    def forward(self, x_dict: Dict[str, torch.Tensor], available_mods: List[str]) -> torch.Tensor:
        # 处理可用/缺失模态
        proj_feats = []
        for mod in self.modalities:
            if mod in available_mods:
                proj_feats.append(self.mod_proj[mod](x_dict[mod].unsqueeze(1)))
            else:
                B = next(iter(x_dict.values())).size(0)
                proj_feats.append(self.mask_token.expand(B, -1, 56, 56))  # 假设下采样后尺寸为56x56

        # 模态注意力融合 [B, C, H, W] -> [HW, B, C]
        feats = torch.stack(proj_feats, dim=0)  # [M, B, C, H, W]
        M, B, C, H, W = feats.shape
        feats = feats.permute(3, 4, 0, 1, 2).reshape(H * W, M, B, C)

        # 注意力计算（使用模态自身作为query）
        attn_out, _ = self.attn(feats, feats, feats)  # [HW, M, B, C]
        fused = attn_out.mean(dim=1)  # [HW, B, C]

        return fused.permute(1, 2, 0).view(B, C, H, W)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Bottleneck],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
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

        # 移除了原生的conv1/maxpool，由MultimodalAttentionFusion替代
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
            elif isinstance(m, nn.BatchNorm2d):
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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = self.layer1(x)
        features.append(x)  # Layer1输出
        x = self.layer2(x)
        features.append(x)  # Layer2输出
        x = self.layer3(x)
        features.append(x)  # Layer3输出
        x = self.layer4(x)
        features.append(x)  # Layer4输出

        return features


class MultimodalResNet(nn.Module):
    """支持多模态输入的ResNet封装"""

    def __init__(self, modalities: List[str], pretrained: bool = True):
        super().__init__()
        # 多模态融合层
        self.fusion = MultimodalAttentionFusion(modalities)

        # 主干网络
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=64 * 2)

        # 加载预训练权重
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['wide_resnet50_2'], progress=True)
            # 移除fusion层不存在的参数
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('conv1') and not k.startswith('bn1')}
            self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x_dict: Dict[str, torch.Tensor], available_mods: List[str]) -> List[torch.Tensor]:
        # 多模态融合
        x = self.fusion(x_dict, available_mods)
        # 特征提取
        return self.backbone(x)


def wide_resnet50_2(modalities: List[str], pretrained: bool = True) -> MultimodalResNet:
    """构造多模态版本的Wide-ResNet-50-2"""
    return MultimodalResNet(modalities, pretrained)