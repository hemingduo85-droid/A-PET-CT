# resnet.py - 完整修复版

import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional, Tuple

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'BN_layer', 'wide_resnet50_2', 'resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class BN_layer(nn.Module):
    """
    Bottleneck layer for feature adaptation.
    """
    def __init__(self, block: Type, layers: int = 4):
        super(BN_layer, self).__init__()
        # 动态适配 Bottleneck (4) 和 BasicBlock (1) 的通道数倍增
        expansion = block.expansion
        
        self.conv1x1_1 = nn.Conv2d(64 * expansion, 128, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(128 * expansion, 256, kernel_size=1, stride=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(256 * expansion, 512, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, features_list):
        """
        Args:
            features_list: list of [f1, f2, f3] from encoder
        Returns:
            list of processed features
        """
        if isinstance(features_list, list) and len(features_list) >= 3:
            f1, f2, f3 = features_list[0], features_list[1], features_list[2]
            out1 = self.relu(self.conv1x1_1(f1))
            out2 = self.relu(self.conv1x1_2(f2))
            out3 = self.relu(self.conv1x1_3(f3))
            return [out1, out2, out3]
        else:
            return features_list


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion: int = 4

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
    ) -> None:
        super(Bottleneck, self).__init__()
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

    def forward(self, x: Tensor) -> Tensor:
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


class BasicBlock(nn.Module):
    expansion: int = 1

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
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            in_channels: int = 3,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                           "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        
        # 支持多通道输入
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                      dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                      dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                      dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
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
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                   stride: int = 1, dilate: bool = False) -> nn.Sequential:
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

    def _forward_impl(self, x: Tensor) -> Union[Tensor, List[Tensor]]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        # 核心拦截逻辑：针对 ReContrast 将 layer4 置为 None 的异常检测应用场景
        # 直接返回特征列表，避免调用 NoneType
        if self.layer4 is None:
            return [f1, f2, f3]

        x = self.layer4(f3)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Union[Tensor, List[Tensor]]:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        in_channels: int = 3,
        **kwargs: Any
) -> Tuple[ResNet, BN_layer]:
    """
    Create ResNet model with bottleneck.
    Returns: (model, bottleneck)
    """
    model = ResNet(block, layers, in_channels=in_channels, **kwargs)
    
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        
        # 处理通道数不匹配
        if in_channels != 3:
            conv1_weight = state_dict['conv1.weight']
            
            if in_channels == 1:
                conv1_weight = conv1_weight.mean(dim=1, keepdim=True)
            elif in_channels == 2:
                # 复制 R 和 G 通道的平均值
                conv1_weight = conv1_weight[:, :2, :, :].mean(dim=1, keepdim=True)
                conv1_weight = conv1_weight.repeat(1, 2, 1, 1)
            elif in_channels > 3:
                conv1_weight = conv1_weight.repeat(1, (in_channels + 2) // 3, 1, 1)[:, :in_channels]
            
            state_dict['conv1.weight'] = conv1_weight
        
        model.load_state_dict(state_dict, strict=False)
    
    # 创建 bottleneck 层
    bn_layer = BN_layer(block, len(layers))
    
    return model, bn_layer


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, in_channels: int = 3, **kwargs: Any) -> Tuple[ResNet, BN_layer]:
    r"""Wide ResNet-50-2 model for multi-channel input.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        in_channels (int): Number of input channels (default: 3 for RGB)
        **kwargs: Additional arguments passed to ResNet
        
    Returns:
        Tuple[encoder, bottleneck]: The ResNet encoder and bottleneck layer
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, in_channels=in_channels, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, in_channels: int = 3, **kwargs: Any) -> Tuple[ResNet, BN_layer]:
    r"""ResNet-50 model."""
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, in_channels=in_channels, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, in_channels: int = 3, **kwargs: Any) -> Tuple[ResNet, BN_layer]:
    r"""Wide ResNet-101-2 model."""
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, in_channels=in_channels, **kwargs)