import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


class DeconvBottleneck(nn.Module):
    """反向Bottleneck块（用于解码器）"""
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            upsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # 反向卷积结构
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)

        if stride == 2:
            self.conv2 = nn.ConvTranspose2d(width, width, kernel_size=3, stride=stride,
                                            padding=1, output_padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        # 增强跳跃连接（ESC机制）
        if skip is not None:
            out += identity + skip  # 添加投影特征作为跳跃连接
        else:
            out += identity

        out = self.relu(out)
        return out


class DeResNet(nn.Module):
    """反向ResNet解码器，支持多层级特征重建"""

    def __init__(
            self,
            block: Type[DeconvBottleneck],
            layers: List[int],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 512 * block.expansion
        self.groups = groups
        self.base_width = width_per_group

        # 反向层级结构（与教师网络对称）
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64 * block.expansion, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, bias=False)  # 最终重建为3通道
        )

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DeconvBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        upsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride,
                                   output_padding=stride - 1, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, 1, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor,
                skip1: Optional[torch.Tensor] = None,
                skip2: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        参数:
            x: 教师网络最高层特征 (layer4输出)
            skip1: 投影层输出的P2特征 (对应layer2)
            skip2: 投影层输出的P3特征 (对应layer3)
        返回:
            多级重建特征 [D1, D2, D3]
        """
        # 第一层解码（无跳跃连接）
        d1 = self.layer1(x)

        # 第二层解码（加入P2跳跃连接）
        d2 = self.layer2(d1)
        if skip1 is not None:
            d2 = torch.cat([d2, skip1], dim=1)

        # 第三层解码（加入P3跳跃连接）
        d3 = self.layer3(d2)
        if skip2 is not None:
            d3 = torch.cat([d3, skip2], dim=1)

        # 最终重建
        recon = self.layer4(d3)

        return [d3, d2, d1]  # 与教师网络特征顺序对应


def de_wide_resnet50_2(pretrained: bool = False, **kwargs: Any) -> DeResNet:
    """构建Wide-ResNet-50-2对应的解码器"""
    model = DeResNet(DeconvBottleneck, [3, 4, 6],
                     width_per_group=64 * 2, **kwargs)

    # 可选：加载预训练解码器权重（如果有）
    if pretrained:
        pass  # 通常解码器不需要预训练

    return model