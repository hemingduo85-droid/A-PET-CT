import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional, Dict

__all__ = ['ESC_DRKD_Multimodal', 'MultimodalResNet', 'wide_resnet50_2', 'de_wide_resnet50_2']

# Pretrained model URLs
model_urls = {
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CentralMaskedConv2d(nn.Conv2d):
    """中心掩码卷积（保持局部上下文）"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class ProjLayer(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=1):
        super().__init__()
        # 保持通道数不变的投影
        self.mc1 = CentralMaskedConv2d(in_c, out_c, kernel_size=3, padding=1)
        self.mc2 = CentralMaskedConv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU()
        )
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False
        ) if scale_factor != 1 else nn.Identity()

    def forward(self, x):
        x = self.mc1(x)
        x = self.mc2(x)
        x = self.conv(x)
        return self.upsample(x)


class MultimodalProjection(nn.Module):
    def __init__(self):
        super().__init__()
        # 保持与教师网络相同的通道数
        self.proj_layers = nn.ModuleDict({
            'layer2': ProjLayer(512, 512, scale_factor=1),  # 保持512通道
            'layer3': ProjLayer(1024, 1024, scale_factor=1)  # 保持1024通道
        })

    def forward(self, feat_dict, available_mods):
        projected = {'layer2': [], 'layer3': []}

        for mod in available_mods:
            if mod not in feat_dict:
                continue

            mod_feats = feat_dict[mod]
            layer2 = mod_feats.get('layer2', torch.zeros_like(mod_feats['layer3']))
            layer3 = mod_feats.get('layer3', torch.zeros_like(mod_feats['layer2']))

            projected['layer2'].append(self.proj_layers['layer2'](layer2))
            projected['layer3'].append(self.proj_layers['layer3'](layer3))

        # 平均所有模态的投影特征
        return {
            'layer2': torch.mean(torch.stack(projected['layer2']), dim=0) if projected['layer2'] else None,
            'layer3': torch.mean(torch.stack(projected['layer3']), dim=0) if projected['layer3'] else None
        }

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
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
                 base_width=64, dilation=1, norm_layer=None):
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
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
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

    def forward(self, x, skip1=None, skip2=None):
        print("\n=== 学生网络处理 ===")
        print(f"输入x: {x.shape}")

        d1 = self.layer1(x)
        print(f"d1 (layer1输出): {d1.shape}")

        d2 = self.layer2(d1)
        print(f"d2 (layer2输出): {d2.shape}")

        if skip1 is not None:
            print(f"skip1输入: {skip1.shape}")
            d2 = self.skip_adjust['layer2'](torch.cat([d2, skip1], dim=1))
            print(f"拼接调整后d2: {d2.shape}")

        d3 = self.layer3(d2)
        print(f"d3 (layer3输出): {d3.shape}")

        if skip2 is not None:
            print(f"skip2输入: {skip2.shape}")
            d3 = self.skip_adjust['layer3'](torch.cat([d3, skip2], dim=1))
            print(f"拼接调整后d3: {d3.shape}")

        recon = self.layer4(d3)
        print(f"最终输出: {recon.shape}")

        return [d3, d2, d1]


class DeResNet(nn.Module):
    def __init__(self, block, layers, width_per_group=64, norm_layer=None, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 512 * block.expansion
        self.groups = 1
        self.base_width = width_per_group

        # 通道调整层
        self.skip_adjust = nn.ModuleDict({
            'layer2': nn.Sequential(
                nn.Conv2d(512 + 256, 512, kernel_size=1),  # 512 (学生) + 256 (投影)
                nn.BatchNorm2d(512)
            ),
            'layer3': nn.Sequential(
                nn.Conv2d(256 + 128, 256, kernel_size=1),  # 256 (学生) + 128 (投影)
                nn.BatchNorm2d(256)
            )
        })

        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64 * block.expansion, 32, kernel_size=3,
                               stride=2, padding=1, output_padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
                            self.base_width, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x, skip1=None, skip2=None):
        d1 = self.layer1(x)  # [B,1024,16,16]
        d2 = self.layer2(d1)  # [B,512,32,32]

        if skip1 is not None:
            d2 = self.skip_adjust['layer2'](torch.cat([d2, skip1], dim=1))

        d3 = self.layer3(d2)  # [B,256,64,64]

        if skip2 is not None:
            d3 = self.skip_adjust['layer3'](torch.cat([d3, skip2], dim=1))

        recon = self.layer4(d3)  # [B,3,256,256]
        return [d3, d2, d1]


class MultimodalAttentionFusion(nn.Module):
    """多模态特征融合层（简化版）"""

    def __init__(self, modalities: List[str], base_width: int = 64):
        super().__init__()
        self.modalities = modalities
        self.mod_proj = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Conv2d(1, base_width, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ) for mod in modalities
        })
        self.weights = nn.Parameter(torch.ones(len(modalities)))
        self.mask_token = nn.Parameter(torch.randn(1, base_width, 1, 1))

    def forward(self, x_dict: Dict[str, torch.Tensor], available_mods: List[str]):
        proj_feats = []
        for i, mod in enumerate(self.modalities):
            if mod in available_mods:
                x_mod = x_dict[mod]
                if x_mod.dim() == 3:
                    x_mod = x_mod.unsqueeze(1)
                proj_feat = self.mod_proj[mod](x_mod)
                proj_feats.append(proj_feat * torch.sigmoid(self.weights[i]))
            else:
                B = next(iter(x_dict.values())).size(0)
                proj_feats.append(self.mask_token.expand(B, -1, 56, 56) * 0)

        return sum(proj_feats) / (len(available_mods) + 1e-6)


class MultimodalResNet(nn.Module):
    def __init__(self, modalities: List[str], pretrained: bool = True):
        super().__init__()
        self.fusion = MultimodalAttentionFusion(modalities)
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=64 * 2)

        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['wide_resnet50_2'], progress=True)
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith('conv1') and not k.startswith('bn1')}
            self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x_dict: Dict[str, torch.Tensor], available_mods: List[str]):
        for mod, tensor in x_dict.items():
            if tensor.dim() == 3:
                x_dict[mod] = tensor.unsqueeze(1)
            assert x_dict[mod].dim() == 4, f"输入应为4D，但得到{x_dict[mod].dim()}D"

        x = self.fusion(x_dict, available_mods)

        # 提取所有 4 层特征
        d1 = self.backbone.layer1(x)
        d2 = self.backbone.layer2(d1)
        d3 = self.backbone.layer3(d2)
        d4 = self.backbone.layer4(d3)  # 新增 layer4 输出

        return [d1, d2, d3, d4]  # 返回 4 层特征

class ESC_DRKD_Multimodal(nn.Module):
    def __init__(self, modalities: List[str]):
        super().__init__()
        self.modalities = modalities
        self.teacher = MultimodalResNet(modalities, pretrained=True)
        self.student = de_wide_resnet50_2()
        self.projection = MultimodalProjection()

        # 冻结教师网络
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.rec_loss = nn.L1Loss()

    def forward(self, x_dict, available_mods, mode='train'):
        t_features = self._extract_teacher_features(x_dict, available_mods)

        if mode == 'train':
            # 修改伪特征结构，使其按模态组织
            pseudo_feats = {
                mod: {
                    'layer2': t_features[1] + 0.1 * torch.randn_like(t_features[1]),
                    'layer3': t_features[2] + 0.1 * torch.randn_like(t_features[2])
                } for mod in available_mods
            }

            p_features = self.projection(pseudo_feats, available_mods)
            s_features = self.student(
                t_features[3],
                skip1=p_features['layer2'],
                skip2=p_features['layer3']
            )
            return {
                't_features': t_features,
                's_features': s_features,
                'p_features': p_features
            }
        else:
            return self.student(t_features[3])
    def _extract_teacher_features(self, x_dict, available_mods):
        processed_dict = {}
        for mod in self.modalities:
            if mod in x_dict:
                x_mod = x_dict[mod]
                if x_mod.dim() == 3:
                    x_mod = x_mod.unsqueeze(1)
                processed_dict[mod] = x_mod
            else:
                B = next(iter(x_dict.values())).size(0)
                processed_dict[mod] = torch.zeros(B, 1, 256, 256).to(x_dict[list(x_dict.keys())[0]].device)
        return self.teacher(processed_dict, available_mods)

    def _adjust_size(self, x, target_size):
        """调整特征图尺寸和通道数"""
        if x.shape[2:] != target_size[2:]:
            x = F.interpolate(x, size=target_size[2:], mode='bilinear', align_corners=False)
        if x.shape[1] != target_size[1]:
            x = nn.Conv2d(x.shape[1], target_size[1], kernel_size=1).to(x.device)(x)
        return x

    def compute_loss(self, outputs, targets):
        losses = []
        s_features = outputs['s_features']  # [d3, d2, d1]
        t_features = outputs['t_features']  # [layer1, layer2, layer3, layer4]
        p_features = outputs['p_features']  # {'layer2': ..., 'layer3': ...}

        # 对齐学生特征和教师特征
        for i in range(len(s_features)):
            s_feat = s_features[i]
            t_feat = t_features[i + 1]  # layer2对应d2, layer3对应d3
            t_feat = self._adjust_size(t_feat, s_feat.shape)
            losses.append(self.cosine_loss(
                s_feat.flatten(1),
                t_feat.flatten(1),
                torch.ones(s_feat.size(0)).to(targets.device)
            ))

        # 对齐投影特征和教师特征
        rec_loss = 0
        for i, layer in enumerate(['layer2', 'layer3']):
            p_feat = p_features[layer]
            t_feat = t_features[i + 1]  # layer2对应教师layer2, layer3对应教师layer3
            t_feat = self._adjust_size(t_feat, p_feat.shape)
            rec_loss += self.rec_loss(p_feat, t_feat)

        return sum(losses) + 0.5 * rec_loss


def wide_resnet50_2(modalities: List[str], pretrained: bool = True) -> MultimodalResNet:
    """构造多模态Wide-ResNet-50-2"""
    return MultimodalResNet(modalities, pretrained)


def de_wide_resnet50_2(pretrained=False, norm_layer=None, **kwargs):
    """构建Wide-ResNet-50-2解码器"""
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    model = DeResNet(DeconvBottleneck, [3, 4, 6],
                    width_per_group=64 * 2,
                    norm_layer=norm_layer,
                    **kwargs)
    return model