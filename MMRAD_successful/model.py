import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class MultiModalAnomalyDetector(nn.Module):
    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 8,
            num_modalities: int = 4,  # 核心控制参数
            embed_dim: int = 384,
            depth: int = 12,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.img_size = img_size

        # 动态计算中间通道数
        base_channels = 64
        self.proj_out_channels = base_channels
        fusion_in_channels = base_channels * num_modalities

        # 多模态特征提取器 (每个模态独立)
        self.modal_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, base_channels // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, padding=1),
                nn.GELU()
            ) for _ in range(num_modalities)
        ])

        # 动态融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_channels, base_channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        )

        # 编码器-解码器
        self.encoder = EnhancedViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=base_channels,  # 动态适应
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )

        self.decoder = MultiScaleDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            num_modalities=num_modalities  # 控制输出通道
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.num_modalities, \
            f"Input channels {C} != initialized modalities {self.num_modalities}"
        assert H == self.img_size and W == self.img_size, \
            f"Input size {H}x{W} != initialized size {self.img_size}x{self.img_size}"

        # 模态特征提取与融合
        modal_features = []
        for i in range(self.num_modalities):
            features = self.modal_projs[i](x[:, i:i + 1])
            modal_features.append(features)

        fused = self.fusion_conv(torch.cat(modal_features, dim=1))

        # 编码解码处理
        encoded = self.encoder(fused)
        output = self.decoder(encoded)

        return output  # (B, num_modalities, H, W)


class MultiScaleDecoder(nn.Module):
    def __init__(self, embed_dim, patch_size, img_size, num_modalities):
        super().__init__()
        self.num_modalities = num_modalities
        self.up_blocks = nn.ModuleList()

        # 动态计算上采样路径
        curr_size = img_size // patch_size
        curr_channels = embed_dim
        min_channels = max(32, num_modalities)  # 确保不小于输出通道

        while curr_size < img_size:
            next_size = min(curr_size * 2, img_size)
            out_channels = max(curr_channels // 2, min_channels)

            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        curr_channels, out_channels,
                        kernel_size=4 if next_size == curr_size * 2 else 3,
                        stride=2 if next_size == curr_size * 2 else 1,
                        padding=1
                    ),
                    nn.GELU(),
                    LayerNorm2d(out_channels)
                )
            )
            curr_channels = out_channels
            curr_size = next_size

        # 动态输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(curr_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, num_modalities, kernel_size=1),
            nn.Sigmoid() if num_modalities == 1 else nn.Identity()
        )

    def forward(self, x):
        for blk in self.up_blocks:
            x = blk(x)
        return self.final_conv(x)  # (B, num_modalities, H, W)

class EnhancedViT(nn.Module):
    """增强版ViT，支持多尺度特征"""

    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 8,
            in_chans: int = 64,
            embed_dim: int = 384,  # Changed back to original value to match pos_embed
            depth: int = 12,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_embed = HierarchicalPatchEmbed(
            img_size=img_size,
            patch_sizes=[patch_size, patch_size * 2],
            in_chans=in_chans,
            embed_dims=[embed_dim // 2, embed_dim // 2]  # This sums to embed_dim
        )

        # 位置编码
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer块
        self.blocks = nn.ModuleList([
            EnhancedTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=0.1 * (i / depth)
            ) for i in range(depth)
        ])

        # 特征增强
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            LayerNorm2d(embed_dim)
        )

    def forward(self, x):
        # 多尺度patch嵌入
        x = self.patch_embed(x)  # (B,H,W,C)
        B, H, W, C = x.shape

        # 展平并添加位置编码
        x = x.view(B, H*W, C) + self.pos_embed  # Changed from x.view(B, -1, C)

        # Transformer处理
        for blk in self.blocks:
            x = blk(x)

        # 恢复空间结构
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.neck(x)
        return x

class EnhancedTransformerBlock(nn.Module):
    """增强的Transformer块，包含残差连接和DropPath"""

    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim)

        # 局部增强
        self.local_enhance = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        # 全局注意力
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # 局部增强
        B, N, C = x.shape
        h = w = int(N ** 0.5)
        x_local = x.transpose(1, 2).view(B, C, h, w)
        x_local = self.local_enhance(x_local)
        x_local = x_local.view(B, C, N).transpose(1, 2)

        x = x + self.drop_path(x_local)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HierarchicalPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_sizes, in_chans, embed_dims):
        super().__init__()
        self.projs = nn.ModuleList()
        for ps, dim in zip(patch_sizes, embed_dims):
            self.projs.append(
                nn.Sequential(
                    nn.Conv2d(in_chans, dim, kernel_size=ps, stride=ps),
                    LayerNorm2d(dim)
                )
            )
        # 关键修改：确保输出通道数等于sum(embed_dims)
        self.fusion = nn.Identity()  # 不再需要融合卷积

    def forward(self, x):
        features = []
        for proj in self.projs:
            features.append(proj(x))

        # 调整所有特征图到相同尺寸
        target_size = features[0].shape[2:]
        resized_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(feat)

        # 在通道维度拼接
        return torch.cat(resized_features, dim=1)  # 输出通道数应为embed_dim


class Attention(nn.Module):
    """标准的多头自注意力"""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(dim, num_heads)

    def forward(self, x):
        return self.attn(x, x, x)[0]


class MLP(nn.Module):
    """全连接层"""

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LayerNorm2d(nn.Module):
    """2D层归一化"""

    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return F.layer_norm(x, x.shape[1:])


class DropPath(nn.Module):
    """DropPath实现"""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.bernoulli(torch.full((x.shape[0],), keep_prob)).to(x.device)
        return x * mask.view(-1, 1, 1)


