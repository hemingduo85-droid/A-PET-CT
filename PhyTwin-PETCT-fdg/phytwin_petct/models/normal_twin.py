
"""Normal Twin PET generator.

A compact residual U-Net predicts the patient-specific normal PET reference from
CT. The default loss is non-negative and stable for one-class training; the
uncertainty head is available but not used by default for model selection.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class NormalTwinUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, uncertainty=True):
        super().__init__()
        self.uncertainty = bool(uncertainty)
        b = int(base_channels)
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.bottleneck = ConvBlock(b * 4, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.pet_head = nn.Conv2d(b, out_channels, 1)
        self.logvar_head = nn.Conv2d(b, 1, 1) if self.uncertainty else None

    @staticmethod
    def _match(skip, x):
        if skip.shape[-2:] == x.shape[-2:]:
            return skip
        return F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, ct):
        e1 = self.enc1(ct)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.bottleneck(self.pool(e3))
        d3 = self.up3(z)
        d3 = self.dec3(torch.cat([d3, self._match(e3, d3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self._match(e2, d2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self._match(e1, d1)], dim=1))
        pred = self.pet_head(d1)
        if self.logvar_head is None:
            return pred, None
        return pred, torch.clamp(self.logvar_head(d1), min=-6.0, max=3.0)

def gradient_loss(pred, target):
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


def normal_twin_loss(pred, logvar, target, l1_weight=1.0, grad_weight=0.25, nll_weight=0.0):
    l1 = F.l1_loss(pred, target)
    grad = gradient_loss(pred, target)
    nll = pred.new_tensor(0.0)
    if logvar is not None and nll_weight > 0:
        err = (pred - target).abs().mean(dim=1, keepdim=True)
        nll = (err * torch.exp(-logvar) + logvar).mean()
    loss = l1_weight * l1 + grad_weight * grad + nll_weight * nll
    return loss, {"l1": float(l1.detach().cpu()), "grad": float(grad.detach().cpu()), "nll": float(nll.detach().cpu())}


def residual_map(pet, pred, logvar=None, uncertainty_floor=0.25):
    residual = (pet - pred).abs().mean(dim=1)
    if logvar is not None:
        sigma = torch.exp(0.5 * logvar.squeeze(1)).clamp_min(uncertainty_floor)
        residual = residual / sigma
    return residual
