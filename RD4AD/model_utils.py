import os

import torch
import torch.nn as nn


def select_device(gpu=None, physical_gpu=None):
    if not torch.cuda.is_available():
        print("device: cpu")
        return torch.device("cpu")

    visible = torch.cuda.device_count()
    if physical_gpu is not None:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    elif gpu is None:
        device = torch.device("cuda")
    else:
        gpu = int(gpu)
        if gpu < 0 or gpu >= visible:
            raise ValueError(
                f"Requested --gpu {gpu}, but PyTorch can see {visible} CUDA device(s). "
                "If CUDA_VISIBLE_DEVICES is set, GPU ids are re-indexed from 0."
            )
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)

    current = torch.cuda.current_device()
    name = torch.cuda.get_device_name(current)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    if physical_gpu is not None:
        print(f"device: physical GPU {physical_gpu} -> cuda:{current} ({name}); CUDA_VISIBLE_DEVICES={visible_devices}")
    else:
        print(f"device: cuda:{current} ({name}); visible cuda devices: {visible}; CUDA_VISIBLE_DEVICES={visible_devices}")
    return device


def adapt_first_conv(model, in_channels):
    old_conv = model.conv1
    if old_conv.in_channels == in_channels:
        return model
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    with torch.no_grad():
        if in_channels % old_conv.in_channels == 0:
            repeat_factor = in_channels // old_conv.in_channels
            weight = old_conv.weight.clone().repeat(1, repeat_factor, 1, 1) / repeat_factor
            new_conv.weight.copy_(weight)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.conv1 = new_conv
    return model
