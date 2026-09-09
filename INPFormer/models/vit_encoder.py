import torch
import timm
from torch.hub import HASH_REGEX, download_url_to_file, urlparse
from dinov1 import vision_transformer
from dinov2.models import vision_transformer as vision_transformer_dinov2
import numpy as np
from scipy import interpolate
import logging
import os

_logger = logging.getLogger(__name__)

_WEIGHTS_DIR = "backbones/weights"
os.makedirs(_WEIGHTS_DIR, exist_ok=True)




def load(name):

    arch, patchsize = name.split("_")[-2], name.split("_")[-1]
    model = vision_transformer.__dict__[f'vit_{arch}'](patch_size=int(patchsize))
    if "dino" in name:
        if "v2" in name:
            if "reg" in name:
                model = vision_transformer_dinov2.__dict__[f'vit_{arch}'](patch_size=int(patchsize), img_size=518,
                                                                          block_chunks=0, init_values=1e-8,
                                                                          num_register_tokens=4,
                                                                          interpolate_antialias=False,
                                                                          interpolate_offset=0.1)

                if arch == "base":
                    ckpt_pth = download_cached_file(
                        f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb{patchsize}/dinov2_vitb{patchsize}_reg4_pretrain.pth")
                elif arch == "small":
                    ckpt_pth = download_cached_file(
                        f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vits{patchsize}/dinov2_vits{patchsize}_reg4_pretrain.pth")
                elif arch == "large":
                    ckpt_pth = download_cached_file(
                        f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl{patchsize}/dinov2_vitl{patchsize}_reg4_pretrain.pth")
                else:
                    raise ValueError("Invalid type of architecture. It must be either 'small' or 'base' or 'large.")
            else:
                model = vision_transformer_dinov2.__dict__[f'vit_{arch}'](patch_size=int(patchsize), img_size=518,
                                                                          block_chunks=0, init_values=1e-8,
                                                                          interpolate_antialias=False,
                                                                          interpolate_offset=0.1)

                if arch == "base":
                    ckpt_pth = download_cached_file(
                        f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb{patchsize}/dinov2_vitb{patchsize}_pretrain.pth")
                elif arch == "small":
                    ckpt_pth = download_cached_file(
                        f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vits{patchsize}/dinov2_vits{patchsize}_pretrain.pth")
                else:
                    raise ValueError("Invalid type of architecture. It must be either 'small' or 'base'.")

            state_dict = torch.load(ckpt_pth, map_location='cpu')
        else:  # dinov1
            if arch == "base":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dino/dino_vit{arch}{patchsize}_pretrain/dino_vit{arch}{patchsize}_pretrain.pth")
            elif arch == "small":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dino/dino_deit{arch}{patchsize}_pretrain/dino_deit{arch}{patchsize}_pretrain.pth")
            else:
                raise ValueError("Invalid type of architecture. It must be either 'small' or 'base'.")

            state_dict = torch.load(ckpt_pth, map_location='cpu')

    if "digpt" in name:
        if arch == 'base':
            state_dict = torch.load(f"{_WEIGHTS_DIR}/D-iGPT_B_PT_1K.pth")['model']
        else:
            raise 'Arch not supported in D-iGPT, must be base.'

    if "moco" in name:
        state_dict = convert_key(download_cached_file(
            f"https://dl.fbaipublicfiles.com/moco-v3/vit-{arch[0]}-300ep/vit-{arch[0]}-300ep.pth.tar"))

    if "mae" in name:
        ckpt_pth = download_cached_file(f"https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_{arch}.pth")
        state_dict = torch.load(ckpt_pth, map_location='cpu')['model']

    if "ibot" in name:
        ckpt_pth = download_cached_file(
            f"https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vit{arch[0]}_{patchsize}_rand_mask/checkpoint_teacher.pth")
        state_dict = torch.load(ckpt_pth, map_location='cpu')['state_dict']

    if "deit" in name:
        if arch == "base":
            ckpt_pth = download_cached_file(
                f"https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth")
        elif arch == 'small':
            ckpt_pth = download_cached_file(
                f"https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth")
        else:
            raise ValueError("Invalid type of architecture. It must be either 'small' or 'base'.")

        state_dict = torch.load(ckpt_pth, map_location='cpu')['model']

    # elif "sup" in name:
    #     try:
    #         state_dict = torch.load(f"{_WEIGHTS_DIR}/vit_{arch}_patch{patchsize}_in1k.pth")
    #     except FileNotFoundError:
    #         state_dict = torch.load(f"{_WEIGHTS_DIR}/vit_{arch}_patchsize_{patchsize}_224.pth")

    model.load_state_dict(state_dict, strict=False)
    return model


def download_cached_file(url, check_hash=True, progress=True):
    """
    Mostly copy-paste from timm library.
    (https://github.com/rwightman/pytorch-image-models/blob/29fda20e6d428bf636090ab207bbcf60617570ca/timm/models/_hub.py#L54)
    """
    if isinstance(url, (list, tuple)):
        url, filename = url
    else:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
    cached_file = os.path.join(_WEIGHTS_DIR, filename)
    if not os.path.exists(cached_file):
        _logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file


def convert_key(ckpt_pth):
    ckpt = torch.load(ckpt_pth, map_location="cpu")
    state_dict = ckpt['state_dict']
    new_state_dict = dict()

    for k, v in state_dict.items():
        if k.startswith('module.base_encoder.'):
            new_state_dict[k[len("module.base_encoder."):]] = v

    return new_state_dict


