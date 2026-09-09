import os

import numpy as np
import torch
import tqdm

from dataset import MRIDataset, get_data_transforms
from decoder import de_wide_resnet50_2
from encoder import wide_resnet50_2
from eval_func import evaluation
from eval_protocol import format_metrics
from loss_function import loss_fucntion


def _log(message):
    print(message, flush=True)


def _expand_first_conv(encoder, new_in_channels):
    old_conv = encoder.conv1
    with torch.no_grad():
        old_w = old_conv.weight
        new_conv = torch.nn.Conv2d(
            new_in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        new_w = old_w.repeat(1, new_in_channels, 1, 1) / new_in_channels
        new_conv.weight.copy_(new_w)
    encoder.conv1 = new_conv


def _encoder_in_channels(modalities, replicate_channels, input_mode):
    if len(modalities) > 1 and input_mode == "pseudo_rgb":
        return 3
    if len(modalities) > 1:
        return len(modalities)
    return replicate_channels


def _device(device_name):
    if device_name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        _log("CUDA 不可用，自动回退到 CPU")
        return torch.device("cpu")

    if device_name == "cuda":
        device_name = "cuda:0"

    device = torch.device(device_name)
    if device.type != "cuda":
        return device
    if device.index is None:
        device = torch.device("cuda:0")
    if device.index >= torch.cuda.device_count():
        raise ValueError(
            f"请求的设备 {device} 不存在；当前 PyTorch 可见 GPU 数量为 {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    _log(
        f"using {device}: {props.name}, "
        f"free={free_mem / 1024**3:.2f}GiB/{total_mem / 1024**3:.2f}GiB"
    )
    return device


def _build_models(modalities, replicate_channels, input_mode, device):
    encoder_in_channels = _encoder_in_channels(modalities, replicate_channels, input_mode)

    _log(f"building encoder: input_channels={encoder_in_channels}, pretrained=True")
    if encoder_in_channels == 3:
        encoder = wide_resnet50_2(pretrained=True, in_channels=3, width_per_group=64 * 2)
    else:
        encoder = wide_resnet50_2(pretrained=True, in_channels=1, width_per_group=64 * 2)
        if encoder_in_channels > 1:
            _expand_first_conv(encoder, encoder_in_channels)

    _log("moving encoder to device")
    encoder = encoder.to(device)
    encoder.eval()

    _log("building decoder")
    decoder = de_wide_resnet50_2(pretrained=False).to(device)
    _log("models ready")
    return encoder, decoder


def _build_dataloaders(data_path, batch_size, modalities, replicate_channels, input_mode, num_workers):
    image_size = 256
    data_transform, _, mean_train, std_train = get_data_transforms(
        image_size,
        image_size,
        num_modalities=len(modalities),
        replicate_channels=replicate_channels,
        input_mode=input_mode,
    )

    train_data = MRIDataset(
        root=data_path,
        transform=data_transform,
        phase="train",
        modalities=modalities,
        mean=mean_train,
        std=std_train,
        replicate_channels=replicate_channels,
        input_mode=input_mode,
    )
    test_data = MRIDataset(
        root=data_path,
        transform=data_transform,
        phase="test",
        modalities=modalities,
        mean=mean_train,
        std=std_train,
        replicate_channels=replicate_channels,
        input_mode=input_mode,
    )
    _log(f"train samples={len(train_data)}, test samples={len(test_data)}, num_workers={num_workers}")

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def _build_test_dataloader(data_path, modalities, replicate_channels, input_mode, num_workers):
    image_size = 256
    data_transform, _, mean_train, std_train = get_data_transforms(
        image_size,
        image_size,
        num_modalities=len(modalities),
        replicate_channels=replicate_channels,
        input_mode=input_mode,
    )
    test_data = MRIDataset(
        root=data_path,
        transform=data_transform,
        phase="test",
        modalities=modalities,
        mean=mean_train,
        std=std_train,
        replicate_channels=replicate_channels,
        input_mode=input_mode,
    )
    _log(f"test samples={len(test_data)}, num_workers={num_workers}")
    return torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )


def _print_metrics(prefix, metrics):
    _log(prefix)
    slice_metrics, pat_metrics = metrics
    _log(format_metrics(slice_metrics, pat_metrics, as_percent=True))


def train(
    class_,
    epochs,
    learning_rate,
    res,
    batch_size,
    data_path,
    save_path,
    score_num,
    print_loss,
    layerloss,
    rate,
    net,
    L2,
    seed,
    modalities,
    replicate_channels=3,
    input_mode="single",
    dataset_name="fdg",
    device_name="cuda",
    eval_interval=0,
    checkpoint_path=None,
    heatmap_count=0,
    save_all_heatmaps=False,
    results_root="./results",
    bootstrap_iters=500,
    ci_pixel_max_samples=0,
    hist_bins=16384,
    num_workers=4,
):
    if isinstance(modalities, str):
        modalities = [modalities]

    device = _device(device_name)
    encoder_in_channels = _encoder_in_channels(modalities, replicate_channels, input_mode)
    _log(f"device={device}")
    _log(f"dataset={dataset_name}, modalities={modalities}, input_mode={input_mode}, encoder_in_channels={encoder_in_channels}")

    os.makedirs(save_path, exist_ok=True)
    if checkpoint_path is None:
        modality_tag = "+".join(modalities)
        checkpoint_path = os.path.join(
            save_path,
            f"{net}_{dataset_name}_{modality_tag}_{input_mode}_epoch{epochs}_seed{seed}.pth",
        )

    train_dataloader, test_dataloader = _build_dataloaders(
        data_path,
        batch_size,
        modalities,
        replicate_channels,
        input_mode,
        num_workers,
    )
    encoder, decoder = _build_models(modalities, replicate_channels, input_mode, device)

    optimizer = torch.optim.Adam(
        list(decoder.parameters()),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    last_metrics = None
    for epoch in range(epochs):
        decoder.train()
        loss_list = []
        for img, _, _, _ in tqdm.tqdm(train_dataloader):
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(inputs[3], inputs[0:3], res)

            if layerloss == 0:
                loss = loss_fucntion(inputs[0:3], outputs, L2)[0]
            else:
                rec_loss, layer_loss = loss_fucntion(inputs[0:3], outputs, L2)
                loss = rec_loss + rate * layer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if print_loss == 1 and ((epoch + 1) % 10 == 0 or epoch + 1 == epochs):
            _log("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, epochs, np.mean(loss_list)))

        if eval_interval > 0 and (epoch + 1) % eval_interval == 0 and epoch + 1 != epochs:
            metrics = evaluation(encoder, decoder, res, test_dataloader, device, score_num)
            _print_metrics(f"epoch {epoch + 1} intermediate evaluation:", metrics)

    last_metrics = evaluation(
        encoder,
        decoder,
        res,
        test_dataloader,
        device,
        score_num,
        heatmap_count=heatmap_count,
        save_all_heatmaps=save_all_heatmaps,
        heatmap_dir=results_root,
        dataset_name=dataset_name,
        modality_tag="+".join(modalities),
        input_mode=input_mode,
        bootstrap_iters=bootstrap_iters,
        ci_pixel_max_samples=ci_pixel_max_samples,
        hist_bins=hist_bins,
    )
    _print_metrics(f"epoch {epochs} final evaluation:", last_metrics)
    torch.save(decoder.state_dict(), checkpoint_path)
    _log(f"Final model saved: {checkpoint_path}")
    return last_metrics[0]["img_aupr"]


def test(
    class_,
    res,
    data_path,
    save_path,
    score_num,
    layerloss,
    rate,
    net,
    L2,
    seed,
    modalities,
    replicate_channels=3,
    input_mode="single",
    dataset_name="fdg",
    device_name="cuda",
    checkpoint_path=None,
    heatmap_count=0,
    save_all_heatmaps=False,
    results_root="./results",
    bootstrap_iters=500,
    ci_pixel_max_samples=0,
    hist_bins=16384,
    num_workers=4,
):
    if isinstance(modalities, str):
        modalities = [modalities]
    if checkpoint_path is None:
        raise ValueError("test 模式必须提供 checkpoint_path 或使用 main.py 的自动推导路径")

    device = _device(device_name)
    test_dataloader = _build_test_dataloader(
        data_path,
        modalities=modalities,
        replicate_channels=replicate_channels,
        input_mode=input_mode,
        num_workers=num_workers,
    )
    encoder, decoder = _build_models(modalities, replicate_channels, input_mode, device)
    _log(f"loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    _log("checkpoint loaded; starting evaluation")

    metrics = evaluation(
        encoder,
        decoder,
        res,
        test_dataloader,
        device,
        score_num,
        heatmap_count=heatmap_count,
        save_all_heatmaps=save_all_heatmaps,
        heatmap_dir=results_root,
        dataset_name=dataset_name,
        modality_tag="+".join(modalities),
        input_mode=input_mode,
        bootstrap_iters=bootstrap_iters,
        ci_pixel_max_samples=ci_pixel_max_samples,
        hist_bins=hist_bins,
    )
    _print_metrics("direct test evaluation:", metrics)
    return metrics
