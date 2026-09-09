import argparse
import json
import os
import sys


def _configure_visible_gpu_from_argv():
    if "--gpu" not in sys.argv:
        return None
    idx = sys.argv.index("--gpu")
    if idx + 1 >= len(sys.argv):
        return None
    gpu = sys.argv[idx + 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return gpu


REQUESTED_PHYSICAL_GPU = _configure_visible_gpu_from_argv()

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

from config import (
    checkpoint_path,
    get_dataset_config,
    input_channels,
    parse_modalities,
    resolve_input_mode,
)
from dataset import MVTecDataset, get_data_transforms
from de_resnet import de_wide_resnet50_2
from eval_protocol import BOOTSTRAP_ITERS, compute_metrics, format_metrics
from model_utils import adapt_first_conv, select_device
from resnet import wide_resnet50_2


def cal_anomaly_map(fs_list, ft_list, out_size=256, amap_mode="a"):
    anomaly_map = np.ones([out_size, out_size], dtype=np.float32) if amap_mode == "mul" else np.zeros(
        [out_size, out_size], dtype=np.float32
    )
    a_map_list = []
    for fs, ft in zip(fs_list, ft_list):
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
        a_map = a_map[0, 0, :, :].detach().cpu().numpy().astype(np.float32)
        a_map_list.append(a_map)
        if amap_mode == "mul":
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def min_max_norm(image):
    image = np.asarray(image, dtype=np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def parse_heatmap_limit(value):
    text = str(value).strip().lower()
    if text in {"all", "-1"}:
        return None
    limit = int(text)
    if limit < 0:
        raise ValueError("--save-heatmaps must be 0, a positive integer, or all")
    return limit


def make_model(device, modalities, input_mode):
    encoder, bn = wide_resnet50_2(pretrained=True)
    in_ch = input_channels(modalities, input_mode)
    if in_ch != 3:
        encoder = adapt_first_conv(encoder, in_ch)
    decoder = de_wide_resnet50_2(pretrained=False)
    return encoder.to(device).eval(), bn.to(device), decoder.to(device)


def load_checkpoint(bn, decoder, ckp_path, device):
    ckp = torch.load(ckp_path, map_location=device)
    bn_state = dict(ckp["bn"])
    for key in list(bn_state):
        if "memory" in key:
            bn_state.pop(key)
    decoder.load_state_dict(ckp["decoder"])
    bn.load_state_dict(bn_state)


def save_checkpoint_metadata(ckp_path, dataset_config, modalities, input_mode, data_root):
    metadata_path = os.path.splitext(ckp_path)[0] + ".json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_config["name"],
                "data_root": data_root,
                "modalities": modalities,
                "input_mode": input_mode,
                "checkpoint": ckp_path,
            },
            f,
            indent=2,
        )


def save_heatmap(original_path, anomaly_map, save_dir, index, label):
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(original_path))[0]
    patient = os.path.basename(os.path.dirname(os.path.dirname(original_path)))
    stem = f"{index:05d}_{label}_{patient}_{base}"

    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        original = np.zeros_like(anomaly_map, dtype=np.uint8)
    original = cv2.resize(original, (anomaly_map.shape[1], anomaly_map.shape[0]))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    normalized = min_max_norm(anomaly_map)
    heatmap = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_rgb, 0.55, heatmap, 0.45, 0)

    cv2.imwrite(os.path.join(save_dir, stem + "_heatmap.png"), heatmap)
    cv2.imwrite(os.path.join(save_dir, stem + "_overlay.png"), overlay)


def evaluate(encoder, bn, decoder, dataloader, device, save_heatmaps="0",
             heatmap_dir="./heatmaps", run_name="run"):
    bn.eval()
    decoder.eval()
    gt_labels = []
    gt_masks = []
    anomaly_maps = []
    image_scores = []
    paths = []
    heatmap_limit = parse_heatmap_limit(save_heatmaps)
    saved = 0
    output_dir = os.path.join(heatmap_dir, run_name)

    with torch.no_grad():
        for idx, (img, gt, label, path) in enumerate(dataloader):
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode="a")
            anomaly_map = gaussian_filter(anomaly_map, sigma=4).astype(np.float32)

            gt_binary = gt.cpu().numpy().astype(np.float32)
            gt_binary = (gt_binary > 0.5).astype(np.float32)
            label_value = int(label.item())
            path_value = path[0] if isinstance(path, (list, tuple)) else str(path)

            gt_labels.append(label_value)
            gt_masks.append(gt_binary[0])
            anomaly_maps.append(anomaly_map)
            image_scores.append(float(np.max(anomaly_map)))
            paths.append(path_value)

            if heatmap_limit is None or saved < heatmap_limit:
                if heatmap_limit != 0:
                    save_heatmap(path_value, anomaly_map, output_dir, idx, "abnormal" if label_value else "normal")
                    saved += 1

    labels_np = np.asarray(gt_labels)
    masks_np = np.asarray(gt_masks)
    maps_np = np.asarray(anomaly_maps)
    image_scores_np = np.asarray(image_scores)
    print(
        f"Cache loaded: labels={labels_np.shape}, masks={masks_np.shape}, "
        f"maps={maps_np.shape}, bootstrap_iters={BOOTSTRAP_ITERS}",
        flush=True,
    )

    slice_metrics, patient_metrics = compute_metrics(
        gt_labels=labels_np,
        gt_masks=masks_np,
        anomaly_maps=maps_np,
        image_scores=image_scores_np,
        paths=paths,
    )
    if heatmap_limit != 0:
        print(f"saved heatmaps: {saved} -> {output_dir}")
    return slice_metrics, patient_metrics


def print_metrics(slice_metrics, patient_metrics):
    print("Results:")
    print(format_metrics(slice_metrics, patient_metrics))


def test(dataset_name="psma", modalities=None, data_root=None, input_mode="auto",
         ckp_path=None, save_heatmaps="0", heatmap_dir="./heatmaps", gpu=None):
    modalities = modalities or ["pet"]
    dataset_config = get_dataset_config(dataset_name)
    data_root = data_root or dataset_config["data_root"]
    input_mode = resolve_input_mode(modalities, input_mode)
    ckp_path = ckp_path or checkpoint_path(dataset_config, modalities, input_mode)

    device = select_device(physical_gpu=REQUESTED_PHYSICAL_GPU)
    print(f"dataset: {dataset_config['name']}")
    print(f"data root: {data_root}")
    print(f"modalities: {modalities}")
    print(f"input mode: {input_mode}")
    print(f"checkpoint: {ckp_path}")

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_data = MVTecDataset(
        root=os.path.join(data_root, "test"),
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
        modalities=modalities,
        input_mode=input_mode,
        image_size=256,
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn, decoder = make_model(device, modalities, input_mode)
    load_checkpoint(bn, decoder, ckp_path, device)
    slice_metrics, patient_metrics = evaluate(
        encoder,
        bn,
        decoder,
        test_dataloader,
        device,
        save_heatmaps=save_heatmaps,
        heatmap_dir=heatmap_dir,
        run_name=f"{dataset_config['name']}_{input_mode}",
    )
    print_metrics(slice_metrics, patient_metrics)
    return slice_metrics, patient_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RD4AD AutoPET direct test")
    parser.add_argument("--dataset", type=str, default="psma", choices=["psma", "fdg"])
    parser.add_argument("--data-root", type=str, default="", help="覆盖配置中的数据集根目录")
    parser.add_argument("--modality", type=str, default="pet", choices=["pet", "ct"])
    parser.add_argument("--modalities", type=str, default="", help="逗号分隔，如 pet,ct")
    parser.add_argument("--input-mode", type=str, default="auto",
                        choices=["auto", "rgb", "ct_pet_pet", "concat"])
    parser.add_argument("--checkpoint", type=str, default="", help="覆盖自动推导的 checkpoint 路径")
    parser.add_argument("--save-heatmaps", type=str, default="0",
                        help="保存热力图数量：0、整数或 all，默认 0")
    parser.add_argument("--heatmap-dir", type=str, default="./heatmaps")
    parser.add_argument("--gpu", type=int, default=None,
                        help="指定 nvidia-smi 中的物理 GPU id，如 --gpu 4")
    args = parser.parse_args()

    selected_modalities = parse_modalities(args.modality, args.modalities)
    test(
        dataset_name=args.dataset,
        modalities=selected_modalities,
        data_root=args.data_root or None,
        input_mode=args.input_mode,
        ckp_path=args.checkpoint or None,
        save_heatmaps=args.save_heatmaps,
        heatmap_dir=args.heatmap_dir,
        gpu=args.gpu,
    )
