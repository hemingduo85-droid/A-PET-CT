import argparse
import os
from typing import List, Optional
"""
python main.py --mode test --dataset psma --cuda 6 --bootstrap_iters 500 > psma_ci.log 2>&1 &

python main.py --mode test --dataset fdg --cuda 2 --bootstrap_iters 500 > fdg_ci.log 2>&1 &

nohup python3 -u main.py \
  --mode test \
  --dataset fdg \
  --modalities ct pet \
  --input_mode pseudo_rgb \
  --cuda 2 \
  --num_workers 4 \
  --bootstrap_iters 500 \
  --ci_pixel_max_samples 0 \
  > fdg_ci.log 2>&1 &

95ci
nohup python3 -u main.py \
  --mode test \
  --dataset psma \
  --modalities ct pet \
  --input_mode pseudo_rgb \
  --cuda 5 \
  --num_workers 4 \
  --bootstrap_iters 500 \
  --ci_pixel_max_samples 0 \
  > psma_npz.log 2>&1 &

nohup python3 -u main.py \
  --mode test \
  --dataset fdg \
  --modalities ct pet \
  --input_mode pseudo_rgb \
  --cuda 6 \
  --num_workers 4 \
  --bootstrap_iters 500 \
  --ci_pixel_max_samples 0 \
  > fdg_npz.log 2>&1 &

"""

DATASET_PATHS = {
    "fdg": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/FDG",
    "psma": "/data/cyf/shared_data/A-PETCT/2d_equal_mask50/PSMA",
}


def resolve_data_path(dataset: str, data_path: Optional[str]) -> str:
    path = data_path or DATASET_PATHS[dataset]
    return validate_data_path(dataset, path)


def validate_data_path(dataset: str, data_path: str) -> str:
    normalized = os.path.normpath(data_path)
    last = os.path.basename(normalized).lower()
    known = set(DATASET_PATHS)
    if last in known and last != dataset:
        raise ValueError(
            f"--dataset {dataset} 与 --data_path 不一致: {data_path}. "
            f"{dataset} 应使用 {DATASET_PATHS[dataset]}"
        )
    parts = [part.lower() for part in normalized.split(os.sep) if part]
    for first, second in zip(parts, parts[1:]):
        if first in known and second in known:
            raise ValueError(
                f"--data_path 看起来拼错了，包含连续数据集名 {first}/{second}: {data_path}. "
                f"{dataset} 应使用 {DATASET_PATHS[dataset]}"
            )
    return normalized


def normalize_input_mode(input_mode: str, dual: bool) -> str:
    if dual:
        return "dual"
    return input_mode


def resolve_save_path(
    checkpoint_root: str,
    dataset: str,
    modalities: List[str],
    input_mode: str,
) -> str:
    modality_tag = "+".join(modalities)
    return os.path.join(checkpoint_root, dataset, f"{modality_tag}_{input_mode}")


def default_checkpoint_path(
    save_path: str,
    net: str,
    dataset: str,
    modalities: List[str],
    input_mode: str,
    epochs: int,
    seed: int,
) -> str:
    modality_tag = "+".join(modalities)
    filename = f"{net}_{dataset}_{modality_tag}_{input_mode}_epoch{epochs}_seed{seed}.pth"
    return os.path.join(save_path, filename)


def resolve_device_name(device: str, cuda: str) -> str:
    if device == "cpu":
        return "cpu"
    if device.startswith("cuda:"):
        return device
    return f"cuda:{cuda}"


def build_run_config(argv=None):
    parser = argparse.ArgumentParser(description="skip-TS PET/CT training and testing")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="train 或直接 test")
    parser.add_argument("--dataset", choices=sorted(DATASET_PATHS), default="fdg", help="数据集配置")
    parser.add_argument("--data_path", type=str, default=None, help="覆盖数据集路径；默认由 --dataset 自动选择")
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints", help="权重保存根目录")
    parser.add_argument("--results_root", type=str, default="./results", help="测试结果/热力图保存根目录")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["ct", "pet"],
        choices=["pet", "ct"],
        help="输入模态。默认 ct pet",
    )
    parser.add_argument(
        "--input_mode",
        choices=["pseudo_rgb", "dual", "single"],
        default="pseudo_rgb",
        help="pseudo_rgb=[CT,PET,PET]；dual=原双通道拼接；single=逐模态单独训练",
    )
    parser.add_argument("--dual", action="store_true", help="兼容旧参数，等价于 --input_mode dual")
    parser.add_argument("--replicate_channels", type=int, default=3, help="single 单模态灰度复制通道数")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数；默认 30")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker 数；默认 4，与 INP 配置一致")
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--res", type=int, default=3)
    parser.add_argument("--score_num", type=int, default=10)
    parser.add_argument("--layerloss", type=int, default=1)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--L2", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--net", type=str, default="wide_res50")
    parser.add_argument("--eval_interval", type=int, default=0, help="0 表示只在最后测试；>0 可额外中途测试")
    parser.add_argument("--print_loss", type=int, default=1)
    parser.add_argument("--heatmap_count", type=int, default=0, help="默认 0 不保存热力图；>0 保存前 N 张")
    parser.add_argument("--save_all_heatmaps", action="store_true", help="保存全部测试热力图")
    parser.add_argument("--bootstrap_iters", type=int, default=500, help="AP/AUPR/F1 的 bootstrap 95%%CI 次数")
    parser.add_argument("--hist_bins", type=int, default=16384, help="像素级 histogram bootstrap 的 score bins 数")
    parser.add_argument(
        "--ci_pixel_max_samples",
        "--pixel_max_samples",
        dest="ci_pixel_max_samples",
        type=int,
        default=0,
        help="像素级 AUROC/AUPR 95%%CI 的异常切片分层像素采样上限；0 表示全量 CI",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="test 模式指定权重；不传则按配置自动推导")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="训练/测试设备：cpu、cuda 或 cuda:N。默认 cuda，并由 --cuda 指定 N",
    )
    parser.add_argument("--cuda", type=str, default="0", help="物理 GPU 编号，例如 0、1、6")

    args = parser.parse_args(argv)
    args.input_mode = normalize_input_mode(args.input_mode, args.dual)
    args.data_path = resolve_data_path(args.dataset, args.data_path)
    return args


def modality_groups(modalities: List[str], input_mode: str) -> List[List[str]]:
    if input_mode == "single":
        return [[m] for m in modalities]
    return [modalities]


def main():
    args = build_run_config()
    device_name = resolve_device_name(args.device, args.cuda)

    from train_and_test import test, train

    for modalities in modality_groups(args.modalities, args.input_mode):
        save_path = resolve_save_path(
            args.checkpoint_root,
            args.dataset,
            modalities,
            args.input_mode if len(modalities) > 1 else "single",
        )
        checkpoint_path = args.checkpoint or default_checkpoint_path(
            save_path,
            args.net,
            args.dataset,
            modalities,
            args.input_mode if len(modalities) > 1 else "single",
            args.epochs,
            args.seed,
        )
        common = dict(
            class_="mri",
            res=args.res,
            data_path=args.data_path,
            save_path=save_path,
            score_num=args.score_num,
            layerloss=args.layerloss,
            rate=args.rate,
            net=args.net,
            L2=args.L2,
            seed=args.seed,
            modalities=modalities,
            replicate_channels=args.replicate_channels,
            input_mode=args.input_mode if len(modalities) > 1 else "single",
            dataset_name=args.dataset,
            device_name=device_name,
            heatmap_count=args.heatmap_count,
            save_all_heatmaps=args.save_all_heatmaps,
            results_root=args.results_root,
            bootstrap_iters=args.bootstrap_iters,
            ci_pixel_max_samples=args.ci_pixel_max_samples,
            hist_bins=args.hist_bins,
            num_workers=args.num_workers,
        )

        if args.mode == "train":
            aupr_img = train(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                eval_interval=args.eval_interval,
                print_loss=args.print_loss,
                checkpoint_path=checkpoint_path,
                **common,
            )
            print(f"Final Image AUPR ({args.dataset}, {'+'.join(modalities)}): {aupr_img:.4f}")
            print(f"Final checkpoint saved: {checkpoint_path}")
        else:
            metrics = test(checkpoint_path=checkpoint_path, **common)
            slice_metrics, pat_metrics = metrics
            print(f"Test checkpoint: {checkpoint_path}")
            print(f"Test Image AUPR: {slice_metrics['img_aupr']:.4f}")
            print(f"Test Patient AUPR: {pat_metrics['pat_aupr']:.4f}")


if __name__ == "__main__":
    main()
