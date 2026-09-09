import os
import argparse

import multiprocessing as mp

import pprint
import yaml

from src.config import resolve_petct_config

"""
cd /data/cyf/codes/A-PET-CT/InversionAD-main

# 双模态 [CT, PET, PET] 伪 RGB（推荐先跑，默认 PSMA；--devices 是物理 GPU 编号）
nohup python -u main.py --task train --fname configs/exp_dit_petct/petct_dual.yml --devices cuda:1 > psma_dual.log 2>&1 &

# FDG 双模态，不需要改 pth 或 yml 文件名
nohup python -u main.py --task train --fname configs/exp_dit_petct/petct_dual.yml --dataset fdg --devices cuda:0 > fdg_dual.log 2>&1 &

# CT 单模态
nohup python -u main.py --task train --fname configs/exp_dit_petct/petct_ct.yml > psma_ct.log 2>&1 &

# PET 单模态
nohup python -u main.py --task train --fname configs/exp_dit_petct/petct_pet.yml > psma_pet.log 2>&1 &

# 直接测试 latest 权重，save_dir 会从配置自动推导
python -u main.py --task test --fname configs/exp_dit_petct/petct_dual.yml --dataset psma --input_mode dual --devices cuda:0

# 保存 20 张热力图；--save_all_heatmaps 保存全部
python -u main.py --task test --fname configs/exp_dit_petct/petct_dual.yml --dataset psma --input_mode dual --save_heatmaps 20 --devices cuda:0

python -u main.py --task test \
  --fname configs/exp_dit_petct/petct_dual.yml \
  --dataset psma \
  --input_mode dual \
  --devices cuda:4

python -u main.py --task test \
  --fname configs/exp_dit_petct/petct_dual.yml \
  --dataset fdg \
  --input_mode dual \
  --devices cuda:1 > fdg_ci.log 2>&1 &

nohup python3 -u main.py \
  --task test \
  --fname configs/exp_dit_petct/petct_dual.yml \
  --dataset fdg \
  --input_mode dual \
  --devices cuda:5 \
  --bootstrap_iters 500 \
  --ci_pixel_max_samples 0 \
  > inversion_fdg_ci.log 2>&1 &

95ci
nohup python3 -u main.py \
  --task test \
  --fname configs/exp_dit_petct/petct_dual.yml \
  --dataset psma \
  --input_mode dual \
  --devices cuda:7 \
  --bootstrap_iters 500 \
  > psma_npz.log 2>&1 &

nohup python3 -u main.py \
  --task test \
  --fname configs/exp_dit_petct/petct_dual.yml \
  --dataset fdg \
  --input_mode dual \
  --devices cuda:4 \
  --bootstrap_iters 500 \
  > fdg_npz.log 2>&1 &
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fname", type=str,
    help="name of config file to load",
    default="configs.yaml"
)
parser.add_argument(
    "--task", type=str, choices=["train_dist", "train", "test"],
)
parser.add_argument(
    "--devices", type=str, nargs="+", default=None,
    help="Physical CUDA devices to use, e.g. cuda:1. If omitted, use the current CUDA default device.",
)
parser.add_argument(
    "--port", type=int, default=29500,
)

# For test
parser.add_argument("--save_dir", type=str, default=None,)
parser.add_argument("--eval_strategy", type=str, default="inversion",)
parser.add_argument("--eval_step", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--category", type=str, default=None, help="Optional category override for evaluation")
parser.add_argument('--noise_step', type=int, default=8, help='Number of noise steps for evaluation')
parser.add_argument('--use_ema_model', action='store_true', help='Use EMA model for evaluation')
parser.add_argument('--use_best_model', action='store_true', help='Use best model for evaluation')
parser.add_argument("--dataset", type=str, choices=["psma", "fdg"], default=None, help="PETCT dataset split to use")
parser.add_argument("--input_mode", type=str, choices=["ct", "pet", "dual"], default=None, help="PETCT input mode")
parser.add_argument("--save_heatmaps", type=int, default=None, help="Number of heatmaps to save; 0 disables, -1 saves all")
parser.add_argument("--save_all_heatmaps", action="store_true", help="Save heatmaps for all evaluated samples")
parser.add_argument("--bootstrap_iters", type=int, default=500, help="Bootstrap iterations for non-AUROC CI")
parser.add_argument("--ci_pixel_max_samples", type=int, default=0, help="Deprecated compatibility option; pixel CI uses abnormal-slice histogram bootstrap")
parser.add_argument("--ci_hist_bins", type=int, default=16384, help="Histogram bins for pixel slice-bootstrap CI")


def process_main(rank, fname, world_size, devices, task, port, args):
    import os
    selected_device = None
    if devices is not None:
        selected_device = int(str(devices[rank]).split(":")[-1])
    
    import torch
    if torch.cuda.is_available():
        if selected_device is None:
            selected_device = torch.cuda.current_device()
        torch.cuda.set_device(selected_device)
        selected_device_name = torch.cuda.get_device_name(selected_device)
        free_mem, total_mem = torch.cuda.mem_get_info(selected_device)
    else:
        selected_device = None
        selected_device_name = "cpu"
        free_mem, total_mem = 0, 0
    import torch.distributed as dist
    from src.utils import init_distributed
    
    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    
    logging.info(f"called-params {fname}")
    logging.info(
        f"rank {rank} uses torch device "
        f"{f'cuda:{selected_device}' if selected_device is not None else 'cpu'} "
        f"({selected_device_name}); current_device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}; "
        f"free={free_mem / 1024**3:.2f}GiB / total={total_mem / 1024**3:.2f}GiB; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
    )
    
    # load params
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        if args.dataset is not None:
            params["data"]["petct_dataset"] = args.dataset
        if args.input_mode is not None:
            params["data"]["input_mode"] = args.input_mode
        params = resolve_petct_config(params)
        if selected_device is not None:
            params.setdefault("meta", {})["device"] = f"cuda:{selected_device}"
        logging.info("loaded params...")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    
    dist_started = False
    try:
        if task == "train_dist":
            world_size, rank = init_distributed(rank_and_world_size=(rank, world_size), port=port)
            dist_started = dist.is_available() and dist.is_initialized()
            logger.info(f"Running distributed train... (rank: {rank}/{world_size})")
            from src.train_distributed import main as train_dist_main
            train_dist_main(params)
        elif task == "train":
            logger.info("Running single-process train without distributed/NCCL init")
            from src.train import main as train_main
            train_main(params)
        elif task == "test":
            logger.info("Running single-process test without distributed/NCCL init")
            from src.evaluate import main as test_main
            test_main(params, args)
        else:
            raise ValueError(f"Task {task} should be specified")
    finally:
        if dist_started:
            dist.destroy_process_group()

if __name__ == "__main__":
    args = parser.parse_args()

    if "test" in args.task and args.save_dir is not None:
        args.fname = os.path.join(args.save_dir, "config.yaml")
    
    if "dist" not in args.task:
        process_main(0, args.fname, 1, args.devices, args.task, args.port, args)
        exit(0)
    
    if args.devices is None:
        raise ValueError("--task train_dist requires --devices, for example: --devices cuda:0 cuda:1")
    num_gpus = len(args.devices)
    mp.set_start_method("spawn", True)
    
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, args.task, args.port, args)
        )
        p.start()
        processes.append(p)
    
    # wait for all processes to finish
    for p in processes:
        p.join()
