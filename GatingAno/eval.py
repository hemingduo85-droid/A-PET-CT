import argparse
import os
"""
python eval.py   --dataset PSMA   --modality petct   --gpu 5 \
        --ckpt /data/cyf/codes/A-PET-CT/GatingAno/checkpoints/1e-20.20.2/PSMA_petct_epoch30.pth \
        --bootstrap_iters 500   --ci_hist_bins 16384

python eval.py   --dataset FDG   --modality petct   --gpu 5 \
        --ckpt /data/cyf/codes/A-PET-CT/GatingAno/checkpoints/1e-20.20.1fdg/FDG_petct_epoch30.pth \
        --bootstrap_iters 500   --ci_hist_bins 16384
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataloader import PETCTAnomalyDataset
from models import GatingAno
from train import (
    CI_BOOTSTRAP_ITERS,
    CI_RANDOM_SEED,
    Config,
    DATASET_CONFIGS,
    evaluate,
    save_metrics,
    validate_checkpoint_metadata,
)


def build_transform(image_size):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_generator(config, ckpt_path):
    generator = GatingAno(
        n_channels=config.input_channels,
        n_classes=config.output_channels,
    ).to(config.device)

    try:
        checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=config.device)
    validate_checkpoint_metadata(checkpoint, config.dataset_name, config.modality, config)
    state_dict = checkpoint['state_dict']
    generator.load_state_dict(state_dict)
    if isinstance(checkpoint, dict) and 'alpha' in checkpoint:
        config.alpha = checkpoint['alpha']
    return generator


def main():
    parser = argparse.ArgumentParser(description='Direct checkpoint evaluation for GatingAno')
    parser.add_argument('--modality', type=str, default='pet', choices=['pet', 'ct', 'petct'],
                        help='pet, ct, or petct for dual-modality PET+CT')
    parser.add_argument('--gpu', type=str, default='2', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset', type=str.upper, default='PSMA', choices=sorted(DATASET_CONFIGS),
                        help='dataset config key; used to choose data root and output names')
    parser.add_argument('--data_root', type=str, default=None,
                        help='custom dataset root containing train/ and test/; overrides --dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='directory used for default checkpoint and metric paths')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint path; defaults to checkpoints/{dataset}_{modality}_epoch30.pth')
    parser.add_argument('--metrics_path', type=str, default=None,
                        help='metrics output path; defaults to checkpoints/{dataset}_{modality}_eval_metrics.txt')
    parser.add_argument('--cache_path', type=str, default=None,
                        help='optional .npz path to save labels, masks, maps, image scores, and paths')
    parser.add_argument('--export_only', action='store_true',
                        help='only export eval cache; skip metric and CI computation')
    parser.add_argument('--save_heatmaps', action='store_true',
                        help='save anomaly heatmaps while evaluating')
    parser.add_argument('--dual_input_mode', type=str, default='pseudo_rgb',
                        choices=['pseudo_rgb', 'ctctpet'],
                        help='petct input: pseudo_rgb=[CT,PET,PET], ctctpet=[CT,CT,PET]')
    parser.add_argument('--bootstrap_iters', type=int, default=CI_BOOTSTRAP_ITERS,
                        help='bootstrap iterations for AP/AUPR/F1 95% CI')
    parser.add_argument('--ci_pixel_max_samples', type=int, default=200000,
                        help='deprecated; point estimates use all abnormal pixels')
    parser.add_argument('--ci_hist_bins', type=int, default=16384,
                        help='histogram bins for abnormal-slice pixel bootstrap CI')
    parser.add_argument('--ci_seed', type=int, default=CI_RANDOM_SEED,
                        help='random seed for CI resampling')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = Config(
        modality=args.modality,
        gpu=args.gpu,
        dataset=args.dataset,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        dual_input_mode=args.dual_input_mode,
        bootstrap_iters=args.bootstrap_iters,
        ci_pixel_max_samples=args.ci_pixel_max_samples,
        ci_hist_bins=args.ci_hist_bins,
        ci_seed=args.ci_seed,
    )
    ckpt_path = args.ckpt or config.final_ckpt_path
    metrics_path = args.metrics_path or config.eval_metrics_path

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"\nDataset:    {config.dataset_name}")
    print(f"Test root:  {config.test_root}")
    print(f"Modality:   {config.modality.upper()}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Metrics:    {metrics_path}")
    if args.cache_path:
        print(f"Cache:      {args.cache_path}")
    if args.export_only:
        print("Mode:       export only, no CI metrics")
    if args.save_heatmaps:
        print(f"Heatmaps:   {config.heatmap_dir}")
    if config.modality == 'petct':
        channel_order = '[CT, CT, PET]' if config.dual_input_mode == 'ctctpet' else '[CT, PET, PET]'
        print(f"PETCT input: {config.dual_input_mode} {channel_order}")
    print(f"Bootstrap:  {config.bootstrap_iters}")
    print("Pixel point estimates: exact full abnormal pixels")
    print(f"Pixel CI histogram bins: {config.ci_hist_bins}")

    test_dataset = PETCTAnomalyDataset(
        root=config.test_root,
        mode='test',
        modality=config.modality,
        dual_input_mode=config.dual_input_mode,
        return_path=True,
        transform=build_transform(config.image_size),
        image_size=config.image_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
    )

    generator = load_generator(config, ckpt_path)
    metrics = evaluate(
        epoch='eval',
        config=config,
        generator=generator,
        test_loader=test_loader,
        save_heatmaps=args.save_heatmaps,
        cache_path=args.cache_path,
        compute_ci=not args.export_only,
    )
    if metrics is None:
        print("Skipped metric computation.")
        return
    save_metrics(metrics_path, 'eval', config, metrics)
    print(f"Saved eval metrics: {metrics_path}")


if __name__ == '__main__':
    main()
