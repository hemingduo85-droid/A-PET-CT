# PhyTwin-PETCT

PhyTwin-PETCT is an independent unsupervised PET/CT anomaly detection framework
for PSMA images. It is not a UniNet modification. UniNet should be kept as a
baseline only.

## Core Idea

The method learns normality from `train/normal` only:

1. **Normal Twin PET generator**: CT -> patient-specific normal PET reference.
2. **Residual modeling**: true PET minus normal twin PET highlights unexpected uptake.
3. **Normal residual patch memory**: normal residual patches form a one-class memory bank.
4. **Physiological uptake prior suppression**: normal PET hot-region statistics suppress
   common PSMA physiological backgrounds such as kidney, urinary tract/bladder, liver/spleen,
   depending on the dataset alignment.
5. **Image scoring**: by default the image-level score is the top-1% mean of the final anomaly map,
   matching the validated TwinPatch-PHYSIO protocol. A lesion-component scorer is available as an ablation.

Labels and masks are used only for evaluation and visualization.

## Expected Data

```text
train/normal/<patient>/pet/*.png
train/normal/<patient>/ct/*.png
test/normal/<patient>/pet/*.png
test/normal/<patient>/ct/*.png
test/abnormal/<patient>/pet/*.png
test/abnormal/<patient>/ct/*.png
test/abnormal/<patient>/label/*.png
```

You may pass either the dataset root ending in `PSMA` or its parent folder. The
loader tries to resolve `PSMA/train/normal` automatically.

## Server Run

```bash
cd /data/cyf/codes/A-PET-CT/PhyTwin-PETCT
nohup python train.py \
  --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
  --epochs 30 \
  --batch_size 8 \
  --image_size 256 \
  --gpu 5 \
  --save_dir ./saved_results_phytwin \
  > phytwin_psma.log 2>&1 &
```

Quick smoke run:

```bash
python train.py \
  --data_root /Users/zongjinying/Desktop/2d_equal_mask50/PSMA \
  --epochs 1 \
  --batch_size 1 \
  --image_size 64 \
  --num_workers 0 \
  --max_train_batches 1 \
  --max_memory_batches 1 \
  --max_test_batches 2 \
  --save_dir ./smoke_results
```

## Outputs

```text
ckpts/PhyTwin_PETCT/BEST_TWIN.pth
ckpts/PhyTwin_PETCT/BEST_PHYTWIN.pth
saved_results_phytwin/PhyTwin_PETCT/log.txt
saved_results_phytwin/PhyTwin_PETCT/physio_prior.png
saved_results_phytwin/PhyTwin_PETCT/visualizations/*.png
```

Each visualization contains:

```text
PET | CT | GT mask | Pred mask | Residual | PhyTwin
```

The predicted mask is generated from the final anomaly map by a uniform top-quantile
threshold and small-component removal. It is for visualization only and is not
trained or tuned with GT masks.

## Key Ablations

- Remove physiological prior: `--no_physio`
- Change suppression strength: `--physio_alpha 0.4`, `0.65`, `0.8`
- Change hot-region quantile: `--physio_quantile 0.98`, `0.985`, `0.99`
- Use only residual map: `--memory_weight 0 --residual_weight 1`
- Use only memory distance: `--memory_weight 1 --residual_weight 0`
- Image scoring ablation: `--score_mode component --component_quantile 0.99` or `0.997`

## Main References

- Hinge et al., *Normal twin PET: personalized generative modeling for confounder correction and anomaly detection in whole-body PET/CT*, Scientific Reports, 2025.
- Klyuzhin et al., *Unsupervised background removal by dual-modality PET/CT guidance: application to PSMA imaging of metastases*, Journal of Nuclear Medicine, 2021.
- Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR, 2022.
- PSMA PET/CT pitfalls and normal-variant reviews describing physiological uptake in kidneys, urinary tract/bladder, liver/spleen, bowel, and salivary/lacrimal glands.

## Suggested First Experiments

Run the full model first:

```bash
nohup python train.py \
  --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
  --epochs 30 \
  --batch_size 8 \
  --image_size 256 \
  --gpu 5 \
  --save_dir ./saved_results_phytwin \
  > phytwin_full.log 2>&1 &
```

Then run a direct physiological-prior ablation:

```bash
nohup python train.py \
  --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
  --epochs 30 \
  --batch_size 8 \
  --image_size 256 \
  --gpu 5 \
  --no_physio \
  --save_dir ./saved_results_phytwin_no_physio \
  > phytwin_no_physio.log 2>&1 &
```

For the optional component-score ablation, try a less aggressive component threshold:

```bash
python train.py \
  --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
  --epochs 30 \
  --batch_size 8 \
  --image_size 256 \
  --gpu 5 \
  --score_mode component \
  --component_quantile 0.99 \
  --save_dir ./saved_results_phytwin_q099
```

## Image-Level Improvement Experiments

The default `top1pct` score reproduces the validated TwinPatch-PHYSIO behavior.
If slice/patient image-level ranking is still below target, run the normal-calibrated
image score. This uses the distribution of final anomaly-map scores from
`train/normal` to normalize test image scores:

```bash
nohup python train.py \
  --data_root /data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA \
  --epochs 30 \
  --batch_size 8 \
  --image_size 256 \
  --gpu 5 \
  --score_mode normal_z \
  --save_dir ./saved_results_phytwin_normalz \
  > phytwin_normalz.log 2>&1 &
```

Prediction masks are visualization-only and use a fair, method-agnostic rule:
per-image top-quantile threshold plus small-component removal.

```bash
--pred_mask_quantile 0.99 --pred_mask_min_area 8
```

If `Pred mask` is too large, increase the quantile to `0.995`. If it is too
small, lower it to `0.98`. Metrics are computed from the continuous anomaly map,
not from this hard mask.

