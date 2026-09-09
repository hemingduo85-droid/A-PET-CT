# GLASS PET-CT

This workspace keeps the PET-CT training path only.

## Data Presets

- `psma`: `/data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA`
- `fdg`: `/data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/FDG`

`petct` modality uses pseudo RGB input as `[CT, PET, PET]`.
Single `pet` and `ct` modes repeat the grayscale modality to RGB.

## Train

Default run: PSMA, PET-CT pseudo RGB, 30 epochs, test once after epoch 30, save no heatmaps.

```bash
bash shell/run-petct.sh
```

Useful overrides:

```bash
DATASET=fdg MODALITY=petct bash shell/run-petct.sh
DATASET=psma MODALITY=pet SAVE_HEATMAPS=20 bash shell/run-petct.sh
DATASET=fdg MODALITY=ct SAVE_HEATMAPS=-1 bash shell/run-petct.sh
```

`SAVE_HEATMAPS=0` saves none, a positive number saves the first N evaluated heatmaps,
and `SAVE_HEATMAPS=-1` saves all heatmaps.

## Test Only

Test the default checkpoint directory for PSMA PET-CT:

```bash
bash shell/test-petct.sh
```

Test another preset or modality:

```bash
DATASET=fdg MODALITY=petct bash shell/test-petct.sh
DATASET=psma MODALITY=pet SAVE_HEATMAPS=20 bash shell/test-petct.sh
```

Test a custom checkpoint/result directory:

```bash
SAVE_DIR=/path/to/results/psma_petct DATASET=psma MODALITY=petct bash shell/test-petct.sh
```

The test-only command reports:

- Image level: AUROC, AP, F1, all with 95% CI.
- Pixel level on abnormal slices: AUROC and AUPR, both with 95% CI.
- Patient level: AUROC, AP, F1, all with 95% CI.

AUROC CI uses DeLong. AP/AUPR/F1 CI uses 500 bootstrap iterations.

## Outputs

If `--save_dir` is not set, checkpoints and results are written under:

```text
results/<dataset>_<modality>/
```

The final epoch checkpoint is:

```text
models/backbone_0/petct_<modality>/ckpt_epoch_30.pth
```

A legacy `ckpt.pth` copy is also written for compatibility.
Heatmaps, when enabled, are saved under:

```text
models/backbone_0/heatmaps/petct_<modality>/
```
