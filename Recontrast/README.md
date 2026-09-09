# PET-CT ReContrast

This workspace keeps the PET-CT ReContrast training and testing path only.

## Data Presets

- `fdg`: `/data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/FDG`
- `psma`: `/data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA`

Use `--data_dir` only when you need to override the preset.

## Train

Train 30 epochs, test once at the end, then save the epoch-30 checkpoint:

```bash
python ReContrast_petct.py --mode train --dataset psma --modality petct --cuda 0
```

Default `petct` input is pseudo RGB `[CT, PET, PET]`. Single-modality runs are still available:

```bash
python ReContrast_petct.py --mode train --dataset fdg --modality pet --cuda 0
python ReContrast_petct.py --mode train --dataset fdg --modality ct --cuda 1
```

The default checkpoint path is:

```text
results/recontrast_<dataset>_<modality>/epoch30_<dataset>_<modality>.pth
```

## Direct Test

Use the default checkpoint name:

```bash
python ReContrast_petct.py --mode test --dataset psma --modality petct --cuda 0
```

Or provide a checkpoint explicitly:

```bash
python ReContrast_petct.py --mode test --dataset psma --modality petct --checkpoint /path/to/model.pth --cuda 0
```

## Heatmaps

Heatmap saving is off by default (`--heatmap_count 0`).

Save the first 20 heatmaps:

```bash
python ReContrast_petct.py --mode test --dataset psma --modality petct --heatmap_count 20 --cuda 0
```

Save all heatmaps:

```bash
python ReContrast_petct.py --mode test --dataset psma --modality petct --save_all_heatmaps --cuda 0
```

Training heatmaps are written to `heatmaps/`; direct-test heatmaps are written to `heatmaps_test/` inside the run directory.
