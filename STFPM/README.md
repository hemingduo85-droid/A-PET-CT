# STFPM PET/CT

This fork trains and evaluates STFPM on AutoPET PET/CT slices.

## Dataset Config

Built-in dataset names:

- `fdg`: `/data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/FDG`
- `psma`: `/data/cyf/shared_data/PET-CT/AutoPET/2d_equal_mask50/PSMA`

Override the root with `--dataset-path` only when needed.

## Train Then Test

Default training is 30 epochs. The script no longer evaluates after every epoch. It saves only the final epoch checkpoint, then runs one test pass unless `--no-test-after-train` is set.

PET/CT pseudo RGB, default for `ct pet`:

```bash
python main.py train --dataset fdg --modalities ct pet --cuda 0 --batch-size 32
```

This uses original-image pseudo RGB input `[CT, PET, PET]` and saves:

```text
snapshots/fdg_ct+pet_pseudo_rgb_epoch30.pth.tar
```

PSMA:

```bash
python main.py train --dataset psma --modalities ct pet --cuda 0 --batch-size 32
```

Single modality:

```bash
python main.py train --dataset fdg --modalities pet --cuda 0 --batch-size 32
python main.py train --dataset psma --modalities ct --cuda 1 --batch-size 32
```

Use the old multi-modal channel concatenation explicitly:

```bash
python main.py train --dataset fdg --modalities ct pet --input-mode concat --cuda 0
```

If the server has no network and hangs while initializing the teacher model, provide a local torchvision ResNet18 weight file:

```bash
python main.py train --dataset fdg --modalities ct pet --teacher-weights /path/to/resnet18-f37072fd.pth --cuda 0
```

For a quick code smoke test only, you can disable ImageNet teacher weights:

```bash
python main.py train --dataset fdg --modalities ct pet --no-pretrained-teacher --cuda 0
```

Use `--num-workers 8` to speed up PNG loading on the server. If shared memory is tight, lower it to `2` or `0`.

## Direct Test

When `--checkpoint` is omitted, the script infers the checkpoint from dataset, modalities, input mode, and epoch count.

```bash
python main.py test --dataset fdg --modalities ct pet --cuda 0
```

Faster test command for large test sets:

```bash
python main.py test --dataset fdg --modalities ct pet --cuda 0 --test-batch-size 32 --num-workers 8 --skip-pixel-metrics
```

Explicit checkpoint:

```bash
python main.py test --dataset psma --modalities ct pet --checkpoint snapshots/psma_ct+pet_pseudo_rgb_epoch30.pth.tar --cuda 1
```

## Heatmaps

Heatmap saving defaults to `0`.

Save the first 20 test heatmaps:

```bash
python main.py test --dataset fdg --modalities ct pet --heatmap-count 20 --heatmap-dir heatmaps --cuda 0
```

Save all test heatmaps:

```bash
python main.py test --dataset fdg --modalities ct pet --save-all-heatmaps --cuda 0
```

## Metrics

The test output includes the original STFPM metrics and an `eval_protocol.py`-style block:

- slice image AUROC/AP/F1
- abnormal-slice pixel AUROC/AP/F1
- patient AUROC/AP/F1 and sensitivity at 90/95 specificity

AUPRO is slow and skipped by default. Add `--compute-pro` only when you need it.

Pixel-level metrics are much slower than image/patient metrics on thousands of 256x256 slices. Add `--skip-pixel-metrics` for quick comparison runs.
