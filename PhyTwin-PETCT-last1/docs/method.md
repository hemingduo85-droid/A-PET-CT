# PhyTwin-PETCT Method

## Motivation

PSMA PET/CT anomaly detection is difficult because abnormal lesions and normal
physiological uptake are both PET-hot. Kidney, urinary tract/bladder, liver,
spleen, bowel, and other normal tissues can dominate PET intensity and cause
false positives. Therefore, a method should not only detect uptake inconsistent
with normal anatomy, but also suppress uptake patterns that are frequently normal.

## Method Overview

Given paired CT and PET slices, PhyTwin learns from normal training pairs only.
The method contains four unsupervised modules.

### 1. Normal Twin PET Generation

A residual U-Net predicts a personalized normal PET reference from CT:

```text
P_hat = G_theta(C)
```

where `C` is CT and `P_hat` is the generated normal twin PET. The model is trained
on `train/normal` pairs with L1 and gradient consistency losses:

```text
L = ||P_hat - P||_1 + lambda_grad * ||grad(P_hat) - grad(P)||_1
```

The default implementation disables uncertainty NLL because prior experiments
showed negative NLL can make train-loss model selection unstable.

### 2. Residual Anomaly Map

At inference, the method computes the PET-to-normal-twin residual:

```text
R = |P - P_hat|
```

This highlights uptake not explained by CT-conditioned normal PET prediction.

### 3. Normal Residual Patch Memory

All normal residual maps are split into patches. A memory bank stores a sampled
set of normalized normal residual patches. Test patches are scored by k-nearest
neighbor distance to this normal memory. This follows the one-class intuition of
PatchCore, but the memory is built on PET/CT residual patches rather than generic
industrial image features.

### 4. Physiological Uptake Prior Suppression

From normal PET images, the method constructs a physiological uptake prior. For
each normal PET slice, the hottest `q` fraction of pixels is selected. The masks
are averaged and smoothed:

```text
A(x) = smooth(mean_i[ P_i(x) >= quantile(P_i, q) ])
```

`A(x)` is high where normal subjects frequently show high uptake. The final
anomaly map is suppressed at such locations:

```text
S(x) = (w_r * z(R)(x) + w_m * z(M)(x)) * max(1 - alpha * A(x), min_factor)
```

where `M` is the memory-distance map. This module is the main PET/CT-specific
innovation. It is unsupervised and does not require organ labels.

### 5. Lesion-Likeness Image Scoring

Pixel-level localization still uses the continuous anomaly map `S(x)`. For
image-level ranking, PhyTwin can use `score_mode=lesion_z`, a lesion-likeness
score calibrated by normal training slices. The motivation is that image-level
AUROC/AP depends on ranking whole slices, where a normal physiological hotspot
can otherwise outrank a weak abnormal lesion.

The score is computed from high-score connected components at multiple
quantiles, by default `0.985, 0.992, 0.997`. Lower quantiles capture weak or
multifocal lesions, while higher quantiles preserve strong focal evidence. Each
candidate component is ranked by five kinds of evidence:

```text
lesion score = anomaly intensity
             + compact/small-lesion and multifocal reward
             + PET-dominant CT/PET mismatch reward
             - physiological-prior overlap penalty
             - body-surface/truncation artifact penalty
             - large organ-like hotspot penalty
```

The rationale is PET/CT-specific. True PSMA lesions are often compact or
multifocal PET-avid foci and may be PET-dominant relative to local CT contrast.
In contrast, kidneys, bladder, urinary tract, bowel, liver/spleen, broad organ
uptake, and edge/truncation artifacts can produce very high PET intensity but
are less lesion-like. Therefore, `lesion_z` does not simply ask whether the
slice has a bright region; it asks whether the high-response regions look more
like disease than normal physiological uptake.

For broad hotspots that strongly overlap the physiological prior, an additional
organ-like penalty is applied. This is designed to reduce kidney, bladder, bowel,
and other large physiological structures at image level without modifying the
pixel-level anomaly map.

The raw multi-scale lesion-likeness score is calibrated with normal training
slices only:

```text
lesion_z = (raw lesion score - mean_train_normal) / std_train_normal
```

This keeps the setting unsupervised: abnormal labels and masks are still used
only for final evaluation and visualization.

## Visual Outputs

Each saved PNG has six panels:

```text
PET | CT | GT mask | Pred mask | Residual | PhyTwin
```

`Pred mask` is derived from the final anomaly map by connected-component
thresholding. It is not used during training.

## Experimental Protocol

- Training uses only `train/normal`.
- Test labels and masks are used only for reporting metrics and drawing figures.
- Report slice image AUROC/AUPR/F1, abnormal-only pixel AUROC/AUPR, and
  patient AUROC/AUPR/F1. All 95% confidence intervals use 500-iteration
  bootstrap resampling with replacement. Image-level CIs resample slices,
  pixel-level CIs resample abnormal slices and use histogram-based weighted
  recomputation for speed, and patient-level CIs resample patients.
- Primary ablations should isolate the physiological prior, residual memory, and
  lesion-component scoring.
