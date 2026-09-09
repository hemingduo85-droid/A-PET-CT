# 95%CI 计算协议：用于所有对比实验统一实现

本文档用于同步各个对比方法的评价指标和 95%CI 计算方式。目标是保证不同方法之间的指标统计口径完全一致。

## 总原则

所有指标的点估计都使用完整测试集计算。

95%CI 全部使用 bootstrap 估计，迭代 500 次。

不同层级指标使用不同的 bootstrap 重采样单位：

- Image-level 指标：按 slice 重采样。
- Pixel-level 指标：按 abnormal slice 重采样。
- Patient-level 指标：按 patient 重采样。

不要把所有 pixel 当作独立样本直接做 pixel-level bootstrap。医学图像中同一张 slice 内的 pixel 高度相关，直接按 pixel 重采样会低估不确定性，使 CI 虚假变窄。

## 输出指标

统一输出以下指标：

```text
[Slice-Img]      AUROC / AP / F1
[Slice-Px(abn)]  AUROC / AUPR
[Patient]        AUROC / AP / F1
```

不输出以下指标：

```text
Sens@90Spec
Sens@95Spec
Pixel AP
Pixel F1
```

注意：pixel-level 的 PR 指标统一命名为 AUPR，不写 AP。

## Image-level 指标

### 点估计

使用所有 test slices：

```python
img_auroc = roc_auc_score(slice_labels, slice_scores)
img_ap = average_precision_score(slice_labels, slice_scores)
img_f1 = max_f1(slice_labels, slice_scores)
```

其中：

- `slice_labels`: 每张 slice 的 0/1 标签。
- `slice_scores`: 每张 slice 的 anomaly score。
- `max_f1`: 在所有 PR 阈值上取最大 F1。

### 95%CI

使用 slice-level bootstrap：

1. 假设测试集有 `N` 张 slice。
2. 每次从 `N` 张 slice 中有放回抽样 `N` 张。
3. 在抽样后的 slice set 上重新计算 AUROC / AP / F1。
4. 重复 500 次。
5. 取 2.5 和 97.5 百分位数作为 95%CI。

伪代码：

```python
values = []
for _ in range(500):
    idx = rng.integers(0, N, size=N)
    if sampled labels contain both classes:
        values.append(metric(labels[idx], scores[idx]))
ci = percentile(values, [2.5, 97.5])
```

## Pixel-level 指标

### 评价范围

Pixel-level 指标只在 abnormal slices 上计算，统一写作：

```text
Slice-Px(abn)
```

即：

```python
keep = slice_labels == 1
pixel_gt = masks[keep].reshape(-1)
pixel_score = anomaly_maps[keep].reshape(-1)
```

### 点估计

Pixel-level AUROC 和 AUPR 的点估计必须基于所有 abnormal slices 的所有 pixels 精确计算：

```python
px_auroc = roc_auc_score(pixel_gt, pixel_score)
px_aupr = average_precision_score(pixel_gt, pixel_score)
```

注意：这里的点估计不是采样结果。

### 95%CI

Pixel-level 95%CI 使用 abnormal-slice-level bootstrap：

1. 假设测试集中有 `M` 张 abnormal slices。
2. 每次从这 `M` 张 abnormal slices 中有放回抽样 `M` 张。
3. 抽到哪些 slice，就展开这些 slice 里的所有 pixels。
4. 在这些 pixels 上重新计算 pixel-level AUROC / AUPR。
5. 重复 500 次。
6. 取 2.5 和 97.5 百分位数作为 95%CI。

伪代码：

```python
abn_idx = np.where(slice_labels == 1)[0]
values = []
for _ in range(500):
    sampled_slices = rng.choice(abn_idx, size=len(abn_idx), replace=True)
    y = masks[sampled_slices].reshape(-1)
    s = anomaly_maps[sampled_slices].reshape(-1)
    values.append(metric(y, s))
ci = percentile(values, [2.5, 97.5])
```

### 加速实现

如果 abnormal pixels 很多，例如 3778 张 256x256 abnormal slices：

```text
3778 * 256 * 256 = 247,595,008 pixels
```

直接在 500 次 bootstrap 中反复展开并排序所有 pixels 会非常慢。

推荐实现：

1. 先对每张 abnormal slice 分别统计正类 pixel 和负类 pixel 的 score histogram。
2. bootstrap 时只重采样 slice。
3. 根据被抽到的 slice 权重，对各 slice 的 histograms 加权求和。
4. 在聚合 histogram 上重新计算 AUROC / AUPR。

这叫：

```text
histogram-based weighted recomputation
```

它只用于 bootstrap CI 的加速。

最终报告的 pixel 点估计仍建议使用完整 abnormal pixels 的 sklearn 精确结果。

论文里可以写：

```text
Pixel-level AUROC and AUPR were computed exactly using all pixels from abnormal slices. Their 95% confidence intervals were estimated by abnormal-slice-level bootstrap resampling with 500 iterations; histogram-based weighted recomputation was used to accelerate repeated metric calculation.
```

中文：

```text
Pixel-level AUROC 和 AUPR 的点估计基于所有异常切片像素精确计算；其 95%CI 通过异常切片级 bootstrap 重采样估计，迭代 500 次，并使用基于直方图的加权重计算以提高重复指标计算效率。
```

## Patient-level 指标

### 点估计

先把 slice-level score 聚合成 patient-level score。

同一 patient 的 score 取最大值：

```python
patient_score = max(slice_scores_of_this_patient)
patient_label = max(slice_labels_of_this_patient)
```

然后计算：

```python
pat_auroc = roc_auc_score(patient_labels, patient_scores)
pat_ap = average_precision_score(patient_labels, patient_scores)
pat_f1 = max_f1(patient_labels, patient_scores)
```

### 95%CI

使用 patient-level bootstrap：

1. 假设测试集中有 `P` 个 patients。
2. 每次从 `P` 个 patients 中有放回抽样 `P` 个。
3. 在抽样后的 patient set 上重新计算 AUROC / AP / F1。
4. 重复 500 次。
5. 取 2.5 和 97.5 百分位数作为 95%CI。

伪代码：

```python
values = []
for _ in range(500):
    idx = rng.integers(0, P, size=P)
    if sampled patient labels contain both classes:
        values.append(metric(patient_labels[idx], patient_scores[idx]))
ci = percentile(values, [2.5, 97.5])
```

## 统一 npz 缓存流程

所有对比方法都建议先导出统一 `.npz` 缓存文件，再用同一个指标脚本计算最终指标和 95%CI。

不要每个方法在自己的训练脚本里单独实现一套指标计算，否则容易出现以下问题：

- pixel-level 是否只用 abnormal slices 不一致。
- patient-level 聚合方式不一致。
- bootstrap 重采样单位不一致。
- 95%CI 迭代次数或随机种子不一致。
- image score 定义不清楚，后续难以复查。

推荐流程：

```text
模型推理 -> 导出统一 npz -> 统一指标脚本读取 npz -> 输出 txt/csv 指标
```

也就是：

```bash
# 第一步：每个方法各自跑推理，只负责导出 npz
python eval_or_infer.py --export_npz xxx_eval_cache.npz

# 第二步：所有方法统一用同一个脚本算指标和 95%CI
python compute_cache_metrics.py --cache xxx_eval_cache.npz --output xxx_metrics.txt --bootstrap_iters 500
```

## npz 缓存格式

每个方法必须导出 `.npz`，字段名保持一致。

`.npz` 文件必须包含：

```text
labels        # shape: (N,), 每张 slice 的 0/1 标签
masks         # shape: (N, H, W), pixel GT mask
maps          # shape: (N, H, W), anomaly map / score map
image_scores  # shape: (N,), slice-level anomaly score
paths         # shape: (N,), 原始图像路径，用于提取 patient id
```

各字段含义：

```python
labels[i]       # 第 i 张 slice 是否异常
masks[i]        # 第 i 张 slice 的 lesion mask
maps[i]         # 第 i 张 slice 的 anomaly score map
image_scores[i] # 第 i 张 slice 的 image-level score
paths[i]        # 第 i 张 slice 的路径
```

推荐 dtype：

```text
labels:       int32 或 int64
masks:        uint8 或 bool
maps:         float32
image_scores: float32 或 float64
paths:        str
```

推荐保存代码：

```python
import numpy as np

np.savez_compressed(
    "METHOD_DATASET_MODALITY_eval_cache.npz",
    labels=np.asarray(labels, dtype=np.int32),
    masks=np.asarray(masks, dtype=np.uint8),
    maps=np.asarray(anomaly_maps, dtype=np.float32),
    image_scores=np.asarray(image_scores, dtype=np.float64),
    paths=np.asarray(paths, dtype=str),
)
```

字段要求：

```text
N 必须是 test slice 数量。
labels、masks、maps、image_scores、paths 的第 0 维必须一一对应。
masks 和 maps 必须已经 resize 到同一 H,W。
labels 中 normal=0，abnormal=1。
masks 中背景=0，病灶=1。
maps 数值越大表示越异常。
paths 必须能提取 patient id。
```

路径命名建议：

```text
{METHOD}_{DATASET}_{MODALITY}_eval_cache.npz
```

例如：

```text
GatingAno_FDG_petct_eval_cache.npz
INP_FDG_ct_pet_eval_cache.npz
DAE_FDG_petct_eval_cache.npz
AEFlow_FDG_petct_eval_cache.npz
```

如果已有路径里 method/dataset/modality 信息很清楚，也可以用：

```text
checkpoints/FDG_petct_eval_cache.npz
```

## npz 导出时各方法需要提供什么

每个方法只需要提供以下五类结果：

### 1. Slice label

```python
labels.append(label)
```

要求：

```text
normal slice = 0
abnormal slice = 1
```

### 2. Pixel mask

```python
masks.append(mask)
```

要求：

```text
shape = (H, W)
background = 0
lesion = 1
```

如果原始 mask 和 anomaly map 尺寸不同，必须先 resize 到 anomaly map 尺寸，或者统一 resize 到 256x256。

### 3. Anomaly map

```python
maps.append(anomaly_map)
```

要求：

```text
shape = (H, W)
score 越大越异常
```

不同方法可以有不同 anomaly map 来源，例如 reconstruction error、feature distance、flow likelihood 等，但最终必须变成同一方向：

```text
larger = more abnormal
```

### 4. Image-level score

```python
image_scores.append(score)
```

要求：

```text
每张 slice 一个 scalar score
score 越大越异常
```

image score 可以来自：

```text
anomaly map mean
anomaly map max
top-k percent mean
模型自己的 image-level score
```

但所有方法要尽量使用各自论文/代码中最合理、已确定的 image-level scoring 方式，并在方法记录里写清楚。

### 5. Path

```python
paths.append(img_path)
```

要求：

```text
能从 path 中解析 patient id。
```

当前 GatingAno 默认 patient id 解析方式是：

```python
patient_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
```

也就是假设路径类似：

```text
.../normal_or_abnormal/PATIENT_ID/modality/slice.png
```

如果其他方法导出的 path 格式不同，需要同步修改 patient id 解析函数，保证所有方法 patient-level 指标一致。

## 统一输出格式

输出格式固定为：

```text
[Slice-Img]  AUROC=xx.xx% (95%CI xx.xx-xx.xx%)  AP=xx.xx% (95%CI xx.xx-xx.xx%)  F1=xx.xx% (95%CI xx.xx-xx.xx%)
[Slice-Px(abn)]  AUROC=xx.xx% (95%CI xx.xx-xx.xx%)  AUPR=xx.xx% (95%CI xx.xx-xx.xx%)
[Patient(a/babn)]  AUROC=xx.xx% (95%CI xx.xx-xx.xx%)  AP=xx.xx% (95%CI xx.xx-xx.xx%)  F1=xx.xx% (95%CI xx.xx-xx.xx%)
```

示例：

```text
[Slice-Img]  AUROC=65.86% (95%CI 64.58-67.02%)  AP=66.34% (95%CI 64.77-67.89%)  F1=67.88% (95%CI 66.95-68.92%)
[Slice-Px(abn)]  AUROC=95.18% (95%CI 95.11-95.25%)  AUPR=24.36% (95%CI 23.82-24.95%)
[Patient(101/202abn)]  AUROC=73.41% (95%CI 66.27-79.75%)  AP=73.87% (95%CI 65.22-82.40%)  F1=71.06% (95%CI 66.67-77.85%)
```

## 当前 GatingAno 示例命令

先导出缓存：

```bash
python eval.py \
  --dataset FDG \
  --modality petct \
  --gpu 4 \
  --ckpt checkpoints/FDG_petct_epoch30.pth \
  --cache_path FDG_petct_eval_cache.npz \
  --export_only
```

再从缓存计算 bootstrap 指标：

```bash
python compute_cache_metrics.py \
  --cache FDG_petct_eval_cache.npz \
  --output FDG_petct_bootstrap_metrics.txt \
  --bootstrap_iters 500 \
  --exact_pixel_auroc 0.9518 \
  --exact_pixel_aupr 0.2436
```

如果没有提前算精确 pixel 点估计，也可以不传 `--exact_pixel_auroc` 和 `--exact_pixel_aupr`。此时脚本会使用 histogram 近似点估计。建议最终论文表格优先使用完整 abnormal pixels 的精确点估计。

## 常见问题

### 点估计是什么？

点估计就是最终报告的指标数值本身。

例如：

```text
AUROC=95.18% (95%CI 95.11-95.25%)
```

其中：

```text
AUROC=95.18% 是点估计
95%CI 95.11-95.25% 是置信区间
```

### Pixel-level 指标是不是基于所有 pixels？

是。

Pixel-level AUROC / AUPR 的点估计基于所有 abnormal slices 的所有 pixels 计算。

区别在于：CI 的 bootstrap 重采样单位是 abnormal slice，不是 pixel。

### 为什么不做 pixel-level bootstrap？

因为同一张医学图像内的 pixel 高度相关。把所有 pixel 当成独立样本会低估 CI。

另外，大数据量下 pixel-level bootstrap 计算成本极高。例如 2.47 亿 pixels 重复 500 次排序，实际不可行。

### 为什么 pixel CI 用 abnormal slice bootstrap？

因为 pixel-level 指标只在 abnormal slices 上计算，所以 bootstrap 也在 abnormal slices 这个独立样本层面进行。

这个做法可以理解为：

```text
指标是 pixel-level；
不确定性估计单位是 slice-level cluster。
```

### AUROC 是否还用 DeLong？

当前统一协议中，所有 95%CI 都使用 bootstrap，包括 AUROC。

原因：

1. 方便所有指标统一。
2. AUROC / AUPR / AP / F1 都可用同一套 bootstrap 框架。
3. 对 pixel-level 指标，slice-level bootstrap 比 pixel-level DeLong 更符合图像内相关性的统计结构。

## 最终方法学表述

英文：

```text
All point estimates were computed using the full test set. The 95% confidence intervals were estimated with 500 bootstrap iterations. For image-level metrics, bootstrap resampling was performed at the slice level. For pixel-level AUROC and AUPR, point estimates were computed exactly using all pixels from abnormal slices, while confidence intervals were estimated by resampling abnormal slices with replacement. Histogram-based weighted recomputation was used to accelerate repeated pixel-level metric calculation. For patient-level metrics, bootstrap resampling was performed at the patient level.
```

中文：

```text
所有点估计均基于完整测试集计算。95%置信区间通过500次bootstrap重采样估计。图像级指标在切片层面重采样；像素级AUROC和AUPR的点估计基于所有异常切片像素精确计算，其置信区间通过对异常切片进行有放回重采样估计，并使用基于直方图的加权重计算以提高重复指标计算效率；患者级指标在患者层面重采样。
```
