## Investigation: Why image_000 fails (low IoU/F1)

### 1. Correlation between n_gt and instance F1 / pixel IoU across 500 images

| Metric | Correlation |
|---|---|
| n_gt vs. pixel-level foreground IoU | 0.0285 (very weak positive) |
| n_gt vs. instance-level F1 (IoU>=0.5) | -0.1447 (weak negative) |

### 2. Normalisation settings and intensity percentiles

#### image_000 statistics (raw float32 image, 512x512):

| Percentile | Value |
|---|---|
| p1 (1st percentile) | 0.8169 |
| p5 (5th percentile) | 0.8650 |
| p10 (10th percentile) | 0.8731 |
| p50 (median/50th) | 0.8997 |
| p90 (90th percentile) | 0.9255 |
| p95 (95th percentile) | 0.9328 |
| p98 (98th percentile) | 0.9410 |
| p99.8 (used by model) | 0.9579 |
| Mean intensity | 0.8931 |
| Min | 0.0480 |
| Max | 0.9919 |

#### Comparison to well-performing real images (top-5 by pixel IoU):

| Image | p1 | p50 | p99.8 | Mean |
|---|---|---|---|---|
| image_456 | 0.2536 | 0.8995 | 0.9572 | 0.8860 |
| image_481 | 0.0989 | 0.8980 | 0.9572 | 0.8524 |
| image_281 | 0.1312 | 0.8987 | 0.9573 | 0.8649 |
| image_364 | 0.0552 | 0.8982 | 0.9575 | 0.8503 |
| image_459 | 0.0784 | 0.8987 | 0.9575 | 0.8641 |

#### Comparison to synthetic training images (10 images from images/overlap_images):

| Metric | Value |
|---|---|
| p1 mean | 0.0722 |
| p50 mean | 0.6985 |
| p99.8 mean | 0.7146 |
| Whole-image mean | 0.6020 +/- 0.0097 |

#### Comparison to other real images (10 random real images):

| Metric | Value |
|---|---|
| p1 mean | 0.0959 |
| p50 mean | 0.8988 |
| p99.8 mean | 0.9571 |
| Whole-image mean | 0.8384 +/- 0.0223 |

### 3. segment() on image_000 with different normalisations

| Normalisation | Pred. particles (n_pred) | Pixel-level IoU | Instance F1 |
|---|---|---|---|
| Default (csbdeep normalize + Otsu refine) | 18 | 0.2758 | 0.1290 |
| Global min-max -> [0,1] | 1 | 0.7098 | — |
| Percentile 1->99.8 -> [0,1] | 1 | 0.7393 | — |
| csbdeep normalize(1, 99.8) only | 1 | 0.0522 | — |
| Raw 0-1 (as-loaded) | 1 | 0.7064 | — |

### 4. Mean particle intensity vs. background intensity

| Metric | image_000 | Typical real (10-image sample) | Synthetic (10-image sample) |
|---|---|---|---|
| Mean predicted particle intensity | 0.2539 | — | — |
| Mean GT particle intensity | 0.2780 | — | — |
| Mean background intensity | 0.8950 | 0.8384 +/- 0.0223 | 0.6020 +/- 0.0097 |
| Background pixels count | 261,422 | — | — |
| p1 intensity | 0.8169 | 0.0959 | 0.0722 |
| p50 intensity | 0.8997 | 0.8988 | 0.6985 |
| p99.8 intensity | 0.9579 | 0.9571 | 0.7146 |