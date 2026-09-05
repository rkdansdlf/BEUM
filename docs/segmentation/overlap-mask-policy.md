# BEUM Canonical Segmentation Policy

## 1. Overview & Policy Definition
BEUM establishes a strict canonical segmentation contract across all model training and evaluation:

```python
CANONICAL_OVERLAP_MASK = False
CANONICAL_MASK_RATIO = 4
BENCHMARK_CONFIDENCE = 0.001
BENCHMARK_IOU = 0.70
```

* `overlap_mask=False`: Canonical standard for all fine-tuning, training, and benchmarking.
* `overlap_mask=True`: Used ONLY for historical baseline reproduction (strictly named `legacy`).
* `mask_ratio=4`: Default downsample ratio to balance GPU memory and boundary accuracy.
* `mask_ratio=1`: Evaluated only as a separate controlled boundary-precision ablation.
* `conf=0.001`: Dedicated cutoff for PR curve & AP calculation (`model.val()`).
* Operational evaluations (`conf=0.20` or higher) are strictly decoupled into separate evaluation tracks.

## 2. Background: The Overlapping Polygon Degradation Problem
When training or evaluating with Ultralytics YOLO segmentation default `overlap_mask=True`:
1. All instance masks in an image are flattened into a single 2D index array `(H, W)` sorted by area descending.
2. Where instances overlap, smaller instances overwrite larger instances.
3. In drainage grating datasets, `drain_full` and `drain_area` co-occur and heavily overlap:
   - In 30 out of 31 Canonical Val images, polygons intersect and contain each other.
   - 30 instances of `drain_full` suffered severe pixel loss (median loss 63.1%, max 99.0%), becoming carved-out "donut" masks.
4. Setting `overlap_mask=False` assigns independent binary channels `(N, H, W)` to each instance, ensuring full polygon integrity without pixel theft.

## 3. Metric Semantics
- **AP50**: Average Precision at IoU=0.50 (`metric.ap50[i]`).
- **AP50-95**: Average Precision from IoU=0.50 to 0.95 (`metric.ap[i]`).
- Do NOT label `metrics.seg.maps` as AP50 (in Ultralytics, `maps` returns AP50-95).
- Do NOT use `metrics.box.p[0]` for overall precision; use `metrics.box.mp` and `metrics.box.mr`.
