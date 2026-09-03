# Model Comparison Results

## Evaluation Date
2026-09-03

## Session Evaluated
`_1-12-2026-09-01_10-42-53` (98 images, 986 GPS readings)

## Detection Results (Session)

| Model | Total Detections | drain_full | drain_area | max_gps_diff_ms | stale_gps |
|-------|------------------|------------|------------|-----------------|-----------|
| `models/best-seg.pt` (baseline) | 123 | 109 | 14 | 5159 | 1 |
| `runs/segment/finetune/copy_paste_merged/weights/best.pt` | 78 | 1 | 77 | 5159 | 1 |
| `runs/segment/finetune/fast_20ep/weights/best.pt` | 294 | 1 | 293 | 5159 | 1 |

## Validation Metrics (Merged Dataset)

| Model | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | Precision | Recall |
|-------|-----------|--------------|------------|---------------|-----------|--------|
| `models/best-seg.pt` (baseline) | 0.302 | 0.200 | 0.056 | 0.017 | 0.229 | 0.028 |
| `runs/segment/finetune/copy_paste_merged/weights/best.pt` | **0.590** | **0.405** | **0.589** | **0.387** | **0.537** | **0.464** |
| `runs/segment/finetune/fast_20ep/weights/best.pt` | 0.557 | 0.337 | 0.555 | 0.310 | 0.566 | 0.399 |

## Key Findings

### ✅ Validation Metrics Improvement
- **Mask mAP50 improved from 0.056 → 0.589** (10.6x improvement) with `copy_paste_merged` model
- **Mask mAP50-95 improved from 0.017 → 0.387** (22.8x improvement)
- **Box mAP50 improved from 0.302 → 0.590** (1.95x improvement)
- Both fine-tuned models significantly outperform the baseline

### ⚠️ Detection Pattern Shift
- Baseline model: heavily biased toward `drain_full` (109 vs 14 `drain_area`)
- Fine-tuned models: heavily biased toward `drain_area` (77/293 vs 1 `drain_full`)
- This suggests the Copy-Paste augmentation may have over-emphasized pothole features (mapped to `drain_area`)

### 📊 Best Model
- **`copy_paste_merged`** has the highest validation metrics across all categories
- **`fast_20ep`** has slightly better precision but lower overall mAP
- Both are significantly better than the baseline

## Recommendations

1. **Deploy `copy_paste_merged` model** for production use (best overall metrics)
2. **Adjust class-specific confidence thresholds** to balance drain_full vs drain_area detections
3. **Consider additional training data** to improve drain_full detection rates
4. **Investigate class imbalance** in fine-tuned models (too many drain_area detections)

## Next Steps

- Apply two-stage post-processing (per-class conf + geometric filter) to fine-tuned models
- Test on additional sessions for generalization
- Consider per-class NMS or class weights to balance predictions
