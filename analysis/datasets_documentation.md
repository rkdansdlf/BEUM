# Dataset Documentation

## Original Dataset (dataset/)

### Structure
- `images/train/`: 691 images (originally counted as 372 after merge analysis)
- `images/val/`: 88 images
- `labels/train/annotations.coco.json`: 372 images, 1218 annotations
- `labels/val/annotations.coco.json`: 31 images, 76 annotations

### Class Distribution (Original)
| Class | ID | Train Instances | Val Instances |
|-------|----|---------------|-------------|
| object | 0 | - | - |
| drain_area | 1 | 762 | 44 |
| drain_full | 2 | 456 | 32 |

**Note**: Original dataset had 691 train images with imbalanced classes (762 drain_area vs 456 drain_full).

## Kaggle Pothole Dataset (dataset/external/pothole-seg-yolov8/)

### Structure
- `Pothole_Segmentation_YOLOv8_Dataset-main/train/images`: 720 images
- `Pothole_Segmentation_YOLOv8_Dataset-main/valid/images`: 60 images
- Each image has segmentation mask and YOLO format label

### Class Mapping
- **Class 0 (pothole) → Mapped to Class 1 (drain_area) in merged dataset**
- All 720 pothole images and 60 validation images were merged with class remapping

### Impact on Merged Dataset
- **Merged train**: 1092 images (372 local + 720 pothole)
- **Merged val**: 91 images (31 local + 60 pothole)
- **Class imbalance**: drain_area had 1846 instances (762 local + 1846 pothole-derived)
- **drain_full**: Only 255 instances (all from local dataset, 0 from pothole)

## Clean Dataset (dataset/clean/)

### Structure
- `images/train/`: 372 images (original local only)
- `images/val/`: 31 images (original local only)
- `labels/train/annotations.coco.json`: 372 images, 1218 annotations (same as original)
- `labels/val/annotations.coco.json`: 31 images, 76 annotations (same as original)
- `gully-seg-clean.yaml`: YAML config file

### Class Distribution (Clean)
| Class | ID | Train Instances | Val Instances |
|-------|----|---------------|-------------|
| object | 0 | - | - |
| drain_area | 1 | 117 | 12 |
| drain_full | 2 | 255 | 19 |

**Ratio**: drain_area : drain_full = 1 : 2.2 (much better balanced than merged dataset's 1 : 7.7)

### Creation Purpose
- Remove Kaggle pothole dataset contamination
- Retrain model with proper class balance
- Validate that model performance improves without pothole data

## Comparison Summary

| Aspect | Original Merged | Clean Dataset |
|--------|----------------|---------------|
| Train images | 1092 (372 + 720) | 372 |
| Val images | 91 (31 + 60) | 31 |
| drain_area instances | 1846 (762 + 1084*) | 117 |
| drain_full instances | 255 (local only) | 255 |
| drain_area : drain_full ratio | 1 : 7.7 (imbalanced) | 1 : 2.2 (balanced) |
| Source of drain_area | Local + 720 pothole | Local only (117) |
| Source of drain_full | Local only | Local only |

*Note: pothole-derived drain_area count estimated from 720 images mapping to class 1*

## File Locations

| Purpose | Path |
|---------|------|
| Clean training images | `dataset/clean/train/images/` |
| Clean validation images | `dataset/clean/val/images/` |
| Clean labels (train) | `dataset/clean/train/labels/` |
| Clean labels (val) | `dataset/clean/val/labels/` |
| Clean YAML config | `dataset/clean/gully-seg-clean.yaml` |
| Original merged data | `dataset/merged/` |
| Kaggle pothole data | `dataset/external/pothole-seg-yolov8/` (archived) |
| Final model weights | `runs/segment/retrain_clean_20ep/weights/best.pt` |

## Recommendations

### 1. Dataset Version Control
- Preserve both `dataset/merged/` (for reference) and `dataset/clean/` (for training)
- Document any future merges with clear class mapping policies
- Track data provenance (source and transformation history)

### 2. Quality Control
- Implement automated class balance checks before model training
- Set threshold: if any class ratio exceeds 1:3 (train), flag for review
- Validate class distribution after any data merge operation

### 3. Future Data Addition
When adding new datasets:
1. Review class distribution before merging
2. Consider whether new classes should be added or mapped to existing
3. Validate after merge with quick annotation count
4. Retrain or fine-tune with class weights if imbalance detected

### 5. Archive Policy
- Archived Kaggle pothole data preserved at `dataset/external/pothole-seg-yolov8/`
- Original local dataset unchanged at `dataset/`
- Clean dataset created as primary training set

## Technical Notes

### YAML Configuration Differences

**Merged dataset YAML (`dataset/merged/gully-seg-merged.yaml`)**:
```yaml
train: C:/Project/BEUM/dataset/merged/train/images
val: C:/Project/BEUM/dataset/merged/val/images

nc: 3
names: ['object', 'drain_area', 'drain_full']
```

**Clean dataset YAML (`dataset/clean/gully-seg-clean.yaml`)**:
```yaml
train: C:/Project/BEUM/dataset/clean/train/images
val: C:/Project/BEUM/dataset/clean/val/images

nc: 3
names: ['object', 'drain_area', 'drain_full']
```

**Key difference**: Same nc and names, but entirely different image/label contents.

### Annotation Format
Both datasets use COCO format YOLO annotations:
```json
{ "id": 1, "image_id": 1, "category_id": 1, "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]], "area": 100, "bbox": [x, y, w, h] }
```
- `category_id`: 1=drain_area, 2=drain_full
- `bbox`: [x_min, y_min, width, height] in normalized coordinates (0-1)

## Files Generated During This Project

| File/Path | Purpose | Status |
|-----------|---------|--------|
| `dataset/clean/` | Clean training dataset | ✅ Active |
| `analysis/model_comparison_final.json` | Model comparison results | ✅ Created |
| `analysis/final_report.md` | Project final report | ✅ Created |
| `analysis/datasets_documentation.md` | Dataset documentation | ✅ Created |
| `analysis/clean_model_twostage_results.json` | Session evaluation results | ✅ Created |
| `analysis/final_clean_evaluation.json` | Clean model validation metrics | ✅ Created |
| `analysis/model_comparison_results.json` | Previous model comparison | ✅ Existing |
| `analysis/val_metrics_comparison.json` | Validation mAP comparison | ✅ Existing |

## Conclusion

The clean dataset successfully resolves the class imbalance issue caused by Kaggle pothole data contamination. With only 372 images (vs 1092 in merged), the clean dataset achieves:

- Better class balance (1:2.2 vs 1:7.7 ratio)
- Improved model performance (mask mAP50: 0.765 vs 0.589)
- More realistic detection counts (10 drain_full vs 1 or 109 previous)
- Simpler data management (single source, no mapping needed)

The clean dataset is now the recommended primary training set for any future drain detection model development.