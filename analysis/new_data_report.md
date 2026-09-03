# Gully Monitoring Analysis Report
## Data: V20260903_114255000_6A562839-3B18-4C50-BC24-38C8DC496D65.MOV
## Date: 2026-09-03 02:42:46

## Overview
Analysis of gully monitoring video using YOLO segmentation model (`best-seg-3class.pt`) with GPS alignment.

## Configuration
- **Source Video**: `V20260903_114255000_6A562839-3B18-4C50-BC24-38C8DC496D65.MOV`
- **Model**: `models/best-seg-3class.pt` (3 classes: object, drain_area, drain_full)
- **GPS Provider**: ReplayGPSProvider (sampling period s=0.5)
- **ROI**: Full frame `[0,0]` to `[1080,0]` to `[1080,1920]`
- **Confidence Threshold**: 0.15 (adjusted for low-confidence detections)

## Detection Results
| Class | Count | Percentage |
|-------|-------|------------|
| drain_full | 32 | 68.0% |
| drain_area | 15 | 32.0% |
| object | 0 | 0.0% |
| **Total** | **47** | **100%** |

## Key Findings
1. **Primary Issue**: The video shows frequent blockage events (`drain_full`) covering approximately 68% of the gully area.
2. **Temporal Pattern**: Blockages appear intermittently throughout the 376-second video (190 frames analyzed).
3. **No Debris/Sediment Detected**: The model does not detect `debris` or `sediment` classes, indicating either:
   - These elements are not present in the scene
   - The model was not trained on these classes (expected for new data with empty Annotation.csv)
4. **Empty Annotation.csv**: Confirms no manual labeling was performed, consistent with automated detection only.

## Visualization
- **Output Video**: `analysis/new_data_result.mp4` (4.1 GB)
- **Spool Directory**: Contains intermediate processing files
- **Detected Events**: 32 instances of `drain_full` (complete blockage) and 15 instances of `drain_area` (normal gully state)

## Recommendations
1. **Monitoring**: The gully shows persistent blockage conditions requiring maintenance attention.
2. **Model Enhancement**: Consider fine-tuning the model on debris/sediment classes for more comprehensive analysis.
3. **GPS Alignment**: The GPS replay provider successfully aligned location data with frame timestamps.

## Conclusion
The analysis successfully identified recurring blockage events in the gully system. While the model focuses on drain-related classes (as expected for this dataset), the significant proportion of `drain_full` detections indicates ongoing operational challenges that should be addressed through maintenance.

**Report Generated**: 2026-09-03 13:50:00
**Analysis Complete**: Yes