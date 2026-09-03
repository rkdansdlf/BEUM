import json
import os
from pathlib import Path

# 수집된 모든 비교 데이터
comparison = {
    'evaluation_date': '2026-09-03',
    'session': '_1-12-2026-09-01_10-42-53',
    
    'models_tested': [
        {
            'name': 'baseline_best-seg',
            'path': 'models/best-seg.pt',
            'description': 'Original model, trained on original 372 images'
        },
        {
            'name': 'copy_paste_merged',
            'path': 'runs/segment/finetune/copy_paste_merged/weights/best.pt',
            'description': 'Fine-tuned on 1092 images (incl. 720 pothole)'
        },
        {
            'name': 'fast_20ep',
            'path': 'runs/segment/finetune/fast_20ep/weights/best.pt',
            'description': '20-epoch fine-tuned version'
        },
        {
            'name': 'retrain_clean_20ep',
            'path': 'runs/segment/retrain_clean_20ep/weights/best.pt',
            'description': 'Fine-tuned on clean 372 images (no pothole contamination)'
        }
    ],
    
    'results': {
        '_best-seg': {
            'detections': 123,
            'drain_full': 109,
            'drain_area': 14,
            'max_gps_diff_ms': 5159,
            'stale_gps': 1,
            'mask_map50': 0.0556,
            'mask_map50_95': 0.0170
        },
        'copy_paste_merged_best': {
            'detections': 78,
            'drain_full': 1,
            'drain_area': 77,
            'max_gps_diff_ms': 5159,
            'stale_gps': 1,
            'mask_map50': 0.5890,
            'mask_map50_95': 0.3867
        },
        'fast_20ep_best': {
            'detections': 294,
            'drain_full': 1,
            'drain_area': 293,
            'max_gps_diff_ms': 5159,
            'stale_gps': 1,
            'mask_map50': 0.5551,
            'mask_map50_95': 0.3100
        },
        'retrain_clean_20ep_two_stage': {
            'detections': 36,
            'drain_full': 10,
            'drain_area': 26,
            'max_gps_diff_ms': 5159,
            'stale_gps': 1,
            'mask_map50': 0.7650,
            'mask_map50_95': 0.6494
        }
    },
    
    'key_findings': [
        'Kaggle pothole 720 images mapped to drain_area caused severe class imbalance',
        'Clean retrain improved mask mAP50 from 0.056 to 0.765 (13.7x improvement)',
        'Two-stage filtering balanced drain_full/drain_area detection (10/26 vs 1/77 or 109/14)',
        'Baseline 109 drain_full were mostly false positives (street lights/fences)',
        'Fine-tuned models nearly missed drain_full entirely (1 detection each)'
    ]
}

# Save comparison file
output_path = Path('analysis/model_comparison_final.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)

print('Final model comparison saved to analysis/model_comparison_final.json')
print()

# Summary table
print('=' * 70)
print('MODEL COMPARISON SUMMARY')
print('=' * 70)
print('Model'.ljust(30), 'Detections'.rjust(10), 'drain_full'.rjust(8), 'drain_area'.rjust(8), 'mAP50'.rjust(6))
print('-' * 70)
for model_name, result in comparison['results'].items():
    det = result['detections']
    df = result['drain_full']
    da = result['drain_area']
    mAP50 = result['mask_map50']
    print(model_name.ljust(30), str(det).rjust(10), str(df).rjust(8), str(da).rjust(8), f'{mAP50:.3f}'.rjust(6))
print('=' * 70)