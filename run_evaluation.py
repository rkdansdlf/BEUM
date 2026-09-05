import sys
sys.path.insert(0, '.')
from tests.test_all_data_sessions import discover_sessions, process_session, GPS_MATCH_LIMIT_MS, CONFIDENCE
from ultralytics import YOLO
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

sessions = discover_sessions()
print(f'Detected sessions: {[s.name for s in sessions]}')

model_paths = [
    'models/best-seg.pt',
    'runs/segment/finetune/copy_paste_merged/weights/best.pt',
    'runs/segment/finetune/fast_20ep/weights/best.pt'
]

results = {}
for model_path in model_paths:
    p = Path(model_path)
    model_name = f"{p.parent.parent.name}_{p.stem}"
    print(f'Evaluating {model_name}')
    model = YOLO(model_path)
    print(f'Classes: {model.names}')
    for session in sessions:
        summary = process_session(model, session)
        results.setdefault(model_name, {})[session.name] = {
            'detections': summary['detections'],
            'classes': summary['detection_classes'],
            'max_gps_diff_ms': summary['max_gps_time_diff_ms'],
            'stale_gps': summary['stale_gps_frames']
        }
        det = summary['detections']
        cls = summary['detection_classes']
        print(f'{session.name}: detections={det} classes={cls}')

with open('analysis/model_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('Results saved to analysis/model_comparison_results.json')

print('\nRunning validation metrics on merged dataset...')

validation_results = {}
for model_path in model_paths:
    p = Path(model_path)
    model_name = f"{p.parent.parent.name}_{p.stem}"
    print(f'Validating {model_name}...')
    model = YOLO(model_path)
    metrics = model.val(data='dataset/merged/gully-seg-merged.yaml', imgsz=640, conf=0.001, overlap_mask=False, verbose=False)
    
    validation_results[model_name] = {
        'box_map50': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0,
        'box_map50_95': float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0,
        'mask_map50': float(metrics.seg.map50) if hasattr(metrics.seg, 'map50') else 0.0,
        'mask_map50_95': float(metrics.seg.map) if hasattr(metrics.seg, 'map') else 0.0,
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
    }
    print(f'  box mAP50: {metrics.box.map50:.4f}')
    print(f'  mask mAP50: {metrics.seg.map50:.4f}')
    print(f'  mask mAP50-95: {metrics.seg.map:.4f}')

with open('analysis/val_metrics_comparison.json', 'w') as f:
    json.dump(validation_results, f, indent=2)
print('Saved validation metrics to analysis/val_metrics_comparison.json')