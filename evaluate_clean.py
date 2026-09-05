from ultralytics import YOLO
import json

# Load the final model (trained on clean dataset)
model = YOLO('runs/segment/retrain_clean_20ep/weights/best.pt')
print(f'Loaded model classes: {model.names}')

# Evaluate on clean validation set
print('\nEvaluating on clean validation set...')
results = model.val(data='dataset/clean/gully-seg-clean.yaml', imgsz=640, conf=0.001, overlap_mask=False, verbose=False)

# Extract metrics
metrics = {
    'box_map50': float(results.box.map50),
    'box_map50_95': float(results.box.map),
    'mask_map50': float(results.seg.map50),
    'mask_map50_95': float(results.seg.map),
    'precision': float(results.box.mp),
    'recall': float(results.box.mr),
}

print(f'  Box mAP50: {metrics["box_map50"]:.4f}')
print(f'  Box mAP50-95: {metrics["box_map50_95"]:.4f}')
print(f'  Mask mAP50: {metrics["mask_map50"]:.4f}')
print(f'  Mask mAP50-95: {metrics["mask_map50_95"]:.4f}')
print(f'  Precision: {metrics["precision"]:.4f}')
print(f'  Recall: {metrics["recall"]:.4f}')

# Save results
output_path = 'analysis/final_clean_evaluation.json'
with open(output_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'\nSaved evaluation results to: {output_path}')

# Compare with previous models
print('\n=== Comparison with Previous Models ===')

# Load baseline comparison
with open('analysis/val_metrics_comparison.json') as f:
    prev = json.load(f)

print('Previous best:')
for model_name, m in prev.items():
    print(f'  {model_name}: mask mAP50={m["mask_map50"]:.4f}')

print(f'\nClean retrain: mask mAP50={metrics["mask_map50"]:.4f}')
print(f'  Improvement over baseline: {(metrics["mask_map50"] - prev["_best-seg"]["mask_map50"]):.4f}')
print(f'  Improvement over copy_paste_merged: {(metrics["mask_map50"] - prev["copy_paste_merged_best"]["mask_map50"]):.4f}')
