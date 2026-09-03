"""Apply two-stage filtering to clean model on session data."""
from ultralytics import YOLO
from pathlib import Path
import json
import cv2
import csv

# Model
model = YOLO('runs/segment/retrain_clean_20ep/weights/best.pt')
print(f'Model classes: {model.names}')

# Session data
session_dir = Path('data/_1-12-2026-09-01_10-42-53')
camera_dir = session_dir / 'Camera'
output_dir = session_dir / 'detection_queue_output_clean'
output_dir.mkdir(parents=True, exist_ok=True)

# Load GPS
gps_data = []
with open(session_dir / 'Location.csv', 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw_time = int(row['time'])
        timestamp_ms = raw_time // 1000000 if abs(raw_time) >= 10**17 else raw_time
        gps_data.append({
            'timestamp_ms': timestamp_ms,
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'speed': max(0.0, float(row.get('speed', 0.0) or 0.0))
        })

# Process images
images = sorted(camera_dir.glob('*.jpg'), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
print(f'Found {len(images)} images')

# Two-stage threshold: drain_area=0.20, drain_full=0.10
CONFIDENCE = 0.10
DA_THRESHOLD = 0.20  # drain_area threshold
DF_THRESHOLD = 0.10  # drain_full threshold

all_detections = []

def is_inside(inner_bbox, outer_bbox):
    """Check if inner bbox is inside outer bbox."""
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    # Predict
    results = model.predict(str(img_path), imgsz=640, conf=CONFIDENCE, verbose=False)
    
    # Parse detections
    drain_full_boxes = []
    drain_area_boxes = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(model.names[class_id])
            conf = float(box.conf[0].item())
            bbox = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            
            if class_name == 'drain_full' and conf >= DF_THRESHOLD:
                drain_full_boxes.append({'bbox': bbox, 'conf': conf})
            elif class_name == 'drain_area' and conf >= DA_THRESHOLD:
                drain_area_boxes.append({'bbox': bbox, 'conf': conf})
    
    # Stage 2: Geometric filter
    # If drain_full exists, keep drain_area only inside drain_full bbox
    filtered_detections = []
    
    if drain_full_boxes:
        for df_box in drain_full_boxes:
            filtered_detections.append({'class': 'drain_full', **df_box})
        
        # Check drain_area inside drain_full bbox
        df_bbox = drain_full_boxes[0]['bbox']
        for da_box in drain_area_boxes:
            if is_inside(da_box['bbox'], df_bbox):
                filtered_detections.append({'class': 'drain_area', **da_box})
    else:
        # No drain_full detected, keep drain_area as is
        for da_box in drain_area_boxes:
            filtered_detections.append({'class': 'drain_area', **da_box})
    
    # Draw and save
    for det in filtered_detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (0, 0, 255) if det['class'] == 'drain_full' else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{det['class']} {det['conf']:.2f}", (x1, max(65, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    cv2.imwrite(str(output_dir / img_path.name), img)
    
    all_detections.append({
        'image': img_path.name,
        'detections': filtered_detections,
        'drain_full_count': len(drain_full_boxes),
        'drain_area_count': len(drain_area_boxes),
        'filtered_count': len(filtered_detections)
    })
    
    class_counts = {}
    for d in filtered_detections:
        class_counts[d['class']] = class_counts.get(d['class'], 0) + 1
    
    print(f"{img_path.name}: drain_full={len(drain_full_boxes)}, drain_area={len(drain_area_boxes)}, filtered={len(filtered_detections)} classes={class_counts}")

# Summary
total_detections = sum(d['filtered_count'] for d in all_detections)
total_drain_full = sum(d['drain_full_count'] for d in all_detections)
total_drain_area = sum(d['drain_area_count'] for d in all_detections)

print(f'\n=== Summary (Two-Stage Filter) ===')
print(f'Total frames: {len(images)}')
print(f'Total detections: {total_detections}')
print(f'drain_full detections: {total_drain_full}')
print(f'drain_area detections: {total_drain_area}')
print(f'Frames with drain_full: {sum(1 for d in all_detections if d["drain_full_count"] > 0)}')
print(f'Frames with drain_area: {sum(1 for d in all_detections if d["drain_area_count"] > 0)}')

# Save results
output_path = 'analysis/clean_model_twostage_results.json'
with open(output_path, 'w') as f:
    json.dump(all_detections, f, indent=2, ensure_ascii=False)
print(f'\nResults saved to: {output_path}')