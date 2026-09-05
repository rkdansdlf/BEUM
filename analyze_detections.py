import json
import os
from collections import Counter

analysis_dir = r'C:\Project\BEUM\analysis'
det_file = os.path.join(analysis_dir, 'new_data_detections.json')

with open(det_file) as f:
    data = json.load(f)

print('=== Detection Summary ===')
print(f'Total frames: {len(data)}')

class_counter = Counter()
for result in data:
    for det in result.get('detections', []):
        class_name = det.get('class', 'unknown')
        class_counter[class_name] += 1

print(f'Class counts: {dict(class_counter)}')

print(f'\nSample first 3 frames:')
for i, result in enumerate(data[:3]):
    n_dets = len(result.get('detections', []))
    ts = result.get('timestamp', 'unknown')
    print(f'  Frame timestamp {ts}: {n_dets} detections')
    for det in result.get('detections', '')[:2]:
        print(f'    - {det["class"]}: {det["confidence"]} @ {det["bbox"]}')