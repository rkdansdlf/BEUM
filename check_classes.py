from pathlib import Path
import json

local = Path('dataset')
for split in ['train', 'val']:
    coco_path = local / 'labels' / split / 'annotations.coco.json'
    if coco_path.exists():
        coco = json.loads(coco_path.read_text())
        class_counts = {}
        for ann in coco['annotations']:
            cls = ann['category_id']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        categories = coco.get('categories', [])
        print(f'{split}: {len(coco["images"])} images, {len(coco["annotations"])} annotations')
        print(f'  Classes: {class_counts}')
        print(f'  Names: {categories}')
    else:
        print(f'{split}: annotations.coco.json not found')
