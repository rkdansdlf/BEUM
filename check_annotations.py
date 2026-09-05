import yaml, json, os

# Check data.yaml
with open('waterDrain-1/data.yaml') as f:
    data = yaml.safe_load(f)
print('data.yaml:')
print(f'  names: {data.get("names")}')
print(f'  nc: {data.get("nc")}')

# Check train annotations
with open('waterDrain-1/train/_annotations.coco.json') as f:
    coco = json.load(f)
print(f'\nTrain: {len(coco["images"])} images, {len(coco["annotations"])} annotations')
print(f'Categories: {coco["categories"]}')

# Show first few annotations
print('\nFirst 3 annotations:')
for ann in coco['annotations'][:3]:
    print(f'  ID {ann["id"]}: category_id={ann["category_id"]}, image_id={ann["image_id"]}, segmentation count={len(ann["segmentation"])}')