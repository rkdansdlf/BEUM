import os, json, yaml

base = "waterDrain-1"

# Check valid annotations
valid_json = os.path.join(base, "valid", "_annotations.coco.json")
if os.path.exists(valid_json):
    with open(valid_json) as f:
        coco = json.load(f)
    print(f"Valid: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"Categories: {coco['categories']}")
    
    # Show first few annotations
    print("\nFirst 3 valid annotations:")
    for ann in coco['annotations'][:3]:
        print(f'  ID {ann["id"]}: category_id={ann["category_id"]}, image_id={ann["image_id"]}')

# Check test annotations  
test_json = os.path.join(base, "test", "_annotations.coco.json")
if os.path.exists(test_json):
    with open(test_json) as f:
        coco = json.load(f)
    print(f"\nTest: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"Categories: {coco['categories']}")

# Check all images across splits
all_images = []
for split in ["train", "valid", "test"]:
    json_path = os.path.join(base, split, "_annotations.coco.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            coco = json.load(f)
        all_images.extend([img["file_name"] for img in coco["images"]])
        
print(f"\nTotal unique images across all splits: {len(set(all_images))}")

# Check for drain_full class presence in train
train_has_drain_full = False
valid_has_drain_full = False
if os.path.exists(train_json):
    with open(train_json) as f:
        coco = json.load(f)
    train_has_drain_full = any(ann["category_id"] == 2 for ann in coco["annotations"]) if len(coco["categories"]) > 2 else False

if os.path.exists(valid_json):
    with open(valid_json) as f:
        coco = json.load(f)
    valid_has_drain_full = any(ann["category_id"] == 2 for ann in coco["annotations"]) if len(coco["categories"]) > 2 else False

print(f"\nTrain has drain_full (cat_id 2): {train_has_drain_full}")
print(f"Valid has drain_full (cat_id 2): {valid_has_drain_full}")