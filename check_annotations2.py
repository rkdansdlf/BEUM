import os, yaml, json

base = "waterDrain-1"

# Check data.yaml location
print("Contents of base:", os.listdir(base))

# Check if data.yaml is in a subdirectory
for root, dirs, files in os.walk(base):
    for f in files:
        if f == "data.yaml":
            path = os.path.join(root, f)
            print(f"Found data.yaml at: {path}")
            with open(path) as fh:
                data = yaml.safe_load(fh)
            print(f"  names: {data.get('names')}")
            print(f"  nc: {data.get('nc')}")

# Check train annotations
train_json = os.path.join(base, "train", "_annotations.coco.json")
if os.path.exists(train_json):
    with open(train_json) as f:
        coco = json.load(f)
    print(f"\nTrain: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"Categories: {coco['categories']}")