"""
Create clean dataset without Kaggle pothole contamination.
Uses only original gully-seg-3class dataset.
"""

import shutil
from pathlib import Path
import json

# Paths
LOCAL = Path("C:/Project/BEUM/dataset")
CLEAN = Path("C:/Project/BEUM/dataset/clean")

print("Creating clean dataset from original gully-seg-3class only...")
CLEAN.mkdir(parents=True, exist_ok=True)
for split in ["train", "val"]:
    (CLEAN / split / "images").mkdir(parents=True, exist_ok=True)
    (CLEAN / split / "labels").mkdir(parents=True, exist_ok=True)

# Copy original dataset (3 classes: object, drain_area, drain_full)
print("\n=== Copying original gully-seg-3class dataset ===")
for split in ["train", "val"]:
    coco_path = LOCAL / "labels" / split / "annotations.coco.json"
    if not coco_path.exists():
        print(f"  {split}: annotations.coco.json not found")
        continue
        
    coco = json.loads(coco_path.read_text())
    img_map = {img["id"]: img for img in coco["images"]}
    print(f"  {split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        fname = img_map[img_id]["file_name"]
        img_w = img_map[img_id]["width"]
        img_h = img_map[img_id]["height"]
        src = LOCAL / "images" / split / fname
        dst = CLEAN / split / "images" / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        seg = ann["segmentation"]
        if isinstance(seg, list) and len(seg) > 0:
            poly = seg[0] if isinstance(seg[0], list) else seg
            if len(poly) >= 6:
                lines = [str(int(ann["category_id"]))]
                for i in range(0, len(poly) - 1, 2):
                    x = poly[i] / img_w
                    y = poly[i + 1] / img_h
                    lines.append(f"{x:.6f} {y:.6f}")
                label_path = CLEAN / split / "labels" / (Path(fname).stem + ".txt")
                label_path.write_text(" ".join(lines) + "\n")

# Create YAML
yaml_content = """train: C:/Project/BEUM/dataset/clean/train/images
val: C:/Project/BEUM/dataset/clean/val/images

nc: 3
names: ['object', 'drain_area', 'drain_full']
"""
(CLEAN / "gully-seg-clean.yaml").write_text(yaml_content)
print(f"\nClean YAML: {CLEAN / 'gully-seg-clean.yaml'}")

# Summary
for split in ["train", "val"]:
    imgs = list((CLEAN / split / "images").glob("*.jpg"))
    lbls = list((CLEAN / split / "labels").glob("*.txt"))
    print(f"  {split}: {len(imgs)} images, {len(lbls)} labels")

print("\nDone. Clean dataset ready for training.")