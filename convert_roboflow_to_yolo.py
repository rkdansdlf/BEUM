"""Convert Roboflow COCO-segmentation data to YOLO format for fine-tuning.
Also tracks batch names to exclude previously fetched data.
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

# Load .env - find it in the project root
PROJECT_ROOT = Path("C:/Project/BEUM").resolve()
load_dotenv(PROJECT_ROOT / ".env")

ROBOFLOW_API = os.environ.get("ROBOFLOW_API")
if not ROBOFLOW_API:
    raise RuntimeError("ROBOFLOW_API is not set in .env or the environment")


def coco_to_yolo_annotation(segmentation, img_width, img_height):
    """Convert COCO polygon segmentation to YOLO format (normalized)."""
    yolo_annotations = []
    for poly in segmentation:
        if isinstance(poly, list) and len(poly) >= 6:
            # Normalize coordinates
            poly_norm = []
            for i in range(0, len(poly) - 1, 2):
                x = poly[i] / img_width
                y = poly[i + 1] / img_height
                poly_norm.extend([x, y])

            # Skip if not enough points
            if len(poly_norm) < 6:  # need at least 3 points (6 values)
                continue

            yolo_annotations.append(poly_norm)
    return yolo_annotations


def download_and_convert_roboflow(
    project_name="waterdrain",
    version=None,
    batch_name="field_blockage_v1",
    exclude_batch=None,
    download_dir="roboflow_downloads",
):
    """Download data from Roboflow and convert to YOLO format.

    Args:
        project_name: Roboflow project name
        version: specific version number (None = latest)
        batch_name: batch name for tracking
        exclude_batch: batch name to exclude (for avoiding duplicate data)
        download_dir: base directory for downloads
    """

    from roboflow import Roboflow

    rf = Roboflow(ROBOFLOW_API)
    project = rf.workspace().project(project_name)

    # If excluding a batch, check what versions/batches were previously used
    if exclude_batch:
        print(f"Excluding previous batch: {exclude_batch}")
        # List all versions to see what's been used
        versions = project.versions()
        version_ids = [v.id for v in versions]
        print(f"Available versions: {version_ids}")

        # If a specific version was previously used, skip it
        if version and version in version_ids:
            print(f"Version {version} was previously used, skipping...")
            return None

    # Download the dataset
    if version:
        latest = project.version(version)
        print(f"Downloading version {version}...")
    else:
        versions = project.versions()
        latest = versions[0]  # latest version
        print(f"Using latest version: {latest.id}")

    dataset = latest.download("coco-segmentation", location=download_dir)

    # Create YOLO format directory structure
    yolo_root = Path("dataset/roboflow_finetune")
    yolo_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (yolo_root / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_root / split / "labels").mkdir(parents=True, exist_ok=True)

    # The download creates data directly in download_dir with train/valid/test subdirs
    # and data.yaml at the top level
    data_dir = download_dir
    print(f"Using download directory as data directory: {data_dir}")

    # =========================
    # Phase 1: Collect all class names
    # =========================
    class_names = set()

    for split in ["train", "val", "test"]:
        json_path = os.path.join(data_dir, split, "_annotations.coco.json")

        if not os.path.exists(json_path):
            print(f"  {split}: no annotations found")
            continue

        with open(json_path) as f:
            coco = json.load(f)

        for ann in coco["annotations"]:
            cat_id = ann["category_id"]
            cat = next(c for c in coco["categories"] if c["id"] == cat_id)
            class_names.add(cat["name"])

    print(f"\nCollected class names: {sorted(class_names)}")

    # Create class ID mapping (0-indexed)
    class_ids = {name: i for i, name in enumerate(sorted(class_names))}
    print(f"Class ID mapping: {class_ids}")

    # =========================
    # Phase 2: Create data.yaml
    # =========================
    # Determine final class names (ensure drain_full is included if needed)
    needed_classes = ["object", "drain_area", "drain_full"]
    final_class_names = []
    for c in needed_classes:
        if c in class_names:
            final_class_names.append(c)
        else:
            # Add as new class with next available id
            final_class_names.append(c)

    nc = len(final_class_names)

    # Train/val/test paths
    yaml_content = f"""train: C:/Project/BEUM/dataset/roboflow_finetune/train/images
val: C:/Project/BEUM/dataset/roboflow_finetune/val/images
test: C:/Project/BEUM/dataset/roboflow_finetune/test/images

nc: {nc}
names: {final_class_names}
"""
    yaml_path = yolo_root / "gully-seg-roboflow.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nCreated YAML: {yaml_path}")
    print(f"Final class names: {final_class_names}")
    print(f"NC: {nc}")

    # =========================
    # Phase 3: Copy images and create labels
    # =========================

    for split in ["train", "val", "test"]:
        json_path = os.path.join(data_dir, split, "_annotations.coco.json")
        src_img_dir = os.path.join(data_dir, split)
        dst_img_dir = yolo_root / split / "images"
        dst_lbl_dir = yolo_root / split / "labels"

        if not os.path.exists(json_path):
            print(f"  {split}: no annotations found")
            continue

        with open(json_path) as f:
            coco = json.load(f)

        img_map = {img["id"]: img for img in coco["images"]}
        print(f"  {split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

        # Copy images and create labels
        for img in coco["images"]:
            img_id = img["id"]
            if img_id not in img_map:
                continue

            img_info = img_map[img_id]
            fname = img_info["file_name"]
            img_width = img_info["width"]
            img_height = img_info["height"]

            # Copy image file
            src_img = os.path.join(src_img_dir, fname)
            dst_img = dst_img_dir / fname
            if os.path.exists(src_img) and not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # Get annotations for this image
            img_anns = [ann for ann in coco["annotations"] if ann["image_id"] == img_id]

            # Get class ID
            if img_anns:
                cat_id = img_anns[0]["category_id"]
                class_name = next(c for c in coco["categories"] if c["id"] == cat_id)["name"]
                class_id = class_ids[class_name]

                # Convert segmentations to YOLO format
                segmentations = []
                for ann in img_anns:
                    seg = coco_to_yolo_annotation(ann["segmentation"], img_width, img_height)
                    segmentations.extend(seg)

                if not segmentations:
                    continue

                # Write YOLO label file
                label_name = Path(fname).stem + ".txt"
                label_path = dst_lbl_dir / label_name

                # Read existing annotations and append
                lines = []
                if label_path.exists():
                    existing = label_path.read_text(encoding="utf-8").strip()
                    if existing:
                        lines = existing.split("\n")

                # Add new annotations with numeric class ID
                for seg in segmentations:
                    # Format: class_id x1 y1 x2 y2 x3 y3 ...
                    line = f"{class_id}" + " " + " ".join(f"{v:.6f}" for v in seg)
                    lines.append(line)

                label_path.write_text("\n".join(lines) + "\n" if lines else "")

    # =========================
    # Summary
    # =========================
    for split in ["train", "val", "test"]:
        img_dir = yolo_root / split / "images"
        lbl_dir = yolo_root / split / "labels"
        if img_dir.exists():
            imgs = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            lbls = len([f for f in lbl_dir.glob("*.txt")]) if lbl_dir.exists() else 0
            print(f"  {split}: {imgs} images, {lbls} labels")

    return {
        "yaml_path": str(yaml_path),
        "class_names": final_class_names,
        "nc": nc,
        "batch_name": batch_name,
        "train_images": len([f for f in os.listdir(yolo_root / "train" / "images")]) if (yolo_root / "train" / "images").exists() else 0
    }


if __name__ == "__main__":
    # Example usage: download version 1, excluding previous batch if needed
    result = download_and_convert_roboflow(
        project_name="waterdrain",
        version=1,
        batch_name="gully_v1",
        exclude_batch=None,  # Set to previous batch name to exclude
        download_dir="roboflow_downloads"
    )

    if result:
        print(f"\nSuccess! Trained on {result['train_images']} images")
        print(f"Dataset: {result['yaml_path']}")