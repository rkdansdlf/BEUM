#!/usr/bin/env python
"""
Build Canonical 3-Class Dataset for BEUM.
Converts verified COCO annotations from roboflow_downloads/v6 to YOLO segmentation format
with explicit class mapping:
  - 0: object
  - 1: drain_area
  - 2: drain_full
Generates dataset/canonical/ and dataset/canonical/canonical-3class.yaml.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path


CANONICAL_SCHEMA = {
    0: "object",
    1: "drain_area",
    2: "drain_full",
}

NAME_TO_CANONICAL_ID = {
    "object": 0,
    "drain_area": 1,
    "drain": 1,
    "drain_full": 2,
    "blockage_area": 2,
}


def coco_to_yolo_polygons(segmentation, width: int, height: int) -> list[list[float]]:
    """Normalizes COCO polygon coordinates to [0, 1]."""
    polys = []
    if isinstance(segmentation, list):
        for item in segmentation:
            if isinstance(item, list) and len(item) >= 6 and len(item) % 2 == 0:
                norm = []
                for i in range(0, len(item) - 1, 2):
                    x = min(1.0, max(0.0, float(item[i]) / width))
                    y = min(1.0, max(0.0, float(item[i + 1]) / height))
                    norm.extend([x, y])
                if len(norm) >= 6:
                    polys.append(norm)
    return polys


def build_canonical_dataset(
    source_dir: Path,
    dest_dir: Path,
) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    split_map = {
        "train": "train",
        "val": "valid",   # Roboflow export folder name
        "test": "test",
    }

    print(f"Building Canonical Dataset from {source_dir} -> {dest_dir}...")

    for canonical_split, roboflow_sub in split_map.items():
        src_split_dir = source_dir / roboflow_sub
        coco_json = src_split_dir / "_annotations.coco.json"

        dst_img_dir = dest_dir / "images" / canonical_split
        dst_lbl_dir = dest_dir / "labels" / canonical_split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not coco_json.is_file():
            print(f"  Warning: {coco_json} not found. Skipping {canonical_split}.")
            continue

        with open(coco_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        category_lookup = {c["id"]: c["name"] for c in coco.get("categories", [])}
        img_map = {img["id"]: img for img in coco.get("images", [])}

        # Group annotations by image_id
        anns_by_img = {}
        for ann in coco.get("annotations", []):
            iid = ann["image_id"]
            anns_by_img.setdefault(iid, []).append(ann)

        class_counts = Counter()
        written_labels = 0
        copied_images = 0

        for img_id, img_info in img_map.items():
            fname = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]

            src_img = src_split_dir / fname
            dst_img = dst_img_dir / fname

            if src_img.is_file() and not dst_img.is_file():
                shutil.copy2(src_img, dst_img)
            copied_images += 1

            lbl_lines = []
            img_anns = anns_by_img.get(img_id, [])

            for ann in img_anns:
                cat_id = ann.get("category_id")
                orig_name = category_lookup.get(cat_id, "")
                target_cls_id = NAME_TO_CANONICAL_ID.get(orig_name)

                if target_cls_id is None:
                    continue

                polys = coco_to_yolo_polygons(ann.get("segmentation"), img_w, img_h)
                for p in polys:
                    line = f"{target_cls_id} " + " ".join(f"{v:.6f}" for v in p)
                    lbl_lines.append(line)
                    class_counts[target_cls_id] += 1

            lbl_name = Path(fname).stem + ".txt"
            dst_lbl = dst_lbl_dir / lbl_name
            dst_lbl.write_text("\n".join(lbl_lines) + ("\n" if lbl_lines else ""), encoding="utf-8")
            written_labels += 1

        named_counts = {CANONICAL_SCHEMA.get(cid, f"cls_{cid}"): cnt for cid, cnt in sorted(class_counts.items())}
        stats[canonical_split] = {
            "images": copied_images,
            "labels": written_labels,
            "class_counts_by_id": dict(sorted(class_counts.items())),
            "class_counts_by_name": named_counts,
        }
        print(f"  [{canonical_split}] Images: {copied_images}, Labels: {written_labels}")
        print(f"         Class counts: {named_counts}")

    # Write Canonical YAML
    yaml_content = f"""path: {dest_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: object
  1: drain_area
  2: drain_full
"""
    yaml_path = dest_dir / "canonical-3class.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\nGenerated Canonical Dataset YAML: {yaml_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build canonical 3-class BEUM dataset from v6 COCO.")
    parser.add_argument("--source", type=str, default="roboflow_downloads/v6", help="Source directory")
    parser.add_argument("--dest", type=str, default="dataset/canonical", help="Destination directory")
    args = parser.parse_args()

    stats = build_canonical_dataset(Path(args.source), Path(args.dest))
    print("\nCanonical dataset build complete.")


if __name__ == "__main__":
    main()
