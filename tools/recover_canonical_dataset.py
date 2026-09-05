#!/usr/bin/env python
"""
P0.6 — Canonical Dataset Recovery and Whole-Project Annotation Audit.
Scans the entire repository for all YOLO label directories and COCO JSON annotation files,
counts class distributions (class 0, 1, 2, ...), identifies original drain_area/drain_full sources,
and outputs a comprehensive canonical dataset recovery plan.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import yaml


PROJECT_ROOT = Path("C:/Project/BEUM")


def audit_coco_file(coco_path: Path) -> dict:
    try:
        data = json.loads(coco_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"path": str(coco_path), "error": str(e)}

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    class_counts = Counter()
    for ann in annotations:
        class_counts[ann.get("category_id")] += 1

    named_counts = {categories.get(cid, f"id_{cid}"): cnt for cid, cnt in class_counts.items()}

    return {
        "path": str(coco_path.relative_to(PROJECT_ROOT)),
        "type": "coco_json",
        "image_count": len(images),
        "annotation_count": len(annotations),
        "categories": categories,
        "class_counts_by_id": dict(sorted(class_counts.items())),
        "class_counts_by_name": named_counts,
    }


def audit_yolo_dir(labels_dir: Path, images_dir: Path | None = None) -> dict:
    label_files = list(labels_dir.glob("*.txt"))
    class_counts = Counter()
    empty_labels = 0
    corrupted_labels = 0
    total_annotations = 0

    for lf in label_files:
        content = lf.read_text(encoding="utf-8").strip()
        if not content:
            empty_labels += 1
            continue

        lines = content.splitlines()
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
                class_counts[cid] += 1
                total_annotations += 1
            except ValueError:
                corrupted_labels += 1

    img_count = len(list(images_dir.glob("*"))) if (images_dir and images_dir.is_dir()) else len(label_files)

    return {
        "path": str(labels_dir.relative_to(PROJECT_ROOT)),
        "type": "yolo_txt",
        "image_count": img_count,
        "label_files": len(label_files),
        "empty_labels": empty_labels,
        "corrupted_labels": corrupted_labels,
        "total_annotations": total_annotations,
        "class_counts_by_id": dict(sorted(class_counts.items())),
    }


def main():
    print("=" * 85)
    print("           P0.6 CANONICAL DATASET RECOVERY & REPO-WIDE ANNOTATION AUDIT           ")
    print("=" * 85)

    all_coco_files = sorted(PROJECT_ROOT.rglob("*annotation*.json")) + sorted(PROJECT_ROOT.rglob("*.coco.json"))
    coco_results = [audit_coco_file(p) for p in set(all_coco_files)]

    all_yolo_dirs = []
    for p in PROJECT_ROOT.rglob("labels"):
        if p.is_dir():
            all_yolo_dirs.append(p)
            for sub in p.iterdir():
                if sub.is_dir():
                    all_yolo_dirs.append(sub)

    yolo_results = []
    for ydir in sorted(set(all_yolo_dirs)):
        # Check if it directly contains txt files
        txt_count = len(list(ydir.glob("*.txt")))
        if txt_count > 0:
            # find parallel images dir
            parts = list(ydir.parts)
            img_dir = None
            if "labels" in parts:
                idx = parts.index("labels")
                parts[idx] = "images"
                candidate_img = Path(*parts)
                if candidate_img.is_dir():
                    img_dir = candidate_img
            yolo_results.append(audit_yolo_dir(ydir, img_dir))

    # Print COCO Audits
    print("\n[1. COCO JSON ANNOTATION DATASETS FOUND]")
    print("-" * 85)
    for c in coco_results:
        print(f"Path: {c['path']}")
        print(f"  Images: {c.get('image_count')}, Annotations: {c.get('annotation_count')}")
        print(f"  Categories: {c.get('categories')}")
        print(f"  Class Counts: {c.get('class_counts_by_name')}")
        print()

    # Print YOLO Audits
    print("\n[2. YOLO TXT LABEL DIRECTORIES FOUND]")
    print("-" * 85)
    print("Directory".ljust(45), "Labels".rjust(8), "Empty".rjust(8), "Class ID Counts".rjust(22))
    print("-" * 85)
    for y in yolo_results:
        print(
            y["path"][:44].ljust(45),
            str(y["label_files"]).rjust(8),
            str(y["empty_labels"]).rjust(8),
            str(y["class_counts_by_id"]).rjust(22),
        )
    print("=" * 85)

    recovery_report = {
        "coco_datasets": coco_results,
        "yolo_datasets": yolo_results,
    }

    out_path = PROJECT_ROOT / "analysis" / "canonical_dataset_recovery.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recovery_report, f, indent=2, ensure_ascii=False)
    print(f"\nFull recovery audit saved to {out_path}")


if __name__ == "__main__":
    main()
