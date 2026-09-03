#!/usr/bin/env python
"""Validate image/label pairs in a YOLO segmentation dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_split(dataset_root: Path, split: str, class_count: int) -> dict[str, object]:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    image_paths = sorted(
        path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    expected_labels = {path.relative_to(images_dir).with_suffix(".txt") for path in image_paths}
    actual_labels = {
        path.relative_to(labels_dir)
        for path in labels_dir.rglob("*.txt")
    } if labels_dir.exists() else set()
    missing = expected_labels - actual_labels
    extra = actual_labels - expected_labels
    classes: Counter[int] = Counter()
    invalid: list[str] = []
    empty = 0
    for relative_path in sorted(expected_labels & actual_labels):
        label_path = labels_dir / relative_path
        rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            empty += 1
            continue
        for row_number, values in enumerate(rows, start=1):
            if len(values) < 7 or (len(values) - 1) % 2:
                invalid.append(f"{label_path}:{row_number}: polygon requires at least 3 points")
                continue
            try:
                class_id = int(values[0])
                coordinates = [float(value) for value in values[1:]]
            except ValueError:
                invalid.append(f"{label_path}:{row_number}: non-numeric value")
                continue
            if not 0 <= class_id < class_count:
                invalid.append(f"{label_path}:{row_number}: class {class_id} is outside 0..{class_count - 1}")
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                invalid.append(f"{label_path}:{row_number}: coordinate is outside 0..1")
            classes[class_id] += 1
    return {
        "split": split,
        "images": len(image_paths),
        "labels": len(actual_labels),
        "missing": len(missing),
        "extra": len(extra),
        "empty": empty,
        "invalid": invalid,
        "classes": dict(sorted(classes.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--classes", type=int, default=5)
    args = parser.parse_args()
    failed = False
    for split in ("train", "val", "test"):
        result = validate_split(args.dataset_root, split, args.classes)
        print(
            f"{split}: images={result['images']} labels={result['labels']} "
            f"missing={result['missing']} extra={result['extra']} empty={result['empty']} "
            f"classes={result['classes']}"
        )
        invalid = result["invalid"]
        if invalid:
            failed = True
            for message in invalid[:20]:
                print(f"  {message}")
            if len(invalid) > 20:
                print(f"  ... {len(invalid) - 20} more invalid rows")
        if result["missing"] or result["extra"]:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
