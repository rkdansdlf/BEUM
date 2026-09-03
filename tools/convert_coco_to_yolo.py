#!/usr/bin/env python
"""Convert COCO polygon annotations to YOLO segmentation labels.

Every source category must be mapped explicitly to a target YOLO class. This
prevents a three-class export from being silently interpreted as the five
runtime classes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _parse_mapping(values: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid mapping {value!r}; use CATEGORY=CLASS_ID")
        category, raw_class_id = value.split("=", 1)
        category = category.strip()
        if not category:
            raise ValueError(f"Invalid mapping {value!r}; category is empty")
        mapping[category] = int(raw_class_id)
    return mapping


def _polygon_segments(segmentation: Any) -> list[list[float]]:
    if isinstance(segmentation, dict):
        raise ValueError("COCO RLE segmentation is not supported; export polygons instead")
    if not isinstance(segmentation, list) or not segmentation:
        return []
    if all(isinstance(value, (int, float)) for value in segmentation):
        return [segmentation]
    return [segment for segment in segmentation if isinstance(segment, list)]


def _normalise_polygon(segment: list[float], width: int, height: int) -> str | None:
    if len(segment) < 6 or len(segment) % 2:
        return None
    values: list[str] = []
    for offset in range(0, len(segment), 2):
        x = float(segment[offset])
        y = float(segment[offset + 1])
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        x = min(1.0, max(0.0, x / width))
        y = min(1.0, max(0.0, y / height))
        values.extend((f"{x:.6f}", f"{y:.6f}"))
    return " ".join(values)


def convert_coco_to_yolo(
    coco_json_path: str | Path,
    output_dir: str | Path,
    category_map: dict[str, int],
) -> int:
    """Convert COCO polygon annotations and return the number of images written."""
    with Path(coco_json_path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = {int(category["id"]): str(category["name"]) for category in data["categories"]}
    used_category_ids = {int(annotation["category_id"]) for annotation in data["annotations"]}
    missing = [
        categories[category_id]
        for category_id in sorted(used_category_ids)
        if categories.get(category_id) not in category_map
    ]
    if missing:
        raise ValueError(f"Missing explicit class mappings for: {', '.join(missing)}")

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in data["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    written = 0
    skipped_polygons = 0
    for image in data["images"]:
        image_id = int(image["id"])
        label_path = output_dir / f"{Path(image['file_name']).stem}.txt"
        rows: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            category_name = categories[int(annotation["category_id"])]
            class_id = category_map[category_name]
            segments = _polygon_segments(annotation.get("segmentation"))
            for segment in segments:
                polygon = _normalise_polygon(segment, int(image["width"]), int(image["height"]))
                if polygon is None:
                    skipped_polygons += 1
                    continue
                rows.append(f"{class_id} {polygon}")
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        written += 1

    if skipped_polygons:
        print(f"Skipped invalid polygons: {skipped_polygons}")
    print(f"Created {written} label files in {output_dir}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coco_json_path")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--map",
        dest="mappings",
        action="append",
        required=True,
        metavar="CATEGORY=CLASS_ID",
        help="Explicit source category to target YOLO class mapping; repeat for each category",
    )
    args = parser.parse_args()
    convert_coco_to_yolo(args.coco_json_path, args.output_dir, _parse_mapping(args.mappings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
