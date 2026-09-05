#!/usr/bin/env python
"""
P0.7-R2: Audit canonical label overlap geometry.
Quantifies actual polygon intersection, containment, and pixel loss ratio
under overlap_mask=True vs overlap_mask=False on the Canonical validation split.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import cv2
import numpy as np
import yaml
from ultralytics.data.augment import polygons2masks, polygons2masks_overlap

DATASET_VAL_IMG = Path("dataset/canonical/images/val")
DATASET_VAL_LBL = Path("dataset/canonical/labels/val")
OUTPUT_DIR = Path("analysis/overlap_mask_audit/geometry")
YAML_PATH = Path("dataset/canonical/canonical-3class.yaml")


def load_yaml_classes(yaml_path: Path) -> dict[int, str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    return {int(k): str(v) for k, v in names.items()}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    class_names = load_yaml_classes(YAML_PATH)

    val_labels = sorted(DATASET_VAL_LBL.glob("*.txt"))
    val_images = sorted(DATASET_VAL_IMG.glob("*.*"))

    val_image_count = len(val_labels)
    images_with_both_classes = 0
    images_with_actual_intersection = 0
    images_with_full_containment = 0

    invalid_labels = []
    per_image_results = []

    instances_losing_pixels_by_class = {0: 0, 1: 0, 2: 0}
    pixel_loss_ratios_by_class = {0: [], 1: [], 2: []}

    for lbl_file in val_labels:
        img_file = DATASET_VAL_IMG / f"{lbl_file.stem}.jpg"
        if not img_file.exists():
            img_file = DATASET_VAL_IMG / f"{lbl_file.stem}.png"
            if not img_file.exists():
                invalid_labels.append({"file": lbl_file.name, "error": "Missing corresponding image"})
                continue

        img = cv2.imread(str(img_file))
        if img is None:
            invalid_labels.append({"file": lbl_file.name, "error": "Cannot read image with cv2"})
            continue
        h, w = img.shape[:2]

        lines = [l.strip().split() for l in lbl_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        instances = []
        classes_present = set()

        for idx, parts in enumerate(lines):
            try:
                cid = int(parts[0])
            except ValueError:
                invalid_labels.append({"file": lbl_file.name, "line": idx, "error": "Non-integer class ID"})
                continue

            if cid not in class_names:
                invalid_labels.append({"file": lbl_file.name, "line": idx, "error": f"Class {cid} out of YAML range"})
                continue

            coords = [float(v) for v in parts[1:]]
            if len(coords) % 2 != 0:
                invalid_labels.append({"file": lbl_file.name, "line": idx, "error": "Odd number of coordinates"})
                continue

            if len(coords) < 6:
                invalid_labels.append({"file": lbl_file.name, "line": idx, "error": "Polygon has less than 3 points"})
                continue

            if any(math.isnan(c) or math.isinf(c) for c in coords):
                invalid_labels.append({"file": lbl_file.name, "line": idx, "error": "NaN or Inf in coordinates"})
                continue

            # Convert to absolute pixel coordinates
            pts = np.array([[coords[k] * w, coords[k + 1] * h] for k in range(0, len(coords) - 1, 2)], dtype=np.float32)

            # Area check
            area = cv2.contourArea(pts.astype(np.int32))
            if area <= 0:
                invalid_labels.append({"file": lbl_file.name, "line": idx, "error": f"Zero or negative area: {area}"})

            classes_present.add(cid)
            instances.append({
                "inst_idx": idx,
                "class_id": cid,
                "class_name": class_names[cid],
                "pts": pts,
                "contour_area": area,
            })

        has_both = (1 in classes_present and 2 in classes_present)
        if has_both:
            images_with_both_classes += 1

        if not instances:
            continue

        # Convert segments for Ultralytics rasterizers
        segments = [inst["pts"] for inst in instances]

        # 1. overlap_mask=False (independent masks)
        indep_masks = polygons2masks((h, w), segments, color=1, downsample_ratio=1)  # (N, H, W)

        # 2. overlap_mask=True (flattened overlap mask)
        overlap_map, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=1)  # (H, W), sorted_idx

        # Calculate actual intersection between class 1 and class 2
        c1_indices = [i for i, inst in enumerate(instances) if inst["class_id"] == 1]
        c2_indices = [i for i, inst in enumerate(instances) if inst["class_id"] == 2]

        image_has_actual_intersection = False
        image_has_full_containment = False
        pair_interactions = []

        for i1 in c1_indices:
            mask1 = indep_masks[i1] > 0
            area1 = mask1.sum()
            for i2 in c2_indices:
                mask2 = indep_masks[i2] > 0
                area2 = mask2.sum()

                inter_px = int(np.logical_and(mask1, mask2).sum())
                union_px = int(np.logical_or(mask1, mask2).sum())
                iou = (inter_px / union_px) if union_px > 0 else 0.0
                c1_contained = (inter_px / area1) if area1 > 0 else 0.0
                c2_contained = (inter_px / area2) if area2 > 0 else 0.0

                if inter_px > 0:
                    image_has_actual_intersection = True
                if c1_contained >= 0.95 or c2_contained >= 0.95:
                    image_has_full_containment = True

                pair_interactions.append({
                    "c1_idx": i1,
                    "c2_idx": i2,
                    "c1_area_px": int(area1),
                    "c2_area_px": int(area2),
                    "intersection_px": inter_px,
                    "union_px": union_px,
                    "iou": round(float(iou), 4),
                    "c1_contained_ratio": round(float(c1_contained), 4),
                    "c2_contained_ratio": round(float(c2_contained), 4),
                })

        if image_has_actual_intersection:
            images_with_actual_intersection += 1
        if image_has_full_containment:
            images_with_full_containment += 1

        # Calculate recovered masks and pixel loss for each instance
        instance_loss_records = []
        max_image_loss_ratio = 0.0

        for rank, orig_idx in enumerate(sorted_idx):
            inst = instances[orig_idx]
            cid = inst["class_id"]
            indep_area = float(indep_masks[orig_idx].sum())

            # In overlap_map, this instance corresponds to rank + 1
            recovered_mask = (overlap_map == (rank + 1))
            recovered_area = float(recovered_mask.sum())

            lost_pixels = max(0.0, indep_area - recovered_area)
            loss_ratio = (lost_pixels / indep_area) if indep_area > 0 else 0.0

            if lost_pixels > 0:
                instances_losing_pixels_by_class[cid] += 1

            pixel_loss_ratios_by_class[cid].append(loss_ratio)
            if loss_ratio > max_image_loss_ratio:
                max_image_loss_ratio = loss_ratio

            instance_loss_records.append({
                "orig_idx": orig_idx,
                "rank": rank,
                "class_id": cid,
                "class_name": inst["class_name"],
                "indep_area_px": int(indep_area),
                "recovered_area_px": int(recovered_area),
                "lost_pixels_px": int(lost_pixels),
                "pixel_loss_ratio": round(loss_ratio, 4),
            })

        per_image_results.append({
            "image": img_file.name,
            "img_h": h,
            "img_w": w,
            "instance_count": len(instances),
            "max_loss_ratio": max_image_loss_ratio,
            "instances": instance_loss_records,
            "pair_interactions": pair_interactions,
            "has_both_classes": has_both,
            "has_actual_intersection": image_has_actual_intersection,
            "has_full_containment": image_has_full_containment,
            "indep_masks": indep_masks,
            "overlap_map": overlap_map,
            "raw_img": img,
            "segments": segments,
        })

    # Sort images by max_loss_ratio descending for Visual QA panels
    per_image_results.sort(key=lambda x: x["max_loss_ratio"], reverse=True)

    # Generate Visual QA Panels for top 10 highest loss images
    print("Generating visual QA panels for top 10 pixel loss images...")
    for rank_qa, rec in enumerate(per_image_results[:10], start=1):
        img = rec["raw_img"]
        h, w = rec["img_h"], rec["img_w"]
        indep_masks = rec["indep_masks"]
        overlap_map = rec["overlap_map"]

        # Panel 1: Original with polygons
        p1 = img.copy()
        for inst in rec["instances"]:
            pts = rec["segments"][inst["orig_idx"]].astype(np.int32)
            color = (0, 255, 0) if inst["class_id"] == 1 else (0, 0, 255)  # Green=drain_area, Red=drain_full
            cv2.polylines(p1, [pts], isClosed=True, color=color, thickness=2)

        # Panel 2: overlap_mask=True GT
        p2 = img.copy()
        color_ov = np.zeros_like(img)
        # Class 1 color (greenish), Class 2 color (reddish)
        for inst in rec["instances"]:
            recovered = (overlap_map == (inst["rank"] + 1))
            c = (0, 200, 0) if inst["class_id"] == 1 else (0, 0, 220)
            color_ov[recovered] = c
        p2 = cv2.addWeighted(p2, 0.65, color_ov, 0.35, 0)

        # Panel 3: overlap_mask=False Class 1 (drain_area)
        p3 = img.copy()
        c1_mask = np.zeros((h, w), dtype=bool)
        for inst in rec["instances"]:
            if inst["class_id"] == 1:
                c1_mask = np.logical_or(c1_mask, indep_masks[inst["orig_idx"]] > 0)
        c1_vis = np.zeros_like(img)
        c1_vis[c1_mask] = (0, 220, 0)
        p3 = cv2.addWeighted(p3, 0.65, c1_vis, 0.35, 0)

        # Panel 4: overlap_mask=False Class 2 (drain_full)
        p4 = img.copy()
        c2_mask = np.zeros((h, w), dtype=bool)
        for inst in rec["instances"]:
            if inst["class_id"] == 2:
                c2_mask = np.logical_or(c2_mask, indep_masks[inst["orig_idx"]] > 0)
        c2_vis = np.zeros_like(img)
        c2_vis[c2_mask] = (0, 0, 220)
        p4 = cv2.addWeighted(p4, 0.65, c2_vis, 0.35, 0)

        # Panel 5: Difference (Lost pixels carved out by overlap_mask=True)
        p5 = img.copy()
        diff_mask = np.zeros((h, w), dtype=bool)
        for inst in rec["instances"]:
            orig_m = indep_masks[inst["orig_idx"]] > 0
            rec_m = (overlap_map == (inst["rank"] + 1))
            lost_m = np.logical_and(orig_m, ~rec_m)
            diff_mask = np.logical_or(diff_mask, lost_m)
        diff_vis = np.zeros_like(img)
        diff_vis[diff_mask] = (255, 255, 0)  # Cyan/Yellow highlighting lost pixels
        p5 = cv2.addWeighted(p5, 0.5, diff_vis, 0.5, 0)

        # Banner for each panel
        target_w = 320
        target_h = int(h * (target_w / w))

        def prep_panel(panel, title):
            res = cv2.resize(panel, (target_w, target_h))
            header = np.zeros((30, target_w, 3), dtype=np.uint8)
            cv2.putText(header, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            return np.vstack([header, res])

        comp = np.hstack([
            prep_panel(p1, "1. Polygons (G:area, R:full)"),
            prep_panel(p2, "2. True GT (Donut Hole)"),
            prep_panel(p3, "3. False GT: drain_area"),
            prep_panel(p4, "4. False GT: drain_full"),
            prep_panel(p5, f"5. Lost Pixels ({rec['max_loss_ratio']*100:.1f}%)")
        ])

        out_name = f"{rank_qa:03d}_{rec['image'].replace('.jpg', '')}_diff.jpg"
        cv2.imwrite(str(OUTPUT_DIR / out_name), comp)

    # Clean up non-serializable objects before JSON dump
    for r in per_image_results:
        del r["indep_masks"]
        del r["overlap_map"]
        del r["raw_img"]
        del r["segments"]

    def calc_stats(lst):
        if not lst:
            return {"median": 0.0, "max": 0.0, "mean": 0.0}
        arr = np.array(lst)
        return {
            "median": round(float(np.median(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "mean": round(float(np.mean(arr)), 4),
        }

    summary = {
        "val_image_count": val_image_count,
        "images_with_both_classes": images_with_both_classes,
        "images_with_actual_intersection": images_with_actual_intersection,
        "images_with_full_containment": images_with_full_containment,
        "invalid_labels_count": len(invalid_labels),
        "invalid_labels": invalid_labels,
        "instances_losing_pixels_by_class": {
            class_names.get(c, str(c)): count for c, count in instances_losing_pixels_by_class.items()
        },
        "pixel_loss_stats_by_class": {
            class_names.get(c, str(c)): calc_stats(lst) for c, lst in pixel_loss_ratios_by_class.items()
        },
        "per_image_results": per_image_results
    }

    out_json = Path("analysis/overlap_mask_audit/canonical_overlap_geometry.json")
    def np_encoder(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    out_json.write_text(json.dumps(summary, indent=2, default=np_encoder), encoding="utf-8")
    print(f"\nAudit complete! Saved to {out_json}")
    print(f"PASS — CANONICAL_OVERLAP_GEOMETRY_QUANTIFIED")
    print(f"Summary:")
    print(f"  val_image_count: {val_image_count}")
    print(f"  images_with_both_classes: {images_with_both_classes}")
    print(f"  images_with_actual_intersection: {images_with_actual_intersection}")
    print(f"  images_with_full_containment: {images_with_full_containment}")
    print(f"  instances_losing_pixels: {summary['instances_losing_pixels_by_class']}")
    print(f"  pixel_loss_stats: {summary['pixel_loss_stats_by_class']}")


if __name__ == "__main__":
    main()
