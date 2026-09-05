#!/usr/bin/env python
"""
P0.7-B: Full Visual and Geometric QA on all 32 drain_full instances in Canonical Val.
Calculates Box IoU and Mask IoU for every instance, generates overlay visualizations
(GT polygon in green, Predicted mask in red/blue), and categorizes failure modes:
  A: GT polygon valid / Prediction mask poor
  B: GT polygon corrupted / degenerated
  C: Class misclassified / confused
  D: Severe mask geometric divergence (high box IoU, low mask IoU)
  E: Small/occluded object
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


DATASET_VAL_IMG = Path("dataset/canonical/images/val")
DATASET_VAL_LBL = Path("dataset/canonical/labels/val")
OUTPUT_DIR = Path("analysis/drain_full_mask_qa")
MODEL_PATH = Path("models/best-seg.pt")


def compute_polygon_area_and_bbox(poly_pts: np.ndarray, img_w: int, img_h: int) -> tuple[float, list[int]]:
    x1, y1 = np.min(poly_pts[:, 0]), np.min(poly_pts[:, 1])
    x2, y2 = np.max(poly_pts[:, 0]), np.max(poly_pts[:, 1])
    area = cv2.contourArea(poly_pts.astype(np.int32))
    return area, [int(x1), int(y1), int(x2), int(y2)]


def compute_box_iou(box1: list[int], box2: list[int]) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    boxBArea = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0


def compute_mask_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    intersection = np.logical_and(gt_mask > 0, pred_mask > 0).sum()
    union = np.logical_or(gt_mask > 0, pred_mask > 0).sum()
    return float(intersection / union) if union > 0 else 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    # Find all validation images with drain_full (class 2)
    val_labels = sorted(DATASET_VAL_LBL.glob("*.txt"))
    drain_full_instances = []

    for lbl_file in val_labels:
        img_file = DATASET_VAL_IMG / f"{lbl_file.stem}.jpg"
        if not img_file.is_file():
            # Try png
            img_file = DATASET_VAL_IMG / f"{lbl_file.stem}.png"
            if not img_file.is_file():
                continue

        lines = [line.split() for line in lbl_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for inst_idx, parts in enumerate(lines):
            cid = int(parts[0])
            if cid == 2:  # drain_full
                coords = [float(v) for v in parts[1:]]
                drain_full_instances.append({
                    "lbl_file": lbl_file.name,
                    "img_file": img_file,
                    "inst_idx": inst_idx,
                    "norm_coords": coords,
                })

    print(f"Found {len(drain_full_instances)} drain_full instances in {DATASET_VAL_LBL}")

    qa_results = []

    for i, item in enumerate(drain_full_instances, start=1):
        img = cv2.imread(str(item["img_file"]))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Convert GT coords to pixel polygon
        raw_coords = item["norm_coords"]
        gt_pts = np.array([[raw_coords[k] * w, raw_coords[k + 1] * h] for k in range(0, len(raw_coords) - 1, 2)], dtype=np.float32)
        gt_area, gt_bbox = compute_polygon_area_and_bbox(gt_pts, w, h)

        # Create GT binary mask
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(gt_mask, [gt_pts.astype(np.int32)], 1)

        # Run inference
        pred = model.predict(img, imgsz=640, conf=0.05, verbose=False)[0]

        best_box_iou = 0.0
        best_mask_iou = 0.0
        best_pred_info = {"cls": None, "conf": 0.0, "has_mask": False}
        matched_pred_mask = None

        if pred.boxes is not None and len(pred.boxes) > 0:
            for b_idx, box in enumerate(pred.boxes):
                p_cls = int(box.cls[0])
                p_conf = float(box.conf[0])
                xyxy = [int(v) for v in box.xyxy[0].tolist()]

                b_iou = compute_box_iou(gt_bbox, xyxy)

                p_mask = None
                m_iou = 0.0
                if pred.masks is not None and b_idx < len(pred.masks.data):
                    # Resize pred mask to image size
                    raw_m = pred.masks.data[b_idx].cpu().numpy()
                    p_mask = cv2.resize(raw_m, (w, h), interpolation=cv2.INTER_NEAREST)
                    p_mask = (p_mask > 0.5).astype(np.uint8)
                    m_iou = compute_mask_iou(gt_mask, p_mask)

                if b_iou > best_box_iou:
                    best_box_iou = b_iou
                    best_mask_iou = m_iou
                    best_pred_info = {
                        "cls": model.names.get(p_cls, str(p_cls)),
                        "cls_id": p_cls,
                        "conf": round(p_conf, 3),
                        "has_mask": p_mask is not None,
                    }
                    matched_pred_mask = p_mask

        # Failure mode classification
        if best_box_iou >= 0.50 and best_mask_iou < 0.20:
            category = "D_HIGH_BOX_LOW_MASK"
        elif best_box_iou < 0.30:
            category = "C_MISSED_OR_LOW_OVERLAP"
        elif best_pred_info["cls"] == "drain_area":
            category = "C_CONFUSED_AS_DRAIN_AREA"
        elif best_mask_iou >= 0.40:
            category = "PASSED_GOOD_MASK"
        else:
            category = "A_PRED_MASK_POOR"

        # Generate Visual QA Overlay
        vis_img = img.copy()

        # Draw GT polygon in GREEN
        cv2.polylines(vis_img, [gt_pts.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw Matched Pred Mask in BLUE/RED
        if matched_pred_mask is not None and matched_pred_mask.sum() > 0:
            color_mask = np.zeros_like(vis_img)
            # Red if confused, Cyan if drain_full
            m_color = (255, 200, 0) if best_pred_info["cls"] == "drain_full" else (0, 165, 255)
            color_mask[matched_pred_mask > 0] = m_color
            vis_img = cv2.addWeighted(vis_img, 0.75, color_mask, 0.25, 0)

        # Add text banner
        text1 = f"Inst #{i:02d} | GT: drain_full (area={int(gt_area)}px)"
        text2 = f"Pred: {best_pred_info['cls']} ({best_pred_info['conf']}) | Box IoU: {best_box_iou:.3f} | Mask IoU: {best_mask_iou:.3f} | [{category}]"
        cv2.rectangle(vis_img, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.putText(vis_img, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(vis_img, text2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255) if "HIGH_BOX" in category else (255, 255, 255), 1)

        out_img_name = f"inst_{i:02d}_{category}_{item['img_file'].stem}.jpg"
        out_img_path = OUTPUT_DIR / out_img_name
        cv2.imwrite(str(out_img_path), vis_img)

        qa_results.append({
            "instance_id": i,
            "image": item["img_file"].name,
            "gt_polygon_points": len(gt_pts),
            "gt_area_px": round(gt_area, 1),
            "best_box_iou": round(best_box_iou, 4),
            "best_mask_iou": round(best_mask_iou, 4),
            "pred_class": best_pred_info["cls"],
            "pred_conf": best_pred_info["conf"],
            "category": category,
            "output_image": out_img_name,
        })

    # Summary analysis
    categories_counter = {}
    for r in qa_results:
        cat = r["category"]
        categories_counter[cat] = categories_counter.get(cat, 0) + 1

    avg_box_iou = sum(r["best_box_iou"] for r in qa_results) / len(qa_results) if qa_results else 0.0
    avg_mask_iou = sum(r["best_mask_iou"] for r in qa_results) / len(qa_results) if qa_results else 0.0

    report = {
        "total_drain_full_instances": len(qa_results),
        "average_box_iou": round(avg_box_iou, 4),
        "average_mask_iou": round(avg_mask_iou, 4),
        "failure_mode_breakdown": categories_counter,
        "instances": qa_results,
    }

    report_json = OUTPUT_DIR / "drain_full_qa_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("             DRAIN_FULL MASK GEOMETRY QA REPORT (ALL 32 INSTANCES)             ")
    print("=" * 80)
    print(f"Total Val drain_full instances audited: {len(qa_results)}")
    print(f"Average Box  IoU: {avg_box_iou:.4f}")
    print(f"Average Mask IoU: {avg_mask_iou:.4f}")
    print("\nFailure Mode Breakdown:")
    for cat, cnt in sorted(categories_counter.items()):
        print(f"  • {cat.ljust(30)}: {cnt:2d} instances ({cnt/len(qa_results)*100:.1f}%)")
    print("=" * 80)
    print(f"Visual QA overlay images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
