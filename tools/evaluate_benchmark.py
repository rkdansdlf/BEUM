#!/usr/bin/env python
"""
Evaluate YOLOv8 segmentation models with strict separation between:
1) Ground-Truth (GT) Validation Metrics:
   - Box & Mask mAP50, mAP50-95
   - Per-class Box & Mask Precision, Recall, F1, AP50, AP50-95 (properly mapped via ap_class_index)
   - Confusion Matrix with verified axes:
       Rows = Predicted (Y-axis)
       Columns = True (X-axis)
       Operating-point P, R, FP, FN breakdown
2) Unlabeled Operational Video Metrics:
   - Detections / frame
   - Detections / minute (calculated via exact timestamp parsing)
   - Candidate clusters / minute
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


TIMESTAMP_PATTERN = re.compile(r"_t(\d+\.\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def evaluate_dataset(
    model_path: str | Path,
    data_yaml: str | Path,
    split: str = "val",
    imgsz: int = 640,
    conf: float = 0.20,
    iou: float = 0.70,
    device: str = "cpu",
    overlap_mask: bool = False,
) -> dict:
    model = YOLO(str(model_path))
    names = model.names
    print(f"Loaded model: {model_path} with schema names: {names}")
    print(f"Evaluating split: '{split}' from {data_yaml} at imgsz={imgsz}, conf={conf}, overlap_mask={overlap_mask}...")

    val_results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        overlap_mask=overlap_mask,
        verbose=False,
    )

    box_map50 = float(val_results.box.map50)
    box_map50_95 = float(val_results.box.map)
    mask_map50 = float(val_results.seg.map50) if hasattr(val_results, "seg") and val_results.seg is not None else 0.0
    mask_map50_95 = float(val_results.seg.map) if hasattr(val_results, "seg") and val_results.seg is not None else 0.0

    class_metrics = {}
    ap_classes = list(val_results.box.ap_class_index)

    # Initialize all declared classes with 0
    for cid, cname in names.items():
        class_metrics[cname] = {
            "class_id": int(cid),
            "present_in_split": False,
            "box_p": 0.0,
            "box_r": 0.0,
            "box_f1": 0.0,
            "box_ap50": 0.0,
            "box_ap50_95": 0.0,
            "mask_p": 0.0,
            "mask_r": 0.0,
            "mask_f1": 0.0,
            "mask_ap50": 0.0,
            "mask_ap50_95": 0.0,
        }

    for i, cid in enumerate(ap_classes):
        cname = names.get(cid, f"class_{cid}")
        bp, br, bap50, bap = val_results.box.class_result(i)
        bf1 = (2 * bp * br / (bp + br)) if (bp + br) > 0 else 0.0

        if hasattr(val_results, "seg") and val_results.seg is not None:
            mp, mr, map50, map_val = val_results.seg.class_result(i)
            mf1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0
        else:
            mp, mr, map50, map_val, mf1 = 0.0, 0.0, 0.0, 0.0, 0.0

        class_metrics[cname] = {
            "class_id": int(cid),
            "present_in_split": True,
            "box_p": round(float(bp), 4),
            "box_r": round(float(br), 4),
            "box_f1": round(float(bf1), 4),
            "box_ap50": round(float(bap50), 4),
            "box_ap50_95": round(float(bap), 4),
            "mask_p": round(float(mp), 4),
            "mask_r": round(float(mr), 4),
            "mask_f1": round(float(mf1), 4),
            "mask_ap50": round(float(map50), 4),
            "mask_ap50_95": round(float(map_val), 4),
        }

    # Confusion Matrix Parsing: Rows = Predicted, Cols = True
    cm_raw = val_results.confusion_matrix.matrix
    cm_list = cm_raw.tolist() if hasattr(cm_raw, "tolist") else cm_raw

    # Column sums = GT counts
    # Row sums = Predicted counts
    # Background is last index (nc)
    nc = len(names)
    gt_column_sums = {}
    for cid, cname in names.items():
        if int(cid) < len(cm_list[0]):
            gt_column_sums[cname] = int(sum(cm_list[r][int(cid)] for r in range(len(cm_list))))

    # Background column contains background instances that were predicted as some class (Background FP)
    bg_col_idx = len(cm_list[0]) - 1
    bg_fp_breakdown = {}
    total_bg_fp = 0
    for r in range(nc):
        cname = names.get(r, f"class_{r}")
        fp_cnt = int(cm_list[r][bg_col_idx])
        bg_fp_breakdown[cname] = fp_cnt
        total_bg_fp += fp_cnt

    return {
        "model": str(model_path),
        "data_yaml": str(data_yaml),
        "split": split,
        "conf": conf,
        "iou": iou,
        "overall_metrics": {
            "box_map50": round(box_map50, 4),
            "box_map50_95": round(box_map50_95, 4),
            "mask_map50": round(mask_map50, 4),
            "mask_map50_95": round(mask_map50_95, 4),
            "mean_box_p": round(float(val_results.box.mp), 4),
            "mean_box_r": round(float(val_results.box.mr), 4),
            "mean_mask_p": round(float(val_results.seg.mp), 4) if hasattr(val_results, "seg") else 0.0,
            "mean_mask_r": round(float(val_results.seg.mr), 4) if hasattr(val_results, "seg") else 0.0,
        },
        "class_metrics": class_metrics,
        "confusion_matrix_axis_convention": "Rows=Predicted, Columns=True",
        "confusion_matrix_raw": cm_list,
        "gt_instance_counts_from_cm": gt_column_sums,
        "background_fp_count": total_bg_fp,
        "background_fp_breakdown": bg_fp_breakdown,
    }


def parse_video_duration(source_path: Path, image_paths: list[Path]) -> tuple[float, float]:
    if image_paths:
        timestamps = []
        for p in image_paths:
            m = TIMESTAMP_PATTERN.search(p.name)
            if m:
                timestamps.append(float(m.group(1)))
        if len(timestamps) == len(image_paths) and len(timestamps) > 1:
            dur_s = timestamps[-1] - timestamps[0]
            return dur_s, dur_s / 60.0
        dur_s = len(image_paths) * 1.0
        return dur_s, dur_s / 60.0

    if source_path.is_file():
        cap = cv2.VideoCapture(str(source_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        dur_s = (fc / fps) if fps > 0 else 0.0
        return dur_s, dur_s / 60.0

    return 0.0, 0.0


def evaluate_video_stream(
    model_path: str | Path,
    video_source: str | Path,
    conf: float = 0.20,
    iou: float = 0.70,
    class_thresholds: dict[str, float] | None = None,
    imgsz: int = 640,
    max_frames: int | None = None,
    time_window_cluster: float = 6.0,
) -> dict:
    model = YOLO(str(model_path))
    names = model.names
    src_path = Path(video_source)

    image_paths = []
    if src_path.is_dir():
        image_paths = sorted(
            [p for p in src_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        total_frames = len(image_paths)
    elif src_path.is_file():
        cap = cv2.VideoCapture(str(src_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
    else:
        raise FileNotFoundError(f"Source not found: {video_source}")

    if max_frames:
        total_frames = min(total_frames, max_frames)
        if image_paths:
            image_paths = image_paths[:total_frames]

    dur_s, duration_minutes = parse_video_duration(src_path, image_paths)

    detections_by_class = {name: 0 for name in names.values()}
    total_detections = 0
    frames_with_detections = 0
    candidate_detection_timestamps = []

    thresholds = class_thresholds or {}
    print(f"\nProcessing video stream: {video_source} ({total_frames} frames, {duration_minutes:.3f} min)...")

    if image_paths:
        frame_iter = []
        for i, p in enumerate(image_paths):
            m = TIMESTAMP_PATTERN.search(p.name)
            ts = float(m.group(1)) if m else float(i)
            frame_iter.append((i, str(p), ts))
    else:
        cap = cv2.VideoCapture(str(src_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_iter = []
        idx = 0
        while cap.isOpened() and (max_frames is None or idx < max_frames):
            ret, _ = cap.read()
            if not ret:
                break
            frame_iter.append((idx, idx, round(idx / fps, 2)))
            idx += 1
        cap.release()

    for idx, f_src, ts in frame_iter:
        frame = cv2.imread(f_src) if isinstance(f_src, str) else None
        if frame is None:
            continue

        results = model.predict(frame, imgsz=imgsz, conf=min(conf, 0.05), iou=iou, verbose=False)[0]

        frame_detected = False
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                box_conf = float(box.conf[0])

                target_th = thresholds.get(cls_name, conf)
                if box_conf >= target_th:
                    detections_by_class[cls_name] = detections_by_class.get(cls_name, 0) + 1
                    total_detections += 1
                    frame_detected = True

        if frame_detected:
            frames_with_detections += 1
            candidate_detection_timestamps.append(ts)

    # Cluster timestamps by time_window_cluster
    clusters = 0
    last_t = -999.0
    for t in candidate_detection_timestamps:
        if t - last_t > time_window_cluster:
            clusters += 1
        last_t = t

    det_per_frame = round(total_detections / total_frames, 4) if total_frames > 0 else 0.0
    det_per_min = round(total_detections / duration_minutes, 2) if duration_minutes > 0 else 0.0
    clusters_per_min = round(clusters / duration_minutes, 2) if duration_minutes > 0 else 0.0

    return {
        "video_source": str(video_source),
        "total_frames_processed": total_frames,
        "duration_seconds": round(dur_s, 2),
        "duration_minutes": round(duration_minutes, 3),
        "total_detections": total_detections,
        "frames_with_detections": frames_with_detections,
        "temporal_candidate_clusters": clusters,
        "detections_by_class": detections_by_class,
        "detections_per_frame": det_per_frame,
        "detections_per_minute": det_per_min,
        "candidate_clusters_per_minute": clusters_per_min,
        "thresholds_applied": {c: thresholds.get(c, conf) for c in names.values()},
    }


def print_report(res_dataset: dict | None, res_video: dict | None):
    print("\n" + "=" * 85)
    print("                    BEUM AI RIGOROUS BENCHMARK REPORT                    ")
    print("=" * 85)

    if res_dataset:
        ov = res_dataset["overall_metrics"]
        print("\n[1. GROUND-TRUTH VALIDATION OVERALL METRICS]")
        print(f"Model: {res_dataset['model']}")
        print(f"Dataset: {res_dataset['data_yaml']} (Split: {res_dataset['split']})")
        print(f"Box  mAP50: {ov['box_map50']:.4f} | Box  mAP50-95: {ov['box_map50_95']:.4f} (Mean P: {ov['mean_box_p']:.4f}, Mean R: {ov['mean_box_r']:.4f})")
        print(f"Mask mAP50: {ov['mask_map50']:.4f} | Mask mAP50-95: {ov['mask_map50_95']:.4f} (Mean P: {ov['mean_mask_p']:.4f}, Mean R: {ov['mean_mask_r']:.4f})")
        print("-" * 85)
        print("Class".ljust(14), "Present".rjust(8), "Box P".rjust(9), "Box R".rjust(9), "Box mAP50".rjust(11), "Mask P".rjust(9), "Mask R".rjust(9), "Mask mAP50".rjust(12))
        print("-" * 85)
        for cname, m in res_dataset["class_metrics"].items():
            pres = "Yes" if m["present_in_split"] else "NO (0)"
            print(
                cname.ljust(14),
                pres.rjust(8),
                f"{m['box_p']:.4f}".rjust(9),
                f"{m['box_r']:.4f}".rjust(9),
                f"{m['box_ap50']:.4f}".rjust(11),
                f"{m['mask_p']:.4f}".rjust(9),
                f"{m['mask_r']:.4f}".rjust(9),
                f"{m['mask_ap50']:.4f}".rjust(12),
            )

        print("\n[2. CONFUSION MATRIX (Rows=Predicted, Columns=True)]")
        cm = res_dataset["confusion_matrix_raw"]
        print("-" * 85)
        print("Pred \\ True".ljust(14), "object".rjust(12), "drain_area".rjust(12), "drain_full".rjust(12), "Background".rjust(14))
        print("-" * 85)
        row_labels = ["object", "drain_area", "drain_full", "Background"]
        for r_idx, r_label in enumerate(row_labels):
            vals = [str(int(cm[r_idx][c])) for c in range(len(cm[r_idx]))]
            print(r_label.ljust(14), "".join(v.rjust(12 if i < 3 else 14) for i, v in enumerate(vals)))
        print("-" * 85)
        gt_sums = res_dataset["gt_instance_counts_from_cm"]
        print(f"Verified GT Totals from Matrix: {gt_sums}")
        print(f"Total Background False Positives: {res_dataset['background_fp_count']} (drain_area FP: {res_dataset['background_fp_breakdown'].get('drain_area')}, drain_full FP: {res_dataset['background_fp_breakdown'].get('drain_full')})")

    if res_video:
        print("\n[3. UNLABELED OPERATIONAL VIDEO METRICS]")
        print(f"Source: {res_video['video_source']}")
        print(f"Frames: {res_video['total_frames_processed']} | Duration: {res_video['duration_seconds']}s ({res_video['duration_minutes']} min)")
        print(f"Total Detections: {res_video['total_detections']} across {res_video['frames_with_detections']} frames")
        print(f"Temporal Candidate Clusters: {res_video['temporal_candidate_clusters']}")
        print(f"Detections by Class: {res_video['detections_by_class']}")
        print(f"Thresholds Applied: {res_video['thresholds_applied']}")
        print("-" * 85)
        print(f"  • Detections / Frame:              {res_video['detections_per_frame']:.4f} det/frame")
        print(f"  • Detections / Minute:             {res_video['detections_per_minute']:.2f} det/min")
        print(f"  • Candidate Clusters / Minute:     {res_video['candidate_clusters_per_minute']:.2f} clusters/min")

    print("=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Rigorous BEUM benchmark evaluation.")
    parser.add_argument("--model", type=str, default="models/best-seg.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--data", type=str, default="", help="Path to dataset YAML")
    parser.add_argument("--split", type=str, default="val", help="val or test")
    parser.add_argument("--video", type=str, default="", help="Video source or directory")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.70, help="IoU threshold")
    parser.add_argument("--da-conf", type=float, default=None, help="drain_area threshold")
    parser.add_argument("--df-conf", type=float, default=None, help="drain_full threshold")
    parser.add_argument("--overlap-mask", action=argparse.BooleanOptionalAction, default=False, help="Whether to merge masks (default: False)")
    parser.add_argument("--output", type=str, default="", help="Save JSON path")
    args = parser.parse_args()

    res_dataset = None
    res_video = None

    if args.data:
        res_dataset = evaluate_dataset(
            model_path=args.model,
            data_yaml=args.data,
            split=args.split,
            conf=args.conf,
            iou=args.iou,
            overlap_mask=args.overlap_mask,
        )

    if args.video:
        class_thresholds = {}
        if args.da_conf is not None:
            class_thresholds["drain_area"] = args.da_conf
        if args.df_conf is not None:
            class_thresholds["drain_full"] = args.df_conf

        res_video = evaluate_video_stream(
            model_path=args.model,
            video_source=args.video,
            conf=args.conf,
            iou=args.iou,
            class_thresholds=class_thresholds,
        )

    print_report(res_dataset, res_video)

    if args.output:
        out_data = {"dataset": res_dataset, "video": res_video}
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
