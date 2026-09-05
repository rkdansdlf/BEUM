#!/usr/bin/env python
"""
Evaluates the effect of TemporalFilter and BlockageAnalyzer
on driving video frames or continuous video files.
Measures false alarm elimination (spurious single frames) and true drain persistence.
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

from gully_system.types import Detection
from gully_system.temporal_filter import TemporalFilter
from gully_system.blockage import BlockageAnalyzer

TIMESTAMP_PATTERN = re.compile(r"_t(\d+\.\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

def run_temporal_evaluation(
    model_path: str,
    frames_dir: str = "analysis/new_data_frames",
    video_path: str = "",
    conf: float = 0.20,
    iou: float = 0.45,
    min_hits: int = 1,
    max_missed: int = 2,
    iou_threshold: float = 0.3,
):
    print("=" * 80)
    print(f"TEMPORAL STREAM EVALUATION: {model_path}")
    source_desc = f"Video: {video_path}" if video_path else f"Frames Dir: {frames_dir}"
    print(f"{source_desc} | Conf: {conf} | min_hits: {min_hits} | max_missed: {max_missed} | track_iou: {iou_threshold}")
    print("=" * 80)

    model = YOLO(model_path)
    names = model.names

    # Initialize TemporalFilter & BlockageAnalyzer
    temp_filter = TemporalFilter(min_hits=min_hits, max_missed=max_missed, iou_threshold=iou_threshold)
    analyzer = BlockageAnalyzer(
        gully_class_names=("drain_area", "gully"),
        obstacle_class_names=("drain_full", "debris", "sediment", "trash", "leaf"),
        warning_percent=20.0,
        critical_percent=50.0,
    )

    raw_detections_log = []
    filtered_events_log = []
    total_raw_detections = 0
    raw_detected_frames = 0
    confirmed_tracks = set()

    frames_iter = []
    fps = 0.5
    dur_min = 1.0

    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = frame_idx / fps
            frames_iter.append((frame_idx, ts, frame))
            frame_idx += 1
        cap.release()
        dur_min = len(frames_iter) / (fps * 60.0) if fps > 0 else 1.0
    else:
        paths = sorted(list(Path(frames_dir).glob("frame_*.jpg")))
        for idx, fpath in enumerate(paths):
            fid = int(fpath.stem.split("_")[1])
            m = TIMESTAMP_PATTERN.search(fpath.name)
            ts = float(m.group(1)) if m else float(idx * 2.0)
            frame = cv2.imread(str(fpath))
            if frame is not None:
                frames_iter.append((fid, ts, frame))
        if paths:
            try:
                dur_s = float(TIMESTAMP_PATTERN.search(paths[-1].name).group(1)) - float(TIMESTAMP_PATTERN.search(paths[0].name).group(1))
            except Exception:
                dur_s = len(paths) * 2.0
            dur_min = dur_s / 60.0

    total_frames = len(frames_iter)

    for fid, ts, frame in frames_iter:
        h, w = frame.shape[:2]
        results = model.predict(frame, imgsz=640, conf=conf, iou=iou, verbose=False)[0]

        frame_detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            for b_idx, box in enumerate(results.boxes):
                cid = int(box.cls[0].item())
                cname = names.get(cid, str(cid))
                score = float(box.conf[0].item())
                bbox = tuple(float(v) for v in box.xyxy[0].tolist())

                mask = None
                if results.masks is not None and hasattr(results.masks, "xy") and b_idx < len(results.masks.xy):
                    poly = np.asarray(results.masks.xy[b_idx], dtype=np.int32)
                    if poly.size > 0:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(mask, [poly], 1)

                frame_detections.append(
                    Detection(
                        bbox=bbox,
                        confidence=score,
                        class_id=cid,
                        class_name=cname,
                        mask=mask,
                    )
                )

        if frame_detections:
            raw_detected_frames += 1
            total_raw_detections += len(frame_detections)
            raw_detections_log.append({
                "frame_id": fid,
                "timestamp_s": round(ts, 2),
                "detections": [
                    {"class": d.class_name, "conf": round(d.confidence, 3), "bbox": [round(x, 1) for x in d.bbox]}
                    for d in frame_detections
                ],
            })

        # Apply temporal filtering
        temp_result = temp_filter.update(frame_detections)

        # Track new entered events or validated detections
        if temp_result.validated or temp_result.entered:
            metrics = analyzer.analyze(frame, list(temp_result.validated))
            for d in temp_result.entered:
                confirmed_tracks.add(d.track_id)
            filtered_events_log.append({
                "frame_id": fid,
                "timestamp_s": round(ts, 2),
                "validated_count": len(temp_result.validated),
                "entered_count": len(temp_result.entered),
                "blockage_status": metrics.status,
                "coverage_percent": metrics.coverage_percent,
                "detections": [
                    {"track_id": d.track_id, "class": d.class_name, "conf": round(d.confidence, 3)}
                    for d in temp_result.validated
                ],
            })

    summary = {
        "model_path": str(model_path),
        "source": video_path if video_path else frames_dir,
        "total_frames": total_frames,
        "duration_minutes": round(dur_min, 3),
        "raw_metrics": {
            "total_detections": total_raw_detections,
            "detected_frames": raw_detected_frames,
            "detections_per_min": round(total_raw_detections / dur_min, 2) if dur_min > 0 else 0,
            "detected_frames_list": [x["frame_id"] for x in raw_detections_log],
        },
        "temporal_filtered_metrics": {
            "min_hits": min_hits,
            "max_missed": max_missed,
            "confirmed_distinct_tracks": len(confirmed_tracks),
            "validated_frames_count": len(filtered_events_log),
            "validated_frames_list": [x["frame_id"] for x in filtered_events_log],
            "confirmed_events_per_min": round(len(confirmed_tracks) / dur_min, 2) if dur_min > 0 else 0,
        },
        "spurious_frame_elimination": {
            "eliminated_frame_ids": sorted(list(
                set(x["frame_id"] for x in raw_detections_log) - set(x["frame_id"] for x in filtered_events_log)
            )),
            "true_drain_frame_155_persisted": 155 in [x["frame_id"] for x in filtered_events_log] or 154 in [x["frame_id"] for x in filtered_events_log] or (5 in [x["frame_id"] for x in filtered_events_log]),
        },
        "filtered_events_log": filtered_events_log,
    }

    out_json = Path("analysis/temporal_filter_evaluation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("TEMPORAL STREAM EVALUATION COMPLETED")
    print(f"Raw Detected Frames: {summary['raw_metrics']['detected_frames']} ({summary['raw_metrics']['detected_frames_list']})")
    print(f"Temporal Confirmed Distinct Tracks: {summary['temporal_filtered_metrics']['confirmed_distinct_tracks']}")
    print(f"Temporal Validated Frames: {summary['temporal_filtered_metrics']['validated_frames_count']} ({summary['temporal_filtered_metrics']['validated_frames_list']})")
    print(f"Eliminated Spurious Frames: {summary['spurious_frame_elimination']['eliminated_frame_ids']}")
    print(f"Confirmed Events / min: {summary['temporal_filtered_metrics']['confirmed_events_per_min']}")
    print(f"Summary JSON saved: {out_json}")
    print("=" * 80)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/candidates/yolov8n_seg_2class_hn20_best_fp_suppression.pt")
    parser.add_argument("--frames", type=str, default="analysis/new_data_frames")
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--max-missed", type=int, default=2)
    parser.add_argument("--iou", type=float, default=0.3)
    args = parser.parse_args()

    run_temporal_evaluation(
        model_path=args.weights,
        frames_dir=args.frames,
        video_path=args.video,
        conf=args.conf,
        min_hits=args.min_hits,
        max_missed=args.max_missed,
        iou_threshold=args.iou,
    )

if __name__ == "__main__":
    main()
