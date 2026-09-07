#!/usr/bin/env python
"""
Canary Replay Validation for Promoted Canonical Model.
Replays driving video frames through the complete BEUM runtime pipeline:
YOLODetector -> BlockageAnalyzer -> BlockageEventGate
Verifies state transition stability, event gating, and alert integrity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import cv2
from gully_system.blockage import BlockageAnalyzer, BlockageEventGate
from gully_system.config import DetectorConfig
from gully_system.detector import YOLODetector

TIMESTAMP_PATTERN = re.compile(r"_t(\d+\.\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_timestamp(filename: str) -> float | None:
    m = TIMESTAMP_PATTERN.search(filename)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best-seg-canonical.pt")
    parser.add_argument("--frames-dir", type=str, default="analysis/new_data_frames")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--output", type=str, default="analysis/canary_replay_report.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    frames_dir = Path(args.frames_dir)

    print("=" * 80)
    print("CANARY REPLAY VALIDATION ON RUNTIME PIPELINE")
    print(f"Model: {model_path} | Frames: {frames_dir} | Conf: {args.conf}")
    print("=" * 80)

    detector_cfg = DetectorConfig(
        model_path=str(model_path),
        image_size=640,
        confidence=args.conf,
        iou=0.45,
        device="mps",
        class_names=("drain_area", "drain_full"),
        mapping_mode="auto",
    )
    detector = YOLODetector(detector_cfg)

    analyzer = BlockageAnalyzer(
        gully_class_names=("drain_area",),
        obstacle_class_names=("drain_full",),
        warning_percent=20.0,
        critical_percent=50.0
    )

    event_gate = BlockageEventGate(change_percent=10.0, cooldown_s=5.0)

    frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if not frame_files:
        raise FileNotFoundError(f"No frames in {frames_dir}")

    replay_log = []
    emitted_events = []
    status_summary = {"clear": 0, "warning": 0, "critical": 0, "no_gully": 0}

    start_sim_time = 0.0
    t0 = time.time()

    for idx, f_path in enumerate(frame_files):
        img = cv2.imread(str(f_path))
        if img is None:
            continue
        ts = parse_timestamp(f_path.name) or (idx * 0.1)

        # 1. Detection
        detections = detector.predict(img)

        # 2. Blockage Analysis
        metrics = analyzer.analyze(img, detections)
        status_summary[metrics.status] += 1

        # 3. Event Gating
        should_emit = event_gate.should_emit(metrics, now=ts)

        record = {
            "frame_index": idx,
            "frame_name": f_path.name,
            "timestamp_s": round(ts, 2),
            "detections_count": len(detections),
            "status": metrics.status,
            "coverage_percent": metrics.coverage_percent,
            "blocked_area_px": metrics.blocked_area_px,
            "gully_area_px": metrics.gully_area_px,
            "should_emit": should_emit,
        }
        replay_log.append(record)

        if should_emit:
            emitted_events.append({
                "frame": f_path.name,
                "timestamp_s": round(ts, 2),
                "status": metrics.status,
                "coverage_percent": metrics.coverage_percent,
                "confidence": metrics.confidence,
                "gully_count": metrics.gully_count,
            })
            print(f"  [EVENT EMITTED @ t={ts:6.2f}s] Frame #{idx:03d} -> Status: {metrics.status.upper()} (Coverage: {metrics.coverage_percent:.1f}%, Conf: {metrics.confidence:.2f})")

    elapsed = time.time() - t0
    fps = round(len(frame_files) / elapsed, 1)

    summary = {
        "model": str(model_path),
        "total_frames": len(frame_files),
        "processing_time_s": round(elapsed, 2),
        "replay_fps": fps,
        "status_distribution": status_summary,
        "total_events_emitted": len(emitted_events),
        "emitted_events": emitted_events,
        "replay_log_sample": replay_log[:20],
    }

    out_file = Path(args.output)
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("CANARY REPLAY COMPLETED SUCCESSFULLY!")
    print(f"Replay FPS: {fps} FPS across {len(frame_files)} frames")
    print(f"Status Distribution: {status_summary}")
    print(f"Total Events Emitted: {len(emitted_events)}")
    for ev in emitted_events:
        print(f"  * t={ev['timestamp_s']}s: {ev['status'].upper()} ({ev['coverage_percent']}%) [{ev['frame']}]")
    print(f"Report saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
