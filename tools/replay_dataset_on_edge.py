#!/usr/bin/env python3
"""
BEUM Edge Replay Tool
---------------------
Replays a recorded video session and GPS trajectory through the complete
edge AI pipeline on the Raspberry Pi 5.
Detects drain area and blockages, renders visualization overlays (boxes, labels, masks, status banner),
and uploads blockage events to the central DrainSight receiver server.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gully_system.blockage import BlockageAnalyzer, BlockageEventGate
from gully_system.config import BlockageConfig, DetectorConfig
from gully_system.detector import YOLODetector
from gully_system.roi import ROI
from gully_system.runtime import GullyRuntime
from gully_system.temporal_filter import TemporalFilter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EdgeReplay")


def load_gps_records(csv_path: Path) -> list[dict[str, float]]:
    """Loads GPS points with elapsed seconds from Location.csv or gps.csv."""
    records = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_time: float | None = None
        for row in reader:
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                speed = float(row.get("speed", 0.0) or 0.0)
                if speed < 0:
                    speed = 0.0

                if "seconds_elapsed" in row and row["seconds_elapsed"]:
                    ts_s = float(row["seconds_elapsed"])
                elif "timestamp" in row and row["timestamp"]:
                    ts_s = float(row["timestamp"])
                elif "time" in row and row["time"]:
                    raw_time = float(row["time"])
                    if first_time is None:
                        first_time = raw_time
                    ts_s = (raw_time - first_time) / 1e9
                else:
                    ts_s = 0.0

                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    records.append({"ts_s": ts_s, "lat": lat, "lon": lon, "speed": speed})
            except (KeyError, TypeError, ValueError):
                continue
    records.sort(key=lambda r: r["ts_s"])
    return records


def find_closest_gps(ts_s: float, records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {"ts_s": ts_s, "lat": 36.8304, "lon": 127.1790, "speed": 0.0}
    return min(records, key=lambda r: abs(r["ts_s"] - ts_s))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay dataset on edge hardware")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to session directory")
    parser.add_argument("--model", type=str, default="models/best-seg-2class_320.onnx", help="Path to ONNX model")
    parser.add_argument("--upload-url", type=str, default="http://172.30.1.46:8000/upload", help="Upload endpoint")
    parser.add_argument("--sample-interval-s", type=float, default=1.0, help="Frame sampling interval in seconds")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--max-events", type=int, default=0, help="Max blockage events to upload (0 for unlimited)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)

    # Locate video file
    video_files = list(dataset_path.glob("*.MOV")) + list(dataset_path.glob("*.mov")) + list(dataset_path.glob("*.mp4"))
    if not video_files:
        logger.error(f"No video file found in {dataset_path}")
        sys.exit(1)
    video_path = video_files[0]

    # Locate GPS file
    gps_path = dataset_path / "Location.csv"
    if not gps_path.exists():
        gps_candidates = list(dataset_path.glob("*gps*.csv")) + list(dataset_path.glob("*Location*.csv"))
        if gps_candidates:
            gps_path = gps_candidates[0]
        else:
            logger.warning("No GPS CSV found; default coordinates will be used")
            gps_path = None

    # Load GPS records
    gps_records = load_gps_records(gps_path) if gps_path and gps_path.exists() else []
    logger.info(f"Loaded {len(gps_records)} GPS points from {gps_path}")

    # Initialize video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps

    orientation_meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    logger.info(f"Video orientation metadata: {orientation_meta} degrees")

    ret, test_frame = cap.read()
    if ret and test_frame is not None:
        if orientation_meta == 90.0:
            test_frame = cv2.rotate(test_frame, cv2.ROTATE_90_CLOCKWISE)
        elif orientation_meta == 180.0:
            test_frame = cv2.rotate(test_frame, cv2.ROTATE_180)
        elif orientation_meta == 270.0:
            test_frame = cv2.rotate(test_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = test_frame.shape[:2]
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1920

    logger.info(f"Video: {video_path.name} | {w}x{h} @ {fps:.2f} FPS | {total_frames} frames ({duration_s:.1f}s)")

    # Initialize Edge AI Pipeline
    logger.info(f"Loading YOLO detector: {args.model} (conf={args.conf}, iou={args.iou})")
    det_cfg = DetectorConfig(
        model_path=str(args.model),
        image_size=320,
        confidence=args.conf,
        iou=args.iou,
        device="cpu",
        class_names=("drain_area", "drain_full"),
        mapping_mode="auto",
        agnostic_nms=True,
    )
    detector = YOLODetector(det_cfg)

    analyzer = BlockageAnalyzer(
        gully_class_names=["drain_area"],
        obstacle_class_names=["drain_full"],
        warning_percent=20.0,
        critical_percent=50.0,
        require_gully_presence=True,
    )
    temporal_filter = TemporalFilter(min_hits=3, max_missed=2, iou_threshold=0.3)
    event_gate = BlockageEventGate(change_percent=10.0, cooldown_s=3.0)

    # Exclude sky/horizon in landscape frames and upper distant scene in portrait frames
    if w > h:
        road_top = float(h) * 0.45
        active_roi = ROI(((0.0, road_top), (float(w), road_top), (float(w), float(h)), (0.0, float(h))))
        logger.info(f"Landscape frame detected: applied Road Surface ROI (y >= {road_top:.0f}px) to suppress sky/treetop false alarms")
    else:
        road_top = float(h) * 0.35
        active_roi = ROI(((0.0, road_top), (float(w), road_top), (float(w), float(h)), (0.0, float(h))))
        logger.info(f"Portrait downward frame detected ({w}x{h}): applied Road Surface ROI (y >= {road_top:.0f}px) to suppress upper vehicles/shadows")
    dummy_runtime = object.__new__(GullyRuntime)

    session_id = dataset_path.name
    step_frames = max(1, int(round(fps * args.sample_interval_s)))
    total_steps = (total_frames + step_frames - 1) // step_frames

    logger.info(f"Starting replay: sampling every {step_frames} frames (~{args.sample_interval_s}s), ~{total_steps} evaluations")

    evaluated_count = 0
    uploaded_count = 0
    start_time = time.time()

    current_frame_idx = 0
    while current_frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if orientation_meta == 90.0:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif orientation_meta == 180.0:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif orientation_meta == 270.0:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        ts_s = current_frame_idx / fps
        evaluated_count += 1

        # 1. Detect objects
        raw_dets = detector.predict(frame)
        in_roi = active_roi.filter(tuple(raw_dets)) if raw_dets else ()
        temporal = temporal_filter.update(list(in_roi))

        # 2. Analyze blockage
        blockage = analyzer.analyze(frame, temporal.validated)
        status_str = blockage.status if isinstance(blockage.status, str) else blockage.status.value

        # 3. Check event gate
        should_upload = event_gate.should_emit(blockage, now=ts_s)

        if should_upload and status_str in ("warning", "critical"):
            # Annotate evidence frame
            active_roi.draw(frame)
            dummy_runtime._draw_detections(frame, tuple(temporal.validated))
            analyzer.draw(frame, blockage)

            # Match GPS
            gps = find_closest_gps(ts_s, gps_records)

            # Prepare event payload
            event_id = f"pi_reproc_{session_id}_f{current_frame_idx:05d}"
            event_payload = {
                "event_id": event_id,
                "event_type": "gully_blockage",
                "source": f"rpi5_replay_{session_id}",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "blockage": {
                    "status": status_str,
                    "coverage_percent": round(blockage.coverage_percent, 2),
                    "occlusion_pct": round(getattr(blockage, "occlusion_pct", blockage.coverage_percent), 2),
                    "gully_count": len([d for d in temporal.validated if d.class_name in analyzer.gully_class_names]),
                    "obstacle_count": len([d for d in temporal.validated if d.class_name in analyzer.obstacle_class_names]),
                },
                "gps": {
                    "latitude": gps["lat"],
                    "longitude": gps["lon"],
                    "speed_kmh": round(gps["speed"] * 3.6, 1),
                    "track_deg": 0.0,
                    "valid": True,
                },
            }

            _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_bytes = enc.tobytes()

            try:
                res = requests.post(
                    args.upload_url,
                    files={
                        "metadata": (None, json.dumps(event_payload), "application/json"),
                        "image": ("evidence.jpg", jpg_bytes, "image/jpeg"),
                    },
                    timeout=10,
                )
                if res.status_code == 201:
                    uploaded_count += 1
                    logger.info(
                        f"[{evaluated_count}/{total_steps}] Frame {current_frame_idx:05d} ({ts_s:.1f}s) | "
                        f"Status: {status_str.upper()} ({blockage.coverage_percent:.1f}%) | "
                        f"GPS: ({gps['lat']:.5f}, {gps['lon']:.5f}) -> Uploaded: {event_id} (HTTP 201)"
                    )
                else:
                    logger.warning(f"Upload failed: HTTP {res.status_code} - {res.text}")
            except Exception as exc:
                logger.error(f"Upload exception: {exc}")

            if args.max_events and uploaded_count >= args.max_events:
                logger.info(f"Reached max events limit ({args.max_events})")
                break

        elif evaluated_count % 30 == 0 or current_frame_idx == 0:
            logger.info(
                f"[{evaluated_count}/{total_steps}] Frame {current_frame_idx:05d} ({ts_s:.1f}s) | "
                f"Detections: {len(temporal.validated)} | Status: {status_str}"
            )

        current_frame_idx += step_frames

    cap.release()
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Replay completed in {elapsed:.1f}s!")
    logger.info(f"Evaluated Frames: {evaluated_count} | Blockage Events Uploaded: {uploaded_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
