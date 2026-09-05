#!/usr/bin/env python3
"""
Benchmark script for evaluating Edge Real-time Video Replay Pipeline.
Runs GullyRuntime with video replay and GPS provider across candidate models
and resolutions, measuring:
- Realtime FPS and processing latency
- Dropped frames and queue drop rate (%)
- Event emission and blockage detection count
- GPS fix acquisition and synchronization age
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gully_system.config import (
    BlockageConfig,
    DetectorConfig,
    GPSConfig,
    PolicyConfig,
    StorageConfig,
    SystemConfig,
    UploadConfig,
)
from gully_system.runtime import GullyRuntime


def run_single_benchmark(
    model_path: str,
    video_path: str,
    gps_path: str,
    max_frames: int = 150,
    image_size: int = 320,
    buffer_size: int = 1,
    conf: float = 0.15,
) -> dict:
    logging.getLogger("gully_system").setLevel(logging.WARNING)

    spool_dir = PROJECT_ROOT / "data" / "spool_benchmark"
    spool_dir.mkdir(parents=True, exist_ok=True)

    config = SystemConfig(
        source=video_path,
        max_frames=max_frames,
        realtime_source=True,
        camera_buffer_size=buffer_size,
        roi_points=((0, 0), (1080, 0), (1080, 1920), (0, 1920)),
        roi_expanded_scale=1.0,
        temporal_hits=1,
        temporal_max_missed=1,
        temporal_iou=0.3,
        policy_interval_s=1.0,
        detector=DetectorConfig(
            model_path=model_path,
            image_size=image_size,
            confidence=conf,
            iou=0.45,
            device="cpu",
            class_names=["drain_area", "drain_full"],
            mapping_mode="auto",
        ),
        storage=StorageConfig(
            spool_dir=str(spool_dir),
            save_evidence=True,
            jpeg_quality=80,
        ),
        upload=UploadConfig(url=""),  # local spool only
        gps=GPSConfig(
            provider="replay",
            csv_path=gps_path,
            sample_period_s=0.1,
            replay_speed=1.0,
        ),
        blockage=BlockageConfig(
            gully_class_names=["drain_area"],
            obstacle_class_names=["drain_full"],
            warning_percent=20.0,
            critical_percent=50.0,
            event_cooldown_s=2.0,
        ),
        policy=PolicyConfig(
            min_safe_battery=20.0,
            critical_battery=8.0,
            mode_intervals={"low": 0.5, "medium": 0.033, "high": 0.033},
        ),
        run_preflight=False,
    )

    runtime = GullyRuntime(config)
    t0 = time.perf_counter()
    stats = runtime.run()
    elapsed = time.perf_counter() - t0

    processed_frames = stats.get("frames", 0)
    dropped_frames = stats.get("dropped_frames", 0)
    inferences = stats.get("inferences", 0)
    events = stats.get("events", 0)
    total_feed_frames = processed_frames + dropped_frames
    drop_rate = (dropped_frames / total_feed_frames * 100.0) if total_feed_frames > 0 else 0.0
    effective_fps = processed_frames / elapsed if elapsed > 0 else 0.0
    inference_fps = inferences / elapsed if elapsed > 0 else 0.0

    # Clean up benchmark spool
    for p in spool_dir.glob("*"):
        if p.is_file():
            p.unlink()

    return {
        "model": Path(model_path).name,
        "model_path": model_path,
        "image_size": image_size,
        "elapsed_s": round(elapsed, 2),
        "processed_frames": processed_frames,
        "dropped_frames": dropped_frames,
        "drop_rate_pct": round(drop_rate, 1),
        "effective_fps": round(effective_fps, 1),
        "inferences": inferences,
        "inference_fps": round(inference_fps, 1),
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Realtime Edge Pipeline")
    parser.add_argument(
        "--video",
        default="data/2026-09-04_04-31-48/V20260904_131914000_79E42EB0-91A0-4F27-A3F0-BF6673616688.MOV",
        help="Path to driving video",
    )
    parser.add_argument(
        "--gps",
        default="data/2026-09-04_04-31-48/Location.csv",
        help="Path to GPS Location.csv",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=150,
        help="Number of frames to process (~5s at 30 FPS)",
    )
    args = parser.parse_args()

    video_path = str(PROJECT_ROOT / args.video)
    gps_path = str(PROJECT_ROOT / args.gps)

    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    benchmarks = [
        {
            "model": "models/edge_exports/best-seg-2class_320.onnx",
            "imgsz": 320,
            "label": "ONNX 320x320 (Lightweight Edge)",
        },
        {
            "model": "models/edge_exports/best-seg-2class_640.onnx",
            "imgsz": 640,
            "label": "ONNX 640x640 (High-Resolution)",
        },
        {
            "model": "models/best-seg-2class.pt",
            "imgsz": 320,
            "label": "PyTorch PT 320x320 (PyTorch CPU)",
        },
    ]

    print("=" * 85)
    print("BEUM REAL-TIME EDGE PIPELINE BENCHMARK")
    print(f"Video Source : {Path(video_path).name}")
    print(f"GPS Replay   : {Path(gps_path).name}")
    print(f"Target Frames: {args.frames} frames (~5 sec at 30 FPS)")
    print("=" * 85)

    results = []
    for b in benchmarks:
        model_full = str(PROJECT_ROOT / b["model"])
        if not Path(model_full).exists():
            print(f"Skipping missing model: {model_full}")
            continue

        print(f"\n[*] Running Benchmark: {b['label']} ...")
        res = run_single_benchmark(
            model_path=model_full,
            video_path=video_path,
            gps_path=gps_path,
            max_frames=args.frames,
            image_size=b["imgsz"],
        )
        res["label"] = b["label"]
        results.append(res)
        print(f"    Processed: {res['processed_frames']} frames | Dropped: {res['dropped_frames']} ({res['drop_rate_pct']}%)")
        print(f"    Effective FPS: {res['effective_fps']} | Inference FPS: {res['inference_fps']} | Events: {res['events']}")

    # Print summary table
    print("\n" + "=" * 85)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 85)
    header = f"{'Configuration':<35} | {'FPS':<6} | {'Proc':<5} | {'Drop':<5} | {'Drop%':<7} | {'Events':<6}"
    print(header)
    print("-" * 85)
    for r in results:
        line = (
            f"{r['label']:<35} | "
            f"{r['effective_fps']:<6} | "
            f"{r['processed_frames']:<5} | "
            f"{r['dropped_frames']:<5} | "
            f"{r['drop_rate_pct']:<6}% | "
            f"{r['events']:<6}"
        )
        print(line)
    print("=" * 85)

    # Save to analysis/realtime_benchmark_results.json
    out_json = PROJECT_ROOT / "analysis" / "realtime_benchmark_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full results saved to: {out_json}")


if __name__ == "__main__":
    main()
