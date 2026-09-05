#!/usr/bin/env python3
"""
Live E2E Demo: Central Receiver Server + Edge GullyRuntime Real-time Pipeline
-----------------------------------------------------------------------------
1. Starts the Central Receiver Server on http://127.0.0.1:8001.
2. Serves DrainSight Web Control Dashboard on http://localhost:8001/dashboard.
3. Feeds real road driving video and synchronized GPS to the edge GullyRuntime.
4. When a drain blockage is detected, it is spooled, uploaded as multipart/form-data,
   and pushed in real-time via SSE directly onto the interactive GIS map.

Run:
    python tools/run_e2e_live_demo.py --port 8001 --max-frames 300
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

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
from receiver_server import create_app


def main():
    parser = argparse.ArgumentParser(description="BEUM Live E2E Demo")
    parser.add_argument("--port", type=int, default=8001, help="Receiver server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Binding host")
    parser.add_argument(
        "--model",
        type=str,
        default="models/edge_exports/best-seg-2class_320.onnx",
        help="Path to edge model (.pt or .onnx)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/2026-09-04_04-31-48/V20260904_131914000_79E42EB0-91A0-4F27-A3F0-BF6673616688.MOV",
        help="Path to driving video",
    )
    parser.add_argument(
        "--gps",
        type=str,
        default="data/2026-09-04_04-31-48/Location.csv",
        help="Path to GPS Location.csv",
    )
    parser.add_argument("--max-frames", type=int, default=300, help="Frames to process")
    parser.add_argument("--open-browser", action="store_true", help="Automatically open browser dashboard")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "received_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    spool_dir = PROJECT_ROOT / "data" / "spool_live_demo"
    spool_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("      [BEUM] Live E2E Gully Monitor & DrainSight Dashboard Demo")
    print("=" * 75)
    print(f"[*] Server Address : http://{args.host}:{args.port}")
    print(f"[*] Dashboard URL  : http://{args.host}:{args.port}/dashboard")
    print(f"[*] Edge Model     : {args.model}")
    print(f"[*] Video Clip     : {Path(args.video).name}")
    print(f"[*] GPS Replay     : {Path(args.gps).name}")
    print(f"[*] Target Frames  : {args.max_frames} (~10 seconds at 30 FPS)")
    print("=" * 75)

    # 1. Start Receiver Server
    app = create_app(data_dir=data_dir, token="")
    server_config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(server_config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    print("\n[1/3] Starting Receiver Server...")
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.05)
    print("      -> Receiver Server is ONLINE ✅")

    dashboard_url = f"http://{args.host}:{args.port}/dashboard"
    if args.open_browser:
        print(f"\n[2/3] Opening browser dashboard: {dashboard_url}")
        webbrowser.open(dashboard_url)
    else:
        print(f"\n[2/3] Web Dashboard available at: {dashboard_url}")
        print("      Open this URL in your web browser to see real-time GIS updates!")

    # 2. Configure Edge Runtime
    print("\n[3/3] Launching Edge GullyRuntime with video feed...")
    edge_config = SystemConfig(
        source=str(PROJECT_ROOT / args.video),
        max_frames=args.max_frames,
        realtime_source=True,
        camera_buffer_size=2,
        roi_points=((0, 0), (1080, 0), (1080, 1920), (0, 1920)),
        detector=DetectorConfig(
            model_path=str(PROJECT_ROOT / args.model),
            image_size=320 if "320" in args.model else 640,
            confidence=0.15,
            iou=0.45,
            device="cpu",
            class_names=["drain_area", "drain_full"],
        ),
        storage=StorageConfig(
            spool_dir=str(spool_dir),
            save_evidence=True,
            jpeg_quality=80,
        ),
        upload=UploadConfig(
            url=f"http://127.0.0.1:{args.port}/upload",
            token="",
            timeout_s=10.0,
            flush_interval_s=1.0,
        ),
        gps=GPSConfig(
            provider="replay",
            csv_path=str(PROJECT_ROOT / args.gps),
            sample_period_s=0.1,
            replay_speed=1.0,
        ),
        blockage=BlockageConfig(
            gully_class_names=["drain_area"],
            obstacle_class_names=["drain_full"],
            warning_percent=20.0,
            critical_percent=50.0,
            event_cooldown_s=1.0,
        ),
        policy=PolicyConfig(
            mode_intervals={"low": 0.5, "medium": 0.033, "high": 0.033},
        ),
        run_preflight=False,
    )

    runtime = GullyRuntime(edge_config)
    try:
        stats = runtime.run()
        print("\n" + "=" * 75)
        print("EDGE RUNTIME FINISHED")
        print("=" * 75)
        print(f"Processed Frames : {stats.get('frames', 0)}")
        print(f"Dropped Frames   : {stats.get('dropped_frames', 0)}")
        print(f"Inferences       : {stats.get('inferences', 0)}")
        print(f"Detected Events  : {stats.get('events', 0)}")
        print(f"Pending in Spool : {stats.get('pending', 0)}")
        print(f"Dashboard Link   : {dashboard_url}")
        print("=" * 75)
        print("\nServer will stay alive for 30 seconds for dashboard viewing (Ctrl+C to quit)...")
        time.sleep(30.0)
    except KeyboardInterrupt:
        print("\nStopping demo...")
    finally:
        server.should_exit = True
        server_thread.join(timeout=2.0)
        print("Demo exited cleanly.")


if __name__ == "__main__":
    main()
