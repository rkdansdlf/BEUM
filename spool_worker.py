#!/usr/bin/env python3
"""
Spool Worker — consumes .job files from data/spool_e2e/pending/,
runs inference, posts events to receiver_server /upload endpoint,
and archives processed jobs.
"""

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SpoolWorker")

PROJECT_ROOT = Path(os.environ.get("BEUM_ROOT", Path(__file__).resolve().parent))
PENDING_DIR = PROJECT_ROOT / "data" / "spool_e2e" / "pending"
PROCESSING_DIR = PROJECT_ROOT / "data" / "spool_e2e" / "processing"
DONE_DIR = PROJECT_ROOT / "data" / "spool_e2e" / "done"
RECEIVER_URL = os.environ.get("BEUM_RECEIVER_URL", "http://127.0.0.1:8000/upload")


def _find_default_model(root: Path) -> Path:
    candidates = [
        root / "models" / "edge_exports" / "best-seg-2class_320.onnx",
        root / "models" / "best-seg-2class.pt",
        root / "models" / "best-seg-3class.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return root / "models" / "best-seg-3class.pt"


MODEL_PATH = Path(os.environ.get("BEUM_MODEL_PATH", str(_find_default_model(PROJECT_ROOT))))
PYTHON_EXEC = Path(os.environ.get("BEUM_PYTHON_EXEC", sys.executable))

POLL_INTERVAL = 10.0
MAX_CONCURRENT = 1


def run_inference(video_path: Path, output_dir: Path, detections_json: Path) -> dict:
    """Run analyze_new_data.py inference via subprocess."""
    cmd = [
        str(PYTHON_EXEC), "-c",
        f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}/tools')
from analyze_new_data import extract_frames, run_inference
from pathlib import Path

video_path = Path('{video_path}')
frames_dir = Path('{output_dir}')
model_path = Path('{MODEL_PATH}')
output_json = Path('{detections_json}')

print('Extracting frames...')
extract_frames(video_path, frames_dir, interval_s=2.0, max_frames=0)
print('Running inference...')
results = run_inference(model_path, frames_dir, output_json, conf=0.35, iou=0.45)
print(f'Detections saved to {{output_json}}')
"""
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=300,  # 5 minutes timeout
        )
        if result.returncode != 0:
            logger.error(f"Inference failed: {result.stderr}")
            return {"error": result.stderr}
        logger.info(f"Inference stdout: {result.stdout[:500]}")
        return {"success": True}
    except subprocess.TimeoutExpired:
        logger.error("Inference timeout")
        return {"error": "timeout"}
    except Exception as e:
        logger.error(f"Inference exception: {e}")
        return {"error": str(e)}


def parse_job_file(job_path: Path) -> dict:
    """Parse .job file for target_dir and video path."""
    data = {}
    with open(job_path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                data[k] = v
    return data


def build_annotation_csv(target_dir: Path, detections_json: Path) -> Path:
    """Convert detections.json to Annotation.csv with proper headers."""
    ann_path = target_dir / "Annotation.csv"
    if not detections_json.exists():
        # Write header only
        with open(ann_path, "w") as f:
            f.write("frame_index,timestamp_s,class_name,confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2,coverage_percent,status\n")
        return ann_path

    with open(detections_json) as f:
        results = json.load(f)

    with open(ann_path, "w") as f:
        f.write("frame_index,timestamp_s,class_name,confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2,coverage_percent,status\n")
        for r in results:
            ts = r.get("timestamp", 0)
            for d in r.get("detections", []):
                bbox = d.get("bbox", [0, 0, 0, 0])
                f.write(
                    f"{r.get('image','unknown')},{ts},{d.get('class','unknown')},"
                    f"{d.get('confidence',0):.4f},{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f},"
                    f"0.0,detected\n"
                )
    return ann_path


def post_event(event_data: dict) -> dict:
    """POST event to receiver_server /upload endpoint."""
    try:
        response = requests.post(RECEIVER_URL, json=event_data, timeout=30)
        return {"status": response.status_code, "data": response.json()}
    except Exception as e:
        logger.error(f"POST failed: {e}")
        return {"error": str(e)}


def get_gps_for_video(target_dir: Path) -> Optional[dict]:
    """Extract latest GPS from Location.csv for event metadata."""
    gps_path = target_dir / "Location.csv"
    if not gps_path.exists():
        return None
    import csv
    with open(gps_path) as f:
        reader = list(csv.DictReader(f))
    if not reader:
        return None
    last = reader[-1]
    return {
        "latitude": float(last.get("latitude", 0)),
        "longitude": float(last.get("longitude", 0)),
        "speed_mps": float(last.get("speed", 0)) if last.get("speed") not in ("-1", "", None) else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def process_job(job_path: Path) -> bool:
    """Process a single .job file."""
    job_data = parse_job_file(job_path)
    target_dir = Path(job_data.get("target_dir", ""))
    video_path = Path(job_data.get("video", ""))

    if not target_dir.exists() or not video_path.exists():
        logger.error(f"Paths not found: {target_dir}, {video_path}")
        return False

    logger.info(f"Processing job: {job_path.name}")
    logger.info(f"  Video: {video_path.name}")
    logger.info(f"  Target: {target_dir.name}")

    # Prepare directories
    frames_dir = target_dir / "frames"
    detections_json = target_dir / "detections.json"

    # Step 1: Run inference
    logger.info("Running inference...")
    result = run_inference(video_path, frames_dir, detections_json)
    if "error" in result:
        logger.error(f"Inference error: {result['error']}")
        return False

    # Step 2: Build Annotation.csv
    logger.info("Writing Annotation.csv...")
    ann_path = build_annotation_csv(target_dir, detections_json)
    ann_size = ann_path.stat().st_size
    logger.info(f"Annotation.csv written: {ann_size} bytes")

    # Step 3: Count detections for coverage
    with open(detections_json) as f:
        results = json.load(f)
    total_dets = sum(len(r.get("detections", [])) for r in results)
    logger.info(f"Total detections: {total_dets}")

    # Step 4: Build event payload
    gps = get_gps_for_video(target_dir)
    event_id = f"spool_{target_dir.name}_{int(time.time())}"
    created_at = datetime.now(timezone.utc).isoformat()

    event_data = {
        "event_id": event_id,
        "event_type": "gully_blockage",
        "source": str(video_path.relative_to(PROJECT_ROOT)),
        "created_at": created_at,
        "blockage": {
            "status": "normal" if total_dets == 0 else "warning",
            "coverage_percent": 0.0,
        },
        "detections": [
            {
                "image": r.get("image"),
                "timestamp": r.get("timestamp"),
                "detections": r.get("detections", []),
            }
            for r in results
        ],
    }
    if gps:
        event_data["gps"] = gps

    # Step 5: POST to receiver
    logger.info(f"Posting event to {RECEIVER_URL}...")
    post_result = post_event(event_data)
    logger.info(f"POST result: {post_result}")

    # Step 6: Update DB directly as fallback (if POST failed)
    if "error" in post_result or post_result.get("status") != 201:
        logger.warning("Falling back to direct DB write...")
        db_path = PROJECT_ROOT / "received_data" / "beum_events.db"
        lat = gps["latitude"] if gps else None
        lon = gps["longitude"] if gps else None
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO events (
                    event_id, event_type, source, created_at, received_at,
                    blockage_status, coverage_percent, occlusion_pct, lat, lon,
                    has_image, image_path, raw_payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_data["event_type"],
                    event_data["source"],
                    created_at,
                    datetime.now(timezone.utc).isoformat(),
                    event_data["blockage"]["status"],
                    event_data["blockage"]["coverage_percent"],
                    0.0,
                    lat,
                    lon,
                    0,
                    None,
                    json.dumps(event_data, ensure_ascii=False),
                ),
            )
            conn.commit()
        logger.info("Direct DB write completed")

    return True


def worker_loop():
    """Main worker loop."""
    PROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Spool Worker started | polling {PENDING_DIR} every {POLL_INTERVAL}s")

    while True:
        try:
            # Find .job files in pending
            job_files = sorted(PENDING_DIR.glob("*.job"))
            if not job_files:
                time.sleep(POLL_INTERVAL)
                continue

            for job_file in job_files:
                # Atomic move to processing (prevents double processing)
                proc_file = PROCESSING_DIR / job_file.name
                try:
                    shutil.move(str(job_file), str(proc_file))
                except Exception as e:
                    logger.error(f"Failed to move to processing: {e}")
                    continue

                logger.info(f"Acquired job: {proc_file.name}")
                success = process_job(proc_file)

                # Move to done
                done_file = DONE_DIR / f"{proc_file.stem}.done"
                shutil.move(str(proc_file), str(done_file))
                logger.info(f"Job completed -> {done_file.name} (success={success})")

        except Exception as e:
            logger.exception(f"Worker loop error: {e}")
            time.sleep(POLL_INTERVAL)


def main():
    import argparse

    global PROJECT_ROOT, PENDING_DIR, PROCESSING_DIR, DONE_DIR, RECEIVER_URL, MODEL_PATH, POLL_INTERVAL

    parser = argparse.ArgumentParser(description="DrainSight Spool Worker")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root directory")
    parser.add_argument("--receiver-url", default=str(RECEIVER_URL), help="Receiver upload URL")
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Model weights path")
    parser.add_argument("--poll-interval", type=float, default=POLL_INTERVAL, help="Polling interval in seconds")
    args = parser.parse_args()

    PROJECT_ROOT = Path(args.project_root)
    PENDING_DIR = PROJECT_ROOT / "data" / "spool_e2e" / "pending"
    PROCESSING_DIR = PROJECT_ROOT / "data" / "spool_e2e" / "processing"
    DONE_DIR = PROJECT_ROOT / "data" / "spool_e2e" / "done"
    RECEIVER_URL = args.receiver_url
    MODEL_PATH = Path(args.model_path)
    POLL_INTERVAL = args.poll_interval


    logger.info("=" * 60)
    logger.info("DrainSight Spool Worker")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Pending dir: {PENDING_DIR}")
    logger.info(f"Receiver: {RECEIVER_URL}")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info("=" * 60)

    worker_loop()


if __name__ == "__main__":
    main()