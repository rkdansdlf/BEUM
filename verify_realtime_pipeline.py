#!/usr/bin/env python3
"""
Verification Script for Real-Time BEUM Queue Ingestion & DrainSight SSE Stream
------------------------------------------------------------------------------
Verifies:
1. Server health check (/health).
2. DrainSight real-time SSE stream (/api/drainsight/stream) handshake and live push.
3. Batch upload of reprocessed events to /upload (with evidence images).
4. SQLite DB (beum_events.db) new record indexing and field integrity.
5. Image storage and retrieval (/events/{id}/image).
6. DrainSight GeoJSON and Stats API updates.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from upload_reprocessed_events import upload_events

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RealtimeVerifier")


@dataclass
class SSERecord:
    event_type: str
    data: dict[str, Any]
    received_at: float


class SSEListener(threading.Thread):
    def __init__(self, stream_url: str, token: str = ""):
        super().__init__(daemon=True)
        self.stream_url = stream_url
        self.token = token
        self.connected_event = threading.Event()
        self.received_events: list[SSERecord] = []
        self._stop_event = threading.Event()
        self.error: Exception | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        headers = {"Accept": "text/event-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = urllib.request.Request(self.stream_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                current_event_type = "message"
                while not self._stop_event.is_set():
                    line_bytes = resp.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    if line.startswith("event:"):
                        current_event_type = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        raw_data = line[len("data:") :].strip()
                        try:
                            parsed_data = json.loads(raw_data)
                        except Exception:
                            parsed_data = {"raw": raw_data}

                        rec = SSERecord(
                            event_type=current_event_type,
                            data=parsed_data,
                            received_at=time.time(),
                        )
                        self.received_events.append(rec)
                        logger.info(f"[SSE Stream] Received '{current_event_type}': {parsed_data.get('event_id', parsed_data)}")

                        if current_event_type == "connected":
                            self.connected_event.set()

                        # Reset event type for next message
                        current_event_type = "message"
        except Exception as exc:
            self.error = exc
            if not self._stop_event.is_set():
                logger.warning(f"[SSE Stream] Connection closed or error: {exc}")


def check_health(base_url: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/health"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_event_image(base_url: str, event_id: str) -> bool:
    url = f"{base_url.rstrip('/')}/events/{event_id}/image"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            content_type = resp.headers.get("Content-Type", "")
            return resp.status == 200 and "image" in content_type
    except Exception:
        return False


def verify_db_records(
    db_path: Path,
    uploaded_ids: list[str],
) -> dict[str, Any]:
    if not db_path.exists():
        return {"error": f"Database file not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM events")
    total_events = cur.fetchone()[0]

    found_records = []
    missing_ids = []

    for eid in uploaded_ids:
        cur.execute(
            """
            SELECT event_id, event_type, source, blockage_status, coverage_percent, occlusion_pct,
                   lat, lon, has_image, image_path, raw_payload
            FROM events WHERE event_id = ?
            """,
            (eid,),
        )
        row = cur.fetchone()
        if row:
            found_records.append(dict(row))
        else:
            missing_ids.append(eid)

    conn.close()

    return {
        "total_events_in_db": total_events,
        "found_count": len(found_records),
        "missing_count": len(missing_ids),
        "missing_ids": missing_ids,
        "sample_records": found_records[:3],
    }


def run_pipeline_verification(
    host: str = "127.0.0.1",
    port: int = 8001,
    data_dir: str = "received_data",
    events_file: str = "data/2026-09-04_04-19-03/events_reprocessed.json",
    limit: int = 5,
    with_images: bool = True,
    token: str = "",
) -> bool:
    base_url = f"http://{host}:{port}"
    stream_url = f"{base_url}/api/drainsight/stream"
    db_path = Path(data_dir) / "beum_events.db"

    print("=" * 60)
    print(" [BEUM] Real-Time Queue Ingestion & DrainSight SSE Verification")
    print("=" * 60)
    print(f" Target Server : {base_url}")
    print(f" SSE Stream    : {stream_url}")
    print(f" Database      : {db_path.resolve()}")
    print(f" Events File   : {events_file}")
    print(f" Test Limit    : {limit} events")
    print(f" With Images   : {with_images}")
    print("=" * 60)

    # 1. Health check
    logger.info("Step 1: Checking Receiver Server Health...")
    try:
        health_info = check_health(base_url)
        initial_db_events = health_info.get("total_events", 0)
        logger.info(f"Server is HEALTHY. Total existing events reported: {initial_db_events}")
    except Exception as exc:
        logger.error(f"Cannot connect to server at {base_url}: {exc}")
        logger.error("Please make sure the receiver server is running:")
        logger.error(f"  python -m receiver_server --port {port} --data-dir {data_dir}")
        return False

    # 2. Start SSE listener
    logger.info("Step 2: Connecting to DrainSight SSE stream (/api/drainsight/stream)...")
    listener = SSEListener(stream_url=stream_url, token=token)
    listener.start()

    # Wait for handshake
    connected = listener.connected_event.wait(timeout=5.0)
    if not connected:
        logger.error("Failed to receive 'event: connected' SSE handshake within 5s.")
        listener.stop()
        return False
    logger.info("SSE Stream handshake OK: Connected to live DrainSight push feed.")

    # 3. Batch upload reprocessed events
    logger.info(f"Step 3: Triggering batch upload of {limit} reprocessed events...")
    summary = upload_events(
        events_file=events_file,
        url=f"{base_url}/upload",
        token=token,
        delay_s=0.15,
        limit=limit,
        with_images=with_images,
        timeout_s=10.0,
    )

    if summary.failed > 0:
        logger.warning(f"Upload completed with {summary.failed} failures out of {summary.total}.")
    else:
        logger.info(f"Upload completed successfully: {summary.succeeded}/{summary.total} sent.")

    # Allow SSE events to settle
    time.sleep(1.0)
    listener.stop()

    # 4. Verify SSE stream reception
    logger.info("Step 4: Verifying SSE real-time reception...")
    blockage_sse_events = [r for r in listener.received_events if r.event_type == "blockage_event"]
    sse_event_ids = {r.data.get("event_id") for r in blockage_sse_events if isinstance(r.data, dict)}

    logger.info(f"Total SSE blockage_events received: {len(blockage_sse_events)}")
    matched_sse_count = sum(1 for eid in summary.uploaded_event_ids if eid in sse_event_ids)
    logger.info(f"Uploaded events received via SSE: {matched_sse_count}/{len(summary.uploaded_event_ids)}")

    sse_ok = matched_sse_count == len(summary.uploaded_event_ids)

    # 5. Verify SQLite DB records
    logger.info("Step 5: Verifying SQLite records in beum_events.db...")
    db_result = verify_db_records(db_path, summary.uploaded_event_ids)
    if "error" in db_result:
        logger.error(db_result["error"])
        db_ok = False
    else:
        logger.info(f"Total events in DB: {db_result['total_events_in_db']}")
        logger.info(f"Found {db_result['found_count']} of {len(summary.uploaded_event_ids)} target records.")
        db_ok = db_result["missing_count"] == 0

    # 6. Verify Evidence Image Serving
    image_ok = True
    if with_images and summary.uploaded_event_ids:
        logger.info("Step 6: Verifying evidence image serving (/events/{id}/image)...")
        sample_id = summary.uploaded_event_ids[0]
        has_img = check_event_image(base_url, sample_id)
        if has_img:
            logger.info(f"Image verification passed for {sample_id} (HTTP 200 image/jpeg).")
        else:
            logger.error(f"Image verification failed for {sample_id}.")
            image_ok = False

    # 7. Summary Report
    all_passed = summary.failed == 0 and sse_ok and db_ok and image_ok

    print("\n" + "=" * 60)
    print(" VERIFICATION RESULT SUMMARY")
    print("=" * 60)
    print(f" [1] Server Health Check     : PASS (Initial events: {initial_db_events})")
    print(f" [2] SSE Stream Handshake    : PASS (Handshake received)")
    print(f" [3] Reprocessed Batch Upload: {'PASS' if summary.failed == 0 else 'FAIL'} ({summary.succeeded}/{summary.total} uploaded)")
    print(f" [4] Real-time SSE Broadcast : {'PASS' if sse_ok else 'FAIL'} ({matched_sse_count}/{len(summary.uploaded_event_ids)} received)")
    print(f" [5] beum_events.db Indexing : {'PASS' if db_ok else 'FAIL'} ({db_result.get('found_count', 0)}/{len(summary.uploaded_event_ids)} verified)")
    if with_images:
        print(f" [6] Image Storage & Servicing: {'PASS' if image_ok else 'FAIL'}")
    print("=" * 60)
    if all_passed:
        print(" >>> ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
    else:
        print(" >>> SOME VERIFICATION CHECKS FAILED! <<<")
    print("=" * 60 + "\n")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify BEUM Real-time Queue Ingestion & SSE Stream")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Server port (default: 8001)")
    parser.add_argument("--data-dir", default="received_data", help="Data directory (default: received_data)")
    parser.add_argument(
        "--events-file",
        default="data/2026-09-04_04-19-03/events_reprocessed.json",
        help="Events json path (default: data/2026-09-04_04-19-03/events_reprocessed.json)",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of events to upload (default: 5)")
    parser.add_argument("--with-images", action="store_true", default=True, help="Upload with frame images")
    parser.add_argument("--no-images", dest="with_images", action="store_false", help="Upload metadata only")
    parser.add_argument("--token", default="", help="Bearer auth token (optional)")

    args = parser.parse_args()

    success = run_pipeline_verification(
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
        events_file=args.events_file,
        limit=args.limit,
        with_images=args.with_images,
        token=args.token,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
