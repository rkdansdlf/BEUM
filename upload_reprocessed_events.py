#!/usr/bin/env python3
"""
Batch Uploader for Reprocessed BEUM Gully Events
------------------------------------------------
Uploads reprocessed events (from events_reprocessed.json) to the BEUM Central Receiver Server (/upload).
Supports automatic evidence frame image matching (multipart/form-data) and pure metadata upload (application/json).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ReprocessedUploader")


def build_frame_index(frames_dir: Path | str | None) -> dict[float, Path]:
    """
    Builds a timestamp -> Path index for extracted frame images.
    Filename pattern expected: frame_00051_t0100.39.jpg -> 100.39
    """
    if not frames_dir:
        return {}
    f_dir = Path(frames_dir)
    if not f_dir.exists() or not f_dir.is_dir():
        return {}

    frame_index: dict[float, Path] = {}
    for p in f_dir.glob("*.jpg"):
        name = p.name
        # Format: frame_XXXXX_tYYYY.YY.jpg
        parts = name.split("_")
        if len(parts) >= 3 and "t" in parts[2]:
            t_str = parts[2].replace("t", "").replace(".jpg", "")
            try:
                ts = float(t_str)
                frame_index[ts] = p
            except ValueError:
                pass
    return frame_index


def find_matching_frame(ts_s: float | None, frame_index: dict[float, Path], max_delta_s: float = 1.5) -> Path | None:
    """Finds the closest frame within max_delta_s seconds."""
    if ts_s is None or not frame_index:
        return None

    closest_ts = min(frame_index.keys(), key=lambda t: abs(t - ts_s))
    if abs(closest_ts - ts_s) <= max_delta_s:
        return frame_index[closest_ts]
    return None


def format_reprocessed_event(raw_event: dict[str, Any], session_name: str, index: int) -> dict[str, Any]:
    """Formats a reprocessed event item into a standard BEUM event payload."""
    frame_no = raw_event.get("frame", index)
    ts_s = raw_event.get("ts_s")
    event_id = raw_event.get("event_id") or f"reproc_{session_name}_f{int(frame_no):05d}"

    status = raw_event.get("status", "unknown")
    coverage_percent = float(raw_event.get("coverage_percent", 0.0) or 0.0)

    gps = raw_event.get("gps")
    created_at = (
        raw_event.get("created_at")
        or (gps.get("timestamp") if isinstance(gps, dict) else None)
        or datetime.now(timezone.utc).isoformat()
    )

    blockage = {
        "status": status,
        "coverage_percent": round(coverage_percent, 2),
        "method": raw_event.get("method", "segmentation_mask"),
        "gully_count": raw_event.get("gully_count", 0),
        "obstacle_count": raw_event.get("obstacle_count", 0),
    }

    payload = {
        "event_id": event_id,
        "event_type": raw_event.get("event_type", "gully_blockage"),
        "source": raw_event.get("source", f"reprocessed_{session_name}"),
        "created_at": created_at,
        "blockage": blockage,
        "occlusion_pct": round(coverage_percent, 2),
        "gps": gps,
        "detections": raw_event.get("detections", []),
        "frame": frame_no,
        "ts_s": ts_s,
    }
    return payload


def send_http_event(
    url: str,
    event_payload: dict[str, Any],
    image_path: Path | None = None,
    token: str = "",
    timeout_s: float = 10.0,
) -> tuple[int, dict[str, Any]]:
    """Sends event to /upload endpoint using multipart or json."""
    metadata_bytes = json.dumps(event_payload, ensure_ascii=False).encode("utf-8")
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if image_path is not None and image_path.exists():
        boundary = f"----BEUMReprocessedUploader{int(time.time()*1000)}"
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        image_bytes = image_path.read_bytes()
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"

        body = b"".join([
            b"--" + boundary.encode("ascii") + b"\r\n",
            b'Content-Disposition: form-data; name="metadata"\r\n',
            b"Content-Type: application/json; charset=utf-8\r\n\r\n",
            metadata_bytes,
            b"\r\n",
            b"--" + boundary.encode("ascii") + b"\r\n",
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("ascii"),
            image_bytes,
            b"\r\n",
            b"--" + boundary.encode("ascii") + b"--\r\n",
        ])
    else:
        headers["Content-Type"] = "application/json; charset=utf-8"
        body = metadata_bytes

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp_bytes = resp.read()
            resp_data = json.loads(resp_bytes.decode("utf-8")) if resp_bytes else {}
            return resp.status, resp_data
    except urllib.error.HTTPError as err:
        err_body = err.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
        except Exception:
            err_json = {"detail": err_body}
        return err.code, err_json
    except urllib.error.URLError as err:
        raise RuntimeError(f"Connection failed: {err}") from err


@dataclass
class UploadSummary:
    total: int
    succeeded: int
    failed: int
    duration_s: float
    uploaded_event_ids: list[str]


def upload_events(
    events_file: Path | str,
    frames_dir: Path | str | None = None,
    url: str = "http://localhost:8001/upload",
    token: str = "",
    delay_s: float = 0.1,
    limit: int | None = None,
    with_images: bool = True,
    timeout_s: float = 10.0,
) -> UploadSummary:
    """Uploads reprocessed events batch."""
    events_path = Path(events_file)
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    with open(events_path, "r", encoding="utf-8") as f:
        raw_events: list[dict[str, Any]] = json.load(f)

    if not isinstance(raw_events, list):
        raise ValueError(f"Expected JSON list in {events_path}, got {type(raw_events)}")

    # Derive session name from directory name
    session_name = events_path.parent.name
    if session_name in ("", "."):
        session_name = "session"

    # Frame indexing
    resolved_frames_dir: Path | None = None
    if with_images:
        if frames_dir:
            resolved_frames_dir = Path(frames_dir)
        else:
            default_frames = events_path.parent / "frames"
            if default_frames.exists() and default_frames.is_dir():
                resolved_frames_dir = default_frames

    frame_index = build_frame_index(resolved_frames_dir) if resolved_frames_dir else {}
    logger.info(
        f"Loaded {len(raw_events)} events from {events_path.name} | "
        f"Frames indexed: {len(frame_index)} from {resolved_frames_dir or 'None'}"
    )

    items_to_upload = raw_events[:limit] if limit is not None and limit > 0 else raw_events
    total_count = len(items_to_upload)
    succeeded = 0
    failed = 0
    uploaded_ids: list[str] = []

    start_time = time.time()
    for idx, raw_item in enumerate(items_to_upload, start=1):
        formatted = format_reprocessed_event(raw_item, session_name=session_name, index=idx)
        evt_id = formatted["event_id"]

        matched_frame: Path | None = None
        if with_images and frame_index:
            matched_frame = find_matching_frame(formatted.get("ts_s"), frame_index)

        try:
            status_code, resp_data = send_http_event(
                url=url,
                event_payload=formatted,
                image_path=matched_frame,
                token=token,
                timeout_s=timeout_s,
            )

            if 200 <= status_code < 300:
                succeeded += 1
                uploaded_ids.append(evt_id)
                img_desc = f"with image ({matched_frame.name})" if matched_frame else "json only"
                logger.info(
                    f"[{idx:03d}/{total_count:03d}] HTTP {status_code} | "
                    f"ID: {evt_id} | Status: {formatted['blockage']['status']} | "
                    f"Cov: {formatted['blockage']['coverage_percent']}% | {img_desc}"
                )
            else:
                failed += 1
                logger.error(
                    f"[{idx:03d}/{total_count:03d}] HTTP {status_code} FAILED | "
                    f"ID: {evt_id} | Detail: {resp_data.get('detail', resp_data)}"
                )
        except Exception as exc:
            failed += 1
            logger.error(f"[{idx:03d}/{total_count:03d}] Exception for {evt_id}: {exc}")

        if delay_s > 0 and idx < total_count:
            time.sleep(delay_s)

    duration = time.time() - start_time
    logger.info(
        f"Batch upload finished in {duration:.2f}s | "
        f"Total: {total_count} | Succeeded: {succeeded} | Failed: {failed}"
    )

    return UploadSummary(
        total=total_count,
        succeeded=succeeded,
        failed=failed,
        duration_s=duration,
        uploaded_event_ids=uploaded_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BEUM Reprocessed Events Batch Uploader")
    parser.add_argument(
        "--events-file",
        default="data/2026-09-04_04-19-03/events_reprocessed.json",
        help="Path to events_reprocessed.json file (default: data/2026-09-04_04-19-03/events_reprocessed.json)",
    )
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Path to frames directory (default: auto-detected next to events file)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8001/upload",
        help="Upload endpoint URL (default: http://localhost:8001/upload)",
    )
    parser.add_argument(
        "--token",
        default="",
        help="Bearer auth token (optional)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between event uploads in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of events to upload (default: all)",
    )
    parser.add_argument(
        "--with-images",
        dest="with_images",
        action="store_true",
        default=True,
        help="Include matched frame images as multipart upload (default: True)",
    )
    parser.add_argument(
        "--no-images",
        dest="with_images",
        action="store_false",
        help="Disable image upload and send pure JSON metadata",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP request timeout in seconds (default: 10.0)",
    )

    args = parser.parse_args()

    try:
        summary = upload_events(
            events_file=args.events_file,
            frames_dir=args.frames_dir,
            url=args.url,
            token=args.token,
            delay_s=args.delay,
            limit=args.limit,
            with_images=args.with_images,
            timeout_s=args.timeout,
        )
        if summary.failed > 0:
            sys.exit(1)
    except Exception as exc:
        logger.critical(f"Upload process terminated with error: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
