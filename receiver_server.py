"""
BEUM Central Receiver Server
---------------------------
Receives preprocessed blockage events and evidence images from edge devices (e.g. Raspberry Pi).

Features:
- Dual payload format support:
    1. application/json (pure metadata)
    2. multipart/form-data (metadata form field + optional image binary)
- Configurable Bearer Token authentication.
- SQLite indexing for search, analytics, and traceability.
- Structured daily directory storage for evidence JPEG images.
- Query APIs for dashboard/monitoring (/health, /events, /events/{id}/image).
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
import uvicorn


class ReceiverApp:
    def __init__(
        self,
        data_dir: str | Path = "received_data",
        token: str = "",
        title: str = "BEUM Central Receiver Server",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.db_path = self.data_dir / "beum_events.db"
        self.token = token.strip()

        self._init_storage()
        self.app = FastAPI(title=title, version="1.0.0")
        self._setup_middleware()
        self._setup_routes()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_storage(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    source TEXT,
                    created_at TEXT,
                    received_at TEXT,
                    blockage_status TEXT,
                    coverage_percent REAL,
                    roi_profile TEXT,
                    lat REAL,
                    lon REAL,
                    has_image BOOLEAN,
                    image_path TEXT,
                    raw_payload TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_received_at ON events(received_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_blockage_status ON events(blockage_status)")
            conn.commit()

    def _setup_middleware(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _verify_token(self, authorization: Annotated[str | None, Header()] = None) -> None:
        if not self.token:
            return
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization Header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        scheme, _, provided_token = authorization.partition(" ")
        if scheme.lower() != "bearer" or provided_token.strip() != self.token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Bearer Token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def _save_event_record(
        self,
        event_id: str,
        event_data: dict[str, Any],
        received_at: str,
        image_path: str | None,
    ) -> None:
        blockage = event_data.get("blockage") or {}
        policy = event_data.get("policy") or {}
        gps = event_data.get("gps") or {}

        if isinstance(gps, dict):
            lat = gps.get("latitude") if gps.get("latitude") is not None else gps.get("lat")
            lon = gps.get("longitude") if gps.get("longitude") is not None else (gps.get("lon") or gps.get("lng"))
        else:
            lat = None
            lon = None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO events (
                    event_id, event_type, source, created_at, received_at,
                    blockage_status, coverage_percent, roi_profile, lat, lon,
                    has_image, image_path, raw_payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    str(event_data.get("event_type", "unknown")),
                    str(event_data.get("source", "")),
                    str(event_data.get("created_at", "")),
                    received_at,
                    str(blockage.get("status", "")),
                    float(blockage.get("coverage_percent", 0.0) or 0.0),
                    str(policy.get("roi_profile", "")),
                    float(lat) if lat is not None else None,
                    float(lon) if lon is not None else None,
                    image_path is not None,
                    image_path,
                    json.dumps(event_data, ensure_ascii=False),
                ),
            )
            conn.commit()

    def _setup_routes(self) -> None:
        @self.app.get("/health", status_code=status.HTTP_200_OK)
        def health_check():
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM events")
                total_events = cur.fetchone()[0]

            return {
                "status": "ok",
                "time": datetime.now(timezone.utc).isoformat(),
                "total_events": total_events,
                "auth_required": bool(self.token),
            }

        @self.app.post("/upload", status_code=status.HTTP_201_CREATED)
        async def upload_event(
            request: Request,
            authorization: Annotated[str | None, Header()] = None,
        ):
            self._verify_token(authorization)

            content_type = request.headers.get("content-type", "")
            event_data: dict[str, Any] | None = None
            image_bytes: bytes | None = None
            image_filename = "evidence.jpg"

            if "multipart/form-data" in content_type:
                try:
                    form = await request.form()
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Failed to parse multipart form: {exc}")

                metadata_field = form.get("metadata")
                if not metadata_field:
                    raise HTTPException(status_code=400, detail="Missing required 'metadata' field in form")

                if isinstance(metadata_field, str):
                    try:
                        event_data = json.loads(metadata_field)
                    except json.JSONDecodeError as exc:
                        raise HTTPException(status_code=400, detail=f"Invalid JSON in 'metadata': {exc}")
                else:
                    try:
                        raw_meta = await metadata_field.read()
                        event_data = json.loads(raw_meta.decode("utf-8"))
                    except Exception as exc:
                        raise HTTPException(status_code=400, detail=f"Invalid JSON in 'metadata': {exc}")

                image_field = form.get("image")
                if image_field is not None:
                    if hasattr(image_field, "read"):
                        image_bytes = await image_field.read()
                        image_filename = getattr(image_field, "filename", "evidence.jpg") or "evidence.jpg"
                    elif isinstance(image_field, bytes):
                        image_bytes = image_field

            elif "application/json" in content_type:
                try:
                    event_data = await request.json()
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")
            else:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported Content-Type: {content_type}. Expected multipart/form-data or application/json",
                )

            if not isinstance(event_data, dict):
                raise HTTPException(status_code=400, detail="Metadata payload must be a JSON object")

            event_id = event_data.get("event_id")
            if not event_id:
                source = str(event_data.get("source", "dev"))
                sensors = event_data.get("sensors")
                ts = sensors.get("timestamp") if isinstance(sensors, dict) else None
                if ts is not None:
                    event_id = f"{source}_{int(ts)}"
                elif event_data.get("created_at"):
                    clean_ts = str(event_data.get("created_at")).replace(":", "-").replace(" ", "_")
                    event_id = f"{source}_{clean_ts}"
                else:
                    event_id = f"evt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
            else:
                event_id = str(event_id)

            received_at = datetime.now(timezone.utc).isoformat()

            saved_image_path: str | None = None
            if image_bytes:
                date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                sub_dir = self.images_dir / date_str
                await run_in_threadpool(sub_dir.mkdir, parents=True, exist_ok=True)
                ext = Path(image_filename).suffix or ".jpg"
                target_image_path = sub_dir / f"{event_id}{ext}"
                await run_in_threadpool(target_image_path.write_bytes, image_bytes)
                saved_image_path = str(target_image_path)

            await run_in_threadpool(self._save_event_record, event_id, event_data, received_at, saved_image_path)

            return {
                "status": "success",
                "event_id": event_id,
                "received_at": received_at,
                "image_stored": saved_image_path is not None,
            }

        @self.app.get("/events", status_code=status.HTTP_200_OK)
        def list_events(
            limit: int = Query(50, ge=1, le=500),
            offset: int = Query(0, ge=0),
            blockage_status: str | None = Query(None),
        ):
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()

                query = "SELECT * FROM events"
                params: list[Any] = []
                if blockage_status:
                    query += " WHERE blockage_status = ?"
                    params.append(blockage_status)

                query += " ORDER BY received_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cur.execute(query, params)
                rows = cur.fetchall()

                cur.execute("SELECT COUNT(*) FROM events")
                total = cur.fetchone()[0]

            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "events": [
                    {
                        "event_id": row["event_id"],
                        "event_type": row["event_type"],
                        "source": row["source"],
                        "created_at": row["created_at"],
                        "received_at": row["received_at"],
                        "blockage_status": row["blockage_status"],
                        "coverage_percent": row["coverage_percent"],
                        "roi_profile": row["roi_profile"],
                        "gps": {"lat": row["lat"], "lon": row["lon"]} if row["lat"] is not None else None,
                        "has_image": bool(row["has_image"]),
                    }
                    for row in rows
                ],
            }

        @self.app.get("/events/{event_id}", status_code=status.HTTP_200_OK)
        def get_event_detail(event_id: str):
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT raw_payload, has_image, image_path FROM events WHERE event_id = ?", (event_id,))
                row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")

            payload = json.loads(row["raw_payload"])
            payload["_has_image"] = bool(row["has_image"])
            return payload

        @self.app.get("/events/{event_id}/image", status_code=status.HTTP_200_OK)
        def get_event_image(event_id: str):
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT image_path FROM events WHERE event_id = ?", (event_id,))
                row = cur.fetchone()

            if not row or not row[0]:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence image not found")

            path = Path(row[0])
            if not path.exists():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file missing from disk")

            return FileResponse(path, media_type="image/jpeg")


def create_app(data_dir: str = "received_data", token: str = "") -> FastAPI:
    receiver = ReceiverApp(data_dir=data_dir, token=token)
    return receiver.app


def main() -> None:
    parser = argparse.ArgumentParser(description="BEUM Central Receiver Server")
    parser.add_argument(
        "--host",
        default=os.getenv("RECEIVER_HOST", "0.0.0.0"),
        help="Host address to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("RECEIVER_PORT", "8000")),
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("RECEIVER_TOKEN", os.getenv("UPLOAD_TOKEN", "")),
        help="Bearer token for authentication (empty to disable auth)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("RECEIVER_DATA_DIR", "received_data"),
        help="Directory to store events DB and evidence images (default: received_data)",
    )

    args = parser.parse_args()

    print("==================================================")
    print("[*] Starting BEUM Central Receiver Server")
    print(f"[*] Binding Address : http://{args.host}:{args.port}")
    print(f"[*] Storage Directory: {Path(args.data_dir).resolve()}")
    print(f"[*] Auth Token       : {'[Configured]' if args.token else '[Disabled - Dev Mode]'}")
    print("[!] Windows Firewall: If edge device connection times out, run PowerShell as Admin:")
    print(f'   New-NetFirewallRule -DisplayName "BEUM Receiver" -Direction Inbound -LocalPort {args.port} -Protocol TCP -Action Allow')
    print("==================================================")

    app_instance = create_app(data_dir=args.data_dir, token=args.token)
    uvicorn.run(app_instance, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
