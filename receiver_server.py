"""
BEUM Central Receiver Server & DrainSight Backend
-------------------------------------------------
Receives preprocessed blockage events, evidence images, detections, and telemetry.

Features:
- Dual payload format support:
    1. application/json (pure metadata)
    2. multipart/form-data (metadata form field + optional image binary)
- Configurable Bearer Token authentication.
- SQLite indexing for events, detections, and vehicle telemetry states.
- Structured daily directory storage for evidence JPEG images.
- Query APIs for dashboard/monitoring (/health, /events, /events/{id}/image).
- DrainSight adapter integration for GeoJSON, SSE, WebSocket streaming, and webhooks.
- Strict schema validation with Pydantic for Detection & Telemetry contracts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from starlette.concurrency import run_in_threadpool
import uvicorn

from drainsight_adapter import DrainSightAdapter, get_dashboard_html
from migrations import migrate_database
from schemas import (
    DetectionCreate,
    TelemetryIn,
    EventSchema,
    validate_detection,
    validate_telemetry,
    validate_event_payload,
)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two coordinates in meters."""
    R = 6371000.0  # WGS84 mean earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


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
        self.token = token.strip() if token else ""
        self.pending_silence_checks: dict[tuple[int | str, str], asyncio.Task] = {}
        self.active_silence_keys: set[tuple[int | str, str]] = set()

        self._setup_logger()
        self._init_storage()
        self.app = FastAPI(title=title, version="1.0.0")
        self.adapter = DrainSightAdapter(
            get_db_conn=self._get_connection,
            images_dir=self.images_dir,
        )
        self.app.include_router(self.adapter.router)
        self._setup_middleware()
        self._setup_routes()

    @contextmanager
    def _get_connection(self, timeout: float = 15.0, autocommit: bool = True):
        conn = sqlite3.connect(self.db_path, timeout=timeout, isolation_level=None if autocommit else "")
        conn.execute("PRAGMA busy_timeout=15000")
        try:
            yield conn
        finally:
            conn.close()

    def _init_storage(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        migrate_database(self.db_path)

    def _setup_logger(self) -> None:
        log_dir = self.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(f"DrainSightReceiver_{id(self)}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = RotatingFileHandler(
            log_dir / "receiver.log",
            maxBytes=10_485_760,
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self.logger = logger
        self.logger.info("Logger initialized (file + console)")

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

        coverage_percent = float(
            blockage.get("coverage_percent")
            if blockage.get("coverage_percent") is not None
            else (event_data.get("coverage_percent", 0.0) or 0.0)
        )
        occlusion_pct = event_data.get("occlusion_pct")
        if occlusion_pct is None:
            occlusion_pct = coverage_percent

        status_val = str(blockage.get("status") or event_data.get("status", "") or "")

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO events (
                    event_id, event_type, source, created_at, received_at,
                    blockage_status, coverage_percent, occlusion_pct, roi_profile, lat, lon,
                    has_image, image_path, raw_payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    str(event_data.get("event_type", "unknown")),
                    str(event_data.get("source", "")),
                    str(event_data.get("created_at", "")),
                    received_at,
                    status_val,
                    float(coverage_percent),
                    float(occlusion_pct) if occlusion_pct is not None else None,
                    str(policy.get("roi_profile", "")),
                    float(lat) if lat is not None else None,
                    float(lon) if lon is not None else None,
                    image_path is not None,
                    image_path,
                    json.dumps(event_data, ensure_ascii=False),
                ),
            )
            conn.commit()

    def _find_nearby_drains(
        self,
        lat: float | None,
        lon: float | None,
        radius_m: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Find drains within radius_m using haversine distance."""
        if lat is None or lon is None:
            return []
        nearby = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            # 1. Registered drains
            cur.execute("SELECT id, lat, lng FROM drains")
            for r in cur.fetchall():
                dist = haversine_m(lat, lon, float(r["lat"]), float(r["lng"]))
                if dist <= radius_m:
                    nearby.append({
                        "id": r["id"],
                        "lat": float(r["lat"]),
                        "lon": float(r["lng"]),
                        "distance_m": dist,
                    })
            # 2. Events table
            cur.execute("SELECT event_id, lat, lon FROM events WHERE lat IS NOT NULL AND lon IS NOT NULL")
            for r in cur.fetchall():
                dist = haversine_m(lat, lon, float(r["lat"]), float(r["lon"]))
                if dist <= radius_m:
                    # Parse as int if numeric, else keep str
                    evt_id = r["event_id"]
                    try:
                        parsed_id = int(evt_id)
                    except (ValueError, TypeError):
                        parsed_id = evt_id
                    nearby.append({
                        "id": parsed_id,
                        "lat": float(r["lat"]),
                        "lon": float(r["lon"]),
                        "distance_m": dist,
                    })
        return nearby

    def _update_vehicle_telemetry_atomic(
        self,
        vehicle_code: str,
        sequence: int,
        timestamp: float,
    ) -> tuple[bool, str | None]:
        """
        Atomically inspects and updates vehicle telemetry state in SQLite.
        Guarantees mutual exclusion and race-free TOCTOU protection across
        concurrent workers via SQLite BEGIN IMMEDIATE transaction and conditional UPSERT.

        Returns:
            (is_accepted: bool, rejection_reason: str | None)
        """
        with self._get_connection(autocommit=True) as conn:
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                cur.execute(
                    "SELECT last_sequence, last_timestamp FROM vehicle_telemetry_states WHERE vehicle_code = ?",
                    (vehicle_code,),
                )
                row = cur.fetchone()
                if row is not None:
                    last_seq = row[0]
                    if sequence == last_seq:
                        cur.execute("ROLLBACK")
                        return False, "duplicate"
                    if sequence < last_seq:
                        cur.execute("ROLLBACK")
                        return False, "out_of_order"

                now_iso = datetime.now(timezone.utc).isoformat()
                cur.execute(
                    """
                    INSERT INTO vehicle_telemetry_states (vehicle_code, last_sequence, last_timestamp, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(vehicle_code) DO UPDATE SET
                        last_sequence = excluded.last_sequence,
                        last_timestamp = excluded.last_timestamp,
                        updated_at = excluded.updated_at
                    WHERE excluded.last_sequence > vehicle_telemetry_states.last_sequence
                    """,
                    (vehicle_code, sequence, timestamp, now_iso),
                )
                cur.execute("COMMIT")
                return True, None
            except Exception:
                cur.execute("ROLLBACK")
                raise

    async def _silence_timer_task(self, drain_id: int | str, vehicle_code: str, timestamp: float) -> None:
        """Silence timer task for a specific (drain_id, vehicle_code) pair."""
        key = (drain_id, vehicle_code)
        try:
            await asyncio.sleep(10.0)
            self.active_silence_keys.discard(key)
        except asyncio.CancelledError:
            pass
        finally:
            if self.pending_silence_checks.get(key) is asyncio.current_task():
                self.pending_silence_checks.pop(key, None)

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

        @self.app.get("/dashboard", response_class=HTMLResponse, summary="DrainSight Web Control Dashboard")
        def dashboard_page():
            return HTMLResponse(content=get_dashboard_html())

        @self.app.websocket("/ws")
        async def root_ws(websocket: WebSocket) -> None:
            if hasattr(self.adapter, "_handle_ws"):
                await self.adapter._handle_ws(websocket, "general")

        @self.app.websocket("/ws/dashboard")
        async def root_ws_dashboard(websocket: WebSocket) -> None:
            if hasattr(self.adapter, "_handle_ws"):
                await self.adapter._handle_ws(websocket, "dashboard")

        @self.app.post("/api/detections", status_code=status.HTTP_201_CREATED, summary="Submit drain detection")
        async def post_detection(
            payload: DetectionCreate,
            authorization: Annotated[str | None, Header()] = None,
        ):
            self._verify_token(authorization)
            created_at = datetime.now(timezone.utc).isoformat()

            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO detections (
                        drain_id, vehicle_code, status, occlusion_pct, reason_code,
                        confidence, lat, lng, source, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload.drain_id,
                        payload.vehicle_code,
                        payload.status.value if hasattr(payload.status, "value") else str(payload.status),
                        payload.occlusion_pct,
                        payload.reason_code,
                        payload.confidence,
                        payload.lat,
                        payload.lng,
                        payload.source.value if hasattr(payload.source, "value") else str(payload.source),
                        created_at,
                    ),
                )
                det_id = cur.lastrowid
                conn.commit()

            ws_event = {
                "type": "detection",
                "id": det_id,
                "drain_id": payload.drain_id,
                "vehicle_code": payload.vehicle_code,
                "status": str(payload.status),
                "occlusion_pct": payload.occlusion_pct,
                "reason_code": payload.reason_code,
                "confidence": payload.confidence,
                "lat": payload.lat,
                "lng": payload.lng,
                "source": str(payload.source),
                "created_at": created_at,
            }
            await self.adapter.broadcast_detection(ws_event)

            return {
                "ok": True,
                "id": det_id,
                "drain_id": payload.drain_id,
                "status": str(payload.status),
                "occlusion_pct": payload.occlusion_pct,
            }

        @self.app.post("/upload", status_code=status.HTTP_201_CREATED)
        async def upload_event(
            request: Request,
            authorization: Annotated[str | None, Header()] = None,
        ):
            self._verify_token(authorization)
            self.logger.info(f"Upload request received | content-type={request.headers.get('content-type', '')} | auth={'yes' if authorization else 'no'}")

            content_type = request.headers.get("content-type", "")
            event_data: dict[str, Any] | None = None
            image_bytes: bytes | None = None
            image_filename = "evidence.jpg"

            if "multipart/form-data" in content_type:
                try:
                    form = await request.form()
                except Exception as exc:
                    self.logger.error(f"Multipart parse failed: {exc}")
                    raise HTTPException(status_code=400, detail=f"Failed to parse multipart form: {exc}")

                metadata_field = form.get("metadata")
                if not metadata_field:
                    self.logger.warning("Upload rejected: missing 'metadata' field")
                    raise HTTPException(status_code=400, detail="Missing required 'metadata' field in form")

                if isinstance(metadata_field, str):
                    try:
                        event_data = json.loads(metadata_field)
                    except json.JSONDecodeError as exc:
                        self.logger.error(f"Invalid JSON in metadata: {exc}")
                        raise HTTPException(status_code=400, detail=f"Invalid JSON in 'metadata': {exc}")
                else:
                    try:
                        raw_meta = await metadata_field.read()
                        event_data = json.loads(raw_meta.decode("utf-8"))
                    except Exception as exc:
                        self.logger.error(f"Invalid metadata read: {exc}")
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

            # Validate occlusion_pct if provided
            blockage = event_data.get("blockage") or {}
            cov = blockage.get("coverage_percent") if blockage.get("coverage_percent") is not None else event_data.get("coverage_percent")
            occ = event_data.get("occlusion_pct")
            if cov is not None:
                try:
                    fcov = float(cov)
                    if fcov < 0.0 or fcov > 100.0:
                        raise HTTPException(status_code=422, detail=f"coverage_percent must be 0-100, got {cov}")
                except (ValueError, TypeError):
                    raise HTTPException(status_code=422, detail="Invalid coverage_percent")
            if occ is not None:
                try:
                    focc = float(occ)
                    if focc < 0.0 or focc > 100.0:
                        raise HTTPException(status_code=422, detail=f"occlusion_pct must be 0-100, got {occ}")
                except (ValueError, TypeError):
                    raise HTTPException(status_code=422, detail="Invalid occlusion_pct")

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

            # Proximity drains
            gps = event_data.get("gps") or {}
            plat = gps.get("lat") or gps.get("latitude")
            plon = gps.get("lon") or gps.get("longitude")
            nearby = self._find_nearby_drains(
                float(plat) if plat is not None else None,
                float(plon) if plon is not None else None,
                radius_m=5.0,
            )

            await self.adapter.broadcast_event(
                event_id=event_id,
                event_data=event_data,
                received_at=received_at,
                has_image=saved_image_path is not None,
                nearby_drains=nearby,
            )

            return {
                "status": "success",
                "event_id": event_id,
                "received_at": received_at,
                "image_stored": saved_image_path is not None,
            }

        async def _handle_telemetry_post(payload: TelemetryIn):
            # Atomic TOCTOU-safe multi-worker state verification & transition
            accepted, reason = await run_in_threadpool(
                self._update_vehicle_telemetry_atomic,
                payload.vehicle_code,
                payload.sequence,
                payload.timestamp,
            )
            if not accepted:
                return {
                    "ok": True,
                    "processed": False,
                    "reason": reason,
                    "sequence": payload.sequence,
                }

            # Proximity calculation (<= 5.0m)
            nearby = self._find_nearby_drains(payload.lat, payload.lng, radius_m=5.0)
            nearby_ids = [d["id"] for d in nearby]

            # Silence Timer Scheduling per (drain_id, vehicle_code)
            for d in nearby:
                timer_key = (d["id"], payload.vehicle_code)
                if timer_key in self.pending_silence_checks:
                    self.pending_silence_checks[timer_key].cancel()
                self.active_silence_keys.add(timer_key)
                try:
                    self.pending_silence_checks[timer_key] = asyncio.create_task(
                        self._silence_timer_task(d["id"], payload.vehicle_code, payload.timestamp)
                    )
                except RuntimeError:
                    pass

            now_iso = datetime.now(timezone.utc).isoformat()

            # WebSocket Broadcast (accepted packet only)
            ws_event = {
                "type": "telemetry",
                "vehicle_code": payload.vehicle_code,
                "lat": payload.lat,
                "lng": payload.lng,
                "timestamp": payload.timestamp,
                "sequence": payload.sequence,
                "speed_mps": payload.speed_mps,
                "nearby_drains": nearby_ids,
                "server_time": now_iso,
            }
            await self.adapter.broadcast_telemetry(ws_event)

            return {
                "ok": True,
                "status": "success",
                "event_id": f"tel_{payload.vehicle_code}_{payload.sequence}",
                "processed": True,
                "nearby_drains": nearby_ids,
                "sequence": payload.sequence,
            }

        @self.app.post("/api/telemetry", status_code=status.HTTP_200_OK, summary="Submit vehicle telemetry")
        async def api_telemetry_endpoint(payload: TelemetryIn):
            return await _handle_telemetry_post(payload)

        @self.app.post("/telemetry", status_code=status.HTTP_200_OK, summary="Telemetry endpoint alias")
        async def telemetry_alias_endpoint(payload: TelemetryIn):
            return await _handle_telemetry_post(payload)

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
                        "occlusion_pct": row["occlusion_pct"] if "occlusion_pct" in row.keys() else None,
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
    host_display = "localhost" if args.host in ("0.0.0.0", "") else args.host
    print(f"[*] Web Dashboard   : http://{host_display}:{args.port}/dashboard")
    print(f"[*] GeoJSON API     : http://{host_display}:{args.port}/api/drainsight/geojson")
    print(f"[*] Storage Directory: {Path(args.data_dir).resolve()}")
    print(f"[*] Auth Token       : {'[Configured]' if args.token else '[Disabled - Dev Mode]'}")
    print("[!] Windows Firewall: If edge device connection times out, run PowerShell as Admin:")
    print(f'   New-NetFirewallRule -DisplayName "BEUM Receiver" -Direction Inbound -LocalPort {args.port} -Protocol TCP -Action Allow')
    print("==================================================")

    app_instance = create_app(data_dir=args.data_dir, token=args.token)
    uvicorn.run(app_instance, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
