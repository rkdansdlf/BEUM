"""
DrainSight Backend Adapter Module
---------------------------------
Provides integration interfaces for the DrainSight Central Monitoring Dashboard,
GIS systems (GeoJSON), real-time SSE streaming, and external webhook notification.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

import websockets

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket

from camera_streamer import LiveCameraStreamer

logger = logging.getLogger("drainsight_adapter")


def calculate_grade(coverage_percent: float) -> tuple[int, str]:
    """
    Classify blockage into 3 operational grades:
    - Grade 1 (0 ~ 30%): Normal / Good condition
    - Grade 2 (30 ~ 60%): Warning / Needs inspection
    - Grade 3 (>= 60%): Critical / Urgent clearing required
    """
    cov = float(coverage_percent or 0.0)
    if cov >= 60.0:
        return 3, "심각 (3등급 - 긴급 준설)"
    elif cov >= 30.0:
        return 2, "주의 (2등급 - 현장 점검)"
    else:
        return 1, "정상 (1등급 - 원활)"


class WebhookConfig(BaseModel):
    url: str = Field(..., description="Target HTTP/HTTPS webhook URL")
    description: str = Field(default="", description="Webhook description or target name")
    min_coverage: float = Field(default=50.0, description="Minimum blockage percentage to trigger")
    min_grade: int = Field(default=2, ge=1, le=3, description="Minimum grade to trigger (1: All, 2: Warning+, 3: Critical)")
    enabled: bool = Field(default=True, description="Whether this webhook is active")


class DrainSightAdapter:
    def __init__(
        self,
        get_db_conn: Callable[[], Any],
        images_dir: Path | str,
        dashboard_template_path: Path | str | None = None,
    ) -> None:
        self._get_db_conn = get_db_conn
        self.images_dir = Path(images_dir)
        self.dashboard_template_path = Path(dashboard_template_path) if dashboard_template_path else None
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._active_ws: list[WebSocket] = []
        self._webhooks: dict[str, dict[str, Any]] = {}
        self.camera_streamer = LiveCameraStreamer()
        self.router = APIRouter(prefix="/api/drainsight", tags=["DrainSight Adapter"])
        self.camera_router = APIRouter(prefix="/api/camera", tags=["Camera Stream & AI Vision"])
        self.gps_router = APIRouter(prefix="/api/gps", tags=["GPS Telemetry Tracking"])
        self._setup_routes()

    def _get_connection(self):
        return self._get_db_conn()

    def _setup_routes(self) -> None:
        @self.router.get("/geojson", summary="Get standard GeoJSON FeatureCollection of all monitored gullies")
        def get_geojson(
            status_filter: str | None = Query(None, alias="status", description="Filter by status: normal, warning, critical"),
            min_grade: int = Query(1, ge=1, le=3, description="Minimum grade filter (1, 2, 3)"),
        ):
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                query = "SELECT * FROM events WHERE lat IS NOT NULL AND lon IS NOT NULL"
                params: list[Any] = []
                if status_filter:
                    query += " AND blockage_status = ?"
                    params.append(status_filter)
                query += " ORDER BY received_at DESC"
                cur.execute(query, params)
                rows = cur.fetchall()

            features = []
            for row in rows:
                coverage = float(row["coverage_percent"] or 0.0)
                try:
                    occlusion_pct = float(row["occlusion_pct"] or coverage)
                except (KeyError, IndexError):
                    occlusion_pct = coverage
                grade_num, grade_label = calculate_grade(occlusion_pct)
                if grade_num < min_grade:
                    continue

                event_id = row["event_id"]
                has_img = bool(row["has_image"])
                image_url = f"/events/{event_id}/image" if has_img else None

                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["lon"]), float(row["lat"])],
                    },
                    "properties": {
                        "id": event_id,
                        "event_type": row["event_type"],
                        "source": row["source"],
                        "created_at": row["created_at"],
                        "received_at": row["received_at"],
                        "status": row["blockage_status"] or "unknown",
                        "coverage_percent": round(coverage, 2),
                        "occlusion_pct": round(occlusion_pct, 2),
                        "grade": grade_num,
                        "grade_label": grade_label,
                        "roi_profile": row["roi_profile"],
                        "has_image": has_img,
                        "image_url": image_url,
                    },
                }
                features.append(feature)

            return {
                "type": "FeatureCollection",
                "features": features,
                "metadata": {
                    "total_features": len(features),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "BEUM Gully Monitoring System",
                },
            }

        @self.router.get("/stats", summary="Get KPI summary statistics for DrainSight control console")
        def get_stats():
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()

                cur.execute("SELECT COUNT(*) as total FROM events")
                total_events = cur.fetchone()["total"]

                cur.execute("""
                    SELECT 
                        COALESCE(blockage_status, 'unknown') as st, 
                        COUNT(*) as cnt 
                    FROM events 
                    GROUP BY blockage_status
                """)
                status_rows = cur.fetchall()
                status_breakdown = {r["st"]: r["cnt"] for r in status_rows}

                cur.execute("""
                    SELECT 
                        AVG(coverage_percent) as avg_cov,
                        MAX(coverage_percent) as max_cov,
                        COUNT(CASE WHEN has_image = 1 THEN 1 END) as with_images
                    FROM events
                """)
                agg_row = cur.fetchone()
                avg_coverage = round(agg_row["avg_cov"] or 0.0, 2)
                max_coverage = round(agg_row["max_cov"] or 0.0, 2)
                with_images = agg_row["with_images"] or 0

                cur.execute("SELECT coverage_percent FROM events")
                all_covs = [r["coverage_percent"] or 0.0 for r in cur.fetchall()]
                grade_counts = {1: 0, 2: 0, 3: 0}
                for cov in all_covs:
                    g, _ = calculate_grade(cov)
                    grade_counts[g] = grade_counts.get(g, 0) + 1

                cur.execute("SELECT DISTINCT source FROM events WHERE source IS NOT NULL AND source != ''")
                sources = [r["source"] for r in cur.fetchall()]

                cur.execute("SELECT received_at FROM events ORDER BY received_at DESC LIMIT 1")
                last_row = cur.fetchone()
                last_event_time = last_row["received_at"] if last_row else None

            return {
                "total_events": total_events,
                "status_breakdown": status_breakdown,
                "grade_breakdown": {
                    "grade_1_normal": grade_counts[1],
                    "grade_2_warning": grade_counts[2],
                    "grade_3_critical": grade_counts[3],
                },
                "average_coverage_percent": avg_coverage,
                "max_coverage_percent": max_coverage,
                "total_with_images": with_images,
                "active_sources": sources,
                "last_event_time": last_event_time,
                "system_status": "operational",
            }

        @self.router.get("/alerts", summary="Get priority actionable blockage alerts")
        def get_alerts(
            min_coverage: float = Query(40.0, description="Minimum coverage percentage threshold"),
            status_filter: str | None = Query(None, alias="status", description="Filter by status (e.g. critical, warning)"),
            limit: int = Query(50, ge=1, le=200, description="Max alerts to return"),
        ):
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                query = "SELECT * FROM events WHERE coverage_percent >= ?"
                params: list[Any] = [min_coverage]
                if status_filter:
                    query += " AND blockage_status = ?"
                    params.append(status_filter)
                query += " ORDER BY coverage_percent DESC, received_at DESC LIMIT ?"
                params.append(limit)
                cur.execute(query, params)
                rows = cur.fetchall()

            alerts = []
            for row in rows:
                cov = float(row["coverage_percent"] or 0.0)
                g_num, g_label = calculate_grade(cov)
                alerts.append({
                    "event_id": row["event_id"],
                    "source": row["source"],
                    "status": row["blockage_status"],
                    "coverage_percent": round(cov, 2),
                    "grade": g_num,
                    "grade_label": g_label,
                    "gps": {"lat": row["lat"], "lon": row["lon"]} if row["lat"] is not None else None,
                    "has_image": bool(row["has_image"]),
                    "image_url": f"/events/{row['event_id']}/image" if row["has_image"] else None,
                    "received_at": row["received_at"],
                })

            return {
                "count": len(alerts),
                "threshold_coverage": min_coverage,
                "alerts": alerts,
            }

        @self.router.get("/stream", summary="Real-time Server-Sent Events (SSE) feed for dashboard live updates")
        async def event_stream(request: Request):
            async def event_generator() -> AsyncGenerator[str, None]:
                queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
                self._subscribers.append(queue)
                try:
                    # Initial connection handshake
                    yield f"event: connected\ndata: {json.dumps({'message': 'Connected to DrainSight Realtime Event Stream', 'time': datetime.now(timezone.utc).isoformat()})}\n\n"
                    while True:
                        if await request.is_disconnected():
                            break
                        try:
                            # Wait for next event with a timeout to send keep-alive heartbeat
                            event_data = await asyncio.wait_for(queue.get(), timeout=15.0)
                            yield f"event: blockage_event\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                        except asyncio.TimeoutError:
                            yield f"event: heartbeat\ndata: {json.dumps({'timestamp': datetime.now(timezone.utc).isoformat()})}\n\n"
                finally:
                    if queue in self._subscribers:
                        self._subscribers.remove(queue)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        async def _handle_ws(websocket: WebSocket, client_type: str = "general") -> None:
            await websocket.accept()
            self._active_ws.append(websocket)
            try:
                await websocket.send_json({
                    "type": "connected",
                    "client": client_type,
                    "message": f"Connected to DrainSight WebSocket ({client_type})",
                    "time": datetime.now(timezone.utc).isoformat(),
                })
                while True:
                    try:
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=25.0)
                    except asyncio.TimeoutError:
                        await websocket.send_json({
                            "type": "heartbeat",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
            except (WebSocketDisconnect, Exception):
                pass
            finally:
                if websocket in self._active_ws:
                    self._active_ws.remove(websocket)

        self._handle_ws = _handle_ws

        @self.router.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket) -> None:
            await _handle_ws(websocket, "general")

        @self.router.websocket("/ws/dashboard")
        async def ws_dashboard_endpoint(websocket: WebSocket) -> None:
            await _handle_ws(websocket, "dashboard")

        @self.router.websocket("/ws/telemetry")
        async def telemetry_ws_endpoint(websocket: WebSocket) -> None:
            await _handle_ws(websocket, "telemetry")

        @self.router.post("/gully-events", summary="Ingest BEUM gully events for DrainSight backend")
        async def ingest_gully_events(request: Request) -> dict[str, Any]:
            """Accept BEUM gully events and ingest them into the DrainSight backend."""
            # Parse incoming JSON payload
            try:
                event_data = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON payload")

            # Extract event properties
            event_id = event_data.get("event_id")
            event_type = event_data.get("event_type", "gully_blockage")
            source = event_data.get("source", "unknown")
            created_at = event_data.get("created_at")
            
            # Extract blockage info
            blockage = event_data.get("blockage", {})
            coverage_percent = blockage.get("coverage_percent", 0.0)
            status = blockage.get("status", "unknown")
            
            # Normalize coverage_percent to occlusion_pct (same scale, just renamed)
            occlusion_pct = round(coverage_percent, 2)
            
            # Calculate grade based on coverage percent
            grade_num, grade_label = calculate_grade(occlusion_pct)
            
            # Prepare event for storage
            event = {
                "event_id": event_id,
                "event_type": event_type,
                "source": source,
                "created_at": created_at,
                "blockage": blockage,
                "gps": blockage.get("gps", {}),
                "coverage_percent": coverage_percent,
                "occlusion_pct": occlusion_pct,
                "grade": grade_num,
                "grade_label": grade_label,
            }
            
            # Save to database
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO events (
                        event_id, event_type, source, created_at, received_at,
                        blockage_status, coverage_percent, occlusion_pct, lat, lon,
                        has_image, image_path, raw_payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        event_type,
                        source,
                        created_at,
                        datetime.now(timezone.utc).isoformat(),
                        status,
                        coverage_percent,
                        occlusion_pct,
                        blockage.get("lat"),
                        blockage.get("lon"),
                        event.get("has_image"),
                        event.get("image_path"),
                        json.dumps(event_data),
                    ),
                )
                conn.commit()
            
            # Broadcast to subscribers via SSE
            await self.broadcast_event(
                event_id=event_id,
                event_data=event,
                received_at=datetime.now(timezone.utc).isoformat(),
                has_image=bool(event.get("has_image")),
            )
            
            return {
                "status": "success",
                "event_id": event_id,
                "event_type": event_type,
                "occlusion_pct": occlusion_pct,
                "grade": grade_num,
                "grade_label": grade_label,
            }

        @self.router.get("/webhooks", summary="List registered outbound webhooks")
        def list_webhooks():
            return {"webhooks": list(self._webhooks.values())}

        @self.router.post("/webhooks", status_code=status.HTTP_201_CREATED, summary="Register a new outbound webhook")
        def register_webhook(webhook: WebhookConfig):
            webhook_id = f"wh_{uuid.uuid4().hex[:8]}"
            record = {
                "id": webhook_id,
                "url": webhook.url,
                "description": webhook.description,
                "min_coverage": webhook.min_coverage,
                "min_grade": webhook.min_grade,
                "enabled": webhook.enabled,
                "registered_at": datetime.now(timezone.utc).isoformat(),
            }
            self._webhooks[webhook_id] = record
            return {"status": "registered", "webhook": record}

        @self.router.delete("/webhooks/{webhook_id}", summary="Delete a registered webhook")
        def delete_webhook(webhook_id: str):
            if webhook_id not in self._webhooks:
                raise HTTPException(status_code=404, detail="Webhook not found")
            deleted = self._webhooks.pop(webhook_id)
            return {"status": "deleted", "webhook_id": webhook_id}

        # Camera & AI Vision Endpoints
        @self.router.get("/camera/stream", summary="Real-time MJPEG camera stream with YOLO object detection")
        @self.camera_router.get("/stream", summary="Real-time MJPEG camera stream with YOLO object detection")
        def camera_stream_endpoint():
            return StreamingResponse(
                self.camera_streamer.get_mjpeg_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @self.router.get("/camera/snapshot", summary="Get latest annotated JPEG frame for Safari/mobile fallback")
        @self.camera_router.get("/snapshot", summary="Get latest annotated JPEG frame for Safari/mobile fallback")
        def camera_snapshot_endpoint():
            if not self.camera_streamer._running:
                self.camera_streamer.start()
            with self.camera_streamer._lock:
                jpeg = self.camera_streamer.latest_jpeg
            if jpeg is None:
                raise HTTPException(status_code=503, detail="Frame not ready")
            return Response(
                content=jpeg,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

        @self.router.get("/camera/status", summary="Get live camera detection status and FPS")
        @self.camera_router.get("/status", summary="Get live camera detection status and FPS")
        def camera_status_endpoint():
            return self.camera_streamer.get_status()

        @self.router.get("/camera/pi-status", summary="Proxy Raspberry Pi hardware diagnostic and GPS status")
        @self.camera_router.get("/pi-status", summary="Proxy Raspberry Pi hardware diagnostic and GPS status")
        def camera_pi_status_endpoint():
            import urllib.request
            try:
                src = getattr(self.camera_streamer, "video_source", "")
                pi_ip = "172.30.1.67"
                if "://" in src:
                    host_part = src.split("://")[1].split("/")[0].split(":")[0]
                    if host_part:
                        pi_ip = host_part
                url = f"http://{pi_ip}:8080/status.json"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=1.2) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                return {"camera_ok": False, "error": str(e), "gps_error": "Connecting to Pi..."}

        @self.router.post("/camera/detect", summary="Single-frame YOLO detection for browser webcam / client uploads")
        @self.camera_router.post("/detect", summary="Single-frame YOLO detection for browser webcam / client uploads")
        async def camera_detect_endpoint(request: Request):
            content_type = request.headers.get("content-type", "")
            image_bytes = None
            conf = None
            if "multipart/form-data" in content_type:
                form = await request.form()
                file_field = form.get("file") or form.get("image")
                if file_field and hasattr(file_field, "read"):
                    image_bytes = await file_field.read()
                conf_str = form.get("conf")
                if conf_str:
                    try:
                        conf = float(conf_str)
                    except ValueError:
                        pass
            elif "application/json" in content_type:
                body = await request.json()
                b64 = body.get("image_base64") or body.get("image") or ""
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                if b64:
                    image_bytes = base64.b64decode(b64)
                if "conf" in body:
                    conf = float(body["conf"])
            else:
                image_bytes = await request.body()

            if not image_bytes:
                raise HTTPException(status_code=400, detail="Missing image data")

            return self.camera_streamer.detect_single_image(image_bytes, conf)

        @self.router.post("/camera/config", summary="Configure camera streamer source and confidence")
        @self.camera_router.post("/config", summary="Configure camera streamer source and confidence")
        async def camera_config_endpoint(request: Request):
            body = await request.json()
            if "source" in body:
                self.camera_streamer.set_source(body["source"])
            if "conf" in body:
                self.camera_streamer.set_confidence(float(body["conf"]))
            return {"ok": True, "status": self.camera_streamer.get_status()}

        # GPS & Telemetry Tracking Endpoints
        @self.router.get("/gps/sources", summary="Get all known GPS transmitting sources and active status")
        @self.gps_router.get("/sources", summary="Get all known GPS transmitting sources and active status")
        def get_gps_sources():
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM vehicle_telemetry_states ORDER BY updated_at DESC")
                states = cur.fetchall()

                cur.execute("SELECT vehicle_code, COUNT(*) as cnt FROM telemetry_logs GROUP BY vehicle_code")
                counts = {r["vehicle_code"]: r["cnt"] for r in cur.fetchall()}

            now_dt = datetime.now(timezone.utc)
            sources = []
            for row in states:
                v_code = row["vehicle_code"]
                updated_at_str = row["updated_at"]
                is_online = False
                age_s = 999999.0
                if updated_at_str:
                    try:
                        clean_ts = updated_at_str.replace("Z", "+00:00")
                        up_dt = datetime.fromisoformat(clean_ts)
                        age_s = max(0.0, (now_dt - up_dt).total_seconds())
                        is_online = age_s <= 30.0
                    except Exception:
                        pass

                lat = row["lat"] if "lat" in row.keys() else None
                lng = row["lng"] if "lng" in row.keys() else None
                speed_mps = row["speed_mps"] if "speed_mps" in row.keys() else 0.0
                client_ip = row["client_ip"] if "client_ip" in row.keys() else "127.0.0.1"

                sources.append({
                    "vehicle_code": v_code,
                    "client_ip": client_ip or "127.0.0.1",
                    "lat": lat,
                    "lng": lng,
                    "speed_mps": speed_mps or 0.0,
                    "speed_kmh": round((speed_mps or 0.0) * 3.6, 1),
                    "last_sequence": row["last_sequence"],
                    "last_timestamp": row["last_timestamp"],
                    "updated_at": updated_at_str,
                    "age_s": round(age_s, 1),
                    "status": "online" if is_online else ("idle" if age_s < 300.0 else "offline"),
                    "packet_count": counts.get(v_code, 1),
                })
            return {"sources": sources, "total": len(sources)}

        @self.router.get("/gps/logs", summary="Query received GPS telemetry logs")
        @self.gps_router.get("/logs", summary="Query received GPS telemetry logs")
        def get_gps_logs(
            vehicle_code: str | None = Query(None, description="Filter by vehicle code"),
            limit: int = Query(50, ge=1, le=500),
            offset: int = Query(0, ge=0),
        ):
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                query = "SELECT * FROM telemetry_logs"
                params: list[Any] = []
                if vehicle_code:
                    query += " WHERE vehicle_code = ?"
                    params.append(vehicle_code)
                query += " ORDER BY received_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cur.execute(query, params)
                rows = cur.fetchall()

                count_q = "SELECT COUNT(*) FROM telemetry_logs"
                count_params: list[Any] = []
                if vehicle_code:
                    count_q += " WHERE vehicle_code = ?"
                    count_params.append(vehicle_code)
                cur.execute(count_q, count_params)
                total = cur.fetchone()[0]

            logs = [
                {
                    "id": r["id"],
                    "vehicle_code": r["vehicle_code"],
                    "client_ip": r["client_ip"],
                    "lat": r["lat"],
                    "lng": r["lng"],
                    "speed_mps": r["speed_mps"],
                    "speed_kmh": round((r["speed_mps"] or 0.0) * 3.6, 1),
                    "sequence": r["sequence"],
                    "received_at": r["received_at"],
                    "nearest_drain_id": r["nearest_drain_id"],
                    "distance_m": round(r["distance_m"], 2) if r["distance_m"] is not None else None,
                }
                for r in rows
            ]
            return {"total": total, "limit": limit, "offset": offset, "logs": logs}

        @self.router.get("/gps/tracks", summary="GeoJSON of recent GPS trajectories")
        @self.gps_router.get("/tracks", summary="GeoJSON of recent GPS trajectories")
        def get_gps_tracks():
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(
                    "SELECT * FROM telemetry_logs WHERE lat IS NOT NULL AND lng IS NOT NULL ORDER BY received_at DESC LIMIT 300"
                )
                rows = cur.fetchall()

            by_vehicle: dict[str, list[Any]] = {}
            for r in reversed(rows):
                by_vehicle.setdefault(r["vehicle_code"], []).append(r)

            features = []
            for v_code, pts in by_vehicle.items():
                coords = [[float(p["lng"]), float(p["lat"])] for p in pts]
                if len(coords) >= 2:
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": {"vehicle_code": v_code, "points_count": len(coords)},
                    })
                latest = pts[-1]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(latest["lng"]), float(latest["lat"])]},
                    "properties": {
                        "vehicle_code": v_code,
                        "speed_kmh": round((latest["speed_mps"] or 0.0) * 3.6, 1),
                        "client_ip": latest["client_ip"],
                        "received_at": latest["received_at"],
                        "sequence": latest["sequence"],
                    },
                })

            return {"type": "FeatureCollection", "features": features}

        @self.router.get("/gps/export", summary="Export received GPS telemetry logs as CSV")
        @self.gps_router.get("/export", summary="Export received GPS telemetry logs as CSV")
        def export_gps_csv():
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM telemetry_logs ORDER BY received_at DESC LIMIT 5000")
                rows = cur.fetchall()

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["id", "received_at", "vehicle_code", "client_ip", "latitude", "longitude", "speed_kmh", "sequence", "nearest_drain_id", "distance_m"])
            for r in rows:
                spd_kmh = round((r["speed_mps"] or 0.0) * 3.6, 2)
                writer.writerow([r["id"], r["received_at"], r["vehicle_code"], r["client_ip"], r["lat"], r["lng"], spd_kmh, r["sequence"], r["nearest_drain_id"], r["distance_m"]])

            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=gps_telemetry_logs.csv"},
            )

    async def broadcast_event(
        self,
        event_id: str,
        event_data: dict[str, Any],
        received_at: str,
        has_image: bool,
        nearby_drains: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Broadcasts newly uploaded event to all active SSE subscribers and triggers eligible webhooks.
        Includes nearby_drains list for telemetry correlation when provided.
        """
        blockage = event_data.get("blockage") or {}
        cov = float(
            blockage.get("coverage_percent")
            if blockage.get("coverage_percent") is not None
            else (event_data.get("coverage_percent") if event_data.get("coverage_percent") is not None else (event_data.get("occlusion_pct", 0.0) or 0.0))
        )
        grade_num, grade_label = calculate_grade(cov)
        gps = event_data.get("gps") or {}

        if isinstance(gps, dict):
            lat = gps.get("latitude") if gps.get("latitude") is not None else gps.get("lat")
            lon = gps.get("longitude") if gps.get("longitude") is not None else (gps.get("lon") or gps.get("lng"))
        else:
            lat, lon = None, None

        message = {
            "event_id": event_id,
            "event_type": event_data.get("event_type", "gully_blockage"),
            "source": event_data.get("source", "unknown"),
            "status": blockage.get("status") or event_data.get("status", "unknown"),
            "coverage_percent": round(cov, 2),
            "grade": grade_num,
            "grade_label": grade_label,
            "lat": float(lat) if lat is not None else None,
            "lon": float(lon) if lon is not None else None,
            "has_image": has_image,
            "image_url": f"/events/{event_id}/image" if has_image else None,
            "received_at": received_at,
            "nearby_drains": nearby_drains or [],
        }

        # 1. Distribute to SSE queues
        dead_subscribers = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except Exception:
                dead_subscribers.append(queue)
        for dead in dead_subscribers:
            if dead in self._subscribers:
                self._subscribers.remove(dead)

        # 2. Distribute to active WebSocket clients
        dead_ws = []
        for ws in list(self._active_ws):
            try:
                await ws.send_json(message)
            except Exception:
                dead_ws.append(ws)
        for dead in dead_ws:
            if dead in self._active_ws:
                self._active_ws.remove(dead)

        # 3. Dispatch to eligible Webhooks asynchronously
        asyncio.create_task(self._dispatch_webhooks(message))

    async def broadcast_telemetry(self, message: dict[str, Any]) -> None:
        """Broadcasts telemetry packet to all connected WebSockets and SSE queues."""
        dead_ws = []
        for ws in list(self._active_ws):
            try:
                await ws.send_json(message)
            except Exception:
                dead_ws.append(ws)
        for dead in dead_ws:
            if dead in self._active_ws:
                self._active_ws.remove(dead)

        dead_subscribers = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except Exception:
                dead_subscribers.append(queue)
        for dead in dead_subscribers:
            if dead in self._subscribers:
                self._subscribers.remove(dead)

    async def broadcast_detection(self, message: dict[str, Any]) -> None:
        """Broadcasts detection packet to all connected WebSockets and SSE queues."""
        dead_ws = []
        for ws in list(self._active_ws):
            try:
                await ws.send_json(message)
            except Exception:
                dead_ws.append(ws)
        for dead in dead_ws:
            if dead in self._active_ws:
                self._active_ws.remove(dead)

        dead_subscribers = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except Exception:
                dead_subscribers.append(queue)
        for dead in dead_subscribers:
            if dead in self._subscribers:
                self._subscribers.remove(dead)

    async def broadcast_dismissal(self, event_id: str) -> None:
        """Broadcasts event dismissal to all connected WebSockets and SSE queues."""
        await self.broadcast_detection({"type": "event_dismissed", "event_id": event_id})

    async def broadcast_dismiss_all(self) -> None:
        """Broadcasts all events dismissal to all connected WebSockets and SSE queues."""
        await self.broadcast_detection({"type": "all_events_dismissed"})

    async def _dispatch_webhooks(self, event_message: dict[str, Any]) -> None:
        """Dispatches event to configured webhook targets in the background."""
        import urllib.request
        cov = event_message.get("coverage_percent", 0.0)
        grade = event_message.get("grade", 1)

        payload_bytes = json.dumps({
            "notification_type": "gully_blockage_alert",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_message,
        }).encode("utf-8")

        for wh_id, wh in list(self._webhooks.items()):
            if not wh.get("enabled", True):
                continue
            if cov < wh.get("min_coverage", 0.0) and grade < wh.get("min_grade", 1):
                continue

            url = wh.get("url")
            try:
                def _send(u=url, p=payload_bytes):
                    req = urllib.request.Request(
                        u,
                        data=p,
                        headers={"Content-Type": "application/json", "User-Agent": "BEUM-DrainSight-Adapter/1.0"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=3.0) as res:
                        return res.status
                await asyncio.to_thread(_send)
            except Exception as e:
                logger.warning("Webhook dispatch to %s failed: %s", url, e)


TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "dashboard.html"


def get_dashboard_html(template_path: Path | str | None = None) -> str:
    """Returns the standalone interactive HTML/JS for the DrainSight Web Control Dashboard."""
    path = Path(template_path) if template_path else TEMPLATE_PATH
    if path.is_file():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Dashboard template not found: {path}")
