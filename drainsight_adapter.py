"""
DrainSight Backend Adapter Module
---------------------------------
Provides integration interfaces for the DrainSight Central Monitoring Dashboard,
GIS systems (GeoJSON), real-time SSE streaming, and external webhook notification.
"""

from __future__ import annotations

import asyncio
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
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket

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
        self.router = APIRouter(prefix="/api/drainsight", tags=["DrainSight Adapter"])
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


def get_dashboard_html() -> str:
    """Returns the standalone interactive HTML/JS for the DrainSight Web Control Dashboard."""
    return """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DrainSight | 스마트 빗물받이 관제 대시보드</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root {
      --bg-main: #0f172a;
      --bg-card: #1e293b;
      --border-color: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --status-crit: #ef4444;
      --status-warn: #f59e0b;
      --status-norm: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg-main); color: var(--text-main); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
    header {
      background-color: var(--bg-card);
      border-bottom: 1px solid var(--border-color);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .logo-area { display: flex; align-items: center; gap: 12px; }
    .logo-icon { width: 32px; height: 32px; background: linear-gradient(135deg, #0284c7, #38bdf8); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white; }
    .title { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.5px; }
    .subtitle { font-size: 0.75rem; color: var(--text-muted); }
    .live-status { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; padding: 6px 12px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 20px; color: var(--status-norm); }
    .pulse-dot { width: 8px; height: 8px; background-color: var(--status-norm); border-radius: 50%; box-shadow: 0 0 8px var(--status-norm); animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
    
    .stats-bar {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      padding: 16px 24px;
      background: var(--bg-main);
    }
    .stat-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
    }
    .stat-title { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px; }
    .stat-value { font-size: 1.7rem; font-weight: 700; }
    .stat-desc { font-size: 0.72rem; margin-top: 4px; color: var(--text-muted); }

    .main-content {
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 16px;
      padding: 0 24px 20px 24px;
      overflow: hidden;
    }
    #map {
      width: 100%;
      height: 100%;
      border-radius: 10px;
      border: 1px solid var(--border-color);
      background: #020617;
    }
    .side-panel {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .panel-header {
      padding: 14px 18px;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .panel-title { font-size: 0.95rem; font-weight: 600; }
    .filter-btn-group { display: flex; gap: 4px; }
    .filter-btn {
      background: #334155;
      border: none;
      color: #cbd5e1;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.75rem;
      cursor: pointer;
    }
    .filter-btn.active { background: var(--accent-blue); color: #0f172a; font-weight: 600; }

    .event-list { flex: 1; overflow-y: auto; padding: 10px 14px; display: flex; flex-direction: column; gap: 8px; }
    .event-card {
      background: #0f172a;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 12px;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .event-card:hover { border-color: var(--accent-blue); transform: translateY(-1px); }
    .event-card-header { display: flex; justify-content: space-between; align-items: center; }
    .badge {
      font-size: 0.7rem;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 12px;
      text-transform: uppercase;
    }
    .badge-critical { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-normal { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); }

    .cov-bar-container { width: 100%; height: 6px; background: #334155; border-radius: 3px; overflow: hidden; margin-top: 4px; }
    .cov-bar-fill { height: 100%; border-radius: 3px; }

    /* Modal for Image Preview */
    .modal {
      display: none;
      position: fixed;
      z-index: 10000;
      left: 0; top: 0; width: 100%; height: 100%;
      background-color: rgba(0,0,0,0.85);
      backdrop-filter: blur(4px);
      align-items: center;
      justify-content: center;
    }
    .modal-content {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      max-width: 800px;
      width: 90%;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .modal-header { padding: 14px 20px; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; }
    .modal-body { padding: 20px; display: flex; flex-direction: column; align-items: center; gap: 14px; }
    .modal-body img { max-width: 100%; max-height: 500px; border-radius: 8px; border: 1px solid var(--border-color); object-fit: contain; }
    .close-btn { background: none; border: none; font-size: 1.5rem; color: var(--text-muted); cursor: pointer; }
    .close-btn:hover { color: white; }

    /* Custom Leaflet Marker Styling */
    .gully-marker-icon {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      border: 2px solid white;
      box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    .leaflet-popup-content-wrapper { background: var(--bg-card) !important; color: var(--text-main) !important; border: 1px solid var(--border-color); border-radius: 8px; }
    .leaflet-popup-tip { background: var(--bg-card) !important; }
  </style>
</head>
<body>

  <header>
    <div class="logo-area">
      <div class="logo-icon">DS</div>
      <div>
        <div class="title">DrainSight™ 관제 대시보드</div>
        <div class="subtitle">스마트 빗물받이 차폐 모니터링 & AI 준설 관리 시스템</div>
      </div>
    </div>
    <div style="display:flex; gap: 12px; align-items: center;">
      <a href="/api/drainsight/geojson" target="_blank" style="color: var(--accent-blue); text-decoration: none; font-size: 0.8rem; border: 1px solid var(--border-color); padding: 6px 12px; border-radius: 6px;">GeoJSON 내보내기</a>
      <div class="live-status" id="liveStatus">
        <div class="pulse-dot"></div>
        <span id="liveText">실시간 연결됨</span>
      </div>
    </div>
  </header>

  <section class="stats-bar">
    <div class="stat-card">
      <div class="stat-title">총 모니터링 배수구</div>
      <div class="stat-value" id="statTotal">-</div>
      <div class="stat-desc" id="statSources">연결 기기: -</div>
    </div>
    <div class="stat-card">
      <div class="stat-title" style="color:#ef4444;">긴급 준설 필요 (심각)</div>
      <div class="stat-value" id="statCritical" style="color:#ef4444;">-</div>
      <div class="stat-desc">차폐율 60% 이상 (3등급)</div>
    </div>
    <div class="stat-card">
      <div class="stat-title" style="color:#f59e0b;">현장 점검 요망 (주의)</div>
      <div class="stat-value" id="statWarning" style="color:#f59e0b;">-</div>
      <div class="stat-desc">차폐율 30%~60% (2등급)</div>
    </div>
    <div class="stat-card">
      <div class="stat-title">평균 차폐율</div>
      <div class="stat-value" id="statAvgCoverage">- %</div>
      <div class="stat-desc" id="statMaxCoverage">최고 차폐율: - %</div>
    </div>
  </section>

  <section class="main-content">
    <div id="map"></div>
    <div class="side-panel">
      <div class="panel-header">
        <div class="panel-title">실시간 차폐 감지 피드</div>
        <div class="filter-btn-group">
          <button class="filter-btn active" onclick="setFilter('all')">전체</button>
          <button class="filter-btn" onclick="setFilter('critical')">심각</button>
          <button class="filter-btn" onclick="setFilter('warning')">주의</button>
        </div>
      </div>
      <div class="event-list" id="eventList">
        <div style="text-align:center; padding: 20px; color: var(--text-muted);">이벤트 로딩 중...</div>
      </div>
    </div>
  </section>

  <!-- Image Preview Modal -->
  <div class="modal" id="imageModal">
    <div class="modal-content">
      <div class="modal-header">
        <div style="font-weight: 600;" id="modalTitle">현장 증거 사진</div>
        <button class="close-btn" onclick="closeModal()">&times;</button>
      </div>
      <div class="modal-body">
        <img id="modalImg" src="" alt="현장 증거 사진" />
        <div id="modalMeta" style="font-size: 0.85rem; color: var(--text-muted); text-align: center;"></div>
      </div>
    </div>
  </div>

  <script>
    let map;
    let markersLayer;
    let currentFilter = 'all';
    let cachedFeatures = [];

    // Initialize Map
    function initMap() {
      map = L.map('map', { zoomControl: true }).setView([36.838, 127.184], 16);
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19
      }).addTo(map);
      markersLayer = L.layerGroup().addTo(map);
    }

    function getStatusColor(status, coverage) {
      if (status === 'critical' || coverage >= 60.0) return '#ef4444';
      if (status === 'warning' || coverage >= 30.0) return '#f59e0b';
      return '#10b981';
    }

    function getBadgeClass(status, coverage) {
      if (status === 'critical' || coverage >= 60.0) return 'badge-critical';
      if (status === 'warning' || coverage >= 30.0) return 'badge-warning';
      return 'badge-normal';
    }

    async function loadStats() {
      try {
        const res = await fetch('/api/drainsight/stats');
        const data = await res.json();
        document.getElementById('statTotal').innerText = data.total_events || 0;
        document.getElementById('statCritical').innerText = data.grade_breakdown.grade_3_critical || 0;
        document.getElementById('statWarning').innerText = data.grade_breakdown.grade_2_warning || 0;
        document.getElementById('statAvgCoverage').innerText = `${data.average_coverage_percent}%`;
        document.getElementById('statMaxCoverage').innerText = `최고 차폐율: ${data.max_coverage_percent}%`;
        document.getElementById('statSources').innerText = `연결 기기: ${data.active_sources.join(', ') || '없음'}`;
      } catch (err) {
        console.error("Failed to load stats:", err);
      }
    }

    async function loadGeoJSON() {
      try {
        const res = await fetch('/api/drainsight/geojson');
        const geojson = await res.json();
        cachedFeatures = geojson.features || [];
        renderMapAndList();
      } catch (err) {
        console.error("Failed to load GeoJSON:", err);
      }
    }

    function renderMapAndList() {
      markersLayer.clearLayers();
      const listEl = document.getElementById('eventList');
      listEl.innerHTML = '';

      const filtered = cachedFeatures.filter(f => {
        const p = f.properties;
        if (currentFilter === 'critical') return p.grade === 3 || p.status === 'critical';
        if (currentFilter === 'warning') return p.grade === 2 || p.status === 'warning';
        return true;
      });

      if (filtered.length === 0) {
        listEl.innerHTML = '<div style="text-align:center; padding: 20px; color: var(--text-muted);">표시할 이벤트가 없습니다.</div>';
        return;
      }

      const bounds = [];

      filtered.forEach(feature => {
        const p = feature.properties;
        const coords = feature.geometry.coordinates; // [lon, lat]
        const latLng = [coords[1], coords[0]];
        bounds.push(latLng);

        const color = getStatusColor(p.status, p.coverage_percent);
        const badgeClass = getBadgeClass(p.status, p.coverage_percent);

        // Marker
        const marker = L.circleMarker(latLng, {
          radius: 9,
          fillColor: color,
          color: '#ffffff',
          weight: 2,
          opacity: 1,
          fillOpacity: 0.85
        });

        const popupHtml = `
          <div style="font-size: 0.85rem; min-width: 200px;">
            <div style="font-weight: 700; margin-bottom: 4px; display:flex; justify-content:space-between;">
              <span>${p.source}</span>
              <span class="badge ${badgeClass}">${p.status}</span>
            </div>
            <div>차폐율: <b>${p.coverage_percent}%</b> (${p.grade_label})</div>
            <div style="color:#94a3b8; font-size:0.75rem; margin: 4px 0;">좌표: ${coords[1].toFixed(5)}, ${coords[0].toFixed(5)}</div>
            ${p.has_image ? `<button onclick="openModal('${p.image_url}', '${p.id}', ${p.coverage_percent})" style="width:100%; margin-top:8px; padding: 6px; background:#0284c7; color:white; border:none; border-radius:6px; cursor:pointer; font-size:0.75rem;">현장 증거 사진 확인</button>` : ''}
          </div>
        `;
        marker.bindPopup(popupHtml);
        markersLayer.addLayer(marker);

        // Feed Card
        const card = document.createElement('div');
        card.className = 'event-card';
        card.innerHTML = `
          <div class="event-card-header">
            <span style="font-weight:600; font-size:0.85rem;">${p.source}</span>
            <span class="badge ${badgeClass}">${p.coverage_percent}%</span>
          </div>
          <div style="font-size:0.75rem; color:#94a3b8;">${p.grade_label}</div>
          <div class="cov-bar-container">
            <div class="cov-bar-fill" style="width: ${Math.min(100, p.coverage_percent)}%; background: ${color};"></div>
          </div>
          <div style="font-size:0.7rem; color:#64748b; margin-top:2px;">수신: ${p.received_at ? p.received_at.split('.')[0] : ''}</div>
        `;
        card.onclick = () => {
          map.flyTo(latLng, 18);
          marker.openPopup();
        };
        listEl.appendChild(card);
      });

      if (bounds.length > 0) {
        map.fitBounds(bounds, { padding: [50, 50], maxZoom: 17 });
      }
    }

    function setFilter(f) {
      currentFilter = f;
      document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
      event.target.classList.add('active');
      renderMapAndList();
    }

    function openModal(imgUrl, eventId, coverage) {
      document.getElementById('modalImg').src = imgUrl;
      document.getElementById('modalTitle').innerText = `이벤트 현장 사진 [${eventId}]`;
      document.getElementById('modalMeta').innerText = `실시간 탐지 차폐율: ${coverage}%`;
      document.getElementById('imageModal').style.display = 'flex';
    }

    function closeModal() {
      document.getElementById('imageModal').style.display = 'none';
      document.getElementById('modalImg').src = '';
    }

    // Realtime Connection: Primary WebSocket with fallback to SSE
    function connectRealtime() {
      const statusText = document.getElementById('liveText');
      const liveStatus = document.getElementById('liveStatus');
      const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${location.host}/ws/dashboard`;

      let ws;
      try {
        ws = new WebSocket(wsUrl);
      } catch (err) {
        console.warn("[WS] Direct constructor failed, falling back to SSE", err);
        return connectSSEFallback();
      }

      ws.onopen = () => {
        statusText.innerText = 'WebSocket 관제 연결됨 (Zero-Polling)';
        liveStatus.style.borderColor = 'rgba(16, 185, 129, 0.5)';
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'connected') {
            console.log("[WS] Handshake ACK:", msg);
          } else if (msg.type === 'telemetry') {
            console.log("[WS] Telemetry Update:", msg);
            loadStats();
          } else if (msg.type === 'detection' || msg.type === 'blockage_event') {
            console.log("[WS] Detection/Blockage Event:", msg);
            loadStats();
            loadGeoJSON();
          }
        } catch (e) {
          console.error("[WS] Message parsing error:", e);
        }
      };

      ws.onclose = () => {
        console.warn("[WS] WebSocket disconnected. Attempting reconnect / SSE fallback...");
        statusText.innerText = '재연결 시도 중...';
        liveStatus.style.borderColor = 'rgba(239, 68, 68, 0.3)';
        setTimeout(connectSSEFallback, 3000);
      };

      ws.onerror = (err) => {
        console.warn("[WS] Error occurred, closing socket:", err);
        ws.close();
      };
    }

    function connectSSEFallback() {
      const statusText = document.getElementById('liveText');
      const liveStatus = document.getElementById('liveStatus');
      const sse = new EventSource('/api/drainsight/stream');

      sse.onopen = () => {
        statusText.innerText = 'SSE 스트리밍 연결됨';
        liveStatus.style.borderColor = 'rgba(16, 185, 129, 0.3)';
      };

      sse.addEventListener('blockage_event', (e) => {
        loadStats();
        loadGeoJSON();
      });

      sse.onerror = () => {
        statusText.innerText = '연결 대기 중';
        liveStatus.style.borderColor = 'rgba(239, 68, 68, 0.3)';
      };
    }

    window.addEventListener('DOMContentLoaded', () => {
      initMap();
      loadStats();
      loadGeoJSON();
      connectRealtime();
    });
  </script>
</body>
</html>
"""
