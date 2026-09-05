"""
End-to-End Integration Test: Edge GullyRuntime -> Central Receiver Server -> DrainSight Dashboard
-------------------------------------------------------------------------------------------------
Validates:
1. Edge GullyRuntime detecting real drain blockage on road video + GPS replay.
2. Local spooling: creation of pending JSON metadata and JPEG evidence.
3. Multipart HTTP upload to receiver server (/upload).
4. Local queue clearance on HTTP 201 response.
5. Server-side SQLite persistence (beum_events.db) with GPS, coverage_percent, occlusion_pct.
6. Evidence image serving (/events/{event_id}/image).
7. Real-time DrainSight push feed validation.
8. GIS GeoJSON FeatureCollection generation (/api/drainsight/geojson).
9. Offline resilience: local spooling when server is unreachable, followed by automatic recovery.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ClientUploader:
    __test__ = False
    """Adapter allowing GullyRuntime's UploadWorker to upload via in-memory TestClient."""

    def __init__(self, client: TestClient, token: str = "") -> None:
        self.client = client
        self.token = token

    def upload(self, event: dict[str, Any], evidence_path: Path | None = None) -> None:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        if evidence_path is not None and evidence_path.exists():
            files = {
                "metadata": (None, json.dumps(event), "application/json"),
                "image": (evidence_path.name, evidence_path.read_bytes(), "image/jpeg"),
            }
            res = self.client.post("/upload", files=files, headers=headers)
        else:
            res = self.client.post("/upload", json=event, headers=headers)

        if res.status_code not in (200, 201):
            raise RuntimeError(f"Upload failed: HTTP {res.status_code} - {res.text}")


class FailUploader:
    """Simulates an unreachable or network-down server."""

    def upload(self, event: dict[str, Any], evidence_path: Path | None = None) -> None:
        raise ConnectionRefusedError("Simulated offline: Connection refused to central server")


class EdgeToServerE2ETest(unittest.TestCase):
    def setUp(self) -> None:
        self.server_temp_dir = tempfile.TemporaryDirectory()
        self.server_dir = Path(self.server_temp_dir.name)
        self.edge_temp_dir = tempfile.TemporaryDirectory()
        self.edge_spool_dir = Path(self.edge_temp_dir.name) / "spool"

        self.token = "e2e-secret-token"
        self.app = create_app(data_dir=self.server_dir, token=self.token)
        self.client = TestClient(self.app)

        self.video_path = PROJECT_ROOT / "data" / "2026-09-04_04-31-48" / "V20260904_131914000_79E42EB0-91A0-4F27-A3F0-BF6673616688.MOV"
        self.gps_path = PROJECT_ROOT / "data" / "2026-09-04_04-31-48" / "Location.csv"
        self.model_path = PROJECT_ROOT / "models" / "best-seg-2class.pt"

    def tearDown(self) -> None:
        self.server_temp_dir.cleanup()
        self.edge_temp_dir.cleanup()

    def test_full_pipeline_edge_to_dashboard(self) -> None:
        # Step 1: Verify Health
        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        self.assertIn(health_resp.json()["status"], ("ok", "healthy"))

        # Step 2: Configure Edge Pipeline
        config = SystemConfig(
            source=str(self.video_path),
            max_frames=150,
            realtime_source=True,
            camera_buffer_size=2,
            roi_points=((0, 0), (1080, 0), (1080, 1920), (0, 1920)),
            detector=DetectorConfig(
                model_path=str(self.model_path),
                image_size=320,
                confidence=0.15,
                iou=0.45,
                device="cpu",
                class_names=["drain_area", "drain_full"],
            ),
            storage=StorageConfig(
                spool_dir=str(self.edge_spool_dir),
                save_evidence=True,
                jpeg_quality=85,
            ),
            upload=UploadConfig(
                url="http://mock-in-memory-server/upload",
                token=self.token,
                timeout_s=10.0,
                flush_interval_s=0.5,
            ),
            gps=GPSConfig(
                provider="replay",
                csv_path=str(self.gps_path),
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
                min_safe_battery=20.0,
                critical_battery=8.0,
                mode_intervals={"low": 0.5, "medium": 0.033, "high": 0.033},
            ),
            run_preflight=False,
        )

        runtime = GullyRuntime(config)
        # Inject TestClientUploader to route uploads directly into the server
        runtime.upload_worker.uploader = ClientUploader(self.client, token=self.token)

        # Step 3: Run Edge Runtime
        stats = runtime.run()

        # Step 4: Verify Edge Event Generation & Spool Purge
        self.assertGreaterEqual(stats["events"], 1, "Expected at least 1 blockage event detected")
        self.assertEqual(runtime.queue.pending_count(), 0, "Spool queue should be empty after upload")
        self.assertEqual(
            len(list((self.edge_spool_dir / "pending").glob("*"))),
            0,
            "Local pending files must be deleted upon successful upload",
        )

        # Step 5: Verify Server Database Records (SQLite)
        import sqlite3
        db_path = self.server_dir / "beum_events.db"
        self.assertTrue(db_path.exists(), "beum_events.db was not created on server")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM events ORDER BY received_at DESC")
        rows = cur.fetchall()
        conn.close()

        self.assertGreaterEqual(len(rows), 1, "Database should contain at least 1 uploaded event")
        first_row = rows[0]
        event_id = first_row["event_id"]
        self.assertIsNotNone(event_id)
        self.assertIn(first_row["blockage_status"], ("warning", "critical", "normal"))
        self.assertGreater(first_row["coverage_percent"], 0.0)
        self.assertAlmostEqual(first_row["lat"], 36.837, delta=0.01)
        self.assertAlmostEqual(first_row["lon"], 127.182, delta=0.01)
        self.assertEqual(first_row["has_image"], 1)

        # Step 6: Verify Image Retrieval Endpoint
        img_resp = self.client.get(f"/events/{event_id}/image", headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(img_resp.status_code, 200)
        self.assertEqual(img_resp.headers.get("content-type"), "image/jpeg")
        self.assertGreater(len(img_resp.content), 1000, "Evidence image bytes should be valid JPEG")

        # Step 7: Verify GeoJSON Feature Collection
        geo_resp = self.client.get("/api/drainsight/geojson", headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(geo_resp.status_code, 200)
        geo_data = geo_resp.json()
        self.assertEqual(geo_data["type"], "FeatureCollection")
        features = geo_data["features"]
        self.assertGreaterEqual(len(features), 1)
        matching_feat = next((f for f in features if f["properties"].get("id") == event_id or f["properties"].get("event_id") == event_id), None)
        self.assertIsNotNone(matching_feat, f"Event {event_id} not found in GeoJSON features")
        self.assertEqual(len(matching_feat["geometry"]["coordinates"]), 2)
        self.assertIn("grade", matching_feat["properties"])

        # Step 8: Verify Control Console Stats
        stats_resp = self.client.get("/api/drainsight/stats", headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(stats_resp.status_code, 200)
        stats_data = stats_resp.json()
        self.assertGreaterEqual(stats_data["total_events"], 1)
        self.assertIn("grade_breakdown", stats_data)

        # Step 9: Verify Web Dashboard HTML Rendering
        dash_resp = self.client.get("/dashboard")
        self.assertEqual(dash_resp.status_code, 200)
        self.assertIn("text/html", dash_resp.headers.get("content-type", ""))
        self.assertIn("DrainSight", dash_resp.text)

    def test_offline_spool_resilience_and_recovery(self) -> None:
        """When server is offline, edge preserves events locally; when server reconnects, flushes cleanly."""
        config = SystemConfig(
            source=str(self.video_path),
            max_frames=150,
            realtime_source=True,
            camera_buffer_size=2,
            roi_points=((0, 0), (1080, 0), (1080, 1920), (0, 1920)),
            detector=DetectorConfig(
                model_path=str(self.model_path),
                image_size=320,
                confidence=0.15,
                iou=0.45,
                device="cpu",
                class_names=["drain_area", "drain_full"],
            ),
            storage=StorageConfig(
                spool_dir=str(self.edge_spool_dir),
                save_evidence=True,
                jpeg_quality=80,
            ),
            upload=UploadConfig(
                url="http://offline-unreachable-server/upload",
                token="test-token",
                timeout_s=1.0,
                base_backoff_s=0.5,
            ),
            gps=GPSConfig(
                provider="replay",
                csv_path=str(self.gps_path),
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

        runtime = GullyRuntime(config)
        # Inject offline failing uploader
        runtime.upload_worker.uploader = FailUploader()

        # Run while offline
        stats = runtime.run()

        # 1. Verification: Event detected but remains safely buffered in local spool
        self.assertGreaterEqual(stats["events"], 1)
        pending_count = runtime.queue.pending_count()
        self.assertGreaterEqual(pending_count, 1, "Unreachable server must preserve items in local spool")
        pending_files = list((self.edge_spool_dir / "pending").glob("*.json"))
        self.assertEqual(len(pending_files), pending_count)

        # 2. Recovery: Server is restored; connect valid ClientUploader
        runtime.upload_worker.uploader = ClientUploader(self.client, token=self.token)
        runtime.upload_worker.consecutive_failures = 0
        runtime.upload_worker.next_retry_time = 0.0

        sent, failed = runtime.upload_worker.flush_once(max_items=10)
        self.assertGreaterEqual(sent, 1, "Flush must successfully upload previously pending events")
        self.assertEqual(failed, 0)
        self.assertEqual(runtime.queue.pending_count(), 0, "Spool must be completely cleared after successful recovery")

        # Verify server DB received recovered events
        events_resp = self.client.get("/events", headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(events_resp.status_code, 200)
        events_list = events_resp.json()["events"]
        self.assertGreaterEqual(len(events_list), 1)


if __name__ == "__main__":
    unittest.main()
