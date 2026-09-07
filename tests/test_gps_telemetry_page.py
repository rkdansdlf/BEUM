"""Unit and integration tests for GPS telemetry reception tracking and page endpoints."""

import csv
import io
import json
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timezone
from starlette.testclient import TestClient

from receiver_server import create_app


class GpsTelemetryPageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app(data_dir=self.temp_dir.name)
        self.client = TestClient(self.app)
        self.db_path = f"{self.temp_dir.name}/beum_events.db"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_telemetry_ingress_and_db_logging(self) -> None:
        # 1. Send first packet from vehicle_alpha
        p1 = {
            "vehicle_code": "vehicle_alpha",
            "lat": 36.8305,
            "lng": 127.1795,
            "speed_mps": 8.5,
            "sequence": 1,
            "timestamp": time.time(),
        }
        res1 = self.client.post("/api/telemetry", json=p1)
        self.assertEqual(res1.status_code, 200)
        self.assertTrue(res1.json().get("ok"))
        self.assertEqual(res1.json().get("client_ip"), "testclient")

        # 2. Send second packet from vehicle_alpha
        p2 = {
            "vehicle_code": "vehicle_alpha",
            "lat": 36.8310,
            "lng": 127.1800,
            "speed_mps": 11.2,
            "sequence": 2,
            "timestamp": time.time() + 1.0,
        }
        res2 = self.client.post("/api/telemetry", json=p2)
        self.assertEqual(res2.status_code, 200)

        # 3. Send packet from vehicle_beta
        p3 = {
            "vehicle_code": "vehicle_beta",
            "lat": 36.8400,
            "lng": 127.1900,
            "speed_mps": 0.0,
            "sequence": 1,
            "timestamp": time.time(),
        }
        res3 = self.client.post("/api/telemetry", json=p3)
        self.assertEqual(res3.status_code, 200)

        # Verify database telemetry_logs table
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM telemetry_logs ORDER BY id ASC")
        rows = cur.fetchall()
        conn.close()

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["vehicle_code"], "vehicle_alpha")
        self.assertEqual(rows[0]["sequence"], 1)
        self.assertEqual(rows[0]["client_ip"], "testclient")
        self.assertAlmostEqual(rows[0]["lat"], 36.8305, places=4)
        self.assertAlmostEqual(rows[0]["lng"], 127.1795, places=4)
        self.assertAlmostEqual(rows[0]["speed_mps"], 8.5, places=2)

        self.assertEqual(rows[2]["vehicle_code"], "vehicle_beta")

    def test_gps_sources_endpoint(self) -> None:
        now_ts = time.time()
        # Ingest telemetry for 2 vehicles
        self.client.post("/api/telemetry", json={
            "vehicle_code": "car_edge_01",
            "lat": 36.8320,
            "lng": 127.1810,
            "speed_mps": 12.0,
            "sequence": 10,
            "timestamp": now_ts,
        })
        self.client.post("/api/telemetry", json={
            "vehicle_code": "car_edge_01",
            "lat": 36.8325,
            "lng": 127.1815,
            "speed_mps": 13.5,
            "sequence": 11,
            "timestamp": now_ts + 1.0,
        })

        resp = self.client.get("/api/gps/sources")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("sources", data)
        self.assertEqual(data["total"], 1)

        src = data["sources"][0]
        self.assertEqual(src["vehicle_code"], "car_edge_01")
        self.assertEqual(src["client_ip"], "testclient")
        self.assertEqual(src["packet_count"], 2)
        self.assertEqual(src["status"], "online")
        self.assertAlmostEqual(src["speed_kmh"], 48.6, places=1)

    def test_gps_logs_endpoint_pagination_and_filter(self) -> None:
        # Ingest telemetry for 2 vehicles
        for seq in range(1, 4):
            self.client.post("/api/telemetry", json={
                "vehicle_code": "car_filter_a",
                "lat": 36.83 + seq * 0.001,
                "lng": 127.18 + seq * 0.001,
                "speed_mps": 5.0,
                "sequence": seq,
                "timestamp": time.time() + seq,
            })
        self.client.post("/api/telemetry", json={
            "vehicle_code": "car_filter_b",
            "lat": 36.85,
            "lng": 127.19,
            "speed_mps": 0.0,
            "sequence": 1,
            "timestamp": time.time(),
        })

        # Test full logs
        resp_all = self.client.get("/api/gps/logs")
        self.assertEqual(resp_all.status_code, 200)
        self.assertEqual(resp_all.json()["total"], 4)

        # Test filter by vehicle_code
        resp_filt = self.client.get("/api/gps/logs?vehicle_code=car_filter_a")
        self.assertEqual(resp_filt.status_code, 200)
        data_filt = resp_filt.json()
        self.assertEqual(data_filt["total"], 3)
        for log in data_filt["logs"]:
            self.assertEqual(log["vehicle_code"], "car_filter_a")

        # Test pagination limit & offset
        resp_page = self.client.get("/api/gps/logs?limit=2&offset=0")
        self.assertEqual(resp_page.status_code, 200)
        self.assertEqual(len(resp_page.json()["logs"]), 2)

    def test_gps_tracks_geojson_endpoint(self) -> None:
        # Ingest 2 points for a vehicle to form a LineString
        self.client.post("/api/telemetry", json={
            "vehicle_code": "track_car_1",
            "lat": 36.8300,
            "lng": 127.1800,
            "speed_mps": 5.0,
            "sequence": 1,
            "timestamp": time.time(),
        })
        self.client.post("/api/telemetry", json={
            "vehicle_code": "track_car_1",
            "lat": 36.8310,
            "lng": 127.1810,
            "speed_mps": 6.0,
            "sequence": 2,
            "timestamp": time.time() + 1.0,
        })

        resp = self.client.get("/api/gps/tracks")
        self.assertEqual(resp.status_code, 200)
        geojson = resp.json()
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertGreaterEqual(len(geojson["features"]), 2)

        # Should contain both LineString and latest Point
        geom_types = [f["geometry"]["type"] for f in geojson["features"]]
        self.assertIn("LineString", geom_types)
        self.assertIn("Point", geom_types)

    def test_gps_export_csv_endpoint(self) -> None:
        self.client.post("/api/telemetry", json={
            "vehicle_code": "export_car",
            "lat": 36.8333,
            "lng": 127.1888,
            "speed_mps": 10.0,
            "sequence": 42,
            "timestamp": time.time(),
        })

        resp = self.client.get("/api/gps/export")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp.headers["content-type"])

        reader = csv.reader(io.StringIO(resp.text))
        header = next(reader)
        self.assertIn("vehicle_code", header)
        self.assertIn("client_ip", header)
        self.assertIn("latitude", header)
        self.assertIn("longitude", header)
        self.assertIn("speed_kmh", header)

        first_row = next(reader)
        self.assertIn("export_car", first_row)
        self.assertIn("testclient", first_row)

    def test_dashboard_page_rendered_clean_without_cluttered_sources(self) -> None:
        resp = self.client.get("/dashboard")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        body = resp.text

        # Check navigation tabs
        self.assertIn("빗물받이 관제", body)
        self.assertIn("실시간 AI 카메라 탐지", body)
        self.assertIn("GPS 수신처 조회", body)

        # Check GPS page components
        self.assertIn("gpsMap", body)
        self.assertIn("gpsLogTableBody", body)
        self.assertIn("gpsSourcesList", body)

        # Ensure cluttered source string is completely removed
        self.assertNotIn("cam-gangnam-01", body)
        self.assertNotIn("cam-gangnam-02", body)
        self.assertNotIn("V20260903", body)
        self.assertNotIn("reprocessed_2026", body)


if __name__ == "__main__":
    unittest.main()
