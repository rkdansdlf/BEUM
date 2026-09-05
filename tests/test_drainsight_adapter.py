from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starlette.testclient import TestClient
from receiver_server import create_app
from drainsight_adapter import calculate_grade


class DrainSightAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.token = "test-token"
        self.app = create_app(data_dir=self.data_dir, token=self.token)
        self.client = TestClient(self.app)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_calculate_grade(self) -> None:
        g1, label1 = calculate_grade(15.0)
        self.assertEqual(g1, 1)
        self.assertIn("1등급", label1)

        g2, label2 = calculate_grade(45.5)
        self.assertEqual(g2, 2)
        self.assertIn("2등급", label2)

        g3, label3 = calculate_grade(85.0)
        self.assertEqual(g3, 3)
        self.assertIn("3등급", label3)

    def _seed_sample_events(self) -> None:
        # Event 1: Normal (Grade 1)
        e1 = {
            "event_id": "evt-norm-01",
            "event_type": "gully_blockage",
            "source": "cam-gangnam-01",
            "created_at": "2026-09-05T01:00:00Z",
            "blockage": {"status": "normal", "coverage_percent": 12.5},
            "gps": {"lat": 37.4980, "lon": 127.0276},
        }
        self.client.post("/upload", json=e1, headers=self.headers)

        # Event 2: Warning (Grade 2)
        e2 = {
            "event_id": "evt-warn-02",
            "event_type": "gully_blockage",
            "source": "cam-gangnam-02",
            "created_at": "2026-09-05T01:10:00Z",
            "blockage": {"status": "warning", "coverage_percent": 48.0},
            "gps": {"latitude": 37.4985, "longitude": 127.0280},
        }
        self.client.post("/upload", json=e2, headers=self.headers)

        # Event 3: Critical (Grade 3) with Image
        e3 = {
            "event_id": "evt-crit-03",
            "event_type": "gully_blockage",
            "source": "cam-gangnam-01",
            "created_at": "2026-09-05T01:20:00Z",
            "blockage": {"status": "critical", "coverage_percent": 88.2},
            "gps": {"lat": 37.4990, "lon": 127.0285},
        }
        files = {
            "metadata": (None, json.dumps(e3), "application/json"),
            "image": ("evidence.jpg", b"\xff\xd8\xff\xe0FAKE_JPG", "image/jpeg"),
        }
        self.client.post("/upload", files=files, headers=self.headers)

    def test_geojson_feature_collection(self) -> None:
        self._seed_sample_events()

        res = self.client.get("/api/drainsight/geojson")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["type"], "FeatureCollection")
        self.assertEqual(len(data["features"]), 3)

        # Verify first feature (newest first: evt-crit-03)
        feat = data["features"][0]
        self.assertEqual(feat["type"], "Feature")
        self.assertEqual(feat["geometry"]["type"], "Point")
        self.assertAlmostEqual(feat["geometry"]["coordinates"][0], 127.0285)  # lon
        self.assertAlmostEqual(feat["geometry"]["coordinates"][1], 37.4990)   # lat

        props = feat["properties"]
        self.assertEqual(props["id"], "evt-crit-03")
        self.assertEqual(props["status"], "critical")
        self.assertEqual(props["grade"], 3)
        self.assertTrue(props["has_image"])
        self.assertEqual(props["image_url"], "/events/evt-crit-03/image")

        # Test filter by status
        crit_res = self.client.get("/api/drainsight/geojson?status=critical")
        self.assertEqual(crit_res.status_code, 200)
        self.assertEqual(len(crit_res.json()["features"]), 1)

        # Test filter by min_grade (Grade 2 and 3 only)
        grade_res = self.client.get("/api/drainsight/geojson?min_grade=2")
        self.assertEqual(grade_res.status_code, 200)
        self.assertEqual(len(grade_res.json()["features"]), 2)

    def test_stats_summary_metrics(self) -> None:
        self._seed_sample_events()

        res = self.client.get("/api/drainsight/stats")
        self.assertEqual(res.status_code, 200)
        stats = res.json()

        self.assertEqual(stats["total_events"], 3)
        self.assertEqual(stats["status_breakdown"]["critical"], 1)
        self.assertEqual(stats["status_breakdown"]["warning"], 1)
        self.assertEqual(stats["status_breakdown"]["normal"], 1)

        grades = stats["grade_breakdown"]
        self.assertEqual(grades["grade_1_normal"], 1)
        self.assertEqual(grades["grade_2_warning"], 1)
        self.assertEqual(grades["grade_3_critical"], 1)

        self.assertAlmostEqual(stats["max_coverage_percent"], 88.2, places=1)
        self.assertEqual(stats["total_with_images"], 1)
        self.assertIn("cam-gangnam-01", stats["active_sources"])
        self.assertIn("cam-gangnam-02", stats["active_sources"])

    def test_priority_alerts(self) -> None:
        self._seed_sample_events()

        # Threshold 50% should return only evt-crit-03 (88.2%)
        res = self.client.get("/api/drainsight/alerts?min_coverage=50.0")
        self.assertEqual(res.status_code, 200)
        alerts = res.json()
        self.assertEqual(alerts["count"], 1)
        self.assertEqual(alerts["alerts"][0]["event_id"], "evt-crit-03")

        # Threshold 30% should return warning and critical
        res2 = self.client.get("/api/drainsight/alerts?min_coverage=30.0")
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["count"], 2)

    def test_webhooks_management(self) -> None:
        # 1. Register webhook
        wh_payload = {
            "url": "http://127.0.0.1:9999/webhook-test",
            "description": "City Hall Disaster Alarm",
            "min_coverage": 65.0,
            "min_grade": 3,
            "enabled": True,
        }
        create_res = self.client.post("/api/drainsight/webhooks", json=wh_payload)
        self.assertEqual(create_res.status_code, 201)
        wh_data = create_res.json()["webhook"]
        wh_id = wh_data["id"]
        self.assertEqual(wh_data["description"], "City Hall Disaster Alarm")

        # 2. List webhooks
        list_res = self.client.get("/api/drainsight/webhooks")
        self.assertEqual(list_res.status_code, 200)
        self.assertEqual(len(list_res.json()["webhooks"]), 1)

        # 3. Delete webhook
        del_res = self.client.delete(f"/api/drainsight/webhooks/{wh_id}")
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(del_res.json()["status"], "deleted")

        # 4. Delete 404
        del_404 = self.client.delete("/api/drainsight/webhooks/invalid-id")
        self.assertEqual(del_404.status_code, 404)

    def test_dashboard_page_rendering(self) -> None:
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("DrainSight", res.text)
        self.assertIn("leaflet", res.text)
        self.assertIn("statTotal", res.text)


if __name__ == "__main__":
    unittest.main()
