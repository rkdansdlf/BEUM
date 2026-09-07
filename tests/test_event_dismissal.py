from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starlette.testclient import TestClient
from receiver_server import create_app


class EventDismissalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.token = "test-token"
        self.app = create_app(data_dir=self.data_dir, token=self.token)
        self.client = TestClient(self.app)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _upload_event(self, event_id: str, coverage: float = 75.0, status_val: str = "critical") -> None:
        payload = {
            "event_id": event_id,
            "event_type": "gully_blockage",
            "source": "cam-test-01",
            "created_at": "2026-09-06T00:00:00Z",
            "blockage": {"status": status_val, "coverage_percent": coverage},
            "gps": {"lat": 37.5, "lon": 127.0},
        }
        res = self.client.post("/upload", json=payload, headers=self.headers)
        self.assertEqual(res.status_code, 201)

    def test_dismiss_single_event_flow(self) -> None:
        self._upload_event("evt_to_dismiss_01", coverage=80.0)

        # Confirm it appears in active feeds
        geojson = self.client.get("/api/drainsight/geojson").json()
        self.assertEqual(len(geojson["features"]), 1)
        stats = self.client.get("/api/drainsight/stats").json()
        self.assertEqual(stats["total_events"], 1)

        # Dismiss as false positive
        dismiss_res = self.client.post(
            "/api/events/evt_to_dismiss_01/dismiss?reason=confirmed_fp",
            headers=self.headers,
        )
        self.assertEqual(dismiss_res.status_code, 200)
        data = dismiss_res.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["event_id"], "evt_to_dismiss_01")
        self.assertEqual(data["status"], "dismissed")

        # Active feeds must now be empty
        geojson_after = self.client.get("/api/drainsight/geojson").json()
        self.assertEqual(len(geojson_after["features"]), 0)
        stats_after = self.client.get("/api/drainsight/stats").json()
        self.assertEqual(stats_after["total_events"], 0)
        events_after = self.client.get("/events").json()
        self.assertEqual(events_after["total"], 0)

        # Check archive
        archived_res = self.client.get("/api/events/archived")
        self.assertEqual(archived_res.status_code, 200)
        archived_data = archived_res.json()
        self.assertEqual(archived_data["total"], 1)
        self.assertEqual(archived_data["archived_events"][0]["event_id"], "evt_to_dismiss_01")
        self.assertEqual(archived_data["archived_events"][0]["dismiss_reason"], "confirmed_fp")

    def test_dismiss_via_delete_method(self) -> None:
        self._upload_event("evt_to_delete_02", coverage=50.0)
        del_res = self.client.delete("/events/evt_to_delete_02", headers=self.headers)
        self.assertEqual(del_res.status_code, 200)
        self.assertTrue(del_res.json()["ok"])

        # Confirm empty
        stats = self.client.get("/api/drainsight/stats").json()
        self.assertEqual(stats["total_events"], 0)

    def test_dismiss_nonexistent_returns_404(self) -> None:
        res = self.client.post("/api/events/does_not_exist/dismiss", headers=self.headers)
        self.assertEqual(res.status_code, 404)

    def test_dismiss_all_events(self) -> None:
        self._upload_event("evt_bulk_1", coverage=20.0, status_val="normal")
        self._upload_event("evt_bulk_2", coverage=55.0, status_val="warning")
        self._upload_event("evt_bulk_3", coverage=90.0, status_val="critical")

        # Verify 3 active
        stats = self.client.get("/api/drainsight/stats").json()
        self.assertEqual(stats["total_events"], 3)

        # Bulk dismiss
        bulk_res = self.client.post("/api/events/dismiss-all?reason=operator_bulk_dismiss", headers=self.headers)
        self.assertEqual(bulk_res.status_code, 200)
        bulk_data = bulk_res.json()
        self.assertTrue(bulk_data["ok"])
        self.assertEqual(bulk_data["dismissed_count"], 3)

        # Verify active is 0
        stats_after = self.client.get("/api/drainsight/stats").json()
        self.assertEqual(stats_after["total_events"], 0)

        # Verify archive has 3
        archived = self.client.get("/api/events/archived").json()
        self.assertEqual(archived["total"], 3)


if __name__ == "__main__":
    unittest.main()
