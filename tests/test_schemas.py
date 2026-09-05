"""Tests for schema validation (occlusion_pct, telemetry timestamp/sequence)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from starlette.testclient import TestClient

from receiver_server import create_app
from schemas import (
    EventSchema,
    TelemetryIn,
    validate_event_payload,
    validate_telemetry,
)
from tests.fixtures_synthetic_gps import (
    haversine_distance_m,
    make_synthetic_gps_csv,
    make_synthetic_proximity_events,
)


class SchemaValidationTest(unittest.TestCase):
    def test_occlusion_pct_valid_range(self) -> None:
        evt = EventSchema(occlusion_pct=50.0)
        self.assertEqual(evt.occlusion_pct, 50.0)

    def test_occlusion_pct_zero(self) -> None:
        evt = EventSchema(occlusion_pct=0.0)
        self.assertEqual(evt.occlusion_pct, 0.0)

    def test_occlusion_pct_100(self) -> None:
        evt = EventSchema(occlusion_pct=100.0)
        self.assertEqual(evt.occlusion_pct, 100.0)

    def test_occlusion_pct_negative_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EventSchema(occlusion_pct=-1.0)

    def test_occlusion_pct_over_100_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EventSchema(occlusion_pct=101.0)

    def test_coverage_percent_valid_range(self) -> None:
        from schemas import BlockageSchema
        blk = BlockageSchema(status="warning", coverage_percent=69.0)
        self.assertEqual(blk.coverage_percent, 69.0)

    def test_coverage_percent_negative_rejected(self) -> None:
        from schemas import BlockageSchema
        with self.assertRaises(ValueError):
            BlockageSchema(status="critical", coverage_percent=-1.0)

    def test_coverage_percent_over_100_rejected(self) -> None:
        from schemas import BlockageSchema
        with self.assertRaises(ValueError):
            BlockageSchema(status="critical", coverage_percent=101.0)

    def test_telemetry_with_timestamp_and_sequence(self) -> None:
        tel = TelemetryIn(vehicle_code="CAR-01", lat=37.5, lng=127.0, timestamp=1700000000.0, sequence=42)
        self.assertEqual(tel.timestamp, 1700000000.0)
        self.assertEqual(tel.sequence, 42)
        self.assertEqual(tel.vehicle_code, "CAR-01")

    def test_telemetry_negative_timestamp_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TelemetryIn(vehicle_code="CAR-01", lat=37.5, lng=127.0, timestamp=-1.0, sequence=0)

    def test_telemetry_negative_sequence_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TelemetryIn(vehicle_code="CAR-01", lat=37.5, lng=127.0, timestamp=1700000000.0, sequence=-1)

    def test_telemetry_vehicle_code_required(self) -> None:
        with self.assertRaises(ValueError):
            TelemetryIn(lat=37.5, lng=127.0, timestamp=1700000000.0, sequence=1)

    def test_telemetry_coordinates_required(self) -> None:
        with self.assertRaises(ValueError):
            TelemetryIn(vehicle_code="CAR-01", timestamp=1700000000.0, sequence=1)


class ValidatePayloadTest(unittest.TestCase):
    def test_validate_valid_event(self) -> None:
        payload = {"event_id": "test-001", "occlusion_pct": 50.0}
        ok, err, model = validate_event_payload(payload)
        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertIsNotNone(model)
        self.assertEqual(model.occlusion_pct, 50.0)

    def test_validate_invalid_occlusion(self) -> None:
        payload = {"event_id": "test-002", "occlusion_pct": 200.0}
        ok, err, model = validate_event_payload(payload)
        self.assertFalse(ok)
        self.assertIsNotNone(err)
        self.assertIn("occlusion_pct", err)

    def test_validate_telemetry_valid(self) -> None:
        payload = {"vehicle_code": "CAR-01", "lat": 37.5, "lng": 127.0, "timestamp": 1700000000.0, "sequence": 1}
        ok, err, model = validate_telemetry(payload)
        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertIsNotNone(model)

    def test_validate_telemetry_missing_sequence(self) -> None:
        payload = {"vehicle_code": "CAR-01", "lat": 37.5, "lng": 127.0, "timestamp": 1700000000.0}
        ok, err, model = validate_telemetry(payload)
        self.assertFalse(ok)
        self.assertIsNotNone(err)

    def test_validate_telemetry_missing_timestamp(self) -> None:
        payload = {"vehicle_code": "CAR-01", "sequence": 1}
        ok, err, model = validate_telemetry(payload)
        self.assertFalse(ok)
        self.assertIsNotNone(err)


class ProximityTest(unittest.TestCase):
    def test_haversine_distance_known_point(self) -> None:
        d = haversine_distance_m(36.838, 127.184, 36.8381, 127.184)
        self.assertGreater(d, 9.0)
        self.assertLess(d, 12.0)

    def test_haversine_zero_distance_same_point(self) -> None:
        d = haversine_distance_m(36.838, 127.184, 36.838, 127.184)
        self.assertAlmostEqual(d, 0.0, places=5)

    def test_synthetic_csv_within_5m(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_synthetic_gps_csv(
                center_lat=36.838,
                center_lon=127.184,
                step_m=1.0,
                num_points=5,
                output_path=Path(tmpdir) / "gps.csv",
            )
            self.assertTrue(csv_path.exists())
            text = csv_path.read_text()
            self.assertIn("timestamp,latitude,longitude,speed", text)
            self.assertGreater(len(text.splitlines()), 2)

    def test_proximity_events_all_within_radius(self) -> None:
        drains = make_synthetic_proximity_events(
            center_lat=36.838,
            center_lon=127.184,
            radius_m=5.0,
            num_drains=4,
        )
        self.assertEqual(len(drains), 4)
        for drain in drains:
            self.assertLessEqual(drain["distance_from_center_m"], 5.0 + 0.1)


class TelemetryEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.token = "test-token"
        self.app = create_app(data_dir=self.data_dir, token=self.token)
        self.client = TestClient(self.app)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_telemetry_endpoint_accepted(self) -> None:
        payload = {
            "vehicle_code": "CAR-01",
            "lat": 37.5,
            "lng": 127.0,
            "timestamp": 1700000000.0,
            "sequence": 1,
        }
        response = self.client.post("/api/telemetry", json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["processed"])

    def test_telemetry_invalid_sequence_rejected(self) -> None:
        payload = {
            "vehicle_code": "CAR-01",
            "lat": 37.5,
            "lng": 127.0,
            "timestamp": 1700000000.0,
            "sequence": -1,
        }
        response = self.client.post("/api/telemetry", json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 422)

    def test_telemetry_missing_timestamp_rejected(self) -> None:
        payload = {
            "vehicle_code": "CAR-01",
            "lat": 37.5,
            "lng": 127.0,
            "sequence": 1,
        }
        response = self.client.post("/api/telemetry", json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 422)

    def test_upload_occlusion_out_of_range_rejected(self) -> None:
        payload = {
            "event_id": "evt-bad-occlusion",
            "event_type": "gully_blockage",
            "source": "test-pi",
            "occlusion_pct": 150.0,
        }
        response = self.client.post("/upload", json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 422)

    def test_upload_occlusion_negative_rejected(self) -> None:
        payload = {
            "event_id": "evt-bad-occlusion2",
            "event_type": "gully_blockage",
            "source": "test-pi",
            "occlusion_pct": -1.0,
        }
        response = self.client.post("/upload", json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 422)

    def test_upload_boundary_0_and_100_accepted(self) -> None:
        for occ in (0.0, 100.0):
            payload = {
                "event_id": f"evt-occ-{int(occ)}",
                "event_type": "gully_blockage",
                "source": "test-pi",
                "occlusion_pct": occ,
            }
            response = self.client.post("/upload", json=payload, headers=self.headers)
            self.assertEqual(response.status_code, 201)


if __name__ == "__main__":
    unittest.main()