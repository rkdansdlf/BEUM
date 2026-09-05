"""
Comprehensive 12-Gate Re-Release Test Suite for DrainSight & BEUM
-----------------------------------------------------------------
Validates all 5 release blockers and the 12 regression gate criteria:
1. occlusion_pct negative and > 100 rejected with HTTP 422
2. occlusion_pct boundary 0.0 and 100.0 accepted with HTTP 200/201
3. UNASSESSABLE behavior regression (occlusion_pct=None, default reason_code)
4. DetectionCreate required fields contract preserved
5. TelemetryIn timestamp (float) and sequence (int) schema validation
6. Telemetry duplicate sequence suppression (processed: False, reason: duplicate)
7. Telemetry out-of-order sequence suppression (processed: False, reason: out_of_order)
8. Accepted telemetry triggers WebSocket broadcast
9. Silence timer concurrency: independent keys per (drain_id, vehicle_code)
10. Proximity exact 5.0m included vs 5.0001m excluded via monkeypatch
11. nearby_drains returns integer DB IDs
12. CSV replay compatibility with auto-sequenced payloads
"""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from starlette.testclient import TestClient
import receiver_server
from receiver_server import create_app, haversine_m
from schemas import Status, Source


class ReleaseGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.token = "release-token"
        self.receiver = receiver_server.ReceiverApp(data_dir=self.data_dir, token=self.token)
        self.app = self.receiver.app
        self.client = TestClient(self.app)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # Gate 1: -1 / 101 -> 422, DB unchanged, WS=0
    def test_gate1_occlusion_pct_negative_and_over_100_rejected_422(self) -> None:
        # 1. DetectionCreate with -1.0
        neg_payload = {
            "drain_id": 101,
            "vehicle_code": "CAR-01",
            "status": "CRITICAL",
            "occlusion_pct": -1.0,
            "lat": 37.5,
            "lng": 127.0,
            "source": "AI_VISION",
        }
        res_neg = self.client.post("/api/detections", json=neg_payload, headers=self.headers)
        self.assertEqual(res_neg.status_code, 422, f"Expected 422, got {res_neg.status_code}")

        # 2. DetectionCreate with 101.0
        over_payload = {
            "drain_id": 101,
            "vehicle_code": "CAR-01",
            "status": "CRITICAL",
            "occlusion_pct": 101.0,
            "lat": 37.5,
            "lng": 127.0,
            "source": "AI_VISION",
        }
        res_over = self.client.post("/api/detections", json=over_payload, headers=self.headers)
        self.assertEqual(res_over.status_code, 422, f"Expected 422, got {res_over.status_code}")

        # 3. Multipart /upload with -1.0
        up_payload = {
            "event_id": "evt-bad-occ",
            "blockage": {"status": "critical", "coverage_percent": -1.0},
        }
        res_up = self.client.post("/upload", json=up_payload, headers=self.headers)
        self.assertEqual(res_up.status_code, 422)

        # 4. Verify DB has 0 detections
        with self.receiver._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM detections")
            self.assertEqual(cur.fetchone()[0], 0)

    # Gate 2: 0 / 100 -> 200/201 accepted
    def test_gate2_occlusion_boundary_accepted(self) -> None:
        p0 = {
            "drain_id": 1,
            "vehicle_code": "CAR-01",
            "status": "NORMAL",
            "occlusion_pct": 0.0,
            "lat": 37.5,
            "lng": 127.0,
            "source": "AI_VISION",
        }
        res0 = self.client.post("/api/detections", json=p0, headers=self.headers)
        self.assertEqual(res0.status_code, 201)
        self.assertEqual(res0.json()["occlusion_pct"], 0.0)

        p100 = {
            "drain_id": 2,
            "vehicle_code": "CAR-01",
            "status": "CRITICAL",
            "occlusion_pct": 100.0,
            "lat": 37.5,
            "lng": 127.0,
            "source": "AI_VISION",
        }
        res100 = self.client.post("/api/detections", json=p100, headers=self.headers)
        self.assertEqual(res100.status_code, 201)
        self.assertEqual(res100.json()["occlusion_pct"], 100.0)

    # Gate 3: UNASSESSABLE behavior regression
    def test_gate3_unassessable_behavior_regression(self) -> None:
        p = {
            "drain_id": 3,
            "vehicle_code": "CAR-01",
            "status": "UNASSESSABLE",
            "occlusion_pct": 50.0,  # Should be forced to None
            "lat": 37.5,
            "lng": 127.0,
            "source": "AI_VISION",
        }
        res = self.client.post("/api/detections", json=p, headers=self.headers)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIsNone(data["occlusion_pct"])

        # Invalid reason_code on NORMAL status
        p_invalid = {
            "drain_id": 4,
            "vehicle_code": "CAR-01",
            "status": "NORMAL",
            "reason_code": "DRAIN_NOT_DETECTED",
            "lat": 37.5,
            "lng": 127.0,
            "source": "AI_VISION",
        }
        res_inv = self.client.post("/api/detections", json=p_invalid, headers=self.headers)
        self.assertEqual(res_inv.status_code, 422)

    # Gate 4: Required fields contract preserved
    def test_gate4_required_fields_contract(self) -> None:
        # Missing drain_id
        bad = {
            "vehicle_code": "CAR-01",
            "status": "NORMAL",
            "lat": 37.5,
            "lng": 127.0,
            "source": "AI_VISION",
        }
        res = self.client.post("/api/detections", json=bad, headers=self.headers)
        self.assertEqual(res.status_code, 422)

        # Missing source
        bad2 = {
            "drain_id": 1,
            "vehicle_code": "CAR-01",
            "status": "NORMAL",
            "lat": 37.5,
            "lng": 127.0,
        }
        res2 = self.client.post("/api/detections", json=bad2, headers=self.headers)
        self.assertEqual(res2.status_code, 422)

    # Gate 5: Telemetry schema requires float timestamp and int sequence
    def test_gate5_telemetry_schema_strictness(self) -> None:
        # Missing sequence
        bad_seq = {
            "vehicle_code": "CAR-01",
            "lat": 37.5,
            "lng": 127.0,
            "timestamp": 1725514800.123,
        }
        res = self.client.post("/api/telemetry", json=bad_seq)
        self.assertEqual(res.status_code, 422)

        # Missing timestamp
        bad_ts = {
            "vehicle_code": "CAR-01",
            "lat": 37.5,
            "lng": 127.0,
            "sequence": 1,
        }
        res2 = self.client.post("/api/telemetry", json=bad_ts)
        self.assertEqual(res2.status_code, 422)

        # Valid
        good = {
            "vehicle_code": "CAR-01",
            "lat": 37.5,
            "lng": 127.0,
            "timestamp": 1725514800.123,
            "sequence": 1,
        }
        res3 = self.client.post("/api/telemetry", json=good)
        self.assertEqual(res3.status_code, 200)
        self.assertTrue(res3.json()["ok"])
        self.assertTrue(res3.json()["processed"])

    # Gate 6: Duplicate sequence -> processed: False, reason: "duplicate"
    def test_gate6_telemetry_duplicate_suppression(self) -> None:
        p1 = {
            "vehicle_code": "TRUCK-99",
            "lat": 37.5,
            "lng": 127.0,
            "timestamp": 1725514800.0,
            "sequence": 50,
        }
        res1 = self.client.post("/api/telemetry", json=p1)
        self.assertEqual(res1.status_code, 200)
        self.assertTrue(res1.json()["processed"])

        # Send same sequence 50 again
        p2 = {
            "vehicle_code": "TRUCK-99",
            "lat": 37.5,
            "lng": 127.0,
            "timestamp": 1725514801.0,
            "sequence": 50,
        }
        res2 = self.client.post("/api/telemetry", json=p2)
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertTrue(data2["ok"])
        self.assertFalse(data2["processed"])
        self.assertEqual(data2["reason"], "duplicate")

    # Gate 7: Out-of-order sequence -> processed: False, reason: "out_of_order"
    def test_gate7_telemetry_out_of_order_handling(self) -> None:
        p1 = {
            "vehicle_code": "TRUCK-88",
            "lat": 37.5,
            "lng": 127.0,
            "timestamp": 1725514800.0,
            "sequence": 100,
        }
        self.client.post("/api/telemetry", json=p1)

        # Arrives with older sequence 99
        p_old = {
            "vehicle_code": "TRUCK-88",
            "lat": 37.5,
            "lng": 127.0,
            "timestamp": 1725514799.0,
            "sequence": 99,
        }
        res_old = self.client.post("/api/telemetry", json=p_old)
        self.assertEqual(res_old.status_code, 200)
        data_old = res_old.json()
        self.assertTrue(data_old["ok"])
        self.assertFalse(data_old["processed"])
        self.assertEqual(data_old["reason"], "out_of_order")

    # Gate 8: Accepted telemetry triggers WebSocket broadcast
    def test_gate8_telemetry_accepted_broadcast(self) -> None:
        with self.client.websocket_connect("/api/drainsight/ws/telemetry") as ws:
            conn_msg = ws.receive_json()
            self.assertEqual(conn_msg["type"], "connected")

            p = {
                "vehicle_code": "BUS-07",
                "lat": 37.51,
                "lng": 127.01,
                "timestamp": 1725514850.0,
                "sequence": 1,
            }
            res = self.client.post("/api/telemetry", json=p)
            self.assertEqual(res.status_code, 200)

            ws_msg = ws.receive_json()
            self.assertEqual(ws_msg["type"], "telemetry")
            self.assertEqual(ws_msg["vehicle_code"], "BUS-07")
            self.assertEqual(ws_msg["sequence"], 1)

    # Gate 9: Silence timer concurrency per (drain_id, vehicle_code)
    def test_gate9_silence_timer_concurrency_keys(self) -> None:
        # Seed Drain 1
        with self.receiver._get_connection() as conn:
            conn.execute("INSERT INTO drains (id, lat, lng) VALUES (1, 37.50000, 127.00000)")
            conn.commit()

        # Vehicle A approaches Drain 1
        p_a = {
            "vehicle_code": "CAR-A",
            "lat": 37.50001,
            "lng": 127.00000,
            "timestamp": 1725514900.0,
            "sequence": 1,
        }
        res_a = self.client.post("/api/telemetry", json=p_a)
        self.assertEqual(res_a.status_code, 200)
        self.assertIn(1, res_a.json()["nearby_drains"])
        self.assertIn((1, "CAR-A"), self.receiver.active_silence_keys)

        # Vehicle B approaches Drain 1
        p_b = {
            "vehicle_code": "CAR-B",
            "lat": 37.50001,
            "lng": 127.00000,
            "timestamp": 1725514901.0,
            "sequence": 1,
        }
        res_b = self.client.post("/api/telemetry", json=p_b)
        self.assertEqual(res_b.status_code, 200)
        self.assertIn(1, res_b.json()["nearby_drains"])

        # Both timers must coexist!
        self.assertIn((1, "CAR-A"), self.receiver.active_silence_keys, "Vehicle A timer must NOT be cancelled by Vehicle B")
        self.assertIn((1, "CAR-B"), self.receiver.active_silence_keys, "Vehicle B timer must be active")

    # Gate 10: Proximity 5m exact boundary via monkeypatch
    def test_gate10_proximity_5m_exact_boundary_monkeypatch(self) -> None:
        with self.receiver._get_connection() as conn:
            conn.execute("INSERT INTO drains (id, lat, lng) VALUES (55, 37.5, 127.0)")
            conn.commit()

        orig_haversine = receiver_server.haversine_m
        try:
            # 1. Exactly 5.0m -> MUST be included (<= 5.0 rule)
            receiver_server.haversine_m = lambda *args: 5.0
            p1 = {
                "vehicle_code": "TEST-PROX",
                "lat": 37.5,
                "lng": 127.0,
                "timestamp": 1725515000.0,
                "sequence": 1,
            }
            res1 = self.client.post("/api/telemetry", json=p1)
            self.assertEqual(res1.status_code, 200)
            self.assertIn(55, res1.json()["nearby_drains"], "5.0m exact boundary must be included")

            # 2. 5.0001m -> MUST be excluded (> 5.0)
            receiver_server.haversine_m = lambda *args: 5.0001
            p2 = {
                "vehicle_code": "TEST-PROX",
                "lat": 37.5,
                "lng": 127.0,
                "timestamp": 1725515001.0,
                "sequence": 2,
            }
            res2 = self.client.post("/api/telemetry", json=p2)
            self.assertEqual(res2.status_code, 200)
            self.assertNotIn(55, res2.json()["nearby_drains"], "5.0001m must be excluded")
        finally:
            receiver_server.haversine_m = orig_haversine

    # Gate 11: nearby_drains returns integer DB IDs
    def test_gate11_nearby_drains_integer_ids(self) -> None:
        with self.receiver._get_connection() as conn:
            conn.execute("INSERT INTO drains (id, lat, lng) VALUES (777, 37.50000, 127.00000)")
            conn.commit()

        p = {
            "vehicle_code": "CAR-INT",
            "lat": 37.50001,
            "lng": 127.00000,
            "timestamp": 1725515100.0,
            "sequence": 1,
        }
        res = self.client.post("/api/telemetry", json=p)
        self.assertEqual(res.status_code, 200)
        drains = res.json()["nearby_drains"]
        self.assertIn(777, drains)
        self.assertIsInstance(drains[0], int, "nearby_drains element must be int, not str")

    # Gate 12: CSV replay compatibility with auto-sequenced payloads
    def test_gate12_csv_replay_compat_with_sequence(self) -> None:
        # Simulate 10 CSV rows being replayed with monotonic sequence and timestamp
        base_time = 1725515200.0
        for i in range(10):
            payload = {
                "vehicle_code": "REPLAY-VEHICLE-01",
                "lat": 37.5000 + (i * 0.0001),
                "lng": 127.0000 + (i * 0.0001),
                "timestamp": base_time + (i * 0.1),
                "sequence": i + 1,
            }
            res = self.client.post("/api/telemetry", json=payload)
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.json()["processed"])
            self.assertEqual(res.json()["sequence"], i + 1)


if __name__ == "__main__":
    unittest.main()
