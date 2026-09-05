"""
Complete regression replay test with 1,305 real-world GPS coordinates
from data/2026-09-04_04-19-03/Location.csv (757 records) and
data/2026-09-04_04-31-48/Location.csv (548 records).

Validates:
1. All 1,305 points are processed with HTTP 200 and sequence validation.
2. DB vehicle state ends at last_sequence = 1305.
3. Duplicate and out-of-order rejection at the end of the full route.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from starlette.testclient import TestClient

from receiver_server import create_app


class FullCsvReplayRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.token = "replay-token"
        self.app = create_app(data_dir=self.data_dir, token=self.token)
        self.client = TestClient(self.app)
        self.headers = {"Authorization": f"Bearer {self.token}"}

        self.csv_paths = [
            Path("data/2026-09-04_04-19-03/Location.csv"),
            Path("data/2026-09-04_04-31-48/Location.csv"),
        ]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_full_1305_csv_replay(self) -> None:
        # 1. Load all 1,305 CSV records
        all_records: list[dict[str, str]] = []
        for p in self.csv_paths:
            self.assertTrue(p.exists(), f"CSV file {p} must exist")
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_records.extend(list(reader))

        total_count = len(all_records)
        self.assertEqual(total_count, 1305, f"Expected exactly 1,305 records, got {total_count}")

        vehicle_code = "REPLAY-BUS-1305"
        base_epoch = 1725420000.0

        # 2. Replay all 1,305 records sequentially
        for i, row in enumerate(all_records, start=1):
            lat = float(row["latitude"])
            lng = float(row["longitude"])
            raw_speed = float(row.get("speed", 0.0))
            speed = max(0.0, raw_speed)
            
            # Timestamp derived monotonically from record index or parsed time
            ts = base_epoch + (i * 0.5)

            payload = {
                "vehicle_code": vehicle_code,
                "lat": lat,
                "lng": lng,
                "speed_mps": speed,
                "timestamp": ts,
                "sequence": i,
            }

            res = self.client.post("/api/telemetry", json=payload, headers=self.headers)
            self.assertEqual(
                res.status_code,
                200,
                f"Row {i}/{total_count} failed with status {res.status_code}: {res.text}",
            )
            data = res.json()
            self.assertTrue(data["ok"])
            self.assertTrue(data["processed"], f"Row {i} should be processed: True")
            self.assertEqual(data["sequence"], i)

        # 3. Verify final DB state has last_sequence = 1305
        db_path = self.data_dir / "beum_events.db"
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT last_sequence, last_timestamp FROM vehicle_telemetry_states WHERE vehicle_code = ?",
                (vehicle_code,),
            )
            row = cur.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], 1305)

        # 4. Verify duplicate suppression on the 1305th packet
        dup_payload = {
            "vehicle_code": vehicle_code,
            "lat": 36.83,
            "lng": 127.18,
            "timestamp": base_epoch + (1305 * 0.5) + 1.0,
            "sequence": 1305,
        }
        res_dup = self.client.post("/api/telemetry", json=dup_payload, headers=self.headers)
        self.assertEqual(res_dup.status_code, 200)
        data_dup = res_dup.json()
        self.assertTrue(data_dup["ok"])
        self.assertFalse(data_dup["processed"])
        self.assertEqual(data_dup["reason"], "duplicate")

        # 5. Verify out-of-order suppression with sequence 500
        old_payload = {
            "vehicle_code": vehicle_code,
            "lat": 36.83,
            "lng": 127.18,
            "timestamp": base_epoch + 250.0,
            "sequence": 500,
        }
        res_old = self.client.post("/api/telemetry", json=old_payload, headers=self.headers)
        self.assertEqual(res_old.status_code, 200)
        data_old = res_old.json()
        self.assertTrue(data_old["ok"])
        self.assertFalse(data_old["processed"])
        self.assertEqual(data_old["reason"], "out_of_order")

        # 6. Verify subsequent sequence 1306 is accepted
        next_payload = {
            "vehicle_code": vehicle_code,
            "lat": 36.83,
            "lng": 127.18,
            "timestamp": base_epoch + (1306 * 0.5),
            "sequence": 1306,
        }
        res_next = self.client.post("/api/telemetry", json=next_payload, headers=self.headers)
        self.assertEqual(res_next.status_code, 200)
        data_next = res_next.json()
        self.assertTrue(data_next["ok"])
        self.assertTrue(data_next["processed"])
        self.assertEqual(data_next["sequence"], 1306)


if __name__ == "__main__":
    unittest.main()
