from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from migrations import migrate_database
from receiver_server import ReceiverApp


class MigrationTest(unittest.TestCase):
    def test_existing_database_gets_event_metrics_and_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            db_path = data_dir / "beum_events.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    source TEXT,
                    created_at TEXT,
                    received_at TEXT,
                    blockage_status TEXT,
                    coverage_percent REAL,
                    roi_profile TEXT,
                    lat REAL,
                    lon REAL,
                    has_image BOOLEAN,
                    image_path TEXT,
                    raw_payload TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO events (event_id, coverage_percent) VALUES (?, ?)",
                ("legacy-event", 42.5),
            )
            conn.commit()
            conn.close()

            ReceiverApp(data_dir=data_dir)

            conn = sqlite3.connect(db_path)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
            self.assertIn("coverage_percent", columns)
            self.assertIn("occlusion_pct", columns)
            row = conn.execute(
                "SELECT coverage_percent, occlusion_pct FROM events WHERE event_id = ?",
                ("legacy-event",),
            ).fetchone()
            self.assertEqual(row, (42.5, 42.5))
            versions = [row[0] for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
            self.assertEqual(versions, ["0001", "0002", "0003"])
            conn.close()

    def test_migrations_are_idempotent_on_fresh_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "fresh.db"
            self.assertEqual(migrate_database(db_path), ["0001", "0002", "0003"])
            self.assertEqual(migrate_database(db_path), [])

            conn = sqlite3.connect(db_path)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            self.assertTrue(
                {"events", "vehicle_telemetry_states", "detections", "drains", "schema_migrations"}
                <= tables
            )
            conn.close()

    def test_legacy_database_missing_both_metric_columns_is_upgraded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "legacy.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    source TEXT,
                    created_at TEXT,
                    received_at TEXT,
                    blockage_status TEXT,
                    roi_profile TEXT,
                    lat REAL,
                    lon REAL,
                    has_image BOOLEAN,
                    image_path TEXT,
                    raw_payload TEXT
                )
                """
            )
            conn.commit()
            conn.close()

            self.assertEqual(migrate_database(db_path), ["0001", "0002", "0003"])

            conn = sqlite3.connect(db_path)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
            self.assertTrue({"coverage_percent", "occlusion_pct"} <= columns)
            conn.close()


if __name__ == "__main__":
    unittest.main()
