from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starlette.testclient import TestClient
from receiver_server import create_app
from upload_reprocessed_events import (
    build_frame_index,
    find_matching_frame,
    format_reprocessed_event,
    upload_events,
)


class ReprocessedUploadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.data_dir = self.base_path / "data"
        self.frames_dir = self.base_path / "frames"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.app = create_app(data_dir=self.data_dir, token="")
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_frame_indexing_and_matching(self) -> None:
        # Create dummy frame files
        f1 = self.frames_dir / "frame_00001_t0010.50.jpg"
        f2 = self.frames_dir / "frame_00002_t0020.00.jpg"
        f1.write_bytes(b"\xff\xd8\xff\xe0DUMMY_JPG_1")
        f2.write_bytes(b"\xff\xd8\xff\xe0DUMMY_JPG_2")

        index = build_frame_index(self.frames_dir)
        self.assertEqual(len(index), 2)
        self.assertIn(10.50, index)
        self.assertIn(20.00, index)

        # Match within delta
        matched1 = find_matching_frame(10.70, index, max_delta_s=0.5)
        self.assertIsNotNone(matched1)
        self.assertEqual(matched1.name, "frame_00001_t0010.50.jpg")

        # Match out of delta
        matched_none = find_matching_frame(15.00, index, max_delta_s=1.0)
        self.assertIsNone(matched_none)

    def test_format_reprocessed_event(self) -> None:
        raw = {
            "frame": 3000,
            "ts_s": 100.09,
            "status": "critical",
            "coverage_percent": 98.73,
            "gully_count": 1,
            "obstacle_count": 1,
            "method": "segmentation_mask",
            "gps": {
                "latitude": 36.8304,
                "longitude": 127.1790,
                "speed_mps": 0.74,
                "timestamp": "2026-09-05T01:10:43Z",
            },
            "detections": [
                {"class": "drain_full", "conf": 0.85, "bbox": [10.0, 20.0, 30.0, 40.0]}
            ],
        }

        formatted = format_reprocessed_event(raw, session_name="session_01", index=1)
        self.assertEqual(formatted["event_id"], "reproc_session_01_f03000")
        self.assertEqual(formatted["blockage"]["status"], "critical")
        self.assertEqual(formatted["blockage"]["coverage_percent"], 98.73)
        self.assertEqual(formatted["occlusion_pct"], 98.73)
        self.assertEqual(formatted["gps"]["latitude"], 36.8304)
        self.assertEqual(len(formatted["detections"]), 1)

    def test_upload_reprocessed_event_via_test_client(self) -> None:
        raw = {
            "frame": 1230,
            "ts_s": 41.04,
            "status": "warning",
            "coverage_percent": 45.5,
            "gps": {"latitude": 37.5665, "longitude": 126.9780},
        }
        payload = format_reprocessed_event(raw, session_name="test_session", index=1)

        # 1. Post JSON
        res = self.client.post("/upload", json=payload)
        self.assertEqual(res.status_code, 201)
        res_data = res.json()
        self.assertEqual(res_data["status"], "success")
        self.assertEqual(res_data["event_id"], payload["event_id"])

        # 2. Check /events list
        list_res = self.client.get("/events")
        self.assertEqual(list_res.status_code, 200)
        events_list = list_res.json()["events"]
        self.assertEqual(len(events_list), 1)

        saved = events_list[0]
        self.assertEqual(saved["event_id"], payload["event_id"])
        self.assertEqual(saved["blockage_status"], "warning")
        self.assertEqual(saved["coverage_percent"], 45.5)
        self.assertEqual(saved["occlusion_pct"], 45.5)
        self.assertAlmostEqual(saved["gps"]["lat"], 37.5665)
        self.assertAlmostEqual(saved["gps"]["lon"], 126.9780)

    def test_upload_with_evidence_image_via_test_client(self) -> None:
        raw = {
            "frame": 5000,
            "ts_s": 166.7,
            "status": "critical",
            "coverage_percent": 85.0,
            "gps": {"lat": 37.5000, "lon": 127.0000},
        }
        payload = format_reprocessed_event(raw, session_name="img_test", index=2)
        image_bytes = b"\xff\xd8\xff\xe0FAKE_EVIDENCE_JPEG"

        files = {
            "metadata": (None, json.dumps(payload), "application/json"),
            "image": ("evidence.jpg", image_bytes, "image/jpeg"),
        }
        res = self.client.post("/upload", files=files)
        self.assertEqual(res.status_code, 201)
        self.assertTrue(res.json()["image_stored"])

        # Verify image retrieval
        img_res = self.client.get(f"/events/{payload['event_id']}/image")
        self.assertEqual(img_res.status_code, 200)
        self.assertEqual(img_res.content, image_bytes)
        self.assertEqual(img_res.headers["content-type"], "image/jpeg")

    def test_sse_generator_newline_format(self) -> None:
        import inspect
        from drainsight_adapter import DrainSightAdapter

        src = inspect.getsource(DrainSightAdapter._setup_routes)
        # Verify no double backslash \\n remains in the yield statements
        yield_lines = [line.strip() for line in src.splitlines() if "yield f\"event:" in line]
        self.assertTrue(len(yield_lines) >= 2)
        for line in yield_lines:
            self.assertNotIn("\\\\n", line, f"Found double-escaped newline in: {line}")


if __name__ == "__main__":
    unittest.main()
