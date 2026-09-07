"""Unit tests for LiveCameraStreamer and /api/camera endpoints."""

import io
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from starlette.testclient import TestClient

from camera_streamer import LiveCameraStreamer
from receiver_server import create_app


class CameraStreamerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.streamer = LiveCameraStreamer()

    def tearDown(self) -> None:
        if self.streamer:
            self.streamer.stop()

    def test_camera_streamer_initial_status(self) -> None:
        status = self.streamer.get_status()
        self.assertIn("fps", status)
        self.assertIn("latency_ms", status)
        self.assertIn("detections", status)
        self.assertIn("occlusion_pct", status)
        self.assertIn("status", status)
        self.assertIn("source", status)
        self.assertIn("model", status)
        self.assertIn("is_running", status)

    def test_camera_streamer_config_updates(self) -> None:
        self.streamer.set_confidence(0.42)
        self.assertAlmostEqual(self.streamer.confidence_threshold, 0.42)

        self.streamer.set_source("test_source_path.mp4")
        self.assertEqual(self.streamer.video_source, "test_source_path.mp4")

    def test_detect_single_image_with_synthetic_frame(self) -> None:
        # Create a synthetic image (480x640x3) with a drawn rectangle
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (200, 200, 200), -1)
        _, buf = cv2.imencode(".jpg", img)
        img_bytes = buf.tobytes()

        result = self.streamer.detect_single_image(img_bytes, conf=0.1)
        self.assertIn("detections", result)
        self.assertIn("occlusion_pct", result)
        self.assertIn("status", result)
        self.assertIn("latency_ms", result)
        self.assertIn("annotated_image_base64", result)
        self.assertGreater(len(result["annotated_image_base64"]), 0)


class CameraApiEndpointsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app(data_dir=self.temp_dir.name)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_camera_status_endpoint(self) -> None:
        # Test /api/camera/status
        resp = self.client.get("/api/camera/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("fps", data)
        self.assertIn("model", data)
        self.assertIn("status", data)

        # Test legacy /api/drainsight/camera/status alias
        resp_alias = self.client.get("/api/drainsight/camera/status")
        self.assertEqual(resp_alias.status_code, 200)

    def test_post_camera_config_endpoint(self) -> None:
        resp = self.client.post("/api/camera/config", json={"conf": 0.35})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("ok"))

    def test_post_camera_detect_endpoint(self) -> None:
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        img_bytes = buf.tobytes()

        # Test multipart upload
        files = {"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        resp = self.client.post("/api/camera/detect", files=files, data={"conf": "0.2"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("detections", data)
        self.assertIn("annotated_image_base64", data)


if __name__ == "__main__":
    unittest.main()
