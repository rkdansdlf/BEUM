"""Integration test for ActiveLearningTriager wired into GullyRuntime."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from gully_system.config import SystemConfig, TriageConfig, DetectorConfig
from gully_system.runtime import GullyRuntime
from gully_system.types import Detection


class TriageRuntimeIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="beum_runtime_triage_test_")
        self.triage_dir = Path(self.temp_dir) / "triage"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_runtime_harvests_uncertain_frame_when_triage_enabled(self) -> None:
        config = SystemConfig(
            source="0",
            max_frames=2,
            run_preflight=False,
            triage=TriageConfig(
                enabled=True,
                output_dir=str(self.triage_dir),
                uncertain_conf_range=(0.15, 0.28),
                cooldown_s=0.01,
                save_images=True,
            ),
        )

        # Mock detector returning an uncertain detection
        mock_detector = MagicMock()
        mock_detector.mapping_result = None
        uncertain_detection = Detection(
            bbox=(100.0, 100.0, 300.0, 300.0),
            confidence=0.22,
            class_id=0,
            class_name="drain_area",
        )
        mock_detector.predict.return_value = [uncertain_detection]

        runtime = GullyRuntime(config, detector=mock_detector)

        # Mock camera returning valid frames
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        runtime.camera.open = MagicMock()
        runtime.camera.read = MagicMock(side_effect=[(True, fake_frame.copy()), (True, fake_frame.copy()), (False, None)])
        runtime.camera.release = MagicMock()

        result = runtime.run(max_frames=2)

        self.assertGreaterEqual(result["harvested_candidates"], 1)

        # Verify disk artifacts
        json_files = list(self.triage_dir.glob("al_*.json"))
        jpg_files = list(self.triage_dir.glob("al_*.jpg"))
        self.assertGreaterEqual(len(json_files), 1)
        self.assertGreaterEqual(len(jpg_files), 1)

        # Verify JSON structure
        payload = json.loads(json_files[0].read_text(encoding="utf-8"))
        self.assertIn("candidate_id", payload)
        self.assertIn("uncertainty_low_conf", payload["reasons"])
        self.assertEqual(payload["detections"][0]["confidence"], 0.22)

        # Verify saved image is valid
        img = cv2.imread(str(jpg_files[0]))
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[:2], (480, 640))

    def test_runtime_does_not_harvest_when_triage_disabled(self) -> None:
        config = SystemConfig(
            source="0",
            max_frames=2,
            run_preflight=False,
            triage=TriageConfig(
                enabled=False,
                output_dir=str(self.triage_dir),
            ),
        )

        mock_detector = MagicMock()
        mock_detector.mapping_result = None
        uncertain_detection = Detection(
            bbox=(100.0, 100.0, 300.0, 300.0),
            confidence=0.22,
            class_id=0,
            class_name="drain_area",
        )
        mock_detector.predict.return_value = [uncertain_detection]

        runtime = GullyRuntime(config, detector=mock_detector)
        self.assertIsNone(runtime.triager)

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        runtime.camera.open = MagicMock()
        runtime.camera.read = MagicMock(side_effect=[(True, fake_frame.copy()), (False, None)])
        runtime.camera.release = MagicMock()

        result = runtime.run(max_frames=1)
        self.assertEqual(result["harvested_candidates"], 0)
        self.assertEqual(len(list(self.triage_dir.glob("al_*"))), 0)


if __name__ == "__main__":
    unittest.main()
