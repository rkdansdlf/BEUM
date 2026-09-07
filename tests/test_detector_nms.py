from __future__ import annotations

import unittest
from gully_system.config import DetectorConfig
from gully_system.detector import YOLODetector
from gully_system.types import Detection


class DetectorNMSTest(unittest.TestCase):
    def test_detector_config_defaults_agnostic_nms_to_true(self) -> None:
        cfg = DetectorConfig()
        self.assertTrue(cfg.agnostic_nms)

        cfg_from_dict = DetectorConfig.from_dict({"agnostic_nms": False})
        self.assertFalse(cfg_from_dict.agnostic_nms)

    def test_cross_class_nms_suppresses_duplicate_overlapping_detections(self) -> None:
        # Near-identical bounding boxes for drain_full (0.88) and drain_area (0.87)
        d_full = Detection(bbox=(100.0, 100.0, 300.0, 300.0), confidence=0.88, class_id=1, class_name="drain_full")
        d_area = Detection(bbox=(105.0, 102.0, 298.0, 295.0), confidence=0.87, class_id=0, class_name="drain_area")

        filtered = YOLODetector._apply_cross_class_nms([d_full, d_area], iou_threshold=0.45)
        # Only the higher confidence detection (d_full, 0.88) should survive
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].class_name, "drain_full")
        self.assertEqual(filtered[0].confidence, 0.88)

    def test_cross_class_nms_preserves_distinct_non_overlapping_detections(self) -> None:
        d1 = Detection(bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.85, class_id=0, class_name="drain_area")
        d2 = Detection(bbox=(200.0, 200.0, 250.0, 250.0), confidence=0.80, class_id=1, class_name="drain_full")

        filtered = YOLODetector._apply_cross_class_nms([d1, d2], iou_threshold=0.45)
        self.assertEqual(len(filtered), 2)


if __name__ == "__main__":
    unittest.main()
