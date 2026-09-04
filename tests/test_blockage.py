from __future__ import annotations
import unittest
import numpy as np
from gully_system.blockage import BlockageAnalyzer, BlockageEventGate, BlockageMetrics
from gully_system.types import Detection

def metrics(status: str, coverage: float) -> BlockageMetrics:
    return BlockageMetrics(status, coverage, 1, 1, 1, 1, 'bbox_estimate', 0.9)

class BlockageEventGateTest(unittest.TestCase):
    def test_emits_hazard_escalation_and_recovery(self) -> None:
        gate = BlockageEventGate(change_percent=10, cooldown_s=300)
        self.assertFalse(gate.should_emit(metrics('clear', 0), now=0))
        self.assertTrue(gate.should_emit(metrics('warning', 25), now=1))
        self.assertFalse(gate.should_emit(metrics('warning', 30), now=2))
        self.assertTrue(gate.should_emit(metrics('critical', 55), now=3))
        self.assertTrue(gate.should_emit(metrics('clear', 0), now=4))

class BlockageAnalyzerTest(unittest.TestCase):
    def test_segmentation_coverage_is_overlap_over_gully_area(self) -> None:
        gully_mask = np.zeros((10, 10), dtype=np.uint8)
        gully_mask[2:8, 2:8] = 1
        debris_mask = np.zeros((10, 10), dtype=np.uint8)
        debris_mask[2:8, 2:5] = 1
        detections = [
            Detection((2, 2, 8, 8), 0.95, 0, 'gully', mask=gully_mask),
            Detection((2, 2, 5, 8), 0.9, 1, 'debris', mask=debris_mask),
        ]
        result = BlockageAnalyzer(warning_percent=20, critical_percent=50).analyze(
            detections, (10, 10, 3)
        )
        self.assertAlmostEqual(result.coverage_percent, 50.0)
        self.assertEqual(result.status, 'critical')
        self.assertEqual(result.method, 'segmentation_mask')

    def test_detection_only_model_uses_explicit_estimate_method(self) -> None:
        detections = [
            Detection((2, 2, 8, 8), 0.95, 0, 'gully'),
            Detection((2, 2, 5, 8), 0.9, 1, 'debris'),
        ]
        result = BlockageAnalyzer().analyze(detections, (10, 10, 3))
        self.assertAlmostEqual(result.coverage_percent, 50.0)
        self.assertEqual(result.method, 'bbox_estimate')

if __name__ == '__main__':
    unittest.main()
