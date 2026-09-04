from __future__ import annotations
import unittest
from gully_system.temporal_filter import TemporalFilter, intersection_over_union
from gully_system.types import Detection

def detection(x: int = 0, y: int = 0) -> Detection:
    return Detection((x, y, x + 10, y + 10), 0.9, 1, 'debris')

class TemporalFilterTest(unittest.TestCase):
    def test_iou(self) -> None:
        self.assertAlmostEqual(
            intersection_over_union(detection(), detection(5, 0)),
            1.0 / 3.0,
        )

    def test_confirmation_and_exit(self) -> None:
        tracker = TemporalFilter(min_hits=2, max_missed=1, iou_threshold=0.2)
        self.assertEqual(tracker.update([detection()]).entered, ())
        result = tracker.update([detection(1, 0)])
        self.assertEqual(len(result.entered), 1)
        self.assertEqual(len(result.validated), 1)
        self.assertEqual(tracker.update([]).exited, ())
        self.assertEqual(len(tracker.update([]).exited), 1)

    def test_new_track_is_not_missed_on_creation_frame(self) -> None:
        tracker = TemporalFilter(min_hits=3, max_missed=0)
        tracker.update([detection()])
        self.assertEqual(len(tracker._tracks), 1)

if __name__ == '__main__':
    unittest.main()
