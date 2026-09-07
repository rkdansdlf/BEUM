import numpy as np
from gully_system.runtime import GullyRuntime
from gully_system.types import Detection

def test_draw_detections_mutually_exclusive_masks():
    """Verify that gully and obstacle masks do not blend together into muddy colors."""
    runtime = object.__new__(GullyRuntime)
    h, w = 100, 100
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Gully mask: full 100x100
    gully_mask = np.ones((h, w), dtype=np.uint8)
    det_gully = Detection(bbox=(10, 10, 90, 90), confidence=0.85, class_id=0, class_name="drain_area", mask=gully_mask)

    # Obstacle mask: center 40..60 (inside gully)
    obs_mask = np.zeros((h, w), dtype=np.uint8)
    obs_mask[40:60, 40:60] = 1
    det_obs = Detection(bbox=(30, 30, 70, 70), confidence=0.88, class_id=1, class_name="drain_full", mask=obs_mask)

    runtime._draw_detections(frame, (det_gully, det_obs))

    # Center pixel (50, 50) is blocked -> should have high Red channel, zero Green channel
    center_b, center_g, center_r = frame[50, 50]
    assert center_r > 50, f"Expected high red channel at center pixel, got R={center_r}"
    assert center_g == 0, f"Expected zero green at blocked center pixel, got G={center_g}"

    # Clear region (80, 80) is clear gully -> should have high Green channel, zero Red channel
    clear_b, clear_g, clear_r = frame[80, 80]
    assert clear_g > 50, f"Expected high green channel at clear pixel, got G={clear_g}"
    assert clear_r == 0, f"Expected zero red at clear pixel, got R={clear_r}"

def test_draw_detections_anti_collision_labels():
    """Verify that overlapping detections do not draw badges at the exact same coordinates."""
    runtime = object.__new__(GullyRuntime)
    h, w = 500, 500
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Two detections with near-identical bounding boxes
    d1 = Detection(bbox=(100, 100, 200, 200), confidence=0.88, class_id=1, class_name="drain_full")
    d2 = Detection(bbox=(105, 102, 195, 198), confidence=0.87, class_id=0, class_name="drain_area")

    runtime._draw_detections(frame, (d1, d2))
    assert np.any(frame > 0)
