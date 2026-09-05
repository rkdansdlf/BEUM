import numpy as np
import pytest
from ultralytics.data.augment import polygons2masks, polygons2masks_overlap

def test_synthetic_overlap_rasterization():
    """
    Test that Ultralytics rasterization reproduces the documented behavior:
    - overlap_mask=True carves a hole (donut) in the larger outer polygon.
    - overlap_mask=False preserves both outer and inner polygons independently.
    """
    img_h, img_w = 100, 100

    # Outer polygon: 80% of image (from 10 to 90 in x and y)
    outer_poly = np.array([
        [10.0, 10.0],
        [90.0, 10.0],
        [90.0, 90.0],
        [10.0, 90.0]
    ], dtype=np.float32)

    # Inner polygon: center 20% of image (from 40 to 60 in x and y, fully contained)
    inner_poly = np.array([
        [40.0, 40.0],
        [60.0, 40.0],
        [60.0, 60.0],
        [40.0, 60.0]
    ], dtype=np.float32)

    segments = [outer_poly, inner_poly]

    # 1. OVERLAP MODE (overlap_mask = True)
    overlap_mask, sorted_idx = polygons2masks_overlap((img_h, img_w), segments, downsample_ratio=1)

    # In overlap mode, instances are sorted by area descending:
    assert sorted_idx[0] == 0
    assert sorted_idx[1] == 1

    # Output mask is 2D (H, W)
    assert overlap_mask.shape == (img_h, img_w)

    # Center pixel (50, 50) is inside both polygons.
    # In overlap mode, it must be overwritten by the smaller inner instance (value = 2)
    assert overlap_mask[50, 50] == 2

    # Outer instance isolated mask (mask == 1)
    outer_recovered = (overlap_mask == 1)
    inner_recovered = (overlap_mask == 2)

    # Verify that outer recovered mask has a HOLE (donut) at the center:
    assert outer_recovered[50, 50] == False, "Outer mask must have lost center pixel to inner instance (donut)"
    assert outer_recovered[20, 20] == True, "Outer mask boundary region should be preserved"
    assert inner_recovered[50, 50] == True, "Inner mask must be preserved at center"

    # 2. INDEPENDENT MODE (overlap_mask = False)
    # polygons2masks returns (N, H, W)
    indep_masks = polygons2masks((img_h, img_w), segments, color=1, downsample_ratio=1)

    assert indep_masks.shape == (2, img_h, img_w)

    # In independent mode, outer mask retains its center pixel!
    assert indep_masks[0, 50, 50] == 1, "Outer mask must retain center pixel in independent mode"
    assert indep_masks[1, 50, 50] == 1, "Inner mask must have center pixel in independent mode"

    # Both instances can simultaneously be 1 at the center pixel:
    assert indep_masks[:, 50, 50].sum() == 2, "Both instances must co-exist at overlapping pixel"

    # 3. Exact pixel loss calculation:
    # Missing pixels in outer recovered mask must exactly equal the overlapping inner mask area
    lost_outer_pixels = indep_masks[0].sum() - outer_recovered.sum()
    assert lost_outer_pixels == indep_masks[1].sum()
