"""Test script with corrected confidence threshold and minimal temporal filtering."""
import cv2, time, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
from pathlib import Path
from dataclasses import replace
from gully_system.config import SystemConfig
from gully_system.detector import YOLODetector
from gully_system.roi import ROI
from gully_system.temporal_filter import TemporalFilter
from gully_system.blockage import BlockageAnalyzer, BlockageEventGate

# Start from the fixed config and override what we need
config = SystemConfig.from_json('config_debug_fixed.json')
config = replace(
    config,
    roi_points=((0,0),(1080,0),(1080,1920),(0,1920)),  # Full frame
    roi_expanded_scale=1.0,
    detector=replace(config.detector, confidence=0.05),  # Lower threshold to catch detections
    temporal_hits=1,  # Require only 1 hit to reduce latency for testing
    temporal_max_missed=0,
)

detector = YOLODetector(config.detector)
roi = ROI(config.roi_points)
temporal_filter = TemporalFilter(
    min_hits=config.temporal_hits,
    max_missed=config.temporal_max_missed,
    iou_threshold=config.temporal_iou,
)
analyzer = BlockageAnalyzer(
    gully_class_names=config.blockage.gully_class_names,
    obstacle_class_names=config.blockage.obstacle_class_names,
    warning_percent=config.blockage.warning_percent,
    critical_percent=config.blockage.critical_percent,
)
event_gate = BlockageEventGate(
    change_percent=config.blockage.event_change_percent,
    cooldown_s=config.blockage.event_cooldown_s,
)

cap = cv2.VideoCapture(config.source)
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)  # Start at frame 60 where drain_area appears

print('Testing with confidence=0.05, temporal_hits=1')
events_found = 0
for frame_idx in range(60, 100):
    ok, frame = cap.read()
    if not ok or frame is None:
        break
    
    raw_dets = detector.predict(frame)
    in_roi = roi.filter(tuple(raw_dets)) if raw_dets else ()
    temporal = temporal_filter.update(list(in_roi))
    blockage = analyzer.analyze(frame, temporal.validated)
    
    if blockage.status != 'no_gully':
        print(f'  Frame {frame_idx}: {blockage.status} {blockage.coverage_percent}% (g={blockage.gully_count}, o={blockage.obstacle_count})')
    
    if event_gate.should_emit(blockage, now=time.monotonic()):
        print(f'  *** EVENT EMITTED at frame {frame_idx}: {blockage.status} {blockage.coverage_percent}% ***')
        events_found += 1
        # Don't break - continue to see if we get multiple events

cap.release()
print(f'Total events found: {events_found}')