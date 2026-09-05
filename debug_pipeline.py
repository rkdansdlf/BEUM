"""Debug script to test the full pipeline with correct config."""
import cv2
import time
import logging
from pathlib import Path
from dataclasses import replace

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
from gully_system.config import SystemConfig
from gully_system.detector import YOLODetector
from gully_system.roi import ROI
from gully_system.temporal_filter import TemporalFilter
from gully_system.blockage import BlockageAnalyzer, BlockageEventGate
from gully_system.gps import ReplayGPSProvider
from gully_system.sensors import SystemSensorProvider

config = SystemConfig.from_json('config_debug_fixed.json')

# Use WIDE ROI that covers most of frame for 1080x1920
wide_roi = ROI(((0, 0), (1080, 0), (1080, 1920), (0, 1920)))
config = replace(config, roi_points=wide_roi.points, roi_expanded_scale=1.0)

print('=== Modified Config ===')
print(f'roi_points: {config.roi_points}')
print(f'confidence: {config.detector.confidence}')
print(f'gully_classes: {config.blockage.gully_class_names}')
print(f'obstacle_classes: {config.blockage.obstacle_class_names}')

detector = YOLODetector(config.detector)
roi = ROI(config.roi_points)
expanded_roi_obj = roi.scaled(config.roi_expanded_scale)
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

gps_provider = ReplayGPSProvider(
    csv_path=config.gps.csv_path,
    sample_period_s=1.0,
    replay_speed=1000.0,
)
gps_provider.start()

cap = cv2.VideoCapture(config.source)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

events_found = 0
for frame_idx in range(100):
    ok, frame = cap.read()
    if not ok or frame is None:
        print(f'Frame {frame_idx}: cap.read() returned False')
        break
    
    now = time.monotonic()
    
    raw_detections = detector.predict(frame)
    if raw_detections:
        print(f'  Frame {frame_idx}: raw_detections={[d.class_name for d in raw_detections]} conf={[d.confidence for d in raw_detections]}')
    
    in_roi = expanded_roi_obj.filter(tuple(raw_detections)) if raw_detections else ()
    if in_roi:
        print(f'  Frame {frame_idx}: in_roi={[d.class_name for d in in_roi]}')
    
    temporal = temporal_filter.update(list(in_roi))
    if temporal.validated:
        print(f'  Frame {frame_idx}: temporal.validated={[d.class_name for d in temporal.validated]}')
    
    blockage = analyzer.analyze(frame, temporal.validated)
    if blockage.status != 'no_gully':
        print(f'  Frame {frame_idx}: blockage={blockage.status} coverage={blockage.coverage_percent}%')
    
    if event_gate.should_emit(blockage, now=now):
        print(f'  Frame {frame_idx}: EVENT EMITTED! blockage={blockage.status} coverage={blockage.coverage_percent}%')
        events_found += 1
        break

cap.release()
gps_provider.stop()
print(f'Events found in 100 frames: {events_found}')