"""Re-evaluate copy_paste_merged model on available sessions."""
import sys
sys.path.insert(0, '.')
from ultralytics import YOLO
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# Model
model_path = 'runs/segment/finetune/copy_paste_merged/weights/best.pt'
model = YOLO(model_path)
LOGGER.info(f"Model classes: {model.names}")

# Discover sessions
from tests.test_all_data_sessions import discover_sessions
sessions = discover_sessions()
LOGGER.info(f"Detected sessions: {[s.name for s in sessions]}")

if not sessions:
    LOGGER.error("No valid data sessions found")
    raise SystemExit(1)

# Process the first session
session = sessions[0]
LOGGER.info(f"Processing session: {session}")

from tests.test_all_data_sessions import process_session
summary = process_session(model, session)

# Print results
det_count = summary['detections']
cls = summary['detection_classes']
det_details = summary['detections_detail']
print(f"\nSession: {summary['session']}")
print(f"  Images: {summary['images']}")
print(f"  GPS readings: {summary['gps_readings']}")
print(f"  Frames processed: {summary['frames_processed']}")
print(f"  Detections: {det_count}")
print(f"  Classes: {cls}")
print(f"  Max GPS diff (ms): {summary['max_gps_time_diff_ms']}")
print(f"  Stale GPS frames: {summary['stale_gps_frames']}")
print(f"  Queue peak: {summary['queue_peak']}")

# Save detailed results
output_path = Path('analysis/copy_paste_merged_reval.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
LOGGER.info(f"Results saved to: {output_path}")