"""Rebuild MP4 from extracted frames using OpenCV VideoWriter."""
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

FRAME_DIR = Path("analysis/new_data_frames")
OUTPUT_PATH = Path("analysis/new_data_result_fixed.mp4")

# Find all frames (format: frame_XXXXX_tXXX.XX.jpg where XXXXX is frame number)
frames = sorted(FRAME_DIR.glob("frame_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
LOGGER.info(f"Found {len(frames)} frames in {FRAME_DIR}")

if not frames:
    LOGGER.error("No frames found!")
    raise SystemExit(1)

# Read first frame to get dimensions
first = cv2.imread(str(frames[0]))
if first is None:
    LOGGER.error(f"Cannot read first frame: {frames[0]}")
    raise SystemExit(1)

height, width = first.shape[:2]
LOGGER.info(f"Frame dimensions: {width}x{height}")

# Write MP4 with OpenCV VideoWriter
# FourCC: mp4v (MPEG-4 Part 2), H264 not always available on Windows
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 2.0  # 2 frames per second

writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))
if not writer.isOpened():
    LOGGER.error("VideoWriter failed to open")
    raise SystemExit(1)

for i, frame_path in enumerate(frames):
    img = cv2.imread(str(frame_path))
    if img is None:
        LOGGER.warning(f"Skipping unreadable frame: {frame_path.name}")
        continue
    writer.write(img)
    if (i + 1) % 50 == 0:
        LOGGER.info(f"Processed {i+1}/{len(frames)} frames")

writer.release()
LOGGER.info(f"MP4 written to: {OUTPUT_PATH}")
LOGGER.info(f"File size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB")

# Verify
cap = cv2.VideoCapture(str(OUTPUT_PATH))
if cap.isOpened():
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_out = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    LOGGER.info(f"Verification: {frame_count} frames, {fps_out:.2f} fps, {w}x{h}")
    cap.release()
else:
    LOGGER.error("Verification failed: cannot open output MP4")
    raise SystemExit(1)