"""Synthesize video from pre-extracted frames and detection results."""

import json
import cv2
import numpy as np
from pathlib import Path


def draw_detections_on_frame(frame, detections, blockage_status="clear"):
    """Draw bounding boxes and labels on frame."""
    # Draw each detection
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Color based on class
            if det['class'] == 'drain_full':
                color = (0, 0, 255)  # Red for full
            elif det['class'] == 'drain_area':
                color = (0, 255, 255)  # Yellow for area
            else:
                color = (0, 255, 0)  # Green for others

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Calculate blockage status based on detections
    has_full = any(d['class'] == 'drain_full' for d in detections)
    has_area = any(d['class'] == 'drain_area' for d in detections)

    if has_full:
        status_text = "BLOCKAGE: drain_full (blocked)"
        status_color = (0, 0, 255)
    elif has_area:
        status_text = "Status: drain_area (visible)"
        status_color = (0, 255, 255)
    else:
        status_text = "Status: clear"
        status_color = (0, 255, 0)

    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    return frame


def main():
    frames_dir = Path("analysis/new_data_frames")
    output_path = Path("analysis/new_data_result.mp4")
    detections_path = Path("analysis/new_data_detections.json")

    # Load detections
    with open(detections_path) as f:
        all_detections = json.load(f)

    # Index detections by frame filename
    det_by_frame = {}
    for item in all_detections:
        img_name = item["image"]
        det_by_frame[img_name] = item.get("detections", [])

    # Get sorted frame files
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        print("No frames found!")
        return

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print("Could not read first frame")
        return

    height, width = first_frame.shape[:2]
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print("Could not open video writer")
        return

    print(f"Writing video: {output_path}, frames={len(frame_files)}")

    for i, f in enumerate(frame_files):
        frame = cv2.imread(str(f))
        if frame is None:
            continue

        # Draw detections for this frame
        dets = det_by_frame.get(f.name, [])
        frame = draw_detections_on_frame(frame, dets)

        writer.write(frame)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(frame_files)} frames")

    writer.release()
    print(f"Video saved: {output_path}")
    print(f"Total frames written: {len(frame_files)}")


if __name__ == "__main__":
    main()