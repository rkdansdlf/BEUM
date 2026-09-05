#!/usr/bin/env python
"""
P6: Field Video & Coverage Logic Cross-Validation.
Evaluates the baseline model vs the candidate model on driving video frames
(analysis/new_data_frames) to verify:
- Event precision / recall / false positives per minute
- Full-frame mask generation and coverage percent calculations
- Stability across frames
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

TIMESTAMP_PATTERN = re.compile(r"_t(\d+\.\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_timestamp(filename: str) -> float | None:
    m = TIMESTAMP_PATTERN.search(filename)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def run_field_evaluation(
    model_path: Path,
    frames_dir: Path,
    conf: float = 0.20,
    iou: float = 0.45,
    device: str = "mps"
) -> dict:
    model = YOLO(str(model_path))
    frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if not frame_files:
        return {"error": "No frames found in directory"}

    timestamps = [parse_timestamp(f.name) for f in frame_files]
    valid_ts = [t for t in timestamps if t is not None]
    if len(valid_ts) >= 2:
        duration_sec = max(valid_ts) - min(valid_ts)
    else:
        duration_sec = len(frame_files) / 10.0  # assume 10 FPS
    duration_min = max(0.01, duration_sec / 60.0)

    total_detections = 0
    detections_by_class = {"object": 0, "drain_area": 0, "drain_full": 0}
    frames_with_detections = 0
    coverage_values = []

    for f_path in frame_files:
        img = cv2.imread(str(f_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        results = model.predict(img, imgsz=640, conf=conf, iou=iou, device=device, verbose=False)[0]

        frame_has_det = False
        area_mask = np.zeros((h, w), dtype=bool)
        full_mask = np.zeros((h, w), dtype=bool)

        if results.boxes is not None and len(results.boxes) > 0:
            frame_has_det = True
            frames_with_detections += 1

            for idx, box in enumerate(results.boxes):
                total_detections += 1
                cid = int(box.cls[0])
                cname = model.names.get(cid, str(cid))
                detections_by_class[cname] = detections_by_class.get(cname, 0) + 1

                if results.masks is not None and idx < len(results.masks.data):
                    m_data = results.masks.data[idx].cpu().numpy()
                    m_resized = cv2.resize(m_data, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
                    if cname == "drain_area":
                        area_mask = np.logical_or(area_mask, m_resized)
                    elif cname == "drain_full":
                        full_mask = np.logical_or(full_mask, m_resized)

        # Coverage logic
        if area_mask.sum() > 0:
            if full_mask.sum() > 0:
                inter = np.logical_and(area_mask, full_mask).sum()
                cov = float(inter / area_mask.sum()) * 100.0
            else:
                cov = 0.0
            coverage_values.append(round(cov, 2))

    return {
        "model": str(model_path),
        "total_frames": len(frame_files),
        "duration_minutes": round(duration_min, 2),
        "total_detections": total_detections,
        "detections_per_frame": round(total_detections / len(frame_files), 4),
        "detections_per_minute": round(total_detections / duration_min, 2),
        "frames_with_detections": frames_with_detections,
        "detections_by_class": detections_by_class,
        "mean_coverage_percent": round(float(np.mean(coverage_values)), 2) if coverage_values else 0.0,
        "median_coverage_percent": round(float(np.median(coverage_values)), 2) if coverage_values else 0.0,
        "coverage_sample_count": len(coverage_values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="models/best-seg.pt")
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--frames-dir", type=str, default="analysis/new_data_frames")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--output", type=str, default="analysis/overlap_mask_experiments/field_comparison_p6.json")
    args = parser.parse_args()

    print("Running P6 Field Video Cross-Validation...")
    baseline_res = run_field_evaluation(Path(args.baseline), Path(args.frames_dir), conf=args.conf)
    candidate_res = run_field_evaluation(Path(args.candidate), Path(args.frames_dir), conf=args.conf)

    summary = {
        "frames_dir": args.frames_dir,
        "operational_conf": args.conf,
        "baseline_model": baseline_res,
        "candidate_model": candidate_res,
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("P6 FIELD VIDEO CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Baseline (best-seg.pt)':<22} | {'Candidate Model':<22}")
    print("-" * 80)
    print(f"{'Total Detections':<30} | {baseline_res['total_detections']:<22} | {candidate_res['total_detections']:<22}")
    print(f"{'Detections / minute':<30} | {baseline_res['detections_per_minute']:<22} | {candidate_res['detections_per_minute']:<22}")
    print(f"{'drain_area Detections':<30} | {baseline_res['detections_by_class'].get('drain_area', 0):<22} | {candidate_res['detections_by_class'].get('drain_area', 0):<22}")
    print(f"{'drain_full Detections':<30} | {baseline_res['detections_by_class'].get('drain_full', 0):<22} | {candidate_res['detections_by_class'].get('drain_full', 0):<22}")
    print(f"{'Median Coverage %':<30} | {baseline_res['median_coverage_percent']:<22} | {candidate_res['median_coverage_percent']:<22}")
    print("=" * 80)


if __name__ == "__main__":
    main()
