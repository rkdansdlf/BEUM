#!/usr/bin/env python
"""
Field Video Session Evaluation.
Evaluates models against driving video sessions with operational confidence (e.g. conf=0.20 or 0.35)
independent of standard validation AP benchmarking.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from tests.test_all_data_sessions import discover_sessions, process_session, GPS_MATCH_LIMIT_MS, CONFIDENCE

logging.basicConfig(level=logging.INFO)


def evaluate_sessions(
    model_path: str | Path,
    conf: float = 0.20,
    output_json: str | Path = "analysis/field_session_results.json"
):
    model = YOLO(str(model_path))
    sessions = discover_sessions()
    print(f"Loaded model: {model_path} with classes {model.names}")
    print(f"Evaluating {len(sessions)} field sessions at operational conf={conf}...")

    results = {}
    for session in sessions:
        summary = process_session(model, session)
        results[session.name] = {
            "detections": summary["detections"],
            "classes": summary["detection_classes"],
            "max_gps_diff_ms": summary["max_gps_time_diff_ms"],
            "stale_gps": summary["stale_gps_frames"]
        }
        print(f"Session {session.name}: detections={summary['detections']}, classes={summary['detection_classes']}")

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved field session evaluation results to {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best-seg.pt")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--output", type=str, default="analysis/field_session_results.json")
    args = parser.parse_args()

    evaluate_sessions(model_path=args.model, conf=args.conf, output_json=args.output)
