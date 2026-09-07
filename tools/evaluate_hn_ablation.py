#!/usr/bin/env python
"""
Dual Evaluation Benchmark for Hard-Negative Ablation Models.
Evaluates Track A (Canonical Val Segmentation AP) and Track B (Driving 190 Frames FP Suppression).
Checks safety critical rule: Frame 155 true drain detection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import cv2
import numpy as np
from ultralytics import YOLO

from tools.segmentation_contract import (
    canonical_val_args,
    extract_class_metrics,
)

TIMESTAMP_PATTERN = re.compile(r"_t(\d+\.\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_timestamp(filename: str) -> float | None:
    m = TIMESTAMP_PATTERN.search(filename)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def evaluate_video_stream(model: YOLO, frames_dir: Path, conf: float = 0.20, iou: float = 0.45) -> dict:
    frame_files = sorted(list(frames_dir.glob("frame_*.jpg")))
    total_frames = len(frame_files)
    if total_frames == 0:
        return {"error": "No frames found"}

    t_start = parse_timestamp(frame_files[0].name) or 0.0
    t_end = parse_timestamp(frame_files[-1].name) or 0.0
    duration_s = max(1.0, t_end - t_start)
    duration_min = duration_s / 60.0

    total_dets = 0
    df_dets = 0
    da_dets = 0
    detected_frames = 0
    frame_155_result = None

    cluster_events = 0
    in_cluster = False
    cluster_gap = 0

    for fpath in frame_files:
        is_frame_155 = "frame_00155" in fpath.name or "155" in fpath.name
        img = cv2.imread(str(fpath))
        if img is None:
            continue

        res = model.predict(img, imgsz=640, conf=conf, iou=iou, device="cpu", verbose=False)[0]
        n_box = len(res.boxes) if res.boxes is not None else 0

        if n_box > 0:
            detected_frames += 1
            total_dets += n_box
            if not in_cluster:
                cluster_events += 1
                in_cluster = True
            cluster_gap = 0

            for b in res.boxes:
                cid = int(b.cls[0].item())
                cname = model.names.get(cid, str(cid))
                if cname == "drain_full":
                    df_dets += 1
                elif cname == "drain_area":
                    da_dets += 1
        else:
            if in_cluster:
                cluster_gap += 1
                if cluster_gap >= 3:  # 3 consecutive empty frames ends cluster
                    in_cluster = False

        if is_frame_155:
            dets_155 = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    dets_155.append({
                        "class": model.names.get(int(b.cls[0].item())),
                        "conf": round(float(b.conf[0].item()), 4),
                        "box": [round(x, 2) for x in b.xyxy[0].tolist()]
                    })
            frame_155_result = {
                "detected": len(dets_155) > 0,
                "detections": dets_155,
                "has_drain_full": any(d["class"] == "drain_full" for d in dets_155),
                "has_drain_area": any(d["class"] == "drain_area" for d in dets_155),
            }

    return {
        "total_frames": total_frames,
        "duration_minutes": round(duration_min, 2),
        "total_detections": total_dets,
        "drain_area_detections": da_dets,
        "drain_full_detections": df_dets,
        "detected_frames": detected_frames,
        "detections_per_minute": round(total_dets / duration_min, 2),
        "clusters_per_minute": round(cluster_events / duration_min, 2),
        "frame_155_safety_check": frame_155_result or {"detected": False, "detections": []},
    }


def evaluate_model_full(model_path: Path, data_yaml: Path, frames_dir: Path) -> dict:
    print(f"\nEvaluating {model_path.name}...")
    model = YOLO(str(model_path))

    # Track A: Canonical Val
    print(f"  [Track A] Canonical Val on {data_yaml}...")
    val_res = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        device="cpu",
        overlap_mask=False,
        conf=0.001,
        iou=0.70,
        verbose=False,
    )
    box_metrics = extract_class_metrics(val_res.box, model.names)
    mask_metrics = extract_class_metrics(val_res.seg, model.names)

    # Track B: Driving Video
    print(f"  [Track B] Driving Frames on {frames_dir} (conf=0.20, iou=0.45)...")
    video_metrics = evaluate_video_stream(model, frames_dir, conf=0.20, iou=0.45)

    return {
        "model_file": model_path.name,
        "val_metrics": {
            "box": box_metrics,
            "mask": mask_metrics,
        },
        "video_metrics": video_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        "models/candidates/best-seg-2class-omfalse-mr4-e50-s44-0182bde4.pt",
        "models/candidates/best-seg-2class-hn5-omfalse-mr4-e50-s44-0182bde4.pt",
        "models/candidates/best-seg-2class-hn10-omfalse-mr4-e50-s44-0182bde4.pt",
    ])
    parser.add_argument("--val-yaml", type=str, default="dataset/canonical_2class/canonical-2class.yaml")
    parser.add_argument("--frames-dir", type=str, default="analysis/new_data_frames")
    parser.add_argument("--output", type=str, default="analysis/hard_negative_canonical_comparison.json")
    args = parser.parse_args()

    val_yaml = Path(args.val_yaml)
    frames_dir = Path(args.frames_dir)

    results = {}
    for m_str in args.models:
        m_path = Path(m_str)
        if not m_path.exists():
            print(f"Warning: Model not found, skipping: {m_path}")
            continue
        key = "hn0" if "hn" not in m_path.name else m_path.name.split("-")[3]
        results[key] = evaluate_model_full(m_path, val_yaml, frames_dir)

    # Print comparative Markdown table
    print("\n" + "=" * 105)
    print("BEUM HARD NEGATIVE ABLATION DUAL-TRACK COMPARISON (overlap_mask=False)")
    print("=" * 105)

    headers = [
        "모델 (Model)",
        "HN 비율",
        "Mask mAP50",
        "DF AP50",
        "DF Recall",
        "DA AP50",
        "Det/min",
        "FP 감소율",
        "Frame 155 참양성",
        "게이트 판정"
    ]

    base_det_min = results.get("hn0", {}).get("video_metrics", {}).get("detections_per_minute", 4.0)

    rows = []
    for k, data in results.items():
        vm = data["val_metrics"]["mask"]
        vid = data["video_metrics"]
        det_min = vid.get("detections_per_minute", 0.0)
        fp_reduction = f"{(1.0 - det_min / base_det_min) * 100:.1f}%" if base_det_min > 0 else "0%"
        f155 = vid.get("frame_155_safety_check", {})
        f155_pass = f155.get("detected", False) and f155.get("has_drain_full", False)
        f155_str = "통과 (100%)" if f155_pass else "실패 ❌"

        df_ap50 = vm["classes"].get("drain_full", {}).get("ap50", 0.0)
        da_ap50 = vm["classes"].get("drain_area", {}).get("ap50", 0.0)
        df_rec = vm["classes"].get("drain_full", {}).get("recall_at_best_f1", 0.0)

        gate_pass = (df_ap50 >= 0.98) and (da_ap50 >= 0.85) and (det_min <= 2.5) and f155_pass
        gate_str = "합격 ✅" if gate_pass else "미달 ❌"

        rows.append([
            data["model_file"][:25] + "...",
            k.upper(),
            f"{vm['overall']['map50']:.4f}",
            f"{df_ap50:.4f}",
            f"{df_rec:.4f}",
            f"{da_ap50:.4f}",
            f"{det_min:.2f}/min",
            fp_reduction,
            f155_str,
            gate_str
        ])

    col_widths = [28, 10, 12, 10, 10, 10, 12, 12, 16, 12]
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    sep_line = "-+-".join("-" * w for w in col_widths)
    print(header_line)
    print(sep_line)
    for r in rows:
        print(" | ".join(f"{col:<{w}}" for col, w in zip(r, col_widths)))
    print("=" * 105)

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDetailed benchmark saved to: {out_file}")


if __name__ == "__main__":
    main()
