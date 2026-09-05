#!/usr/bin/env python
"""
Dataset and Class Mapping Audit Tool for BEUM.
Audits checkpoints, YAML configs, train/val class distributions, image-label correspondence,
and video timestamp intervals. Outputs analysis/dataset_audit.json.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
import yaml
from ultralytics import YOLO


TIMESTAMP_PATTERN = re.compile(r"_t(\d+\.\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def audit_models(model_paths: list[str | Path]) -> dict:
    model_audit = {}
    for p in model_paths:
        path = Path(p)
        if not path.is_file():
            continue
        try:
            m = YOLO(str(path))
            model_audit[str(path)] = {
                "exists": True,
                "names": m.names,
                "task": getattr(m, "task", "unknown"),
            }
        except Exception as e:
            model_audit[str(path)] = {
                "exists": True,
                "error": str(e),
            }
    return model_audit


def audit_yaml_and_labels(yaml_path: str | Path) -> dict:
    ypath = Path(yaml_path)
    if not ypath.is_file():
        return {"error": f"YAML not found: {ypath}"}

    with open(ypath, "r", encoding="utf-8") as f:
        ydata = yaml.safe_load(f)

    names = ydata.get("names", {})
    if isinstance(names, list):
        names_dict = {i: name for i, name in enumerate(names)}
    else:
        names_dict = {int(k): v for k, v in names.items()}

    split_audit = {}
    root_dir = Path(ydata.get("path", ypath.parent))

    for split in ["train", "val", "test"]:
        split_img_rel = ydata.get(split)
        if not split_img_rel:
            continue

        img_dir = Path(split_img_rel)
        if not img_dir.is_absolute():
            img_dir = root_dir / img_dir

        lbl_dir = img_dir.parent / "labels"
        if not lbl_dir.is_dir():
            # Try parallel structure
            lbl_dir = img_dir.parents[1] / "labels" / split
            if not lbl_dir.is_dir():
                lbl_dir = img_dir

        images = [p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}] if img_dir.is_dir() else []
        labels = [p for p in lbl_dir.glob("*.txt")] if lbl_dir.is_dir() else []

        class_counts = Counter()
        empty_labels = 0
        missing_labels = 0
        corrupt_labels = 0

        for img in images:
            lbl_path = lbl_dir / f"{img.stem}.txt"
            if not lbl_path.is_file():
                missing_labels += 1
                continue

            content = lbl_path.read_text(encoding="utf-8").strip()
            if not content:
                empty_labels += 1
                continue

            for line in content.splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                    class_counts[cid] += 1
                except ValueError:
                    corrupt_labels += 1

        named_counts = {names_dict.get(cid, f"unknown_class_{cid}"): count for cid, count in sorted(class_counts.items())}

        split_audit[split] = {
            "images_dir": str(img_dir),
            "labels_dir": str(lbl_dir),
            "total_images": len(images),
            "total_labels": len(labels),
            "missing_labels": missing_labels,
            "empty_labels": empty_labels,
            "corrupt_labels": corrupt_labels,
            "class_counts_by_id": dict(sorted(class_counts.items())),
            "class_counts_by_name": named_counts,
        }

    return {
        "yaml_path": str(ypath),
        "names": names_dict,
        "splits": split_audit,
    }


def audit_frame_directory(frames_dir: str | Path) -> dict:
    fdir = Path(frames_dir)
    if not fdir.is_dir():
        return {"error": f"Frames directory not found: {fdir}"}

    images = sorted([p for p in fdir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        return {"error": "No images found"}

    timestamps = []
    for img in images:
        m = TIMESTAMP_PATTERN.search(img.name)
        if m:
            timestamps.append(float(m.group(1)))

    has_embedded_timestamps = len(timestamps) == len(images)
    total_frames = len(images)

    if has_embedded_timestamps and total_frames > 1:
        start_t = timestamps[0]
        end_t = timestamps[-1]
        duration_s = end_t - start_t
        intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0.0
        effective_fps = 1.0 / avg_interval if avg_interval > 0 else 0.0
    else:
        start_t = 0.0
        end_t = total_frames
        duration_s = float(total_frames)
        avg_interval = 1.0
        effective_fps = 1.0

    return {
        "frames_dir": str(fdir),
        "total_frames": total_frames,
        "has_embedded_timestamps": has_embedded_timestamps,
        "start_timestamp_s": round(start_t, 2),
        "end_timestamp_s": round(end_t, 2),
        "actual_duration_seconds": round(duration_s, 2),
        "actual_duration_minutes": round(duration_s / 60.0, 3),
        "average_frame_interval_seconds": round(avg_interval, 3),
        "effective_sampling_fps": round(effective_fps, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit datasets, models, and video frame sequences for BEUM.")
    parser.add_argument("--output", type=str, default="analysis/dataset_audit.json", help="Path to save JSON audit")
    args = parser.parse_args()

    # Models to audit
    model_candidates = [
        "models/best-seg-3class.pt",
        "models/best-seg.pt",
        "models/best-seg-3class-backup.pt",
        "runs/segment/finetune_roboflow_30ep-3/weights/best.pt",
    ]
    model_audit = audit_models(model_candidates)

    # YAMLs to audit
    yaml_candidates = [
        "dataset/roboflow_finetune/gully-seg-roboflow.yaml",
        "dataset/blockage/gully-blockage.yaml",
    ]
    yaml_audits = [audit_yaml_and_labels(y) for y in yaml_candidates]

    # Video frames directory
    frames_audit = audit_frame_directory("analysis/new_data_frames")

    audit_result = {
        "model_classes": model_audit,
        "dataset_yamls": yaml_audits,
        "frames_analysis": frames_audit,
        "negative_only_video_verified": False,
        "audit_summary_notes": [
            "Actual duration of new_data_frames is based on filename timestamp regex _t(\\d+\\.\\d+).",
            "Unlabeled operational video metrics must not be termed FP until confirmed negative-only.",
            "Class ID mapping between checkpoints and YAMLs must be verified before retraining.",
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("                    BEUM DATASET & MODEL AUDIT REPORT                    ")
    print("=" * 80)
    print("\n[1. CHECKPOINT MODEL CLASSES]")
    for mpath, info in model_audit.items():
        print(f"  • {mpath}: {info.get('names')}")

    print("\n[2. DATASET YAML & CLASS HISTOGRAMS]")
    for ya in yaml_audits:
        print(f"  • YAML: {ya.get('yaml_path')}")
        print(f"    Declared Names: {ya.get('names')}")
        for split, sinfo in ya.get("splits", {}).items():
            print(f"    [{split}] Images: {sinfo['total_images']}, Labels: {sinfo['total_labels']}, Missing: {sinfo['missing_labels']}")
            print(f"            By ID:   {sinfo['class_counts_by_id']}")
            print(f"            By Name: {sinfo['class_counts_by_name']}")

    print("\n[3. VIDEO FRAMES DURATION AUDIT]")
    print(f"  • Frames: {frames_audit.get('total_frames')}")
    print(f"  • Has Timestamps: {frames_audit.get('has_embedded_timestamps')}")
    print(f"  • Range: {frames_audit.get('start_timestamp_s')}s ~ {frames_audit.get('end_timestamp_s')}s")
    print(f"  • Actual Duration: {frames_audit.get('actual_duration_seconds')}s ({frames_audit.get('actual_duration_minutes')} min)")
    print(f"  • Frame Interval: {frames_audit.get('average_frame_interval_seconds')}s (Effective FPS: {frames_audit.get('effective_sampling_fps')})")
    print("=" * 80)
    print(f"\nAudit saved to {out_path}")


if __name__ == "__main__":
    main()
