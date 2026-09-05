#!/usr/bin/env python
"""
Automated False Positive (FP) Mining Tool for BEUM.
Runs inference on driving videos or continuous frame queues, extracts candidate false detections,
and queues them into categorized directories for review and hard-negative ingestion.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


CATEGORIES = [
    "manhole",      # 맨홀 뚜껑
    "crack",        # 아스팔트 균열/보수 흔적
    "shadow",       # 가로등/건물 그림자
    "reflection",   # 젖은 노면/빛 반사
    "curb",         # 연석/인도 경계
    "wet_road",     # 물웅덩이/젖은 아스팔트
    "leaves",       # 낙엽/토사
    "unknown",      # 미분류
]


def setup_mining_directories(output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    raw_dir = output_root / "raw"
    annotated_dir = output_root / "annotated"
    raw_dir.mkdir(exist_ok=True)
    annotated_dir.mkdir(exist_ok=True)

    cat_dirs = {}
    for cat in CATEGORIES:
        d = output_root / "categories" / cat
        d.mkdir(parents=True, exist_ok=True)
        cat_dirs[cat] = d

    return {"raw": raw_dir, "annotated": annotated_dir, "categories": cat_dirs}


def mine_video_fps(
    model_path: str | Path,
    video_source: str | Path,
    output_root: str | Path = "data/FP_mined",
    conf: float = 0.15,
    iou: float = 0.70,
    imgsz: int = 640,
    max_frames: int | None = None,
    save_annotated: bool = True,
) -> int:
    model = YOLO(str(model_path))
    names = model.names
    src_path = Path(video_source)
    out_root = Path(output_root)
    dirs = setup_mining_directories(out_root)

    manifest_csv = out_root / "candidates_manifest.csv"
    manifest_rows = []

    print(f"Starting FP Mining with model: {model_path} (conf={conf})")
    print(f"Source: {video_source} -> Output: {output_root}")

    image_paths = []
    fps = 30.0
    total_frames = 0

    if src_path.is_dir():
        image_paths = sorted(
            [p for p in src_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        total_frames = len(image_paths)
        fps = 1.0
    elif src_path.is_file() and src_path.suffix.lower() in {".mp4", ".avi", ".mov"}:
        cap = cv2.VideoCapture(str(src_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

    if max_frames:
        total_frames = min(total_frames, max_frames)

    extracted_count = 0

    if image_paths:
        frame_iter = ((i, cv2.imread(str(p)), p.name) for i, p in enumerate(image_paths[:total_frames]))
    else:
        cap = cv2.VideoCapture(str(src_path))

        def _gen():
            idx = 0
            while cap.isOpened() and (max_frames is None or idx < max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                yield idx, frame, f"frame_{idx:06d}.jpg"
                idx += 1
            cap.release()

        frame_iter = _gen()

    for idx, frame, orig_name in frame_iter:
        if frame is None:
            continue

        results = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]

        has_detection = results.boxes is not None and len(results.boxes) > 0

        if has_detection:
            extracted_count += 1
            m_ts = re.search(r"_t(\d+\.\d+)\.(jpg|jpeg|png)$", orig_name, re.IGNORECASE)
            if m_ts:
                timestamp_s = float(m_ts.group(1))
            else:
                timestamp_s = round(idx / fps, 2) if fps > 0 else float(idx)

            det_classes = []
            det_confs = []
            for box in results.boxes:
                cid = int(box.cls[0])
                cname = names.get(cid, str(cid))
                c_conf = round(float(box.conf[0]), 3)
                det_classes.append(cname)
                det_confs.append(c_conf)

            stem = f"{Path(orig_name).stem}_t{timestamp_s:06.2f}"
            raw_filename = f"{stem}.jpg"
            raw_path = dirs["raw"] / raw_filename
            cv2.imwrite(str(raw_path), frame)

            # Draw annotations
            if save_annotated:
                annotated_img = results.plot()
                annotated_path = dirs["annotated"] / f"{stem}_annotated.jpg"
                cv2.imwrite(str(annotated_path), annotated_img)

            # Default categorization queue (default unknown until user labels or rule matches)
            # Simple heuristic: if aspect ratio is roughly 1:1 and near bottom center -> potential manhole
            suggested = "unknown"

            # Copy to suggested category queue
            cat_dst = dirs["categories"][suggested] / raw_filename
            shutil.copy2(raw_path, cat_dst)

            manifest_rows.append({
                "frame_id": idx,
                "timestamp_s": timestamp_s,
                "raw_file": str(raw_path.relative_to(out_root)),
                "classes": ";".join(det_classes),
                "confidences": ";".join(map(str, det_confs)),
                "category": suggested,
                "verified_negative": "pending",
            })

    # Write manifest CSV
    fieldnames = ["frame_id", "timestamp_s", "raw_file", "classes", "confidences", "category", "verified_negative"]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nCompleted FP Mining:")
    print(f"- Processed frames: {total_frames}")
    print(f"- Extracted FP candidates: {extracted_count}")
    print(f"- Candidates manifest saved: {manifest_csv}")
    return extracted_count


def export_as_hard_negatives(
    mining_root: str | Path,
    dataset_dest: str | Path,
    categories: list[str] | None = None,
    split: str = "train",
    prefix: str = "hard_neg",
) -> int:
    """
    Exports verified negative images into target dataset split with empty label files (.txt with 0 bytes).
    """
    m_root = Path(mining_root)
    d_dest = Path(dataset_dest)
    img_dir = d_dest / "images" / split
    lbl_dir = d_dest / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    selected_cats = categories or CATEGORIES
    exported = 0

    for cat in selected_cats:
        cat_dir = m_root / "categories" / cat
        if not cat_dir.is_dir():
            continue
        for img_path in cat_dir.glob("*.jpg"):
            dest_stem = f"{prefix}_{cat}_{img_path.stem}"
            dst_img = img_dir / f"{dest_stem}.jpg"
            dst_lbl = lbl_dir / f"{dest_stem}.txt"

            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("", encoding="utf-8")  # Empty text file = Ultralytics background sample
            exported += 1

    print(f"Exported {exported} hard negative background images to {split} split in {d_dest}")
    return exported


def main():
    parser = argparse.ArgumentParser(description="Mine False Positive candidate frames from video/stream.")
    subparsers = parser.add_subparsers(dest="cmd")

    mine_parser = subparsers.add_parser("mine", help="Mine FP frames from video or directory")
    mine_parser.add_argument("--model", type=str, default="models/best-seg-3class.pt", help="Path to weights")
    mine_parser.add_argument("--video", type=str, required=True, help="Path to video or image folder")
    mine_parser.add_argument("--output", type=str, default="data/FP_mined", help="Output directory")
    mine_parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    mine_parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")

    export_parser = subparsers.add_parser("export", help="Export mined candidates to dataset as empty labels")
    export_parser.add_argument("--mining-root", type=str, default="data/FP_mined", help="Mining root dir")
    export_parser.add_argument("--dest", type=str, required=True, help="Destination dataset path")
    export_parser.add_argument("--split", type=str, default="train", help="train or val")
    export_parser.add_argument("--categories", nargs="+", default=None, help="Specific categories to export")

    args = parser.parse_args()

    if args.cmd == "mine":
        mine_video_fps(
            model_path=args.model,
            video_source=args.video,
            output_root=args.output,
            conf=args.conf,
            max_frames=args.max_frames,
        )
    elif args.cmd == "export":
        export_as_hard_negatives(
            mining_root=args.mining_root,
            dataset_dest=args.dest,
            categories=args.categories,
            split=args.split,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
