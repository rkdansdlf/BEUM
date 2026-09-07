#!/usr/bin/env python3
"""
Human-in-the-Loop Review Tool for Active Learning Candidates.
Reviews harvested frames from data/active_learning/triage/, categorizes them
into approved positives (with YOLO labels), approved hard negatives (empty labels),
or discarded samples.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_directories(base_dir: Path) -> dict[str, Path]:
    triage_dir = base_dir / "triage"
    pos_img = base_dir / "approved_positives" / "images"
    pos_lbl = base_dir / "approved_positives" / "labels"
    neg_img = base_dir / "approved_negatives" / "images"
    neg_lbl = base_dir / "approved_negatives" / "labels"
    discard_dir = base_dir / "discarded"

    for d in [triage_dir, pos_img, pos_lbl, neg_img, neg_lbl, discard_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "triage": triage_dir,
        "pos_img": pos_img,
        "pos_lbl": pos_lbl,
        "neg_img": neg_img,
        "neg_lbl": neg_lbl,
        "discard": discard_dir,
    }


def detection_to_yolo_line(det: dict[str, Any], class_mapping: dict[str, int]) -> str | None:
    cname = det.get("class_name")
    if cname not in class_mapping:
        return None
    cid = class_mapping[cname]

    # Prefer polygon mask if available
    poly = det.get("polygon")
    if poly and len(poly) >= 6:
        # Normalized coordinates x1 y1 x2 y2 ...
        poly_str = " ".join(f"{coord:.5f}" for coord in poly)
        return f"{cid} {poly_str}"

    # Otherwise fallback to normalized bbox: class xc yc w h
    bbox = det.get("bbox")
    if bbox and len(bbox) == 4:
        # Assuming normalized [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        w = max(0.001, x2 - x1)
        h = max(0.001, y2 - y1)
        return f"{cid} {xc:.5f} {yc:.5f} {w:.5f} {h:.5f}"

    return None


def process_candidate(
    cand_json_path: Path,
    dirs: dict[str, Path],
    decision: str,
    class_mapping: dict[str, int],
) -> dict[str, Any]:
    with open(cand_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cand_id = data["candidate_id"]
    img_name = data.get("image_filename", f"{cand_id}.jpg")
    src_img = cand_json_path.parent / img_name

    if decision == "positive":
        # Positive: Copy image and write YOLO labels
        if src_img.exists():
            shutil.copy2(src_img, dirs["pos_img"] / img_name)
        lines = []
        for det in data.get("detections", []):
            line = detection_to_yolo_line(det, class_mapping)
            if line:
                lines.append(line)
        lbl_file = dirs["pos_lbl"] / f"{cand_id}.txt"
        lbl_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        # Archive JSON
        shutil.move(str(cand_json_path), str(dirs["pos_img"].parent / f"{cand_id}.json"))
        if src_img.exists():
            src_img.unlink()
        return {"id": cand_id, "status": "approved_positive", "detections": len(lines)}

    elif decision == "negative":
        # Hard Negative: Copy image and write empty 0-byte label
        if src_img.exists():
            shutil.copy2(src_img, dirs["neg_img"] / img_name)
        lbl_file = dirs["neg_lbl"] / f"{cand_id}.txt"
        lbl_file.write_text("", encoding="utf-8")

        # Archive JSON
        shutil.move(str(cand_json_path), str(dirs["neg_img"].parent / f"{cand_id}.json"))
        if src_img.exists():
            src_img.unlink()
        return {"id": cand_id, "status": "approved_negative"}

    else:
        # Discard
        shutil.move(str(cand_json_path), str(dirs["discard"] / f"{cand_id}.json"))
        if src_img.exists():
            shutil.move(str(src_img), str(dirs["discard"] / img_name))
        return {"id": cand_id, "status": "discarded"}


def main():
    parser = argparse.ArgumentParser(description="BEUM Active Learning Review Tool")
    parser.add_argument("--base-dir", type=str, default="data/active_learning")
    parser.add_argument("--auto-decision", type=str, choices=["positive", "negative", "discard"],
                        help="Automated decision for non-interactive batch/testing mode")
    parser.add_argument("--limit", type=int, default=0, help="Max candidates to review")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    dirs = setup_directories(base_dir)

    class_mapping = {
        "drain_area": 0,
        "drain_full": 1,
    }

    candidates = sorted(list(dirs["triage"].glob("al_*.json")))
    if not candidates:
        print(f"No pending active learning candidates found in {dirs['triage']}.")
        return

    print("=" * 80)
    print(f"BEUM ACTIVE LEARNING REVIEW TOOL: {len(candidates)} pending candidates")
    print(f"Base Directory: {base_dir}")
    print("=" * 80)

    reviewed_count = 0
    stats = {"positive": 0, "negative": 0, "discard": 0}

    for idx, cpath in enumerate(candidates):
        if args.limit > 0 and reviewed_count >= args.limit:
            break

        with open(cpath, "r", encoding="utf-8") as f:
            cand = json.load(f)

        cid = cand.get("candidate_id")
        reasons = cand.get("reasons", [])
        cov = cand.get("coverage_percent", 0.0)
        dets = cand.get("detections", [])
        gps = cand.get("gps")

        if args.auto_decision:
            decision = args.auto_decision
        else:
            print(f"\n[{idx + 1}/{len(candidates)}] Candidate: {cid}")
            print(f"  -> Trigger Reasons : {', '.join(reasons)}")
            print(f"  -> Coverage %       : {cov:.1f}% ({cand.get('status')})")
            print(f"  -> Detections Count : {len(dets)}")
            for d in dets:
                print(f"     * {d.get('class_name')} (conf={d.get('confidence'):.2f})")
            if gps and gps.get("latitude"):
                print(f"  -> GPS Location    : ({gps.get('latitude'):.5f}, {gps.get('longitude'):.5f})")

            prompt = "Action [ (p)ositive / (n)egative / (d)iscard / (q)uit ]: "
            try:
                choice = input(prompt).strip().lower()
            except EOFError:
                choice = "q"

            if choice in ("p", "pos", "positive"):
                decision = "positive"
            elif choice in ("n", "neg", "negative"):
                decision = "negative"
            elif choice in ("d", "disc", "discard"):
                decision = "discard"
            elif choice in ("q", "quit", "exit"):
                print("Exiting review tool.")
                break
            else:
                print("Unknown command, defaulting to discard.")
                decision = "discard"

        res = process_candidate(cpath, dirs, decision, class_mapping)
        stats[decision] += 1
        reviewed_count += 1
        print(f"  ✅ Processed {cid} -> {res['status']}")

    print("\n" + "=" * 80)
    print(f"Review session completed! Reviewed: {reviewed_count}")
    print(f"  - Approved Positives : {stats['positive']}")
    print(f"  - Approved Negatives : {stats['negative']}")
    print(f"  - Discarded          : {stats['discard']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
