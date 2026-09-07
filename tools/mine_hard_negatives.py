#!/usr/bin/env python3
"""
Hard Negative Mining Tool for BEUM
---------------------------------
Extracts confirmed false-positive events from `events_false_positive_archive` in the SQLite database,
and formats them as YOLO background (negative) images with empty label files.
This ensures future model training rounds suppress false positives on road markings, car tires, shadows, etc.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MineHardNegatives")


def mine_hard_negatives(
    db_path: Path | str = "received_data/beum_events.db",
    output_dir: Path | str = "dataset/hard_negatives",
    project_root: Path | str = ".",
) -> int:
    db_file = Path(db_path)
    if not db_file.exists():
        logger.error(f"Database not found: {db_file}")
        return 0

    root = Path(project_root).resolve()
    out_dir = Path(output_dir)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT event_id, image_path, dismiss_reason, dismissed_at 
        FROM events_false_positive_archive 
        WHERE image_path IS NOT NULL AND image_path != ''
        """
    )
    rows = cur.fetchall()
    conn.close()

    logger.info(f"Found {len(rows)} archived false-positive events with images")
    extracted_count = 0

    for r in rows:
        event_id = r["event_id"]
        rel_img_path = r["image_path"]
        candidate_paths = [
            Path(rel_img_path),
            root / rel_img_path,
        ]
        src_img: Path | None = None
        for cand in candidate_paths:
            if cand.exists() and cand.is_file():
                src_img = cand
                break

        if src_img is None:
            logger.warning(f"Image for event {event_id} not found: {rel_img_path}")
            continue

        dest_name = f"fp_{event_id}{src_img.suffix.lower()}"
        dest_img = img_dir / dest_name
        dest_lbl = lbl_dir / f"fp_{event_id}.txt"

        # Copy image
        shutil.copy2(src_img, dest_img)
        # Create empty label file (YOLO standard for negative background sample)
        dest_lbl.write_text("")
        extracted_count += 1
        logger.info(f"Mined negative sample: {dest_name} (reason: {r['dismiss_reason']})")

    logger.info(f"Mining complete. Successfully extracted {extracted_count} background samples to {out_dir}")
    return extracted_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives from false positive archive")
    parser.add_argument("--db-path", type=str, default="received_data/beum_events.db", help="Path to SQLite DB")
    parser.add_argument("--output-dir", type=str, default="dataset/hard_negatives", help="Destination directory")
    args = parser.parse_args()

    count = mine_hard_negatives(db_path=args.db_path, output_dir=args.output_dir)
    print(f"Total hard negative samples generated: {count}")


if __name__ == "__main__":
    main()
