#!/usr/bin/env python3
"""
Tool to dismiss / clear false positive events from the active dashboard feed.
Archives events to SQLite `events_false_positive_archive`, backs up the database,
and exports candidate records to `data/FP_mined/` for Hard Negative active learning.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from migrations import migrate_database

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DismissFeeds")


def backup_database(db_path: Path) -> Path:
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.bak_{ts_str}")
    shutil.copy2(db_path, backup_path)
    logger.info(f"Database backup created: {backup_path}")
    return backup_path


def dismiss_all_events(
    db_path: Path | str = "received_data/beum_events.db",
    reason: str = "confirmed_fp",
    export_fp_mined: bool = True,
) -> dict[str, Any]:
    db_path = Path(db_path)
    if not db_path.exists():
        logger.error(f"Database does not exist: {db_path}")
        return {"success": False, "error": "Database not found"}

    # Ensure migrations are applied (table exists)
    migrate_database(db_path)

    # 1. Backup DB
    backup_file = backup_database(db_path)

    # 2. Extract and transfer
    now_iso = datetime.now(timezone.utc).isoformat()
    archived_records: list[dict[str, Any]] = []

    with sqlite3.connect(db_path, timeout=15.0) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute("SELECT * FROM events")
            rows = cur.fetchall()

            for r in rows:
                rec = dict(r)
                archived_records.append(rec)

                cur.execute(
                    """
                    INSERT OR REPLACE INTO events_false_positive_archive (
                        event_id, event_type, source, created_at, received_at,
                        blockage_status, coverage_percent, occlusion_pct,
                        roi_profile, lat, lon, has_image, image_path,
                        raw_payload, dismissed_at, dismiss_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rec["event_id"],
                        rec.get("event_type"),
                        rec.get("source"),
                        rec.get("created_at"),
                        rec.get("received_at"),
                        rec.get("blockage_status"),
                        rec.get("coverage_percent"),
                        rec.get("occlusion_pct"),
                        rec.get("roi_profile"),
                        rec.get("lat"),
                        rec.get("lon"),
                        rec.get("has_image"),
                        rec.get("image_path"),
                        rec.get("raw_payload"),
                        now_iso,
                        reason,
                    ),
                )

            # Clear active events
            cur.execute("DELETE FROM events")
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to dismiss events: {exc}")
            raise

    logger.info(f"Successfully archived {len(archived_records)} events to events_false_positive_archive.")

    # 3. Export to data/FP_mined for Hard Negative training
    fp_export_path = None
    if export_fp_mined and archived_records:
        fp_dir = PROJECT_ROOT / "data" / "FP_mined"
        fp_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fp_export_path = fp_dir / f"dashboard_confirmed_fp_{ts_str}.json"

        export_data = []
        for rec in archived_records:
            payload = {}
            if rec.get("raw_payload"):
                try:
                    payload = json.loads(rec["raw_payload"])
                except Exception:
                    pass
            export_data.append({
                "event_id": rec["event_id"],
                "source": rec.get("source"),
                "coverage_percent": rec.get("coverage_percent"),
                "occlusion_pct": rec.get("occlusion_pct"),
                "blockage_status": rec.get("blockage_status"),
                "has_image": bool(rec.get("has_image")),
                "image_path": rec.get("image_path"),
                "lat": rec.get("lat"),
                "lon": rec.get("lon"),
                "received_at": rec.get("received_at"),
                "review_status": "confirmed_fp",
                "notes": f"Dismissed from dashboard feed: {reason}",
                "raw_payload": payload,
            })

        fp_export_path.write_text(json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Exported {len(export_data)} false positives manifest to: {fp_export_path}")

    return {
        "success": True,
        "archived_count": len(archived_records),
        "backup_path": str(backup_file),
        "fp_export_path": str(fp_export_path) if fp_export_path else None,
        "dismissed_at": now_iso,
    }


def main():
    parser = argparse.ArgumentParser(description="Dismiss false positive events from dashboard feed")
    parser.add_argument("--db", default="received_data/beum_events.db", help="Path to beum_events.db")
    parser.add_argument("--reason", default="confirmed_fp", help="Dismissal reason label")
    parser.add_argument("--no-export", action="store_true", help="Skip exporting FP_mined JSON manifest")
    args = parser.parse_args()

    res = dismiss_all_events(
        db_path=Path(args.db),
        reason=args.reason,
        export_fp_mined=not args.no_export,
    )
    print("==================================================")
    print(f"[*] Dismiss Result : {'SUCCESS' if res['success'] else 'FAILED'}")
    print(f"[*] Archived Events: {res['archived_count']}")
    print(f"[*] Backup Created : {res['backup_path']}")
    if res.get("fp_export_path"):
        print(f"[*] FP Mined Export: {res['fp_export_path']}")
    print("==================================================")


if __name__ == "__main__":
    main()
