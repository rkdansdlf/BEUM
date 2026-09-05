import os
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Reconcile")


DB_PATH = Path("received_data/beum_events.db")
DATA_DIR = Path("data")
SPOOL_PENDING = Path("data/spool_e2e/pending")
EXCLUDED_DIRS = {"logs", "spool_e2e", "received_data", "FP_mined"}


def reconcile_orphans(auto_enqueue: bool = True) -> list[tuple[Path, Path]]:
    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return []

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT DISTINCT source FROM events WHERE source IS NOT NULL")
        registered_sources = {row[0] for row in cursor}

    orphans: list[tuple[Path, Path]] = []
    SPOOL_PENDING.mkdir(parents=True, exist_ok=True)

    for target_dir in sorted(DATA_DIR.iterdir()):
        if not target_dir.is_dir():
            continue
        if target_dir.name in EXCLUDED_DIRS:
            continue

        media_files = list(target_dir.glob("*.MOV")) + list(target_dir.glob("*.mp4"))
        gps_files = list(target_dir.glob("Location.csv"))
        ann_files = list(target_dir.glob("Annotation.csv"))

        if not media_files or not gps_files:
            continue

        is_registered = any(
            target_dir.name in src or media_files[0].name in src
            for src in registered_sources
        )

        if is_registered:
            ann_size = ann_files[0].stat().st_size if ann_files else 0
            logger.info(
                f"[REGISTERED] {target_dir.name} | video={media_files[0].name} | "
                f"ann_size={ann_size}B | coverage=detection_required"
            )
        else:
            ann_status = ann_files[0].stat().st_size if ann_files else 0
            logger.warning(
                f"[ORPHAN] {target_dir.name} | video={media_files[0].name} | "
                f"ann_size={ann_status}B (UNPROCESSED)"
            )
            orphans.append((target_dir, media_files[0]))

    logger.info(f"Orphan scan complete: {len(orphans)} unprocessed directories found")

    for dir_path, media_path in orphans:
        logger.info(f"Enqueuing orphan: {dir_path.name}")
        if auto_enqueue:
            queue_marker = SPOOL_PENDING / f"{dir_path.name}.job"
            queue_marker.write_text(
                f"target_dir={dir_path.resolve()}\n"
                f"video={media_path.resolve()}\n"
                f"discovered_at={datetime.now().isoformat()}\n"
            )
            logger.info(f"  -> Spool queued: {queue_marker.name}")

    return orphans


def run_full_audit() -> None:
    logger.info("=" * 60)
    logger.info("DrainSight Orphan Audit Report")
    logger.info(f"Generated: {datetime.now().isoformat()}")
    logger.info(f"DB: {DB_PATH}")
    logger.info(f"Data root: {DATA_DIR}")
    logger.info("=" * 60)

    total_dirs = 0
    total_videos = 0
    total_with_gps = 0
    total_with_ann = 0
    total_ann_nonzero = 0

    for target_dir in sorted(DATA_DIR.iterdir()):
        if not target_dir.is_dir() or target_dir.name in EXCLUDED_DIRS:
            continue
        total_dirs += 1

        media_files = list(target_dir.glob("*.MOV")) + list(target_dir.glob("*.mp4"))
        gps_files = list(target_dir.glob("Location.csv"))
        ann_files = list(target_dir.glob("Annotation.csv"))

        total_videos += len(media_files)
        if gps_files:
            total_with_gps += 1
        if ann_files:
            total_with_ann += 1
            if ann_files[0].stat().st_size > 2:
                total_ann_nonzero += 1

    logger.info(f"Total data directories: {total_dirs}")
    logger.info(f"Total video files: {total_videos}")
    logger.info(f"With GPS (Location.csv): {total_with_gps}")
    logger.info(f"With Annotation.csv: {total_with_ann}")
    logger.info(f"Annotation.csv non-empty: {total_ann_nonzero}")

    orphans = reconcile_orphans(auto_enqueue=True)

    logger.info("=" * 60)
    logger.info(f"Summary: {len(orphans)} orphan(s) enqueued to {SPOOL_PENDING}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_full_audit()
