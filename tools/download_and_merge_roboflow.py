#!/usr/bin/env python
"""Download and merge Roboflow drain/cover segmentation datasets into BEUM dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

EXTERNAL_DIR = Path(r"C:\Project\BEUM\dataset\external")
DATASET_ROOT = Path(r"C:\Project\BEUM\dataset")

# Mapping to BEUM 3-class system:
# 0: object
# 1: drain_area
# 2: drain_full
TARGET_CLASS_ID = 2  # drain_full


def merge_cover_sv2a4(
    src_root: Path = EXTERNAL_DIR / "cover-sv2a4",
    dataset_root: Path = DATASET_ROOT,
) -> dict[str, int]:
    """Merge cover-sv2a4 (class 0: cover -> class 2: drain_full)."""
    counts = {"train": 0, "val": 0}
    split_map = {
        "train": "train",
        "valid": "val",
    }

    for src_split, dest_split in split_map.items():
        src_img_dir = src_root / src_split / "images"
        src_lbl_dir = src_root / src_split / "labels"
        if not src_img_dir.is_dir():
            continue

        dest_img_dir = dataset_root / "images" / dest_split
        dest_lbl_dir = dataset_root / "labels" / dest_split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(src_img_dir.glob("*.*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.is_file():
                continue

            # Read and remap labels (0 -> TARGET_CLASS_ID)
            remapped_lines: list[str] = []
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                # Replace class id with TARGET_CLASS_ID
                parts[0] = str(TARGET_CLASS_ID)
                remapped_lines.append(" ".join(parts))

            # New destination files
            dest_stem = f"rf_cover_{dest_split}_{counts[dest_split]:04d}"
            dest_img = dest_img_dir / f"{dest_stem}{img_path.suffix.lower()}"
            dest_lbl = dest_lbl_dir / f"{dest_stem}.txt"

            shutil.copy2(img_path, dest_img)
            dest_lbl.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")
            counts[dest_split] += 1

    return counts


def merge_curb_grates(
    src_root: Path = EXTERNAL_DIR / "curb-wc7oy",
    dataset_root: Path = DATASET_ROOT,
) -> dict[str, int]:
    """Merge curb-wc7oy (class 1: curb_grates -> class 2: drain_full). Filter out other classes."""
    counts = {"train": 0}
    src_img_dir = src_root / "train" / "images"
    src_lbl_dir = src_root / "train" / "labels"
    if not src_img_dir.is_dir():
        return counts

    dest_img_dir = dataset_root / "images" / "train"
    dest_lbl_dir = dataset_root / "labels" / "train"

    for img_path in sorted(src_img_dir.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            continue

        grate_lines: list[str] = []
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            # curb_grates is class 1 in curb-wc7oy
            if parts[0] == "1":
                parts[0] = str(TARGET_CLASS_ID)
                grate_lines.append(" ".join(parts))

        # Only copy if there's at least one grate annotation
        if grate_lines:
            dest_stem = f"rf_grate_train_{counts['train']:04d}"
            dest_img = dest_img_dir / f"{dest_stem}{img_path.suffix.lower()}"
            dest_lbl = dest_lbl_dir / f"{dest_stem}.txt"

            shutil.copy2(img_path, dest_img)
            dest_lbl.write_text("\n".join(grate_lines) + "\n", encoding="utf-8")
            counts["train"] += 1

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    print("Merging cover-sv2a4 dataset into BEUM...")
    c_stats = merge_cover_sv2a4()
    print(f"cover-sv2a4 merged: train={c_stats['train']}, val={c_stats['val']}")

    print("Merging curb-wc7oy grate samples into BEUM...")
    g_stats = merge_curb_grates()
    print(f"curb-wc7oy merged: train={g_stats['train']}")

    total_added = c_stats['train'] + c_stats['val'] + g_stats['train']
    print(f"Total Roboflow drain/grate samples added: {total_added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
