#!/usr/bin/env python
"""Extract negative background samples from Kaggle Road Damage dataset into BEUM dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

DEFAULT_KAGGLE_DIR = Path(
    r"C:\Users\user\.cache\kagglehub\datasets\lorenzoarcioni\road-damage-dataset-potholes-cracks-and-manholes\versions\4\data\images"
)
DEFAULT_DATASET_ROOT = Path(r"C:\Project\BEUM\dataset")


def setup_negative_samples(
    kaggle_dir: Path = DEFAULT_KAGGLE_DIR,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    total_samples: int = 250,
    val_ratio: float = 0.12,
    seed: int = 42,
) -> dict[str, int]:
    if not kaggle_dir.is_dir():
        raise FileNotFoundError(f"Kaggle images directory not found: {kaggle_dir}")

    all_images = sorted(
        [p for p in kaggle_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if len(all_images) < total_samples:
        raise ValueError(f"Found only {len(all_images)} images, requested {total_samples}")

    random.seed(seed)
    selected_images = random.sample(all_images, total_samples)

    val_count = int(total_samples * val_ratio)
    train_count = total_samples - val_count

    splits = {
        "train": selected_images[:train_count],
        "val": selected_images[train_count:],
    }

    results = {"train": 0, "val": 0}

    for split, images in splits.items():
        img_dest_dir = dataset_root / "images" / split
        lbl_dest_dir = dataset_root / "labels" / split
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)

        for i, src_img in enumerate(images, start=1):
            file_stem = f"kaggle_neg_{split}_{i:04d}"
            dest_img = img_dest_dir / f"{file_stem}{src_img.suffix.lower()}"
            dest_lbl = lbl_dest_dir / f"{file_stem}.txt"

            shutil.copy2(src_img, dest_img)
            # Create an empty text file (0 bytes) as negative sample
            dest_lbl.write_text("", encoding="utf-8")
            results[split] += 1

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=250, help="Total negative samples to copy")
    parser.add_argument("--val-ratio", type=float, default=0.12, help="Ratio for validation set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    print(f"Setting up {args.count} negative background samples...")
    stats = setup_negative_samples(
        total_samples=args.count,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Done! Added negative samples: train={stats['train']}, val={stats['val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
