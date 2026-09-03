#!/usr/bin/env python
"""Fine-tune YOLOv8 segmentation model on BEUM 3-class dataset (including negative & external samples)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.validate_yolo_seg import validate_split

DATASET_ROOT = ROOT / "dataset"
DATA_YAML = DATASET_ROOT / "gully-seg-3class.yaml"
MODELS_DIR = ROOT / "models"
BASE_MODEL = MODELS_DIR / "best-seg-3class.pt"
if not BASE_MODEL.is_file():
    BASE_MODEL = ROOT / "yolov8n-seg.pt"


def ensure_dataset_ready() -> None:
    print("Validating dataset integrity...")
    for split in ("train", "val"):
        res = validate_split(DATASET_ROOT, split, class_count=3)
        if res["missing"] or res["extra"] or res["invalid"]:
            raise RuntimeError(
                f"Split '{split}' has issues: missing={res['missing']}, extra={res['extra']}, invalid={len(res['invalid'])}"
            )
        print(f"  [{split}] images={res['images']}, labels={res['labels']}, empty(negatives)={res['empty']}, classes={res['classes']}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (default: auto)")
    args = parser.parse_args()

    ensure_dataset_ready()

    device = 0 if torch.cuda.is_available() else "cpu"
    batch_size = args.batch if args.batch else (8 if torch.cuda.is_available() else 4)
    epochs = args.epochs

    print(f"\n==========================================")
    print(f"Starting Fine-Tuning Pipeline")
    print(f"Base model: {BASE_MODEL}")
    print(f"Device: {device} (CUDA: {torch.cuda.is_available()})")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")
    print(f"==========================================\n")

    model = YOLO(str(BASE_MODEL))

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        project=str(MODELS_DIR),
        name="yolo_seg_3class_finetuned",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        patience=10,
        verbose=True,
        workers=2,
        cache=False,
        plots=True,
        save=True,
        save_period=5,
    )

    print("\nTraining completed successfully!")
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    last_pt = Path(results.save_dir) / "weights" / "last.pt"
    target_pt = MODELS_DIR / "best-seg-3class.pt"
    backup_pt = MODELS_DIR / "best-seg-3class-backup.pt"

    chosen_pt = best_pt if best_pt.is_file() else (last_pt if last_pt.is_file() else None)

    if chosen_pt:
        if target_pt.is_file():
            shutil.copy2(target_pt, backup_pt)
            print(f"Previous model backed up to: {backup_pt}")
        shutil.copy2(chosen_pt, target_pt)
        print(f"Updated production model at: {target_pt}")
    else:
        print("Warning: No weights found in output directory.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
