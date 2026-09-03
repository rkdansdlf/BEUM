"""Train the controlled two-class drain blockage segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from tools.validate_yolo_seg import validate_split


CLASS_COUNT = 2
REQUIRED_CLASSES = {0, 1}
DEFAULT_DATA = Path("dataset/gully-blockage.yaml")
DEFAULT_MODEL = Path("yolov8n-seg.pt")


def ensure_dataset_ready(dataset_root: Path) -> None:
    summaries = {}
    for split in ("train", "val"):
        result = validate_split(dataset_root, split, class_count=CLASS_COUNT)
        summaries[split] = result
        if result["missing"] or result["extra"] or result["invalid"]:
            raise RuntimeError(
                f"Invalid {split} split: missing={result['missing']} "
                f"extra={result['extra']} invalid={len(result['invalid'])}"
            )
        if not result["images"]:
            raise RuntimeError(f"{split} split has no images")

    train_classes = set(summaries["train"]["classes"])
    if not REQUIRED_CLASSES.issubset(train_classes):
        raise RuntimeError(
            "Training split must contain both classes 0 (drain) and 1 (blockage_area); "
            f"found {sorted(train_classes)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="blockage_2class")
    parser.add_argument("--project", type=Path, default=Path("runs/segment"))
    args = parser.parse_args()

    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("epochs, imgsz, and batch must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(f"Segmentation base model not found: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    dataset_root = Path("dataset/blockage")
    ensure_dataset_ready(dataset_root)
    model = YOLO(str(args.model))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        patience=20,
        workers=0,
        cache=False,
        plots=True,
        save=True,
        seed=42,
    )
    print(f"Training complete: {results.save_dir}")
    print(f"Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
