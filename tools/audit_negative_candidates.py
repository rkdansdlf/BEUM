"""Use the legacy model only to prioritize manual negative review.

This tool writes an audit CSV. It never creates labels and never promotes an
image into a training split because model predictions are not ground truth.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def audit_candidates(
    model_path: str | Path,
    image_dir: str | Path,
    output_csv: str | Path,
    confidence: float = 0.10,
) -> dict[str, int]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for candidate auditing") from exc

    model_path = Path(model_path)
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No candidate images found in {image_dir}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    counts: dict[str, int] = {}
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=640,
        conf=confidence,
        device="cpu",
        stream=True,
        verbose=False,
    )

    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "image",
                "detection_count",
                "classes",
                "max_confidence",
                "review_status",
            ),
        )
        writer.writeheader()
        for result in results:
            detections = []
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = str(model.names[class_id])
                confidence_value = float(box.conf[0].item())
                detections.append((class_name, confidence_value))
                counts[class_name] = counts.get(class_name, 0) + 1
            writer.writerow(
                {
                    "image": Path(result.path).name,
                    "detection_count": len(detections),
                    "classes": ";".join(name for name, _ in detections),
                    "max_confidence": (
                        f"{max(confidence_value for _, confidence_value in detections):.4f}"
                        if detections else ""
                    ),
                    "review_status": "review_required",
                }
            )

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/best-seg-3class.pt"))
    parser.add_argument("--confidence", type=float, default=0.10)
    args = parser.parse_args()

    counts = audit_candidates(
        args.model,
        args.image_dir,
        args.output_csv,
        confidence=args.confidence,
    )
    print(f"Audit complete: {args.output_csv}")
    print(f"Model hits by class (manual confirmation still required): {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
