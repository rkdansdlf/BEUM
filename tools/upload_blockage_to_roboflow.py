"""Safely upload reviewed two-class segmentation data to Roboflow.

The command is dry-run by default. It refuses to upload when the target project
has a class schema other than ``drain`` and ``blockage_area``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPECTED_CLASSES = {"drain", "blockage_area"}


def _validate_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    pairs: list[tuple[Path, Path]] = []
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise ValueError(f"Missing label for {image.name}: {label}")
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            values = line.split()
            if not values:
                continue
            if len(values) < 7 or (len(values) - 1) % 2:
                raise ValueError(f"Invalid polygon at {label}:{line_number}")
            try:
                class_id = int(values[0])
                coordinates = [float(value) for value in values[1:]]
            except ValueError as exc:
                raise ValueError(f"Non-numeric label at {label}:{line_number}") from exc
            if class_id not in (0, 1):
                raise ValueError(f"Unexpected class {class_id} at {label}:{line_number}")
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                raise ValueError(f"Coordinate outside 0..1 at {label}:{line_number}")
        pairs.append((image, label))
    return pairs


def upload_reviewed_data(
    image_dir: str | Path,
    label_dir: str | Path,
    project_name: str = "waterdrain",
    split: str = "train",
    batch_name: str = "field_blockage_v1",
    tags: list[str] | None = None,
    perform_upload: bool = False,
) -> int:
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    pairs = _validate_pairs(image_dir, label_dir)

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.environ.get("ROBOFLOW_API")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API is not set in .env or the environment")
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError("Install the Roboflow SDK first") from exc

    project = Roboflow(api_key=api_key).workspace().project(project_name)
    project_classes = set(project.classes)
    if project_classes != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Project {project_name!r} has classes {sorted(project_classes)}; "
            f"expected exactly {sorted(EXPECTED_CLASSES)}. Rename/reset the project "
            "before uploading new labels."
        )

    print(f"Validated {len(pairs)} image/label pairs for project {project_name}")
    if not perform_upload:
        print("Dry run only. Add --upload to send data.")
        return len(pairs)

    for image, label in pairs:
        project.upload(
            str(image),
            annotation_path=str(label),
            split=split,
            batch_name=batch_name,
            tag_names=tags or [],
        )
    print(f"Uploaded {len(pairs)} image/label pairs")
    return len(pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("--project", default="waterdrain")
    parser.add_argument("--split", default="train", choices=("train", "valid", "test"))
    parser.add_argument("--batch", default="field_blockage_v1")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--upload", action="store_true", help="Actually upload; default is dry-run")
    args = parser.parse_args()

    upload_reviewed_data(
        args.image_dir,
        args.label_dir,
        project_name=args.project,
        split=args.split,
        batch_name=args.batch,
        tags=args.tag,
        perform_upload=args.upload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
