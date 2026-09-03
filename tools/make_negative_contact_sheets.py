"""Create contact sheets for manual review of negative candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def make_contact_sheets(
    image_dir: str | Path,
    audit_csv: str | Path,
    output_dir: str | Path,
    per_sheet: int = 25,
    thumb_width: int = 180,
) -> int:
    if per_sheet <= 0 or thumb_width <= 0:
        raise ValueError("per_sheet and thumb_width must be positive")

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required") from exc

    image_dir = Path(image_dir)
    audit_csv = Path(audit_csv)
    output_dir = Path(output_dir)
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    if not audit_csv.is_file():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv}")

    audit = {}
    with audit_csv.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            audit[row["image"]] = row

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_sheet in output_dir.glob("sheet_*.jpg"):
        old_sheet.unlink()

    cell_height = int(round(thumb_width * 1707 / 960)) + 34
    columns = 5
    rows_per_sheet = max(1, (per_sheet + columns - 1) // columns)
    sheet_count = 0
    for start in range(0, len(image_paths), per_sheet):
        batch = image_paths[start:start + per_sheet]
        sheet = np.full(
            (rows_per_sheet * cell_height, columns * thumb_width, 3),
            32,
            dtype=np.uint8,
        )
        for position, image_path in enumerate(batch):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            scale = thumb_width / image.shape[1]
            thumb = cv2.resize(
                image,
                (thumb_width, max(1, int(round(image.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            row = position // columns
            column = position % columns
            top = row * cell_height
            left = column * thumb_width
            sheet[top:top + thumb.shape[0], left:left + thumb.shape[1]] = thumb

            audit_row = audit.get(image_path.name, {})
            hit = int(audit_row.get("detection_count", "0") or 0) > 0
            color = (0, 0, 220) if hit else (100, 100, 100)
            cv2.rectangle(
                sheet,
                (left, top),
                (left + thumb_width - 1, top + thumb.shape[0] - 1),
                color,
                3,
            )
            label = f"{image_path.stem.split('_t')[0]} {'HIT' if hit else 'NOHIT'}"
            cv2.rectangle(
                sheet,
                (left, top + thumb.shape[0]),
                (left + thumb_width, top + cell_height),
                (32, 32, 32),
                -1,
            )
            cv2.putText(
                sheet,
                label,
                (left + 3, top + thumb.shape[0] + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

        output_path = output_dir / f"sheet_{sheet_count:03d}.jpg"
        if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError(f"Could not write contact sheet: {output_path}")
        sheet_count += 1

    return sheet_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("audit_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-sheet", type=int, default=25)
    parser.add_argument("--thumb-width", type=int, default=180)
    args = parser.parse_args()

    count = make_contact_sheets(
        args.image_dir,
        args.audit_csv,
        args.output_dir,
        per_sheet=args.per_sheet,
        thumb_width=args.thumb_width,
    )
    print(f"Created {count} contact sheets in {args.output_dir}")
    print("Red border means legacy-model hit; all images still require human review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
