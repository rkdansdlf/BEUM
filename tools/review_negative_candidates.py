"""Interactively review field-video negative candidates.

Keys:
  N: confirm no drain and copy to the reviewed-negative staging set
  S: not a negative; keep out of the negative staging set
  U: uncertain; keep out of the negative staging set
  P: go to the previous candidate
  Q: save and quit
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
STATUSES = {"confirmed_negative", "needs_label", "uncertain"}


def _load_review(review_csv: Path) -> dict[str, str]:
    if not review_csv.is_file():
        return {}
    with review_csv.open(encoding="utf-8", newline="") as stream:
        return {
            row["image"]: row["status"]
            for row in csv.DictReader(stream)
            if row.get("image") and row.get("status") in STATUSES
        }


def _save_review(review_csv: Path, statuses: dict[str, str]) -> None:
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = review_csv.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "status"))
        writer.writeheader()
        for image in sorted(statuses):
            writer.writerow({"image": image, "status": statuses[image]})
    temporary.replace(review_csv)


def review_candidates(
    image_dir: str | Path,
    output_dir: str | Path,
    audit_csv: str | Path | None = None,
    window_width: int = 1200,
) -> dict[str, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for candidate review") from exc

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    review_csv = image_dir / "review.csv"
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No candidate images found in {image_dir}")
    if window_width <= 0:
        raise ValueError("window_width must be positive")

    audit = {}
    if audit_csv is not None and Path(audit_csv).is_file():
        with Path(audit_csv).open(encoding="utf-8", newline="") as stream:
            audit = {row["image"]: row for row in csv.DictReader(stream)}

    statuses = _load_review(review_csv)
    image_output = output_dir / "images"
    label_output = output_dir / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    window = "negative candidate review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, window_width, window_width)
    index = next(
        (position for position, path in enumerate(image_paths) if path.name not in statuses),
        len(image_paths),
    )
    try:
        while 0 <= index < len(image_paths):
            image_path = image_paths[index]
            image = cv2.imread(str(image_path))
            if image is None:
                statuses[image_path.name] = "uncertain"
                index += 1
                continue

            scale = min(1.0, window_width / image.shape[1])
            display = cv2.resize(
                image,
                (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            audit_row = audit.get(image_path.name, {})
            hit = audit_row.get("detection_count", "0")
            header = (
                f"{index + 1}/{len(image_paths)} {image_path.name} "
                f"legacy_hit={hit} | N negative S needs_label U uncertain P previous Q quit"
            )
            cv2.putText(
                display,
                header,
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                index = max(0, index - 1)
                continue
            if key not in (ord("n"), ord("s"), ord("u")):
                continue

            status = {
                ord("n"): "confirmed_negative",
                ord("s"): "needs_label",
                ord("u"): "uncertain",
            }[key]
            statuses[image_path.name] = status
            if status == "confirmed_negative":
                shutil.copy2(image_path, image_output / image_path.name)
                (label_output / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
            else:
                (image_output / image_path.name).unlink(missing_ok=True)
                (label_output / f"{image_path.stem}.txt").unlink(missing_ok=True)
            _save_review(review_csv, statuses)
            index += 1
    finally:
        cv2.destroyWindow(window)
        _save_review(review_csv, statuses)

    counts = {status: sum(value == status for value in statuses.values()) for status in STATUSES}
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--audit-csv", type=Path)
    parser.add_argument("--window-width", type=int, default=1200)
    args = parser.parse_args()

    counts = review_candidates(
        args.image_dir,
        args.output_dir,
        audit_csv=args.audit_csv,
        window_width=args.window_width,
    )
    print(f"Review state: {counts}")
    print(f"Review log: {args.image_dir / 'review.csv'}")
    print(f"Confirmed-negative staging set: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
