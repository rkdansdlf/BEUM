"""Extract evenly spaced field-video frames for manual negative review.

The output is deliberately a review queue, not a training split. Empty labels
are not created until a person confirms that a candidate contains no drain.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def extract_candidates(
    video_path: str | Path,
    output_dir: str | Path,
    count: int = 400,
    resize_width: int | None = 960,
    jpeg_quality: int = 92,
    overwrite: bool = False,
) -> int:
    if count <= 0:
        raise ValueError("count must be positive")
    if resize_width is not None and resize_width <= 0:
        raise ValueError("resize_width must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for frame extraction") from exc

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_images = [
        path for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    manifest_path = output_dir / "manifest.csv"
    if (existing_images or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace its candidates."
        )
    if overwrite:
        for path in existing_images:
            path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        capture.release()
        raise RuntimeError("Could not determine video frame count")

    selected_count = min(count, total_frames)
    if selected_count == 1:
        selected_indices = {0}
    else:
        step = (total_frames - 1) / (selected_count - 1)
        selected_indices = {
            min(total_frames - 1, int(round(index * step)))
            for index in range(selected_count)
        }

    saved = 0
    frame_index = 0
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("image", "source", "frame", "time_s", "status"),
        )
        writer.writeheader()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index not in selected_indices:
                frame_index += 1
                continue

            time_s = frame_index / fps
            if resize_width is not None and frame.shape[1] != resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (resize_width, max(1, int(round(frame.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            filename = f"candidate_{saved:04d}_t{time_s:08.2f}.jpg"
            image_path = output_dir / filename
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not ok:
                capture.release()
                raise RuntimeError(f"Could not encode frame {frame_index}")
            image_path.write_bytes(bytes(encoded))
            writer.writerow(
                {
                    "image": filename,
                    "source": str(video_path),
                    "frame": frame_index,
                    "time_s": f"{time_s:.3f}",
                    "status": "review_required",
                }
            )
            saved += 1
            frame_index += 1

    capture.release()
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    count = extract_candidates(
        args.video,
        args.output_dir,
        count=args.count,
        resize_width=args.resize_width,
        jpeg_quality=args.quality,
        overwrite=args.overwrite,
    )
    print(f"Extracted {count} negative-review candidates")
    print(f"Manual review required before creating empty labels: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
