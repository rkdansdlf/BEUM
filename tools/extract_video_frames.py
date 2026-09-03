from __future__ import annotations

import argparse
import csv
from pathlib import Path


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    interval_s: float = 1.0,
    resize_width: int | None = 960,
    jpeg_quality: int = 92,
    max_frames: int = 0,
) -> int:
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    if resize_width is not None and resize_width <= 0:
        raise ValueError("resize_width must be positive")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for frame extraction") from exc

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    next_capture_time = 0.0
    frame_index = 0
    saved = 0
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "frame", "time_s"))
        writer.writeheader()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time_s = frame_index / fps
            if time_s + 1e-9 < next_capture_time:
                frame_index += 1
                continue

            if resize_width is not None and frame.shape[1] != resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (resize_width, max(1, int(round(frame.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            filename = f"frame_{saved:05d}_t{time_s:08.2f}.jpg"
            image_path = output_dir / filename
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not ok:
                raise RuntimeError(f"Could not encode frame {frame_index}")
            image_path.write_bytes(bytes(encoded))
            writer.writerow({"image": filename, "frame": frame_index, "time_s": f"{time_s:.3f}"})
            saved += 1
            next_capture_time += interval_s
            if max_frames > 0 and saved >= max_frames:
                break
            frame_index += 1

    capture.release()
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract representative frames for segmentation labeling")
    parser.add_argument("video")
    parser.add_argument("output_dir")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between saved frames")
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    count = extract_frames(
        args.video,
        args.output_dir,
        interval_s=args.interval,
        resize_width=args.resize_width,
        jpeg_quality=args.quality,
        max_frames=args.max_frames,
    )
    print(f"Extracted {count} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
