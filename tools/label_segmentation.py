from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_CLASS_NAMES = ("drain", "blockage_area")


class PolygonLabeler:
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        window_width: int = 1200,
        class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for the labeler") from exc

        self.cv2 = cv2
        self.image_paths = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
        self.label_dir = label_dir
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.window_width = window_width
        if not class_names:
            raise ValueError("At least one class name is required")
        self.class_names = class_names
        self.class_keys = {str(index): index for index in range(len(class_names))}
        self.index = 0
        self.class_id = 0
        self.polygons: list[tuple[int, list[tuple[int, int]]]] = []
        self.current: list[tuple[int, int]] = []
        self.image = None
        self.display = None
        self.scale = 1.0

    @property
    def label_path(self) -> Path:
        return self.label_dir / f"{self.image_paths[self.index].stem}.txt"

    def _load_existing(self) -> None:
        self.polygons = []
        self.current = []
        path = self.label_path
        if not path.exists():
            return
        height, width = self.image.shape[:2]
        for line in path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) < 7 or (len(values) - 1) % 2:
                continue
            class_id = int(values[0])
            if not 0 <= class_id < len(self.class_names):
                continue
            points = []
            for offset in range(1, len(values), 2):
                points.append((int(float(values[offset]) * width), int(float(values[offset + 1]) * height)))
            self.polygons.append((class_id, points))

    def _render(self) -> None:
        height, width = self.image.shape[:2]
        self.scale = min(1.0, self.window_width / width)
        display_width = max(1, int(width * self.scale))
        display_height = max(1, int(height * self.scale))
        self.display = self.cv2.resize(self.image, (display_width, display_height))

        colors = ((255, 180, 0), (0, 0, 255))
        for class_id, points in self.polygons:
            scaled = [(int(x * self.scale), int(y * self.scale)) for x, y in points]
            color = colors[class_id % len(colors)]
            self.cv2.polylines(self.display, [self._array(scaled)], True, color, 2)
            self.cv2.putText(
                self.display,
                self.class_names[class_id],
                scaled[0],
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                self.cv2.LINE_AA,
            )
        if self.current:
            scaled = [(int(x * self.scale), int(y * self.scale)) for x, y in self.current]
            self.cv2.polylines(self.display, [self._array(scaled)], False, (255, 255, 255), 2)
            for point in scaled:
                self.cv2.circle(self.display, point, 4, (255, 255, 255), -1)

        instruction = f"Frame {self.index + 1}/{len(self.image_paths)} | class={self.class_id}:{self.class_names[self.class_id]}"
        self.cv2.putText(self.display, instruction, (10, 25), self.cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    def _array(self, points):
        import numpy as np

        return np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))

    def _mouse(self, event, x, y, flags, userdata) -> None:
        if event == self.cv2.EVENT_LBUTTONDOWN:
            self.current.append((int(x / self.scale), int(y / self.scale)))
            self._render()

    def _save(self) -> None:
        height, width = self.image.shape[:2]
        rows = []
        for class_id, points in self.polygons:
            normalized = []
            for x, y in points:
                normalized.extend((x / width, y / height))
            rows.append(" ".join([str(class_id)] + [f"{value:.6f}" for value in normalized]))
        self.label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    def _open(self) -> None:
        path = self.image_paths[self.index]
        self.image = self.cv2.imread(str(path))
        if self.image is None:
            raise RuntimeError(f"Could not read image: {path}")
        self._load_existing()
        self._render()

    def run(self) -> None:
        window = "gully segmentation labeler"
        self.cv2.namedWindow(window, self.cv2.WINDOW_NORMAL)
        self.cv2.setMouseCallback(window, self._mouse)
        self._open()
        print("Click polygon points. Enter=finish, W=save, N=save/next, P=save/previous, R=reset, Q=quit")
        print("Number keys select class: " + ", ".join(
            f"{index}={name}" for index, name in enumerate(self.class_names)
        ))

        while True:
            self.cv2.imshow(window, self.display)
            key = self.cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if chr(key) in self.class_keys:
                self.class_id = self.class_keys[chr(key)]
            elif key in (13, 32):
                if len(self.current) >= 3:
                    self.polygons.append((self.class_id, self.current))
                    self.current = []
                    self._render()
            elif key == ord("w"):
                self._save()
            elif key == ord("n"):
                self._save()
                if self.index + 1 >= len(self.image_paths):
                    break
                self.index += 1
                self._open()
            elif key == ord("p"):
                self._save()
                if self.index > 0:
                    self.index -= 1
                    self._open()
            elif key == ord("r"):
                self.polygons = []
                self.current = []
                self._render()
            elif key == ord("q"):
                break
        self.cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive YOLO segmentation polygon labeler")
    parser.add_argument("image_dir")
    parser.add_argument("label_dir")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=list(DEFAULT_CLASS_NAMES),
        help="Class names in numeric key order",
    )
    args = parser.parse_args()
    PolygonLabeler(
        Path(args.image_dir),
        Path(args.label_dir),
        class_names=tuple(args.class_names),
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
