"""ROI definition and inside-polygon filtering."""

from __future__ import annotations

from dataclasses import dataclass
from gully_system.types import Detection


def point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    x, y = point
    inside = False
    count = len(polygon)
    if count < 3:
        return False
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        dot = (x - x1) * (x - x2) + (y - y1) * (y - y2)
        if abs(cross) < 1e-9 and dot <= 1e-9:
            return True
        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
    return inside


@dataclass(frozen=True)
class ROI:
    points: tuple[tuple[float, float], ...]

    def contains_detection(self, detection: Detection) -> bool:
        return point_in_polygon(detection.center, self.points)

    def filter(self, detections: tuple[Detection, ...]) -> tuple[Detection, ...]:
        return tuple(item for item in detections if self.contains_detection(item))

    def scaled(self, scale: float) -> ROI:
        center_x = sum(point[0] for point in self.points) / len(self.points)
        center_y = sum(point[1] for point in self.points) / len(self.points)
        points = tuple(
            (
                center_x + (x - center_x) * scale,
                center_y + (y - center_y) * scale,
            )
            for x, y in self.points
        )
        return ROI(points)

    def is_normalized(self) -> bool:
        return is_normalized(self.points)

    def scale_to_resolution(
        self,
        target_width: int,
        target_height: int,
        base_resolution: tuple[int, int] | None = None,
    ) -> ROI:
        if target_width <= 0 or target_height <= 0:
            return self
        scaled_points = scale_points(
            self.points,
            target_width=target_width,
            target_height=target_height,
            base_resolution=base_resolution,
        )
        return ROI(scaled_points)

    def draw(self, frame: object, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> None:
        try:
            import cv2
            import numpy as np
            points = np.asarray(self.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to draw an ROI") from exc


def is_normalized(points: tuple[tuple[float, float], ...]) -> bool:
    """Check if all polygon coordinates are normalized between 0.0 and 1.0."""
    if not points:
        return False
    return all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)


def scale_points(
    points: tuple[tuple[float, float], ...],
    target_width: int,
    target_height: int,
    base_resolution: tuple[int, int] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Scale polygon points to target width and height."""
    if not points or target_width <= 0 or target_height <= 0:
        return points

    if is_normalized(points):
        return tuple(
            (
                min(float(target_width), max(0.0, x * target_width)),
                min(float(target_height), max(0.0, y * target_height)),
            )
            for x, y in points
        )

    if base_resolution is not None and base_resolution[0] > 0 and base_resolution[1] > 0:
        base_w, base_h = float(base_resolution[0]), float(base_resolution[1])
    else:
        max_x = max(point[0] for point in points)
        max_y = max(point[1] for point in points)
        if max_x <= 640 and max_y <= 480:
            base_w, base_h = 640.0, 480.0
        elif max_x <= 1080 and max_y <= 1920:
            base_w, base_h = 1080.0, 1920.0
        elif max_x <= 1920 and max_y <= 1080:
            base_w, base_h = 1920.0, 1080.0
        else:
            base_w = max(1.0, float(max_x))
            base_h = max(1.0, float(max_y))

    scale_x = float(target_width) / base_w
    scale_y = float(target_height) / base_h

    return tuple(
        (
            min(float(target_width), max(0.0, x * scale_x)),
            min(float(target_height), max(0.0, y * scale_y)),
        )
        for x, y in points
    )
