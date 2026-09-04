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

    def draw(self, frame: object, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> None:
        try:
            import cv2
            import numpy as np
            points = np.asarray(self.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to draw an ROI") from exc
