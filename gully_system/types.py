"""Domain types for detection results."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """A backend-independent detection result."""

    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    track_id: str | None = None
    mask: object | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def with_track_id(self, track_id: str) -> Detection:
        return replace(self, track_id=track_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(float(item), 2) for item in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }
