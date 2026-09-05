"""Drain blockage and coverage measurement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from gully_system.types import Detection


@dataclass(frozen=True)
class BlockageMetrics:
    status: str
    coverage_percent: float
    blocked_area_px: int
    gully_area_px: int
    gully_count: int
    obstacle_count: int
    method: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "coverage_percent": round(float(self.coverage_percent), 2),
            "blocked_area_px": int(self.blocked_area_px),
            "gully_area_px": int(self.gully_area_px),
            "gully_count": int(self.gully_count),
            "obstacle_count": int(self.obstacle_count),
            "method": self.method,
            "confidence": round(float(self.confidence), 4),
        }


class BlockageAnalyzer:
    def __init__(
        self,
        gully_class_names: tuple[str, ...] = ("gully",),
        obstacle_class_names: tuple[str, ...] = ("debris", "sediment", "trash", "leaf"),
        warning_percent: float = 20.0,
        critical_percent: float = 50.0,
    ) -> None:
        if critical_percent < warning_percent:
            raise ValueError("critical_percent must be >= warning_percent")
        self.gully_class_names = {name.lower() for name in gully_class_names}
        self.obstacle_class_names = {name.lower() for name in obstacle_class_names}
        self.warning_percent = float(warning_percent)
        self.critical_percent = float(critical_percent)

    @staticmethod
    def _bbox_mask(bbox: tuple[float, float, float, float], height: int, width: int) -> object:
        import numpy as np
        x1, y1, x2, y2 = bbox
        left = max(0, min(width, int(x1)))
        top = max(0, min(height, int(y1)))
        right = max(left, min(width, int(x2 + 0.999)))
        bottom = max(top, min(height, int(y2 + 0.999)))
        mask = np.zeros((height, width), dtype=np.uint8)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 1
        return mask

    def _valid_mask(self, mask: object, height: int, width: int) -> bool:
        if mask is None or not hasattr(mask, "shape"):
            return False
        return mask.shape[:2] == (height, width)

    def _detection_mask(self, detection: Detection, height: int, width: int) -> tuple[object, str]:
        if self._valid_mask(detection.mask, height, width):
            return detection.mask, "segmentation_mask"
        return self._bbox_mask(detection.bbox, height, width), "bbox_estimate"

    def _union(self, masks: list[object], height: int, width: int) -> object:
        import numpy as np
        out = np.zeros((height, width), dtype=np.uint8)
        for m in masks:
            out = np.bitwise_or(out, (m > 0).astype(np.uint8))
        return out

    def analyze(self, frame: Any, detections: Any = None) -> BlockageMetrics:
        import numpy as np

        if isinstance(frame, (list, tuple)) and (detections is None or isinstance(detections, (tuple, list))):
            actual_detections = frame
            shape = detections if detections is not None else (720, 1280)
            height, width = int(shape[0]), int(shape[1])
        elif hasattr(frame, "shape"):
            height, width = int(frame.shape[0]), int(frame.shape[1])
            actual_detections = detections or ()
        else:
            height, width = 720, 1280
            actual_detections = detections or ()

        gullies = [d for d in actual_detections if d.class_name.lower() in self.gully_class_names]
        obstacles = [d for d in actual_detections if d.class_name.lower() in self.obstacle_class_names]

        if not gullies:
            return BlockageMetrics(
                status="no_gully",
                coverage_percent=0.0,
                blocked_area_px=0,
                gully_area_px=0,
                gully_count=0,
                obstacle_count=len(obstacles),
                method="none",
                confidence=0.0,
            )

        gully_masks: list[object] = []
        methods: set[str] = set()
        confidences = [d.confidence for d in gullies]

        for g in gullies:
            m, method = self._detection_mask(g, height, width)
            gully_masks.append(m)
            methods.add(method)

        gully_union = self._union(gully_masks, height, width)
        gully_area_px = int(np.count_nonzero(gully_union))

        if gully_area_px == 0:
            return BlockageMetrics(
                status="no_gully",
                coverage_percent=0.0,
                blocked_area_px=0,
                gully_area_px=0,
                gully_count=len(gullies),
                obstacle_count=len(obstacles),
                method="none",
                confidence=float(np.mean(confidences)) if confidences else 0.0,
            )

        obstacle_masks: list[object] = []
        for o in obstacles:
            m, method = self._detection_mask(o, height, width)
            obstacle_masks.append(m)
            methods.add(method)
            confidences.append(o.confidence)

        if obstacle_masks:
            obstacle_union = self._union(obstacle_masks, height, width)
            blocked_in_gully = np.bitwise_and(gully_union, obstacle_union)
            blocked_area_px = int(np.count_nonzero(blocked_in_gully))
        else:
            blocked_area_px = 0

        coverage_percent = min(100.0, (blocked_area_px / gully_area_px) * 100.0)
        if coverage_percent >= self.critical_percent:
            status = "critical"
        elif coverage_percent >= self.warning_percent:
            status = "warning"
        else:
            status = "clear"

        method_name = "segmentation_mask" if "segmentation_mask" in methods else "bbox_estimate"
        confidence = float(np.mean(confidences)) if confidences else 0.0

        return BlockageMetrics(
            status=status,
            coverage_percent=coverage_percent,
            blocked_area_px=blocked_area_px,
            gully_area_px=gully_area_px,
            gully_count=len(gullies),
            obstacle_count=len(obstacles),
            method=method_name,
            confidence=confidence,
        )

    def draw(self, frame: object, metrics: BlockageMetrics) -> None:
        try:
            import cv2
            text = f"Blockage: {metrics.status.upper()} ({metrics.coverage_percent:.1f}%)"
            colors = {
                "clear": (0, 255, 0),
                "warning": (0, 200, 255),
                "critical": (0, 0, 255),
                "no_gully": (180, 180, 180),
            }
            color = colors.get(metrics.status, (255, 255, 255))
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except Exception:
            pass


class BlockageEventGate:
    def __init__(self, change_percent: float = 10.0, cooldown_s: float = 300.0) -> None:
        self.change_percent = change_percent
        self.cooldown_s = cooldown_s
        self._last_status: str | None = None
        self._last_coverage: float | None = None
        self._last_emit_at: float = float("-inf")

    def should_emit(self, metrics: BlockageMetrics, now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        if metrics.status == "no_gully":
            return False

        previous_status = self._last_status
        previous_coverage = self._last_coverage
        self._last_status = metrics.status
        self._last_coverage = metrics.coverage_percent

        if previous_status is None:
            candidate = metrics.status in ("warning", "critical")
        elif metrics.status != previous_status:
            candidate = (metrics.status in ("warning", "critical")) or (previous_status in ("warning", "critical"))
        else:
            candidate = (
                metrics.status in ("warning", "critical")
                and previous_coverage is not None
                and abs(metrics.coverage_percent - previous_coverage) >= self.change_percent
            )

        if not candidate:
            return False

        escalation = (metrics.status == "critical") and (previous_status != "critical")
        recovery = (metrics.status == "clear") and (previous_status in ("warning", "critical"))

        if (current_time - self._last_emit_at) < self.cooldown_s and not escalation and not recovery:
            return False

        self._last_emit_at = current_time
        return True
