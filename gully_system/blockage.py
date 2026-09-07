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
        gully_class_names: tuple[str, ...] = ("drain_area", "gully"),
        obstacle_class_names: tuple[str, ...] = ("drain_full", "debris", "sediment", "trash", "leaf"),
        warning_percent: float = 20.0,
        critical_percent: float = 50.0,
        require_gully_presence: bool = True,
        standalone_min_conf: float = 0.85,
    ) -> None:
        if critical_percent < warning_percent:
            raise ValueError("critical_percent must be >= warning_percent")
        self.gully_class_names = {name.lower() for name in gully_class_names}
        self.obstacle_class_names = {name.lower() for name in obstacle_class_names}
        self.warning_percent = float(warning_percent)
        self.critical_percent = float(critical_percent)
        self.require_gully_presence = bool(require_gully_presence)
        self.standalone_min_conf = float(standalone_min_conf)

    @staticmethod
    def _box_intersects(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float], margin_ratio: float = 0.20) -> bool:
        x1_1, y1_1, x2_1, y2_1 = b1
        x1_2, y1_2, x2_2, y2_2 = b2
        w1 = max(1.0, x2_1 - x1_1)
        h1 = max(1.0, y2_1 - y1_1)
        mx = w1 * margin_ratio
        my = h1 * margin_ratio
        return not (x2_1 + mx < x1_2 or x2_2 < x1_1 - mx or y2_1 + my < y1_2 or y2_2 < y1_1 - my)

    @staticmethod
    def _box_iou(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0.0 else 0.0

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

        if not gullies and not obstacles:
            return BlockageMetrics(
                status="no_gully",
                coverage_percent=0.0,
                blocked_area_px=0,
                gully_area_px=0,
                gully_count=0,
                obstacle_count=0,
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

        obstacle_masks: list[object] = []
        for o in obstacles:
            m, method = self._detection_mask(o, height, width)
            obstacle_masks.append(m)
            methods.add(method)
            confidences.append(o.confidence)

        gully_union = self._union(gully_masks, height, width) if gully_masks else np.zeros((height, width), dtype=np.uint8)
        obstacle_union = self._union(obstacle_masks, height, width) if obstacle_masks else np.zeros((height, width), dtype=np.uint8)

        if gullies and obstacles:
            # Filter obstacles to only those that spatially associate with at least one gully
            associated_obstacles = []
            for o in obstacles:
                o_mask, _ = self._detection_mask(o, height, width)
                intersects_mask = bool(np.any(np.bitwise_and(gully_union, (o_mask > 0).astype(np.uint8))))
                intersects_box = any(self._box_intersects(g.bbox, o.bbox) for g in gullies)
                if intersects_mask or intersects_box:
                    associated_obstacles.append(o)
            obstacles = associated_obstacles

            # Recalculate obstacle masks and union with only associated obstacles
            obstacle_masks = []
            for o in obstacles:
                m, method = self._detection_mask(o, height, width)
                obstacle_masks.append(m)
                methods.add(method)
            obstacle_union = self._union(obstacle_masks, height, width) if obstacle_masks else np.zeros((height, width), dtype=np.uint8)

        total_drain_union = np.bitwise_or(gully_union, obstacle_union)
        total_drain_area_px = int(np.count_nonzero(total_drain_union))

        if total_drain_area_px == 0:
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

        if not gullies and obstacles:
            # Standalone obstacle without a detected drain grating:
            # Low-confidence false alarms on sky, trees, road markings, car wheels, or asphalt must be rejected.
            max_conf = max((getattr(o, "confidence", 0.0) for o in obstacles), default=0.0)
            if (not self.require_gully_presence) and max_conf >= self.standalone_min_conf:
                blocked_area_px = total_drain_area_px
                coverage_percent = 100.0
                gully_count = len(obstacles)
            else:
                return BlockageMetrics(
                    status="no_gully",
                    coverage_percent=0.0,
                    blocked_area_px=0,
                    gully_area_px=0,
                    gully_count=0,
                    obstacle_count=len(obstacles),
                    method="none",
                    confidence=max_conf,
                )
        elif gullies and not obstacles:
            # Clean drain (only drain_area)
            blocked_area_px = 0
            coverage_percent = 0.0
            gully_count = len(gullies)
        else:
            # Both detected: check if any gully and obstacle are co-located duplicate predictions
            # on the same physical structure (e.g. non-slip textured steel plate)
            is_colocated_duplicate = False
            for g in gullies:
                for o in obstacles:
                    if self._box_iou(g.bbox, o.bbox) >= 0.55:
                        # High bbox IoU between drain_area and drain_full indicates class ambiguity on texture
                        if g.confidence >= 0.70 and abs(g.confidence - o.confidence) < 0.25:
                            is_colocated_duplicate = True
                            break
                if is_colocated_duplicate:
                    break

            if is_colocated_duplicate:
                # When co-located duplicate predictions compete on a clean textured plate,
                # only count true non-gully occlusion area rather than double-counting
                unique_obstacle_mask = np.bitwise_and(obstacle_union, np.bitwise_not(gully_union))
                blocked_area_px = int(np.count_nonzero(unique_obstacle_mask))
                coverage_percent = min(100.0, (blocked_area_px / total_drain_area_px) * 100.0)
            else:
                blocked_area_px = int(np.count_nonzero(obstacle_union))
                coverage_percent = min(100.0, (blocked_area_px / total_drain_area_px) * 100.0)

            gully_count = max(len(gullies), len(obstacles))

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
            gully_area_px=total_drain_area_px,
            gully_count=gully_count,
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
