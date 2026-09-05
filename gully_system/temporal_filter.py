"""Temporal track filtering to suppress false positives across frames."""

from __future__ import annotations

from dataclasses import dataclass, replace
from gully_system.types import Detection


def intersection_over_union(first: Detection, second: Detection) -> float:
    ax1, ay1, ax2, ay2 = first.bbox
    bx1, by1, bx2, by2 = second.bbox
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return (intersection / union) if union > 0 else 0.0


@dataclass
class _Track:
    track_id: str
    detection: Detection
    hits: int = 1
    missed: int = 0
    confirmed: bool = False


@dataclass(frozen=True)
class TemporalResult:
    validated: tuple[Detection, ...]
    entered: tuple[Detection, ...]
    exited: tuple[Detection, ...]


class TemporalFilter:
    def __init__(self, min_hits: int = 3, max_missed: int = 2, iou_threshold: float = 0.3) -> None:
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._tracks: dict[str, _Track] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def _new_track(self, detection: Detection) -> _Track:
        track_id = f"local-{self._next_id}"
        self._next_id += 1
        return _Track(track_id=track_id, detection=detection.with_track_id(track_id))

    def update(self, detections: list[Detection]) -> TemporalResult:
        remaining = list(detections)
        matched: dict[str, Detection] = {}
        for track_id, track in list(self._tracks.items()):
            best_idx = -1
            best_iou = self.iou_threshold
            for i, det in enumerate(remaining):
                if det.class_name == track.detection.class_name:
                    score = intersection_over_union(det, track.detection)
                    if score >= best_iou:
                        best_iou = score
                        best_idx = i
            if best_idx >= 0:
                matched[track_id] = remaining.pop(best_idx)

        validated: list[Detection] = []
        entered: list[Detection] = []
        exited: list[Detection] = []

        for track_id, track in list(self._tracks.items()):
            if track_id in matched:
                det = matched[track_id].with_track_id(track_id)
                track.detection = det
                track.hits += 1
                track.missed = 0
                if not track.confirmed and track.hits >= self.min_hits:
                    track.confirmed = True
                    entered.append(det)
                if track.confirmed:
                    validated.append(det)
            else:
                track.missed += 1
                if track.missed > self.max_missed:
                    if track.confirmed:
                        exited.append(track.detection)
                    del self._tracks[track_id]
                elif track.confirmed:
                    validated.append(track.detection)

        for det in remaining:
            track = self._new_track(det)
            if track.hits >= self.min_hits:
                track.confirmed = True
                entered.append(track.detection)
                validated.append(track.detection)
            self._tracks[track.track_id] = track

        return TemporalResult(
            validated=tuple(validated),
            entered=tuple(entered),
            exited=tuple(exited),
        )
