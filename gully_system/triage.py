"""Active Learning and Hard Negative Auto-Triage Module for BEUM.

Monitors edge inference results in real-time and harvests high-value training
candidates (uncertain predictions, borderline coverage, temporal flickers)
with GPS telemetry and visual evidence for continuous MLOps ingestion.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gully_system.blockage import BlockageMetrics
from gully_system.gps import GPSFix
from gully_system.types import Detection

LOGGER = logging.getLogger(__name__)


class TriageReason:
    UNCERTAINTY = "uncertainty_low_conf"
    BORDERLINE_WARNING = "borderline_warning_coverage"
    BORDERLINE_CRITICAL = "borderline_critical_coverage"
    TEMPORAL_FLICKER = "temporal_flicker"
    HIGH_AREA_MISMATCH = "high_area_mismatch"


@dataclass
class TriageCandidate:
    candidate_id: str
    timestamp: float
    reasons: list[str]
    detections: list[dict[str, Any]]
    coverage_percent: float
    status: str
    gps: dict[str, Any] | None
    image_filename: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActiveLearningTriager:
    """Evaluates frames and harvests active learning candidates under rate limiting."""

    def __init__(
        self,
        output_dir: str | Path = "data/active_learning/triage",
        uncertain_conf_range: tuple[float, float] = (0.15, 0.28),
        borderline_warning_range: tuple[float, float] = (15.0, 28.0),
        borderline_critical_range: tuple[float, float] = (45.0, 55.0),
        cooldown_s: float = 2.0,
        max_candidates: int = 1000,
        save_images: bool = True,
        jpeg_quality: int = 85,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.uncertain_conf_range = uncertain_conf_range
        self.borderline_warning_range = borderline_warning_range
        self.borderline_critical_range = borderline_critical_range
        self.cooldown_s = cooldown_s
        self.max_candidates = max_candidates
        self.save_images = save_images
        self.jpeg_quality = jpeg_quality

        self._last_harvest_time = 0.0
        self._harvest_count = 0
        self._prev_frame_had_detection = False

    def evaluate_frame(
        self,
        frame: Any,
        raw_detections: tuple[Detection, ...] | list[Detection],
        blockage: BlockageMetrics | None = None,
        gps_fix: GPSFix | None = None,
        is_flicker: bool = False,
    ) -> TriageCandidate | None:
        """Evaluates whether the current frame warrants harvesting for active learning."""
        now = time.time()
        if now - self._last_harvest_time < self.cooldown_s:
            return None
        if self._harvest_count >= self.max_candidates:
            return None

        reasons: list[str] = []

        # 1. Uncertainty / Low Confidence Detection Check
        has_uncertain = any(
            self.uncertain_conf_range[0] <= d.confidence <= self.uncertain_conf_range[1]
            for d in raw_detections
        )
        if has_uncertain:
            reasons.append(TriageReason.UNCERTAINTY)

        # 2. Borderline Coverage Check
        cov = blockage.coverage_percent if blockage else 0.0
        if self.borderline_warning_range[0] <= cov <= self.borderline_warning_range[1]:
            reasons.append(TriageReason.BORDERLINE_WARNING)
        elif self.borderline_critical_range[0] <= cov <= self.borderline_critical_range[1]:
            reasons.append(TriageReason.BORDERLINE_CRITICAL)

        # 3. Temporal Flicker Check
        if is_flicker:
            reasons.append(TriageReason.TEMPORAL_FLICKER)

        if not reasons:
            return None

        # Build candidate
        cand_id = f"al_{int(now)}_{uuid.uuid4().hex[:8]}"
        img_name = f"{cand_id}.jpg"

        # Encode and save image
        if self.save_images and frame is not None:
            try:
                import cv2
                img_path = self.output_dir / img_name
                cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            except Exception as e:
                LOGGER.warning("Failed to save triage candidate image: %s", e)
                return None

        candidate = TriageCandidate(
            candidate_id=cand_id,
            timestamp=now,
            reasons=reasons,
            detections=[d.to_dict() for d in raw_detections],
            coverage_percent=round(cov, 2),
            status=blockage.status if blockage else "unknown",
            gps=gps_fix.to_dict() if gps_fix else None,
            image_filename=img_name,
            metadata={
                "gully_count": blockage.gully_count if blockage else 0,
                "obstacle_count": blockage.obstacle_count if blockage else 0,
            }
        )

        # Save JSON metadata
        json_path = self.output_dir / f"{cand_id}.json"
        json_path.write_text(json.dumps(candidate.to_dict(), indent=2), encoding="utf-8")

        self._last_harvest_time = now
        self._harvest_count += 1
        LOGGER.info(
            "Harvested active learning candidate [%s]: reasons=%s, cov=%.1f%%",
            cand_id,
            reasons,
            cov,
        )
        return candidate
