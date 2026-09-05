"""
Schema validation for DrainSight & BEUM events, detections, and telemetry.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Status(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNASSESSABLE = "UNASSESSABLE"


class Source(str, Enum):
    AI_VISION = "AI_VISION"
    EDGE_CAMERA = "EDGE_CAMERA"
    MANUAL = "MANUAL"
    SIMULATOR = "SIMULATOR"


class DetectionCreate(BaseModel):
    """
    Detection schema with strict contract preservation.
    drain_id, vehicle_code, status, source, lat, lng are required.
    occlusion_pct is bounded to [0.0, 100.0].
    """
    model_config = ConfigDict(extra="allow")

    drain_id: int
    vehicle_code: str = Field(min_length=1, max_length=50)
    status: Status
    occlusion_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    reason_code: Optional[Literal["DRAIN_NOT_DETECTED"]] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    lat: float = Field(ge=-90.0, le=90.0)
    lng: float = Field(ge=-180.0, le=180.0)
    source: Source

    @model_validator(mode="after")
    def _validate_assessment(self):
        if self.status == Status.UNASSESSABLE or self.status == "UNASSESSABLE":
            self.occlusion_pct = None
            self.reason_code = self.reason_code or "DRAIN_NOT_DETECTED"
        elif self.reason_code is not None:
            raise ValueError("reason_code는 UNASSESSABLE 상태에서만 사용할 수 있습니다")
        return self


class TelemetryIn(BaseModel):
    """
    Telemetry contract with timestamp (Unix epoch seconds) and per-vehicle sequence.
    Strictly requires vehicle_code, lat, lng, timestamp, and sequence.
    """
    model_config = ConfigDict(extra="allow")

    vehicle_code: str = Field(..., min_length=1, max_length=50)
    lat: float = Field(..., ge=-90.0, le=90.0)
    lng: float = Field(..., ge=-180.0, le=180.0)
    timestamp: float = Field(..., ge=0.0, description="Unix epoch seconds (e.g. 1725514800.123)")
    sequence: int = Field(..., ge=0, description="Per-vehicle monotonically increasing sequence")
    speed_mps: Optional[float] = Field(default=None, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # source -> vehicle_code
            if "vehicle_code" not in data and "source" in data:
                data["vehicle_code"] = data["source"]
            # gps dict mapping
            gps = data.get("gps")
            if isinstance(gps, dict):
                if "lat" not in data and ("latitude" in gps or "lat" in gps):
                    data["lat"] = gps.get("latitude", gps.get("lat"))
                if "lng" not in data and ("longitude" in gps or "lon" in gps or "lng" in gps):
                    data["lng"] = gps.get("longitude", gps.get("lon", gps.get("lng")))
                if "speed_mps" not in data and "speed" in gps:
                    data["speed_mps"] = gps["speed"]
            # lon -> lng
            if "lng" not in data and "lon" in data:
                data["lng"] = data["lon"]
        return data


TelemetrySchema = TelemetryIn


class BlockageSchema(BaseModel):
    status: str = "unknown"
    coverage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    method: str = "segmentation_mask"


class EventSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str = ""
    event_type: str = "gully_blockage"
    source: str = "unknown"
    created_at: str = ""
    blockage: BlockageSchema = Field(default_factory=BlockageSchema)
    gps: dict | None = None
    occlusion_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)


def validate_detection(payload: dict[str, Any]) -> tuple[bool, str | None, DetectionCreate | None]:
    try:
        det = DetectionCreate(**payload)
        return True, None, det
    except Exception as exc:
        return False, str(exc), None


def validate_telemetry(payload: dict[str, Any]) -> tuple[bool, str | None, TelemetryIn | None]:
    try:
        tel = TelemetryIn(**payload)
        return True, None, tel
    except Exception as exc:
        return False, str(exc), None


def validate_event_payload(payload: dict[str, Any]) -> tuple[bool, str | None, EventSchema | None]:
    try:
        evt = EventSchema(**payload)
        return True, None, evt
    except Exception as exc:
        return False, str(exc), None
