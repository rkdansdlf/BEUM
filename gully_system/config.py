"""System and component configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectorConfig:
    model_path: str = "models/best.pt"
    image_size: int = 320
    confidence: float = 0.4
    iou: float = 0.45
    device: str = "cpu"
    class_names: tuple[str, ...] = ("drain_area", "drain_full")
    mapping_mode: str = "auto"

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> DetectorConfig:
        if not value:
            return cls()
        return cls(
            model_path=str(value.get("model_path", cls.model_path)),
            image_size=int(value.get("image_size", cls.image_size)),
            confidence=float(value.get("confidence", cls.confidence)),
            iou=float(value.get("iou", cls.iou)),
            device=str(value.get("device", cls.device)),
            class_names=tuple(str(item) for item in value.get("class_names", cls.class_names)),
            mapping_mode=str(value.get("mapping_mode", cls.mapping_mode)),
        )


@dataclass(frozen=True)
class StorageConfig:
    spool_dir: str = "data/spool"
    max_gb: float = 1.0
    save_evidence: bool = True
    jpeg_quality: int = 80

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> StorageConfig:
        if not value:
            return cls()
        return cls(
            spool_dir=str(value.get("spool_dir", cls.spool_dir)),
            max_gb=float(value.get("max_gb", cls.max_gb)),
            save_evidence=bool(value.get("save_evidence", cls.save_evidence)),
            jpeg_quality=int(value.get("jpeg_quality", cls.jpeg_quality)),
        )


@dataclass(frozen=True)
class UploadConfig:
    url: str = ""
    token: str = ""
    timeout_s: float = 10.0
    flush_interval_s: float = 5.0
    base_backoff_s: float = 5.0
    max_backoff_s: float = 60.0

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> UploadConfig:
        if not value:
            return cls()
        return cls(
            url=str(value.get("url", cls.url)),
            token=str(value.get("token", cls.token)),
            timeout_s=float(value.get("timeout_s", cls.timeout_s)),
            flush_interval_s=float(value.get("flush_interval_s", cls.flush_interval_s)),
            base_backoff_s=float(value.get("base_backoff_s", cls.base_backoff_s)),
            max_backoff_s=float(value.get("max_backoff_s", cls.max_backoff_s)),
        )


@dataclass(frozen=True)
class GPSConfig:
    provider: str = "none"
    csv_path: str = "data/gps.csv"
    sample_period_s: float = 1.0
    replay_speed: float = 1.0
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    serial_timeout_s: float = 1.0
    udp_host: str = "0.0.0.0"
    udp_port: int = 9000

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> GPSConfig:
        if not value:
            return cls()
        return cls(
            provider=str(value.get("provider", cls.provider)),
            csv_path=str(value.get("csv_path", cls.csv_path)),
            sample_period_s=float(value.get("sample_period_s", cls.sample_period_s)),
            replay_speed=float(value.get("replay_speed", cls.replay_speed)),
            serial_port=str(value.get("serial_port", cls.serial_port)),
            baudrate=int(value.get("baudrate", cls.baudrate)),
            serial_timeout_s=float(value.get("serial_timeout_s", cls.serial_timeout_s)),
            udp_host=str(value.get("udp_host", cls.udp_host)),
            udp_port=int(value.get("udp_port", cls.udp_port)),
        )


@dataclass(frozen=True)
class BlockageConfig:
    gully_class_names: tuple[str, ...] = ("drain_area", "gully")
    obstacle_class_names: tuple[str, ...] = ("drain_full", "debris", "sediment", "trash", "leaf")
    warning_percent: float = 20.0
    critical_percent: float = 50.0
    event_change_percent: float = 10.0
    event_cooldown_s: float = 300.0

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> BlockageConfig:
        if not value:
            return cls()
        return cls(
            gully_class_names=tuple(str(item) for item in value.get("gully_class_names", cls.gully_class_names)),
            obstacle_class_names=tuple(str(item) for item in value.get("obstacle_class_names", cls.obstacle_class_names)),
            warning_percent=float(value.get("warning_percent", cls.warning_percent)),
            critical_percent=float(value.get("critical_percent", cls.critical_percent)),
            event_change_percent=float(value.get("event_change_percent", cls.event_change_percent)),
            event_cooldown_s=float(value.get("event_cooldown_s", cls.event_cooldown_s)),
        )


@dataclass(frozen=True)
class PolicyConfig:
    model_path: str = ""
    min_safe_battery: float = 20.0
    critical_battery: float = 8.0
    emergency_rain_level: int = 2
    emergency_water_level: float = 0.85
    mode_intervals: dict[str, float] = field(default_factory=lambda: {"low": 5.0, "medium": 1.0, "high": 0.2})

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> PolicyConfig:
        if not value:
            return cls()
        intervals = cls().mode_intervals.copy()
        if "mode_intervals" in value and isinstance(value["mode_intervals"], dict):
            intervals.update({str(k): float(v) for k, v in value["mode_intervals"].items()})
        return cls(
            model_path=str(value.get("model_path", cls.model_path)),
            min_safe_battery=float(value.get("min_safe_battery", cls.min_safe_battery)),
            critical_battery=float(value.get("critical_battery", cls.critical_battery)),
            emergency_rain_level=int(value.get("emergency_rain_level", cls.emergency_rain_level)),
            emergency_water_level=float(value.get("emergency_water_level", cls.emergency_water_level)),
            mode_intervals=intervals,
        )


@dataclass(frozen=True)
class SystemConfig:
    source: str = "0"
    output_path: str = ""
    max_frames: int = 0
    realtime_source: bool = False
    camera_buffer_size: int = 1
    roi_points: tuple[tuple[float, float], ...] = (
        (100.0, 400.0),
        (540.0, 400.0),
        (600.0, 480.0),
        (40.0, 480.0),
    )
    roi_base_resolution: tuple[int, int] | None = None
    roi_expanded_scale: float = 1.15
    temporal_hits: int = 3
    temporal_max_missed: int = 2
    temporal_iou: float = 0.3
    policy_interval_s: float = 30.0
    run_preflight: bool = True
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    blockage: BlockageConfig = field(default_factory=BlockageConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SystemConfig:
        roi_points = tuple((float(point[0]), float(point[1])) for point in value.get("roi_points", cls.roi_points))
        roi_base_resolution = None
        if "roi_base_resolution" in value and value["roi_base_resolution"]:
            roi_base_resolution = (
                int(value["roi_base_resolution"][0]),
                int(value["roi_base_resolution"][1]),
            )
        return cls(
            source=str(value.get("source", cls.source)),
            output_path=str(value.get("output_path", cls.output_path)),
            max_frames=int(value.get("max_frames", cls.max_frames)),
            realtime_source=bool(value.get("realtime_source", cls.realtime_source)),
            camera_buffer_size=int(value.get("camera_buffer_size", cls.camera_buffer_size)),
            roi_points=roi_points,
            roi_base_resolution=roi_base_resolution,
            roi_expanded_scale=float(value.get("roi_expanded_scale", cls.roi_expanded_scale)),
            temporal_hits=int(value.get("temporal_hits", cls.temporal_hits)),
            temporal_max_missed=int(value.get("temporal_max_missed", cls.temporal_max_missed)),
            temporal_iou=float(value.get("temporal_iou", cls.temporal_iou)),
            policy_interval_s=float(value.get("policy_interval_s", cls.policy_interval_s)),
            run_preflight=bool(value.get("run_preflight", cls.run_preflight)),
            detector=DetectorConfig.from_dict(value.get("detector", {})),
            storage=StorageConfig.from_dict(value.get("storage", {})),
            upload=UploadConfig.from_dict(value.get("upload", {})),
            gps=GPSConfig.from_dict(value.get("gps", {})),
            blockage=BlockageConfig.from_dict(value.get("blockage", {})),
            policy=PolicyConfig.from_dict(value.get("policy", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> SystemConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))
