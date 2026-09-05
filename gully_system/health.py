"""Comprehensive health check utilities for the gully monitoring runtime."""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from gully_system.config import SystemConfig
from gully_system.sensors import SensorProvider, SystemSensorProvider
from gully_system.validator import (
    ModelMappingError,
    extract_model_classes,
    validate_and_resolve_mapping,
)

LOGGER = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class SystemHealthError(RuntimeError):
    """Raised when the system fails critical pre-flight health checks."""


@dataclass(frozen=True)
class ComponentHealth:
    name: str
    status: HealthStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class SystemHealthReport:
    status: HealthStatus
    can_start: bool
    components: dict[str, ComponentHealth]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "can_start": self.can_start,
            "timestamp": self.timestamp,
            "components": {k: v.to_dict() for k, v in self.components.items()},
        }

    def summary(self) -> str:
        lines = [
            f"=== Gully System Health Check Report ===",
            f"Overall Status: {self.status.value.upper()}",
            f"Can Start: {'YES' if self.can_start else 'NO'}",
            f"Timestamp: {self.timestamp}",
            "-" * 40,
        ]
        for name, comp in self.components.items():
            lines.append(f"[{comp.status.value.upper()}] {name}: {comp.message}")
            for k, v in comp.details.items():
                lines.append(f"    - {k}: {v}")
        return "\n".join(lines)


def check_detector_health(config: SystemConfig) -> ComponentHealth:
    """Checks model path existence, loading capability, and class mapping."""
    det_config = config.detector
    model_path = Path(det_config.model_path)
    if not model_path.exists():
        return ComponentHealth(
            name="detector",
            status=HealthStatus.UNHEALTHY,
            message=f"Model file not found: {model_path}",
            details={"model_path": str(model_path)},
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        return ComponentHealth(
            name="detector",
            status=HealthStatus.UNHEALTHY,
            message="Ultralytics YOLO package is not installed",
            details={"model_path": str(model_path)},
        )

    try:
        model = YOLO(str(model_path))
        model_classes = extract_model_classes(model)
        mapping_result = validate_and_resolve_mapping(
            model_classes=model_classes,
            config_class_names=det_config.class_names,
            gully_class_names=config.blockage.gully_class_names,
            obstacle_class_names=config.blockage.obstacle_class_names,
            mapping_mode=det_config.mapping_mode,
        )

        details = {
            "model_path": str(model_path),
            "model_classes": list(model_classes),
            "config_classes": list(det_config.class_names),
            "resolved_gully": list(mapping_result.resolved_gully_classes),
            "resolved_obstacle": list(mapping_result.resolved_obstacle_classes),
            "mapping_mode": det_config.mapping_mode,
            "exact_match": mapping_result.is_exact_match,
        }

        if mapping_result.warnings:
            details["warnings"] = list(mapping_result.warnings)

        if not mapping_result.is_valid:
            return ComponentHealth(
                name="detector",
                status=HealthStatus.UNHEALTHY,
                message=f"Model class mapping failed: {mapping_result.errors[0] if mapping_result.errors else 'unknown'}",
                details=details,
            )

        if not mapping_result.is_exact_match:
            return ComponentHealth(
                name="detector",
                status=HealthStatus.DEGRADED,
                message="Model loaded successfully with class mapping adjustments/warnings",
                details=details,
            )

        return ComponentHealth(
            name="detector",
            status=HealthStatus.HEALTHY,
            message="Model loaded and class mapping is exact match",
            details=details,
        )

    except ModelMappingError as exc:
        return ComponentHealth(
            name="detector",
            status=HealthStatus.UNHEALTHY,
            message=f"Model mapping validation error: {exc}",
            details={"model_path": str(model_path), "error": str(exc)},
        )
    except Exception as exc:
        return ComponentHealth(
            name="detector",
            status=HealthStatus.UNHEALTHY,
            message=f"Failed to load YOLO model: {exc}",
            details={"model_path": str(model_path), "error": str(exc)},
        )


def check_camera_health(config: SystemConfig) -> ComponentHealth:
    """Checks camera or video source availability and attempts a 1-frame read."""
    source_str = config.source.strip()
    is_picamera = source_str.lower() == "picamera2"
    is_device = source_str.isdigit()

    if is_picamera:
        try:
            from picamera2 import Picamera2
        except ImportError:
            return ComponentHealth(
                name="camera",
                status=HealthStatus.UNHEALTHY,
                message="python3-picamera2 is required for picamera2 source (run 'sudo apt install python3-picamera2')",
                details={"source": source_str, "type": "csi"},
            )
        try:
            picam = Picamera2()
            cam_config = picam.create_video_configuration(main={"size": (640, 480)})
            picam.configure(cam_config)
            picam.start()
            frame = picam.capture_array()
            picam.stop()
            picam.close()
            h, w = frame.shape[:2]
            return ComponentHealth(
                name="camera",
                status=HealthStatus.HEALTHY,
                message=f"Picamera2 CSI camera ready (resolution: {w}x{h})",
                details={"source": source_str, "type": "csi", "width": w, "height": h},
            )
        except Exception as exc:
            return ComponentHealth(
                name="camera",
                status=HealthStatus.DEGRADED,
                message=f"Picamera2 device check failed: {exc}",
                details={"source": source_str, "type": "csi", "error": str(exc)},
            )

    if not is_device and not Path(source_str).exists():
        return ComponentHealth(
            name="camera",
            status=HealthStatus.UNHEALTHY,
            message=f"Video file source does not exist: {source_str}",
            details={"source": source_str, "type": "file"},
        )

    try:
        import cv2
    except ImportError:
        return ComponentHealth(
            name="camera",
            status=HealthStatus.UNHEALTHY,
            message="OpenCV (cv2) is not installed",
            details={"source": source_str},
        )

    try:
        source_val: int | str = int(source_str) if is_device else source_str
        cap = cv2.VideoCapture(source_val)
        if not cap.isOpened():
            status = HealthStatus.UNHEALTHY if not is_device else HealthStatus.DEGRADED
            return ComponentHealth(
                name="camera",
                status=status,
                message=f"Failed to open video source: {source_str}",
                details={"source": source_str, "type": "device" if is_device else "file"},
            )

        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            return ComponentHealth(
                name="camera",
                status=HealthStatus.DEGRADED,
                message=f"Opened source {source_str} but failed to capture a test frame",
                details={"source": source_str},
            )

        height, width = frame.shape[:2]
        return ComponentHealth(
            name="camera",
            status=HealthStatus.HEALTHY,
            message=f"Camera source ready (resolution: {width}x{height})",
            details={"source": source_str, "width": width, "height": height},
        )
    except Exception as exc:
        return ComponentHealth(
            name="camera",
            status=HealthStatus.UNHEALTHY,
            message=f"Camera check error: {exc}",
            details={"source": source_str, "error": str(exc)},
        )


def check_storage_health(config: SystemConfig) -> ComponentHealth:
    """Checks spool storage directory write permissions and remaining disk space."""
    spool_path = Path(config.storage.spool_dir)
    try:
        spool_path.mkdir(parents=True, exist_ok=True)
        # Test write and delete
        test_file = spool_path / f".health_test_{os.getpid()}_{int(time.time())}.tmp"
        test_file.write_text("health_check_ok", encoding="utf-8")
        test_file.unlink()

        total, used, free = shutil.disk_usage(spool_path)
        free_mb = free / (1024 * 1024)

        details = {
            "spool_dir": str(spool_path),
            "free_mb": round(free_mb, 1),
            "writable": True,
        }

        # Warn if less than 200MB free
        if free_mb < 200.0:
            return ComponentHealth(
                name="storage",
                status=HealthStatus.DEGRADED,
                message=f"Storage writable but low disk space: {free_mb:.1f}MB available (<200MB)",
                details=details,
            )

        return ComponentHealth(
            name="storage",
            status=HealthStatus.HEALTHY,
            message=f"Storage directory writable and disk space adequate ({free_mb:.1f}MB free)",
            details=details,
        )
    except Exception as exc:
        return ComponentHealth(
            name="storage",
            status=HealthStatus.UNHEALTHY,
            message=f"Storage check failed: {exc}",
            details={"spool_dir": str(spool_path), "error": str(exc)},
        )


def check_sensors_health(
    config: SystemConfig,
    sensor_provider: SensorProvider | None = None,
) -> ComponentHealth:
    """Checks sensor readings and GPS configuration."""
    provider = sensor_provider or SystemSensorProvider()
    details: dict[str, Any] = {"gps_provider": config.gps.provider}

    # Test sensor snapshot
    try:
        snapshot = provider.read()
        details["sensor_snapshot"] = {
            "battery_pct": snapshot.battery_pct,
            "cpu_temp_c": snapshot.cpu_temp_c,
            "network_ok": snapshot.network_ok,
        }
    except Exception as exc:
        return ComponentHealth(
            name="sensors",
            status=HealthStatus.DEGRADED,
            message=f"Sensor reading error: {exc}",
            details=details,
        )

    # Test GPS provider configuration
    gps_provider_name = config.gps.provider.lower()
    if gps_provider_name == "replay":
        csv_path = Path(config.gps.csv_path)
        if not csv_path.exists():
            return ComponentHealth(
                name="sensors",
                status=HealthStatus.DEGRADED,
                message=f"Replay GPS csv file does not exist: {csv_path}",
                details=details,
            )
    elif gps_provider_name == "serial":
        port = config.gps.serial_port
        details["serial_port"] = port
        details["baudrate"] = config.gps.baudrate
        try:
            import serial
        except ImportError:
            return ComponentHealth(
                name="sensors",
                status=HealthStatus.DEGRADED,
                message="pyserial package is not installed (run 'pip install pyserial')",
                details=details,
            )

        port_path = Path(port)
        if not port_path.exists():
            return ComponentHealth(
                name="sensors",
                status=HealthStatus.DEGRADED,
                message=f"GPS serial port device not found: {port} (check USB/UART cable or run 'ls /dev/ttyUSB*')",
                details=details,
            )
        if not os.access(str(port_path), os.R_OK):
            return ComponentHealth(
                name="sensors",
                status=HealthStatus.DEGRADED,
                message=f"Permission denied accessing {port}. Run 'sudo usermod -aG dialout $USER' and re-login",
                details=details,
            )

    return ComponentHealth(
        name="sensors",
        status=HealthStatus.HEALTHY,
        message="Sensors and GPS configuration verified",
        details=details,
    )


def check_upload_health(config: SystemConfig, check_network: bool = False) -> ComponentHealth:
    """Checks upload configuration and optional connectivity."""
    url = config.upload.url.strip()
    if not url:
        return ComponentHealth(
            name="upload",
            status=HealthStatus.HEALTHY,
            message="Upload URL not configured (offline mode)",
            details={"configured": False},
        )

    details: dict[str, Any] = {"url": url, "timeout_s": config.upload.timeout_s}
    if not check_network:
        return ComponentHealth(
            name="upload",
            status=HealthStatus.HEALTHY,
            message="Upload URL configured (network probe skipped)",
            details=details,
        )

    # If network check requested, probe upload server
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD")
        if config.upload.token:
            req.add_header("Authorization", f"Bearer {config.upload.token}")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            details["status_code"] = resp.status
            return ComponentHealth(
                name="upload",
                status=HealthStatus.HEALTHY,
                message=f"Upload endpoint reached (HTTP {resp.status})",
                details=details,
            )
    except Exception as exc:
        # Edge device can still operate offline with spooling, so degraded rather than unhealthy
        return ComponentHealth(
            name="upload",
            status=HealthStatus.DEGRADED,
            message=f"Could not reach upload server (will spool locally): {exc}",
            details=details,
        )


def run_health_check(
    config: SystemConfig,
    check_network: bool = False,
    sensor_provider: SensorProvider | None = None,
) -> SystemHealthReport:
    """Runs all component health checks and compiles a SystemHealthReport."""
    components: dict[str, ComponentHealth] = {
        "detector": check_detector_health(config),
        "camera": check_camera_health(config),
        "storage": check_storage_health(config),
        "sensors": check_sensors_health(config, sensor_provider=sensor_provider),
        "upload": check_upload_health(config, check_network=check_network),
    }

    has_unhealthy = any(c.status == HealthStatus.UNHEALTHY for c in components.values())
    has_degraded = any(c.status == HealthStatus.DEGRADED for c in components.values())

    # can_start: detector or storage UNHEALTHY blocks start
    can_start = not (
        components["detector"].status == HealthStatus.UNHEALTHY
        or components["storage"].status == HealthStatus.UNHEALTHY
        or (components["camera"].status == HealthStatus.UNHEALTHY and not config.source.strip().isdigit())
    )

    if has_unhealthy:
        overall_status = HealthStatus.UNHEALTHY
    elif has_degraded:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.HEALTHY

    return SystemHealthReport(
        status=overall_status,
        can_start=can_start,
        components=components,
    )
