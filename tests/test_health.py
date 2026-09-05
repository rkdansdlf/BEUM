"""Tests for gully monitoring system health checks."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from gully_system.config import DetectorConfig, GPSConfig, StorageConfig, SystemConfig
from gully_system.health import (
    HealthStatus,
    SystemHealthError,
    check_camera_health,
    check_detector_health,
    check_sensors_health,
    check_storage_health,
    check_upload_health,
    run_health_check,
)
from gully_system.main import build_parser, main


def test_storage_health_success(tmp_path):
    config = SystemConfig(storage=StorageConfig(spool_dir=str(tmp_path / "spool")))
    health = check_storage_health(config)
    assert health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
    assert health.details["writable"] is True


def test_camera_health_missing_file():
    config = SystemConfig(source="non_existent_video_file_12345.mp4")
    health = check_camera_health(config)
    assert health.status == HealthStatus.UNHEALTHY
    assert "Video file source does not exist" in health.message


def test_detector_health_missing_file():
    config = SystemConfig(detector=DetectorConfig(model_path="non_existent_model_12345.pt"))
    health = check_detector_health(config)
    assert health.status == HealthStatus.UNHEALTHY
    assert "Model file not found" in health.message


def test_sensors_health_missing_replay_csv():
    config = SystemConfig(gps=GPSConfig(provider="replay", csv_path="non_existent_gps.csv"))
    health = check_sensors_health(config)
    assert health.status == HealthStatus.DEGRADED
    assert "does not exist" in health.message


def test_camera_health_picamera2_missing_import():
    config = SystemConfig(source="picamera2")
    with patch.dict("sys.modules", {"picamera2": None}):
        health = check_camera_health(config)
        assert health.status == HealthStatus.UNHEALTHY
        assert "python3-picamera2 is required" in health.message


def test_sensors_health_serial_missing_device(tmp_path):
    non_existent = str(tmp_path / "ttyUSB999")
    config = SystemConfig(gps=GPSConfig(provider="serial", serial_port=non_existent))
    with patch.dict("sys.modules", {"serial": MagicMock()}):
        health = check_sensors_health(config)
        assert health.status == HealthStatus.DEGRADED
        assert "GPS serial port device not found" in health.message


def test_upload_health_offline():
    config = SystemConfig()
    health = check_upload_health(config, check_network=False)
    assert health.status == HealthStatus.HEALTHY
    assert health.details["configured"] is False


def test_run_health_check_summary():
    config = SystemConfig(detector=DetectorConfig(model_path="non_existent_model.pt"))
    report = run_health_check(config, check_network=False)
    assert report.status == HealthStatus.UNHEALTHY
    assert report.can_start is False
    summary_text = report.summary()
    assert "Overall Status: UNHEALTHY" in summary_text
    assert "Can Start: NO" in summary_text


def test_main_cli_health_check_argument():
    parser = build_parser()
    args = parser.parse_args(["--health-check", "--strict", "--mapping-mode", "auto"])
    assert args.health_check is True
    assert args.strict is True
    assert args.mapping_mode == "auto"


def test_runtime_blockage_class_sync():
    """Verify that GullyRuntime synchronizes BlockageAnalyzer classes with detector mapping."""
    from gully_system.runtime import GullyRuntime
    from gully_system.validator import MappingResult

    mock_detector = MagicMock()
    mock_detector.mapping_result = MappingResult(
        is_exact_match=False,
        model_classes=("drain_area", "drain_full"),
        config_classes=("gully", "debris"),
        missing_in_config=("drain_area", "drain_full"),
        missing_in_model=("gully", "debris"),
        resolved_class_names=("drain_area", "drain_full"),
        resolved_gully_classes=("drain_area",),
        resolved_obstacle_classes=("drain_full",),
    )

    # Disable preflight for pure unit test
    config = SystemConfig(
        source="0",
        run_preflight=False,
        detector=DetectorConfig(model_path="dummy.pt"),
    )

    runtime = GullyRuntime(config, detector=mock_detector)
    assert "drain_area" in runtime.analyzer.gully_class_names
    assert "drain_full" in runtime.analyzer.obstacle_class_names
