"""Synthetic GPS fixtures for 5m proximity testing."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between two GPS coordinates."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def make_synthetic_gps_csv(
    center_lat: float = 36.838,
    center_lon: float = 127.184,
    num_points: int = 10,
    step_m: float = 1.0,
    output_path: Path | str | None = None,
) -> Path:
    """Create a synthetic GPS CSV with waypoints around a center point.

    The waypoints are placed at approximately `step_m` meter intervals,
    suitable for testing 5m proximity rules.

    Args:
        center_lat: Center latitude.
        center_lon: Center longitude.
        num_points: Number of GPS waypoints to generate.
        step_m: Distance between waypoints in meters.
        output_path: Optional path; if None, writes to a temp file.

    Returns:
        Path to the created CSV file.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmp.close()
        output_path = Path(tmp.name)
    output_path = Path(output_path)

    lines = ["timestamp,latitude,longitude,speed"]
    for i in range(num_points):
        t = float(i)
        offset = step_m * i
        dlat = offset / 111320.0
        lat = center_lat + dlat
        lon = center_lon
        speed = 1.0
        lines.append(f"{t},{lat:.8f},{lon:.8f},{speed}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def make_synthetic_proximity_events(
    center_lat: float = 36.838,
    center_lon: float = 127.184,
    radius_m: float = 5.0,
    num_drains: int = 4,
) -> list[dict[str, Any]]:
    """Create a set of drain locations around a center, within radius_m.

    Returns:
        List of drain dicts with lat, lon, and approximate distance from center.
    """
    drains = []
    for i in range(num_drains):
        angle = (2 * math.pi * i) / num_drains
        dlat = (radius_m * math.cos(angle)) / 111320.0
        dlon = (radius_m * math.sin(angle)) / (111320.0 * math.cos(math.radians(center_lat)))
        lat = center_lat + dlat
        lon = center_lon + dlon
        drains.append(
            {
                "drain_id": f"drain_{i:03d}",
                "lat": lat,
                "lon": lon,
                "distance_from_center_m": haversine_distance_m(center_lat, center_lon, lat, lon),
            }
        )
    return drains


def make_telemetry_payload(
    source: str = "pi5-curb-cam",
    timestamp: float = 1700000000.0,
    sequence: int = 1,
    lat: float = 36.838,
    lon: float = 127.184,
    battery_pct: float = 85.0,
    rain_level: int = 0,
) -> dict[str, Any]:
    """Create a synthetic telemetry payload conforming to TelemetrySchema."""
    return {
        "source": source,
        "timestamp": timestamp,
        "sequence": sequence,
        "battery_pct": battery_pct,
        "rain_level": rain_level,
        "gps": {"latitude": lat, "longitude": lon, "speed": 0.0},
    }