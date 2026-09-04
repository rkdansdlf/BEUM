"""System and mock sensor snapshots."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class SensorSnapshot:
    battery_pct: float | None = None
    rain_level: int = 0
    water_level: float | None = None
    cpu_temp_c: float | None = None
    network_ok: bool | None = None
    timestamp: float = 0.0


class SensorProvider(Protocol):
    def read(self) -> SensorSnapshot:
        ...


class SystemSensorProvider:
    """Reads available system values and leaves external sensors optional."""

    def __init__(self, battery_file: str | Path | None = None) -> None:
        if battery_file:
            self.battery_file = Path(battery_file)
        else:
            self.battery_file = None

    def _battery(self) -> float | None:
        if not self.battery_file or not self.battery_file.exists():
            return None
        try:
            return max(0.0, min(100.0, float(self.battery_file.read_text().strip())))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _cpu_temp() -> float | None:
        path = Path("/sys/class/thermal/thermal_zone0/temp")
        try:
            return float(path.read_text().strip()) / 1000.0
        except (OSError, ValueError):
            return None

    def read(self) -> SensorSnapshot:
        return SensorSnapshot(
            battery_pct=self._battery(),
            cpu_temp_c=self._cpu_temp(),
            timestamp=time.time(),
        )


class MockSensorProvider:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = values or {}

    def read(self) -> SensorSnapshot:
        values = dict(self.values)
        values.setdefault("timestamp", time.time())
        return SensorSnapshot(**values)
