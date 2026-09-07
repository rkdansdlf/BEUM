"""System and mock sensor snapshots."""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)


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
    """Reads available system and hardware values with auto-detection and graceful fallback."""

    def __init__(
        self,
        battery_file: str | Path | None = None,
        rain_file: str | Path | None = None,
        water_file: str | Path | None = None,
        rain_pin: int | None = None,
        water_pin: int | None = None,
        network_check_host: str = "8.8.8.8",
        network_check_port: int = 53,
        network_check_timeout_s: float = 0.5,
    ) -> None:
        self.battery_file = Path(battery_file) if battery_file else None
        self.rain_file = Path(rain_file) if rain_file else None
        self.water_file = Path(water_file) if water_file else None
        self.rain_pin = rain_pin
        self.water_pin = water_pin
        self.network_check_host = network_check_host
        self.network_check_port = network_check_port
        self.network_check_timeout_s = network_check_timeout_s

        self._network_last_check = 0.0
        self._network_cache: bool | None = None
        self._network_cache_ttl = 5.0

    def _battery(self) -> float | None:
        # 1. Custom battery file (e.g. UPS script or test mock)
        if self.battery_file and self.battery_file.exists():
            try:
                return max(0.0, min(100.0, float(self.battery_file.read_text(encoding="utf-8").strip())))
            except (OSError, ValueError):
                pass

        # 2. Linux standard power_supply sysfs
        power_supply_dir = Path("/sys/class/power_supply")
        if power_supply_dir.exists():
            for cap_file in power_supply_dir.glob("*/capacity"):
                try:
                    return max(0.0, min(100.0, float(cap_file.read_text(encoding="utf-8").strip())))
                except (OSError, ValueError):
                    continue

        return None

    @staticmethod
    def _cpu_temp() -> float | None:
        path = Path("/sys/class/thermal/thermal_zone0/temp")
        try:
            return float(path.read_text(encoding="utf-8").strip()) / 1000.0
        except (OSError, ValueError):
            return None

    def _rain(self) -> int:
        if self.rain_file and self.rain_file.exists():
            try:
                return max(0, int(self.rain_file.read_text(encoding="utf-8").strip()))
            except (OSError, ValueError):
                pass
        return 0

    def _water(self) -> float | None:
        if self.water_file and self.water_file.exists():
            try:
                return max(0.0, float(self.water_file.read_text(encoding="utf-8").strip()))
            except (OSError, ValueError):
                pass
        return None

    def _check_network(self) -> bool | None:
        now = time.monotonic()
        if now - self._network_last_check < self._network_cache_ttl:
            return self._network_cache

        self._network_last_check = now
        try:
            with socket.create_connection(
                (self.network_check_host, self.network_check_port),
                timeout=self.network_check_timeout_s,
            ):
                self._network_cache = True
                return True
        except (OSError, TimeoutError):
            self._network_cache = False
            return False

    def read(self) -> SensorSnapshot:
        return SensorSnapshot(
            battery_pct=self._battery(),
            rain_level=self._rain(),
            water_level=self._water(),
            cpu_temp_c=self._cpu_temp(),
            network_ok=self._check_network(),
            timestamp=time.time(),
        )


class MockSensorProvider:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = values or {}

    def read(self) -> SensorSnapshot:
        values = dict(self.values)
        values.setdefault("timestamp", time.time())
        return SensorSnapshot(**values)

