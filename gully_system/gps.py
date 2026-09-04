"""GPS providers for replay, serial UART, and UDP socket."""

from __future__ import annotations

import csv
import json
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GPSFix:
    latitude: float
    longitude: float
    speed_mps: float = 0.0
    received_at: float = field(default_factory=time.time)
    source: str = "unknown"

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        current_time = time.time() if now is None else now
        value = asdict(self)
        value["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.received_at))
        value["age_s"] = round(max(0.0, current_time - self.received_at), 3)
        del value["received_at"]
        return value


class BaseGPSProvider(ABC):
    """Common lifecycle and latest-fix interface for replay and hardware GPS."""

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def latest(self) -> GPSFix | None:
        pass


class NullGPSProvider(BaseGPSProvider):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def latest(self) -> GPSFix | None:
        return None


@dataclass(frozen=True)
class _ReplayRecord:
    elapsed_s: float
    latitude: float
    longitude: float
    speed_mps: float


class ReplayGPSProvider(BaseGPSProvider):
    def __init__(self, csv_path: str | Path, sample_period_s: float = 1.0, replay_speed: float = 1.0) -> None:
        self.csv_path = Path(csv_path)
        self.sample_period_s = max(0.01, float(sample_period_s))
        self.replay_speed = max(0.1, float(replay_speed))
        self._records: list[_ReplayRecord] = []
        self._latest_fix: GPSFix | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._load_csv()

    def _load_csv(self) -> None:
        if not self.csv_path.exists():
            return
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first_ts: float | None = None
            for row in reader:
                try:
                    lat = float(row["latitude"])
                    lon = float(row["longitude"])
                    speed = float(row.get("speed", 0.0) or 0.0)
                    ts = float(row["timestamp"]) if "timestamp" in row else 0.0
                except (KeyError, TypeError, ValueError):
                    continue
                if first_ts is None:
                    first_ts = ts
                elapsed = max(0.0, ts - first_ts)
                self._records.append(_ReplayRecord(elapsed, lat, lon, speed))

    def start(self) -> None:
        if not self._records:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest(self) -> GPSFix | None:
        return self._latest_fix

    def _run(self) -> None:
        start_time = time.monotonic()
        index = 0
        total = len(self._records)
        while not self._stop_event.is_set() and index < total:
            rec = self._records[index]
            self._latest_fix = GPSFix(
                latitude=rec.latitude,
                longitude=rec.longitude,
                speed_mps=rec.speed_mps,
                source="csv",
            )
            index += 1
            if index < total:
                next_rec = self._records[index]
                wait = (next_rec.elapsed_s - rec.elapsed_s) / self.replay_speed
                if wait > 0:
                    time.sleep(wait)
            else:
                break


def _nmea_coordinate(raw: str, hemisphere: str, deg_digits: int) -> float:
    degrees = float(raw[:deg_digits])
    minutes = float(raw[deg_digits:])
    coordinate = degrees + minutes / 60.0
    return -coordinate if hemisphere in ("S", "W") else coordinate


def parse_nmea_sentence(sentence: str) -> GPSFix | None:
    """Parse RMC or GGA NMEA sentences from common u-blox receivers."""
    fields = sentence.strip().split(",")
    if not fields or not fields[0].startswith("$"):
        return None
    sentence_type = fields[0][3:6]
    try:
        if sentence_type == "RMC" and len(fields) >= 9 and fields[2] == "A":
            lat = _nmea_coordinate(fields[3], fields[4], 2)
            lon = _nmea_coordinate(fields[5], fields[6], 3)
            speed = float(fields[7] or 0.0) * 0.514444
            return GPSFix(latitude=lat, longitude=lon, speed_mps=speed, source="nmea")
        if sentence_type == "GGA" and len(fields) >= 7 and fields[6] != "0":
            lat = _nmea_coordinate(fields[2], fields[3], 2)
            lon = _nmea_coordinate(fields[4], fields[5], 3)
            return GPSFix(latitude=lat, longitude=lon, source="nmea")
    except (IndexError, ValueError):
        return None
    return None


class SerialGPSProvider(BaseGPSProvider):
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 9600, timeout_s: float = 1.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self._latest_fix: GPSFix | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest(self) -> GPSFix | None:
        return self._latest_fix

    def _run(self) -> None:
        try:
            import serial
            with serial.Serial(self.port, self.baudrate, timeout=self.timeout_s) as ser:
                while not self._stop_event.is_set():
                    line = ser.readline().decode("ascii", errors="replace")
                    fix = parse_nmea_sentence(line)
                    if fix:
                        self._latest_fix = fix
        except Exception:
            pass


class UDPGPSProvider(BaseGPSProvider):
    def __init__(self, host: str = "0.0.0.0", port: int = 9000) -> None:
        self.host = host
        self.port = port
        self._latest_fix: GPSFix | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest(self) -> GPSFix | None:
        return self._latest_fix

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(1.0)
        while not self._stop_event.is_set():
            try:
                data, _ = sock.recvfrom(4096)
                text = data.decode("utf-8", errors="replace").strip()
                if text.startswith("$"):
                    fix = parse_nmea_sentence(text)
                    if fix:
                        self._latest_fix = fix
                elif text.startswith("{"):
                    payload = json.loads(text)
                    lat = float(payload.get("latitude") or payload.get("lat"))
                    lon = float(payload.get("longitude") or payload.get("lon"))
                    speed = float(payload.get("speed", 0.0) or 0.0)
                    self._latest_fix = GPSFix(latitude=lat, longitude=lon, speed_mps=speed, source="udp")
            except (socket.timeout, ValueError, KeyError):
                continue
            except Exception:
                break
        sock.close()
