"""Convert SENVIS-style Location.csv (nanosecond epoch, speed m/s) to a simple
GPS CSV that the runtime's ReplayGPSProvider can consume (timestamp seconds,
latitude, longitude, speed_mps)."""

from __future__ import annotations

import csv
from pathlib import Path

SOURCE = Path("data/_1-15-2026-09-03_02-42-46/Location.csv")
DEST = Path("data/_1-15-2026-09-03_02-42-46/gps_aligned.csv")


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"Source not found: {SOURCE}")
    rows: list[tuple[float, float, float, float]] = []
    with SOURCE.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            try:
                ts_ns = float(row["time"])
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            speed = float(row.get("speed") or 0.0)
            if speed < 0:
                speed = 0.0
            ts_s = ts_ns / 1_000_000_000.0
            rows.append((ts_s, lat, lon, speed))

    if not rows:
        raise SystemExit("No usable rows in source GPS CSV")
    rows.sort(key=lambda item: item[0])
    first_ts = rows[0][0]
    rows = [(ts - first_ts, lat, lon, sp) for ts, lat, lon, sp in rows]

    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "latitude", "longitude", "speed"])
        for ts, lat, lon, sp in rows:
            writer.writerow([f"{ts:.3f}", f"{lat:.8f}", f"{lon:.8f}", f"{sp:.3f}"])
    print(f"Wrote {len(rows)} rows to {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())