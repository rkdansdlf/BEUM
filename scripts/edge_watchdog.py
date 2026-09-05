#!/usr/bin/env python3
"""
Edge Hardware & Service Watchdog for Raspberry Pi 5
---------------------------------------------------
Periodically inspects:
1. beum-edge service process liveness.
2. CPU thermal status (/sys/class/thermal/thermal_zone0/temp).
3. Disk space on data/spool.
4. Camera & detector preflight health.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gully_system.config import SystemConfig
from gully_system.health import run_health_check

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (Watchdog) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("edge_watchdog")


def get_cpu_temp_c() -> float | None:
    """Reads Raspberry Pi 5 thermal zone."""
    thermal_file = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_file.exists():
        try:
            raw = thermal_file.read_text().strip()
            return float(raw) / 1000.0
        except Exception:
            return None
    return None


def get_disk_usage_pct(path: Path) -> float:
    """Calculates disk usage percentage."""
    total, used, free = shutil.disk_usage(str(path))
    return (used / total) * 100.0 if total > 0 else 0.0


def check_systemd_service(service_name: str = "beum-edge.service") -> bool:
    """Checks if systemd service is active."""
    try:
        res = subprocess.run(
            ["systemctl", "is-active", "--quiet", service_name],
            check=False,
        )
        return res.returncode == 0
    except FileNotFoundError:
        # systemctl not available (e.g. macOS / non-systemd)
        return True


def run_watchdog_cycle(config_path: Path, max_temp_c: float = 82.0, max_disk_pct: float = 90.0) -> bool:
    logger.info("Starting edge health inspection cycle...")
    is_healthy = True

    # 1. Check process liveness
    service_active = check_systemd_service("beum-edge.service")
    if not service_active:
        logger.error("[ALERT] beum-edge.service is NOT active! Attempting restart...")
        try:
            subprocess.run(["sudo", "systemctl", "restart", "beum-edge.service"], check=False)
        except Exception as e:
            logger.error(f"Failed to trigger service restart: {e}")
        is_healthy = False
    else:
        logger.info("[OK] beum-edge.service is ACTIVE")

    # 2. Check CPU temperature
    temp = get_cpu_temp_c()
    if temp is not None:
        if temp >= max_temp_c:
            logger.warning(f"[THERMAL WARNING] CPU temperature high: {temp:.1f}°C (Threshold: {max_temp_c}°C)")
            is_healthy = False
        else:
            logger.info(f"[OK] CPU Temperature: {temp:.1f}°C")
    else:
        logger.debug("CPU temperature thermal zone not accessible (non-Linux/Pi host).")

    # 3. Check Disk Storage
    spool_dir = PROJECT_ROOT / "data" / "spool"
    spool_dir.mkdir(parents=True, exist_ok=True)
    disk_pct = get_disk_usage_pct(spool_dir)
    if disk_pct >= max_disk_pct:
        logger.error(f"[DISK WARNING] Disk space critical: {disk_pct:.1f}% used (Threshold: {max_disk_pct}%)")
        is_healthy = False
    else:
        logger.info(f"[OK] Disk usage: {disk_pct:.1f}%")

    # 4. Check Config & Component Preflight
    if config_path.exists():
        config = SystemConfig.from_json(config_path)
        report = run_health_check(config, check_network=False)
        if not report.can_start:
            logger.error(f"[COMPONENT FAILURE] Preflight health check failed: {report.status.value}")
            for comp, h in report.components.items():
                if h.status.value == "unhealthy":
                    logger.error(f"  - {comp}: {h.message}")
            is_healthy = False
        else:
            logger.info("[OK] Core components preflight check PASSED")

    return is_healthy


def main():
    parser = argparse.ArgumentParser(description="BEUM Edge Watchdog")
    parser.add_argument("--config", default="config.pi.json", help="Path to config.pi.json")
    parser.add_argument("--interval-s", type=float, default=60.0, help="Loop interval in seconds (0 = run once)")
    parser.add_argument("--max-temp", type=float, default=82.0, help="Max CPU temperature threshold (°C)")
    parser.add_argument("--max-disk", type=float, default=90.0, help="Max disk usage threshold (%)")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config

    if args.interval_s <= 0:
        healthy = run_watchdog_cycle(config_path, max_temp_c=args.max_temp, max_disk_pct=args.max_disk)
        sys.exit(0 if healthy else 1)

    logger.info(f"Watchdog daemon started. Inspecting every {args.interval_s}s...")
    try:
        while True:
            run_watchdog_cycle(config_path, max_temp_c=args.max_temp, max_disk_pct=args.max_disk)
            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        logger.info("Watchdog terminated by operator.")


if __name__ == "__main__":
    main()
