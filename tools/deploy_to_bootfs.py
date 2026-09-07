#!/usr/bin/env python3
"""
BEUM Edge Deployment Tool for Raspberry Pi bootfs partition.
Copies essential runtime modules, quantized ONNX models, configs,
and bootstrap automation scripts directly into the FAT32 bootfs partition.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Deploy BEUM Edge runtime to SD card bootfs")
    parser.add_argument("--bootfs", default="/Volumes/bootfs", help="Path to mounted bootfs partition")
    parser.add_argument("--model", default="models/edge_exports/best-seg-2class_320.onnx", help="Path to ONNX model")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    bootfs_path = Path(args.bootfs)

    print("======================================================================")
    print("            [BEUM] Deploying Edge Runtime to SD bootfs")
    print("======================================================================")
    print(f"[*] Project Root : {project_root}")
    print(f"[*] Bootfs Path  : {bootfs_path}")

    if not bootfs_path.exists() or not bootfs_path.is_dir():
        print(f"[-] Error: bootfs mount point not found at {bootfs_path}")
        print("    Please ensure the Raspberry Pi SD card is plugged in and mounted.")
        sys.exit(1)

    # 1. Check disk space
    stat = os.statvfs(bootfs_path)
    free_bytes = stat.f_bavail * stat.f_frsize
    free_mb = free_bytes / (1024 * 1024)
    print(f"[*] Free Space on bootfs : {free_mb:.1f} MB")
    if free_mb < 50.0:
        print(f"[-] Warning: Low disk space ({free_mb:.1f} MB < 50 MB)")

    # 2. Enable SSH on first boot
    ssh_file = bootfs_path / "ssh"
    if not ssh_file.exists():
        ssh_file.touch()
        print(f"[+] Created SSH enablement flag: {ssh_file}")
    else:
        print(f"[*] SSH enablement flag already present: {ssh_file}")

    # 3. Configure hardware UART in config.txt
    config_txt = bootfs_path / "config.txt"
    if config_txt.exists():
        content = config_txt.read_text(encoding="utf-8", errors="replace")
        lines_to_add = []
        if "enable_uart=1" not in content:
            lines_to_add.append("enable_uart=1")
        if "dtoverlay=disable-bt" not in content and "# dtoverlay=disable-bt" not in content:
            # Commented note for serial GPS on AMA0
            lines_to_add.append("# dtoverlay=disable-bt # (Optional: uncomment if using GPIO pin 8/10 for GPS)")
        
        if lines_to_add:
            with open(config_txt, "a", encoding="utf-8") as f:
                f.write("\n# BEUM Edge Hardware Interfaces\n")
                for line in lines_to_add:
                    f.write(f"{line}\n")
            print(f"[+] Updated {config_txt} with UART settings.")
        else:
            print(f"[*] {config_txt} already contains UART settings.")

    # 4. Target package directory
    target_dir = bootfs_path / "beum_edge"
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Target Package Directory: {target_dir}")

    # 5. Copy essential directories and files
    dirs_to_copy = ["gully_system", "scripts", "systemd", "migrations", "tools"]
    for d in dirs_to_copy:
        src = project_root / d
        dst = target_dir / d
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
            print(f"[+] Copied directory: {d} -> {dst}")

    # Copy sample GPS replay data
    data_dst_dir = target_dir / "data"
    data_dst_dir.mkdir(parents=True, exist_ok=True)
    sample_gps = project_root / "data/e2e_gps_replay.csv"
    if sample_gps.exists():
        shutil.copy2(sample_gps, data_dst_dir / "gps.csv")
        shutil.copy2(sample_gps, data_dst_dir / "e2e_gps_replay.csv")
        print(f"[+] Copied sample GPS data: {sample_gps} -> {data_dst_dir / 'gps.csv'}")

    files_to_copy = [
        ("config.pi.json", "config.pi.json"),
        ("config.example.json", "config.example.json"),
        ("requirements.txt", "requirements.txt"),
        ("requirements-pi.txt", "requirements-pi.txt"),
        ("requirements-server.txt", "requirements-server.txt"),
        ("receiver_server.py", "receiver_server.py"),
        ("drainsight_adapter.py", "drainsight_adapter.py"),
        ("schemas.py", "schemas.py"),
        ("README.md", "README.md"),
        ("scripts/bootstrap_pi.sh", "bootstrap_pi.sh"),
        ("scripts/install_edge_service.sh", "install_edge_service.sh"),
        ("scripts/install_receiver_service.sh", "install_receiver_service.sh"),
        ("scripts/uninstall_edge_service.sh", "uninstall_edge_service.sh"),
        ("scripts/uninstall_receiver_service.sh", "uninstall_receiver_service.sh"),
        ("scripts/edge_watchdog.py", "edge_watchdog.py"),
    ]

    for rel_src, rel_dst in files_to_copy:
        src = project_root / rel_src
        dst = target_dir / rel_dst
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[+] Copied file: {rel_src} -> {dst}")

    # 6. Copy ONNX Model
    model_src = project_root / args.model
    if not model_src.exists():
        # Fallback search
        candidate = project_root / "models/edge_exports/best-seg-2class_320.onnx"
        if candidate.exists():
            model_src = candidate
        else:
            candidate2 = project_root / "models/deploy/best-seg-canonical.onnx"
            if candidate2.exists():
                model_src = candidate2

    if model_src.exists():
        models_dst_dir = target_dir / "models"
        models_dst_dir.mkdir(exist_ok=True)
        dst_model = models_dst_dir / "best-seg-2class_320.onnx"
        shutil.copy2(model_src, dst_model)
        # Also copy directly under target_dir for bootstrap script convenience
        shutil.copy2(model_src, target_dir / "best-seg-2class_320.onnx")
        model_size_mb = dst_model.stat().st_size / (1024 * 1024)
        print(f"[+] Copied ONNX model: {model_src} -> {dst_model} ({model_size_mb:.2f} MB)")
    else:
        print(f"[-] Warning: Model file not found at {model_src}")

    # 7. Write Quickstart Guide in bootfs
    guide_file = bootfs_path / "BEUM_INSTALL_GUIDE.txt"
    guide_content = f"""======================================================================
           BEUM Full System - Raspberry Pi 5 Quickstart Guide
======================================================================

1. Insert this SD card into your Raspberry Pi 5 and power it on.

2. Access the Raspberry Pi terminal:
   - Option A: Monitor & Keyboard connected directly
   - Option B: Headless SSH via Ethernet / Wi-Fi
     ssh pi@raspberrypi.local
     (or ssh <your-user>@<ip-address>)

3. Run the one-click bootstrap installer:
   sudo bash /boot/firmware/beum_edge/bootstrap_pi.sh

4. What the installer does automatically:
   - Copies complete BEUM system to /home/<user>/BEUM
   - Installs system packages (picamera2, libgl, libglib, venv, serial)
   - Creates Python virtual environment (.venv)
   - Installs all dependencies (ultralytics, onnxruntime, fastapi, etc.)
   - Registers and starts the Edge Monitoring Service (beum-edge.service)

5. Operational Verification Commands:
   - Edge Service Status : sudo systemctl status beum-edge.service
   - Edge Live Logs      : journalctl -u beum-edge.service -f
   - Edge Health Check   : python -m gully_system.main --config config.pi.json --health-check
   - Live Camera Preview : python tools/hardware_check.py --config config.pi.json --web --web-port 8080

6. [Optional] Run Receiver Server & DrainSight Dashboard directly on Pi:
   - Start receiver as a background service:
     sudo bash ~/BEUM/scripts/install_receiver_service.sh
   - Check Receiver Status: sudo systemctl status beum-receiver.service
   - Open Web Dashboard in browser: http://<pi-ip>:8001/dashboard
======================================================================
"""
    guide_file.write_text(guide_content, encoding="utf-8")
    print(f"[+] Generated Guide: {guide_file}")

    # Summary
    total_size = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    print("======================================================================")
    print(f"[+] Deployment complete! Total payload size: {total_size_mb:.2f} MB")
    print(f"[*] Target Location: {target_dir}")
    print("======================================================================")

if __name__ == "__main__":
    main()
