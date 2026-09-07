#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# BEUM Receiver Service Uninstaller
# ==============================================================================

SERVICE_NAME="beum-receiver.service"
TARGET_SERVICE="/etc/systemd/system/${SERVICE_NAME}"

echo "======================================================================"
echo "          [BEUM] Uninstalling Central Receiver Systemd Service"
echo "======================================================================"

if [ -f "${TARGET_SERVICE}" ]; then
    echo "[*] Stopping and disabling ${SERVICE_NAME}..."
    if [ "$(id -u)" -eq 0 ]; then
        systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
        systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
        rm -f "${TARGET_SERVICE}"
        systemctl daemon-reload
    else
        sudo systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
        sudo systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
        sudo rm -f "${TARGET_SERVICE}"
        sudo systemctl daemon-reload
    fi
    echo "[+] Successfully uninstalled ${SERVICE_NAME}"
else
    echo "[!] Service file ${TARGET_SERVICE} does not exist. Nothing to remove."
fi
echo "======================================================================"
