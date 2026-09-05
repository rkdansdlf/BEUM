#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# BEUM Edge Service Installer for Raspberry Pi 5 / Linux
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CURRENT_USER="$(id -un)"
CURRENT_GROUP="$(id -gn)"
SERVICE_NAME="beum-edge.service"
TARGET_SERVICE="/etc/systemd/system/${SERVICE_NAME}"
TEMPLATE_FILE="${PROJECT_DIR}/systemd/beum-edge.service"

echo "======================================================================"
echo "          [BEUM] Installing Edge Monitoring Systemd Service"
echo "======================================================================"
echo "[*] Project Directory : ${PROJECT_DIR}"
echo "[*] Target User/Group : ${CURRENT_USER}:${CURRENT_GROUP}"
echo "[*] Service File      : ${TARGET_SERVICE}"
echo "======================================================================"

# 1. Check Python virtual environment
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[-] Error: Python virtual environment not found at ${PYTHON_BIN}"
    echo "    Please create it first: python3 -m venv --system-site-packages .venv"
    exit 1
fi
echo "[+] Verified Python runtime: $(${PYTHON_BIN} --version)"

# 2. Ensure data and log directories exist
mkdir -p "${PROJECT_DIR}/data/logs"
mkdir -p "${PROJECT_DIR}/data/spool/pending"
mkdir -p "${PROJECT_DIR}/.ultralytics"
mkdir -p "${PROJECT_DIR}/.matplotlib"
chmod -R u+rw "${PROJECT_DIR}/data"
echo "[+] Log and Spool directories created"

# 3. Generate systemd service file with resolved paths and user
echo "[*] Generating systemd unit file..."
TEMP_SERVICE="$(mktemp)"
sed \
    -e "s|/home/pi/BEUM|${PROJECT_DIR}|g" \
    -e "s|User=pi|User=${CURRENT_USER}|g" \
    -e "s|Group=pi|Group=${CURRENT_GROUP}|g" \
    "${TEMPLATE_FILE}" > "${TEMP_SERVICE}"

if [ "$(id -u)" -eq 0 ]; then
    cp "${TEMP_SERVICE}" "${TARGET_SERVICE}"
    chmod 644 "${TARGET_SERVICE}"
else
    echo "[*] Requesting sudo permission to install service to /etc/systemd/system/..."
    sudo cp "${TEMP_SERVICE}" "${TARGET_SERVICE}"
    sudo chmod 644 "${TARGET_SERVICE}"
fi
rm -f "${TEMP_SERVICE}"
echo "[+] Installed ${TARGET_SERVICE}"

# 4. Reload systemd daemon and enable service
echo "[*] Reloading systemd daemon..."
if [ "$(id -u)" -eq 0 ]; then
    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}"
    systemctl restart "${SERVICE_NAME}"
    echo "[+] Service enabled and started successfully!"
    echo "----------------------------------------------------------------------"
    systemctl status "${SERVICE_NAME}" --no-pager || true
else
    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}"
    sudo systemctl restart "${SERVICE_NAME}"
    echo "[+] Service enabled and started successfully!"
    echo "----------------------------------------------------------------------"
    sudo systemctl status "${SERVICE_NAME}" --no-pager || true
fi

echo "======================================================================"
echo "Useful Commands:"
echo "  View live logs    : journalctl -u ${SERVICE_NAME} -f"
echo "  Tail log file     : tail -f ${PROJECT_DIR}/data/logs/beum-edge.log"
echo "  Stop service      : sudo systemctl stop ${SERVICE_NAME}"
echo "  Restart service   : sudo systemctl restart ${SERVICE_NAME}"
echo "  Uninstall service : bash ${SCRIPT_DIR}/uninstall_edge_service.sh"
echo "======================================================================"
