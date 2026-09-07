#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# BEUM Edge Runtime Bootstrap Installer for Raspberry Pi 5 (Bookworm)
# Runs directly on the Raspberry Pi from /boot/firmware/beum_edge
# ==============================================================================

if [ "$(id -u)" -ne 0 ]; then
    echo "[-] Error: This script must be run with sudo."
    echo "    Usage: sudo bash $0"
    exit 1
fi

TARGET_USER="${SUDO_USER:-$(logname 2>/dev/null || echo "pi")}"
if [ "${TARGET_USER}" = "root" ]; then
    # Fallback to the first standard user in /home
    FIRST_HOME=$(find /home -maxdepth 1 -mindepth 1 -type d | head -n 1)
    if [ -n "${FIRST_HOME}" ]; then
        TARGET_USER="$(basename "${FIRST_HOME}")"
    else
        TARGET_USER="pi"
    fi
fi

USER_HOME=$(getent passwd "${TARGET_USER}" | cut -d: -f6)
if [ -z "${USER_HOME}" ] || [ ! -d "${USER_HOME}" ]; then
    USER_HOME="/home/${TARGET_USER}"
    mkdir -p "${USER_HOME}"
    chown "${TARGET_USER}:${TARGET_USER}" "${USER_HOME}"
fi

INSTALL_DIR="${USER_HOME}/BEUM"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "          [BEUM] Edge Runtime One-Click Installer"
echo "======================================================================"
echo "[*] Source Directory  : ${SOURCE_DIR}"
echo "[*] Target User       : ${TARGET_USER}"
echo "[*] Target Home       : ${USER_HOME}"
echo "[*] Install Directory : ${INSTALL_DIR}"
echo "======================================================================"

# 1. Copy source files to target user directory
echo "[1/6] Deploying BEUM package to ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}"
cp -R "${SOURCE_DIR}"/* "${INSTALL_DIR}/" || true
mkdir -p "${INSTALL_DIR}/data/logs"
mkdir -p "${INSTALL_DIR}/data/spool/pending"
mkdir -p "${INSTALL_DIR}/models"
mkdir -p "${INSTALL_DIR}/.ultralytics"
mkdir -p "${INSTALL_DIR}/.matplotlib"

# If model was placed in source root or models/, ensure it exists in INSTALL_DIR/models
if [ -f "${SOURCE_DIR}/models/best-seg-2class_320.onnx" ]; then
    cp -u "${SOURCE_DIR}/models/best-seg-2class_320.onnx" "${INSTALL_DIR}/models/best-seg-2class_320.onnx"
elif [ -f "${SOURCE_DIR}/best-seg-2class_320.onnx" ]; then
    cp -u "${SOURCE_DIR}/best-seg-2class_320.onnx" "${INSTALL_DIR}/models/best-seg-2class_320.onnx"
fi

chown -R "${TARGET_USER}:${TARGET_USER}" "${INSTALL_DIR}"
echo "[+] Package deployed successfully."

# 2. Update APT and install essential system libraries
echo "[2/6] Installing system dependencies via APT..."
apt-get update -y
apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    i2c-tools \
    python3-serial
echo "[+] System dependencies installed."

# 3. Create Python virtual environment with system site packages (for picamera2 access)
echo "[3/6] Setting up Python virtual environment..."
VENV_DIR="${INSTALL_DIR}/.venv"
if [ ! -d "${VENV_DIR}" ]; then
    sudo -u "${TARGET_USER}" python3 -m venv --system-site-packages "${VENV_DIR}"
    echo "[+] Virtual environment created at ${VENV_DIR}"
else
    echo "[*] Virtual environment already exists at ${VENV_DIR}"
fi

# 4. Install Python requirements
echo "[4/6] Installing Python packages inside virtual environment..."
sudo -u "${TARGET_USER}" "${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
if [ -f "${INSTALL_DIR}/requirements.txt" ]; then
    sudo -u "${TARGET_USER}" "${VENV_DIR}/bin/pip" install -r "${INSTALL_DIR}/requirements.txt"
fi

# Ensure onnxruntime is installed for edge ONNX inference
if ! sudo -u "${TARGET_USER}" "${VENV_DIR}/bin/python" -c "import onnxruntime" 2>/dev/null; then
    echo "[*] Installing onnxruntime..."
    sudo -u "${TARGET_USER}" "${VENV_DIR}/bin/pip" install onnxruntime
fi
echo "[+] Python dependencies configured."

# 5. Fix config.pi.json model path if needed
CONFIG_FILE="${INSTALL_DIR}/config.pi.json"
if [ -f "${CONFIG_FILE}" ]; then
    echo "[5/6] Validating ${CONFIG_FILE}..."
    sed -i \
        -e 's|"models/deploy/best-seg-canonical.onnx"|"models/best-seg-2class_320.onnx"|g' \
        "${CONFIG_FILE}"
    echo "[+] Configuration updated to target models/best-seg-2class_320.onnx"
fi

# 6. Install and enable systemd services
echo "[6/6] Registering and starting Systemd services..."
chmod +x "${INSTALL_DIR}/scripts/install_edge_service.sh"
chmod +x "${INSTALL_DIR}/scripts/uninstall_edge_service.sh"
chmod +x "${INSTALL_DIR}/scripts/install_receiver_service.sh" 2>/dev/null || true
chmod +x "${INSTALL_DIR}/scripts/uninstall_receiver_service.sh" 2>/dev/null || true
chmod +x "${INSTALL_DIR}/scripts/edge_watchdog.py"

# Run install_edge_service.sh as target user with sudo
sudo -u "${TARGET_USER}" bash "${INSTALL_DIR}/scripts/install_edge_service.sh"

echo "======================================================================"
echo "          [BEUM] Edge Runtime Installation COMPLETE!"
echo "======================================================================"
echo "Edge Status    : sudo systemctl status beum-edge.service"
echo "Edge Live Logs : journalctl -u beum-edge.service -f"
echo "Application Dir: ${INSTALL_DIR}"
echo "----------------------------------------------------------------------"
echo "To also run Receiver Server & Dashboard directly on this Pi:"
echo "  sudo bash ${INSTALL_DIR}/scripts/install_receiver_service.sh"
echo "  (Access at http://<pi-ip>:8001/dashboard)"
echo "======================================================================"
