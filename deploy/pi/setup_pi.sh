#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "          BEUM Edge Runtime Setup for Raspberry Pi 4 / 5 (ARM64)               "
echo "================================================================================"

# 1. Architecture Check
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    echo "⚠️  Warning: Current architecture is $ARCH. 64-bit OS (aarch64) is strongly recommended for ONNXRuntime."
fi

# 2. System Packages Installation
echo "\n[1/5] Updating package index & installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    v4l-utils

# 3. User Group Permissions for Camera and Serial GPS
echo "\n[2/5] Configuring user permissions for video and serial devices..."
sudo usermod -a -G video,dialout "$USER" || true

# 4. Virtual Environment Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "\n[3/5] Creating Python virtual environment in $REPO_ROOT/.venv-pi..."
if [ ! -d "$REPO_ROOT/.venv-pi" ]; then
    python3 -m venv "$REPO_ROOT/.venv-pi"
fi

source "$REPO_ROOT/.venv-pi/bin/activate"

# 5. Pip Dependencies
echo "\n[4/5] Installing Python edge dependencies..."
pip install --upgrade pip wheel setuptools
pip install -r "$SCRIPT_DIR/requirements-pi.txt"

# Optional: Try installing picamera2 if system libcamera is present
if python3 -c "import picamera2" 2>/dev/null; then
    echo "      -> picamera2 system library detected."
else
    echo "      -> Note: Install picamera2 via 'sudo apt install -y python3-picamera2' if using CSI camera module."
fi

# 6. Verify ONNX Model and Configurations
echo "\n[5/5] Verifying model deployment artifacts..."
if [ -f "$REPO_ROOT/models/deploy/best-seg-canonical.onnx" ]; then
    echo "      -> ONNX model verified: models/deploy/best-seg-canonical.onnx ✅"
else
    echo "      ⚠️  Warning: models/deploy/best-seg-canonical.onnx not found. Please sync deploy models."
fi

echo "\n================================================================================"
echo "✅ BEUM Edge setup completed successfully!"
echo "To start the edge monitor manually:"
echo "    source .venv-pi/bin/activate"
echo "    python -m gully_system.main --config config.pi.json"
echo "================================================================================"
