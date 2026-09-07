# Raspberry Pi 5 Gully Monitor

This directory contains the deployable runtime extracted from the Colab prototype.
The OpenSCAD enclosure files in the project root are kept separate from the Python runtime.

## Important Model Rule

Do not deploy `yolov8n.pt` as a debris model. It is a COCO model and class `0` means
`person`. For accurate coverage measurement, deploy the validated custom 2-class segmentation model
`models/edge_exports/best-seg-2class_320.onnx` (classes: `drain_area`, `drain_full`).


Training is performed on a PC or Colab. The Raspberry Pi runs inference only.

```bash
yolo segment train model=yolov8n-seg.pt data=dataset/gully-seg.yaml imgsz=640 epochs=100
yolo export model=runs/segment/train/weights/best.pt format=ncnn imgsz=320
```

Benchmark the original `.pt`, ONNX, and NCNN exports on the target Pi. Do not assume
that the highest advertised FPS is achievable with the chosen camera resolution.
TensorRT is not a target backend for a plain Raspberry Pi 5 because it requires an
NVIDIA execution platform. If higher throughput is required, evaluate a supported
external accelerator separately.

## Local Development & Testing

### 1. Run Receiver Server (Central Backend)
```bash
# Start receiver server (default port 8001 or 8000)
python receiver_server.py --port 8001 --data-dir received_data

# Or run with Docker Compose (One-Click Containerized Deployment)
docker compose up -d

# View container logs
docker compose logs -f receiver
```

The receiver applies pending SQLite migrations before serving requests. Migration
state is recorded in `schema_migrations`, so restarting the server is safe and
does not re-run completed changes. To migrate a database before starting the
server, stop any running receiver and run:

```bash
python -m migrations.runner --db received_data/beum_events.db
```

Back up the database before a production migration. The event-metrics migration
adds missing `coverage_percent` and `occlusion_pct` columns and backfills
historical `occlusion_pct` values from `coverage_percent`.

The server exposes:
- `GET /dashboard` - Integrated interactive DrainSight Web Control Dashboard (Leaflet Map + Real-time SSE Feed).
- `GET /api/drainsight/geojson` - RFC 7946 GeoJSON FeatureCollection for GIS map integration (supports `?status=` and `?min_grade=`).
- `GET /api/drainsight/stats` - Control console KPI summary metrics (grade distribution, averages, sources).
- `GET /api/drainsight/alerts` - Actionable priority alerts filtered by coverage threshold (`?min_coverage=`).
- `GET /api/drainsight/stream` - Real-time Server-Sent Events (SSE) feed for live push notifications.
- `GET /api/drainsight/webhooks` & `POST /api/drainsight/webhooks` - Outbound webhook dispatcher configuration.
- `POST /upload` - Accepts multipart/form-data (`metadata` JSON + `image` JPEG) or plain JSON.
- `GET /events` - Lists received blockage events.
- `GET /events/{event_id}` - Detailed event record.
- `GET /events/{event_id}/image` - Download stored evidence photo.
- `GET /health` - Service health status.

### 2. Run Unit Tests
```bash
python -m unittest discover tests -v
```

### 3. Run Gully Edge Pipeline
```bash
# Run pipeline with default 2-class ONNX model
python -m gully_system.main --config config.example.json --model models/edge_exports/best-seg-2class_320.onnx --source video.mp4 --max-frames 100

# Realtime mode with latest-frame queue
python -m gully_system.main --config config.example.json --model models/edge_exports/best-seg-2class_320.onnx --source video.mp4 --realtime --max-frames 100
```

The example configuration uses the custom two-class segmentation model at `models/edge_exports/best-seg-2class_320.onnx`. A
detection-only model can run, but its coverage is explicitly reported as `bbox_estimate`
and is less accurate than `segmentation_mask`. Continuous output

video is disabled by default because it can fill edge storage; enable it only for short
debug runs. Events are written to `data/spool/pending` and are deleted only after a
successful HTTP upload. With no URL configured, the uploader is disabled and events
remain local.

## Raspberry Pi Setup

Use Raspberry Pi OS 64-bit with active cooling and a stable 5V/5A power supply.
Install the camera and OpenCV system packages using `apt`, then create a virtual
environment. Install only the selected inference backend and runtime dependencies.
Do not copy the Colab CUDA packages to the Pi.

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-picamera2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-pi.txt
```

For a CSI camera, set `source` to `picamera2`. For a USB camera, use a numeric
device index such as `0`. Picamera2 requires the Raspberry Pi camera stack packages.
For MP4 replay, use `--realtime`; frames are produced in a background buffer and
the oldest queued frame is dropped when processing falls behind. The final result
reports `dropped_frames`.

The repository includes `config.pi.json` pre-configured for the Raspberry Pi 5 with
the lightweight `best-seg-2class_320.onnx` model, CSI Picamera2, and UART GPS.

### Systemd Auto-Start Service & Watchdog

To configure the monitoring pipeline to start automatically on boot and recover
from unexpected crashes or power cycles:

```bash
# 1. Install and enable systemd service
sudo bash scripts/install_edge_service.sh

# 2. Check service status
systemctl status beum-edge.service

# 3. View live journal logs
journalctl -u beum-edge.service -f

# 4. Run hardware thermal & health watchdog
python scripts/edge_watchdog.py --config config.pi.json --interval-s 60
```

To remove the service:
```bash
sudo bash scripts/uninstall_edge_service.sh
```

## Hardware Verification & Live Preview (Camera & GPS)

To verify camera capture and GPS reception on the target Raspberry Pi without loading heavy AI models, use the dedicated hardware inspector:

```bash
# 1. System Pre-flight Check (Detector, Camera, Storage, GPS, Upload)
python -m gully_system.main --config config.example.json --health-check

# 2. Live Terminal Monitor (CLI / SSH mode)
python tools/hardware_check.py --config config.example.json

# 3. Live Web Preview (Best for field testing via Smartphone / Laptop browser)
python tools/hardware_check.py --config config.example.json --web --web-port 8080
# Open http://<RaspberryPi-IP>:8080 on your phone or laptop

# 4. Local GUI Window (Requires HDMI monitor or VNC desktop)
python tools/hardware_check.py --config config.example.json --display
```

## GPS Providers

GPS is accessed through `BaseGPSProvider`, so the runtime does not depend on whether
the source is a file or hardware. Configure one of the following providers:

```json
{"gps": {"provider": "replay", "csv_path": "data/gps.csv", "sample_period_s": 1.0}}
```

The replay provider updates its latest fix on a background thread. A detection event
uses the latest fix at the moment the event is created and records `age_s`, which
makes stale GPS mappings visible in the server payload.

For a u-blox UART receiver, use `provider: "serial"` and set `serial_port` such as
`/dev/ttyUSB0`. The provider parses common RMC and GGA NMEA sentences. Install
`pyserial` and add the `pi` user to the appropriate serial-device group.

For an iPhone or PC live test, use `provider: "udp"`. Send either NMEA text or JSON:

```json
{"latitude": 37.5665, "longitude": 126.9780, "speed": 0.0}
```

Set `udp_port` to the port used by the sender.

## Central Server Contract

Set `upload.url` to an HTTPS endpoint such as `/api/v1/gully-events`. When an event
has an evidence image, the client sends `multipart/form-data` with these fields:

```text
metadata: application/json
image: image/jpeg
```

The `metadata` object contains `event_id`, detections, blockage coverage, sensor
values, GPS, and the selected policy. A 2xx response is treated as a successful
upload and removes the local JSON/JPEG pair. Any timeout or non-2xx response leaves
the event pending for a later retry. The server should use `event_id` as an idempotency
key because a retry can deliver the same event more than once.

Example metadata shape:

```json
{
  "event_type": "gully_blockage",
  "blockage": {
    "status": "critical",
    "coverage_percent": 67.4,
    "method": "segmentation_mask"
  },
  "gps": {
    "latitude": 37.5665,
    "longitude": 126.978,
    "age_s": 0.12
  }
}
```

Use TLS, a device-specific bearer token, request size limits, and server-side image
retention rules in production. The current client sends the image only for queued
events, not for every camera frame.

## Offline RL Training

The RL environment and training script are under `training/`. They are run on a
PC or Colab and are not part of the edge runtime.

```bash
python -m training.train_dqn --steps 100000 --output models/gully_dqn
python -m training.export_policy_table models/gully_dqn.zip --output models/gully_policy.json
```

After validation, set `policy.model_path` to the generated JSON table. This avoids
installing stable-baselines3 on the Pi. The runtime can also load the DQN `.zip` on
CPU, but falls back to the rule policy if the file or RL dependencies are missing.
Emergency rain/water rules and critical-battery limits always override it.

## systemd Deployment

Copy the project to `/opt/gully-monitor`, create `/opt/gully-monitor/config.json`,
edit the service user and paths if necessary, then install the service:

```bash
sudo cp systemd/gully-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gully-monitor.service
sudo journalctl -u gully-monitor.service -f
```

## Runtime Behavior

- ROI filtering is performed before events are queued.
- Detections must persist across multiple inference cycles before confirmation.
- Only state-transition events are queued, preventing one object from being uploaded every frame.
- The local spool is bounded by the configured 1GB limit and prunes oldest pending events when full.
- The rule-based policy is the fallback and hard safety constraints always override RL.
- A DQN policy is optional and is loaded only for inference; it is never trained on the Pi.

## Next Hardware Integration

Replace the placeholder sensor provider with actual battery, rain, and water-level
drivers. Register the runtime as a `systemd` service with automatic restart and a
watchdog after camera and power tests pass.
