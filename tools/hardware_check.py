#!/usr/bin/env python3
"""Hardware diagnostic and live preview tool for camera and GPS on Raspberry Pi."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gully_system.camera import CameraSource
from gully_system.config import SystemConfig
from gully_system.gps import (
    BaseGPSProvider,
    GPSFix,
    NullGPSProvider,
    ReplayGPSProvider,
    SerialGPSProvider,
    UDPGPSProvider,
    parse_nmea_sentence,
)

LOGGER = logging.getLogger("hardware_check")


@dataclass
class DiagnosticState:
    camera_ok: bool = False
    camera_source: str = "0"
    resolution: tuple[int, int] = (0, 0)
    fps: float = 0.0
    frame_count: int = 0
    camera_error: str = ""

    gps_provider: str = "none"
    gps_port: str = ""
    gps_sentences_received: int = 0
    gps_fix: GPSFix | None = None
    last_nmea_line: str = ""
    gps_error: str = ""

    current_frame_jpeg: bytes | None = None
    lock: threading.Lock = threading.Lock()


GLOBAL_STATE = DiagnosticState()


class DiagnosticGPSMonitor:
    """Monitors GPS data with raw sentence inspection."""

    def __init__(self, config: SystemConfig, state: DiagnosticState) -> None:
        self.config = config
        self.state = state
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.provider_name = config.gps.provider.lower()

    def start(self) -> None:
        self._stop_event.clear()
        self.state.gps_provider = self.provider_name
        if self.provider_name == "serial":
            self.state.gps_port = f"{self.config.gps.serial_port} @ {self.config.gps.baudrate}bps"
            self._thread = threading.Thread(target=self._run_serial, daemon=True)
            self._thread.start()
        elif self.provider_name == "udp":
            self.state.gps_port = f"UDP {self.config.gps.udp_host}:{self.config.gps.udp_port}"
            self._thread = threading.Thread(target=self._run_udp, daemon=True)
            self._thread.start()
        elif self.provider_name == "replay":
            self.state.gps_port = f"Replay {self.config.gps.csv_path}"
            self._thread = threading.Thread(target=self._run_replay, daemon=True)
            self._thread.start()
        else:
            self.state.gps_port = "Disabled (provider=none)"

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run_serial(self) -> None:
        port = self.config.gps.serial_port
        baudrate = self.config.gps.baudrate
        try:
            import serial
        except ImportError:
            with self.state.lock:
                self.state.gps_error = "pyserial not installed. Run 'pip install pyserial'"
            return

        if not Path(port).exists():
            with self.state.lock:
                self.state.gps_error = f"Device not found: {port}"
            return

        try:
            with serial.Serial(port, baudrate, timeout=1.0) as ser:
                while not self._stop_event.is_set():
                    raw_line = ser.readline().decode("ascii", errors="replace").strip()
                    if not raw_line:
                        continue
                    fix = parse_nmea_sentence(raw_line)
                    with self.state.lock:
                        self.state.last_nmea_line = raw_line
                        self.state.gps_sentences_received += 1
                        if fix:
                            self.state.gps_fix = fix
                        self.state.gps_error = ""
        except Exception as exc:
            with self.state.lock:
                self.state.gps_error = str(exc)

    def _run_udp(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.config.gps.udp_host, self.config.gps.udp_port))
        sock.settimeout(1.0)
        while not self._stop_event.is_set():
            try:
                data, _ = sock.recvfrom(4096)
                text = data.decode("utf-8", errors="replace").strip()
                fix = None
                if text.startswith("$"):
                    fix = parse_nmea_sentence(text)
                elif text.startswith("{"):
                    payload = json.loads(text)
                    lat = float(payload.get("latitude") or payload.get("lat"))
                    lon = float(payload.get("longitude") or payload.get("lon"))
                    speed = float(payload.get("speed", 0.0) or 0.0)
                    fix = GPSFix(latitude=lat, longitude=lon, speed_mps=speed, source="udp")
                with self.state.lock:
                    self.state.last_nmea_line = text[:80]
                    self.state.gps_sentences_received += 1
                    if fix:
                        self.state.gps_fix = fix
            except (socket.timeout, ValueError, KeyError):
                continue
            except Exception as exc:
                with self.state.lock:
                    self.state.gps_error = str(exc)
                break
        sock.close()

    def _run_replay(self) -> None:
        provider = ReplayGPSProvider(
            csv_path=self.config.gps.csv_path,
            sample_period_s=self.config.gps.sample_period_s,
            replay_speed=self.config.gps.replay_speed,
        )
        provider.start()
        while not self._stop_event.is_set():
            fix = provider.latest()
            if fix:
                with self.state.lock:
                    self.state.gps_fix = fix
                    self.state.gps_sentences_received += 1
            time.sleep(0.5)
        provider.stop()


class WebViewerHandler(BaseHTTPRequestHandler):
    """Simple MJPEG streaming & status web server."""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._serve_index()
        elif self.path == "/stream.mjpg":
            self._serve_stream()
        elif self.path == "/status.json":
            self._serve_status()
        else:
            self.send_error(404, "Not Found")

    def _serve_index(self) -> None:
        html = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEUM - 라즈베리파이 하드웨어 진단</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #0f172a; color: #f8fafc; padding: 16px; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 16px; }
        h1 { font-size: 1.25rem; font-weight: 700; color: #38bdf8; }
        .grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
        @media (min-width: 768px) { .grid { grid-template-columns: 3fr 2fr; } }
        .card { background: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .card-title { font-size: 1rem; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; }
        .badge { padding: 4px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }
        .badge-ok { background: #059669; color: #ecfdf5; }
        .badge-warn { background: #d97706; color: #fffbeb; }
        .badge-err { background: #dc2626; color: #fef2f2; }
        .video-box { width: 100%; border-radius: 8px; overflow: hidden; background: #000; min-height: 240px; display: flex; align-items: center; justify-content: center; }
        .video-box img { width: 100%; height: auto; display: block; }
        .info-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #334155; font-size: 0.9rem; }
        .info-label { color: #94a3b8; }
        .info-val { font-weight: 600; font-family: monospace; }
        .nmea-box { background: #0b0f19; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 0.8rem; color: #10b981; overflow-x: auto; white-space: nowrap; margin-top: 10px; }
        .map-btn { display: inline-block; margin-top: 12px; padding: 10px 16px; background: #2563eb; color: white; text-decoration: none; border-radius: 8px; font-size: 0.85rem; font-weight: 600; text-align: center; width: 100%; }
        .map-btn:hover { background: #1d4ed8; }
    </style>
</head>
<body>
    <header>
        <h1>BEUM Hardware Inspector</h1>
        <span id="sys-clock" style="font-size: 0.85rem; color: #94a3b8;">--:--:--</span>
    </header>

    <div class="grid">
        <div class="card">
            <div class="card-title">
                <span>카메라 실시간 뷰 (Camera Live)</span>
                <span id="cam-badge" class="badge badge-warn">확인 중</span>
            </div>
            <div class="video-box">
                <img src="/stream.mjpg" alt="Live Stream" />
            </div>
            <div style="margin-top: 12px;">
                <div class="info-row"><span class="info-label">입력 소스</span><span id="cam-src" class="info-val">-</span></div>
                <div class="info-row"><span class="info-label">해상도</span><span id="cam-res" class="info-val">-</span></div>
                <div class="info-row"><span class="info-label">실시간 FPS</span><span id="cam-fps" class="info-val">0.0</span></div>
            </div>
        </div>

        <div class="card">
            <div class="card-title">
                <span>GPS 모듈 상태 (GPS Fix)</span>
                <span id="gps-badge" class="badge badge-warn">수신 대기</span>
            </div>
            <div class="info-row"><span class="info-label">연결 방식 / 포트</span><span id="gps-port" class="info-val">-</span></div>
            <div class="info-row"><span class="info-label">위도 (Latitude)</span><span id="gps-lat" class="info-val">-</span></div>
            <div class="info-row"><span class="info-label">경도 (Longitude)</span><span id="gps-lon" class="info-val">-</span></div>
            <div class="info-row"><span class="info-label">속도 (Speed)</span><span id="gps-speed" class="info-val">0.0 km/h</span></div>
            <div class="info-row"><span class="info-label">수신된 NMEA 패킷</span><span id="gps-packets" class="info-val">0</span></div>
            <div class="info-row"><span class="info-label">데이터 갱신 경과</span><span id="gps-age" class="info-val">-</span></div>

            <div style="margin-top: 14px;">
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 4px;">최근 수신 NMEA 원시 문장:</div>
                <div id="gps-raw" class="nmea-box">Waiting for NMEA sentence...</div>
            </div>

            <a id="map-link" href="#" target="_blank" class="map-btn" style="display: none;">Google Maps에서 현재 위치 확인</a>
        </div>
    </div>

    <script>
        setInterval(() => {
            const now = new Date();
            document.getElementById('sys-clock').innerText = now.toLocaleTimeString();
        }, 1000);

        async function updateStatus() {
            try {
                const res = await fetch('/status.json');
                const data = await res.json();

                const camBadge = document.getElementById('cam-badge');
                if (data.camera_ok) {
                    camBadge.innerText = 'ONLINE';
                    camBadge.className = 'badge badge-ok';
                } else {
                    camBadge.innerText = data.camera_error ? 'ERROR' : 'OFFLINE';
                    camBadge.className = 'badge badge-err';
                }
                document.getElementById('cam-src').innerText = data.camera_source;
                document.getElementById('cam-res').innerText = data.resolution[0] + ' x ' + data.resolution[1];
                document.getElementById('cam-fps').innerText = data.fps.toFixed(1) + ' FPS';

                const gpsBadge = document.getElementById('gps-badge');
                document.getElementById('gps-port').innerText = data.gps_port;
                document.getElementById('gps-packets').innerText = data.gps_sentences_received;
                document.getElementById('gps-raw').innerText = data.last_nmea_line || '(NMEA 수신 대기 중)';

                if (data.gps_fix) {
                    gpsBadge.innerText = 'FIXED (정상 수신)';
                    gpsBadge.className = 'badge badge-ok';
                    const lat = data.gps_fix.latitude.toFixed(6);
                    const lon = data.gps_fix.longitude.toFixed(6);
                    const speedKmh = (data.gps_fix.speed_mps * 3.6).toFixed(1);

                    document.getElementById('gps-lat').innerText = lat;
                    document.getElementById('gps-lon').innerText = lon;
                    document.getElementById('gps-speed').innerText = speedKmh + ' km/h';
                    document.getElementById('gps-age').innerText = (data.gps_fix_age || 0).toFixed(1) + '초 전';

                    const mapBtn = document.getElementById('map-link');
                    mapBtn.href = `https://www.google.com/maps?q=${lat},${lon}`;
                    mapBtn.style.display = 'block';
                } else {
                    if (data.gps_sentences_received > 0) {
                        gpsBadge.innerText = 'NO FIX (위성 신호 탐색 중)';
                        gpsBadge.className = 'badge badge-warn';
                    } else if (data.gps_error) {
                        gpsBadge.innerText = 'PORT ERROR';
                        gpsBadge.className = 'badge badge-err';
                    } else {
                        gpsBadge.innerText = 'NO DATA';
                        gpsBadge.className = 'badge badge-warn';
                    }
                    document.getElementById('gps-lat').innerText = '-';
                    document.getElementById('gps-lon').innerText = '-';
                    document.getElementById('gps-age').innerText = '-';
                }
            } catch (err) {
                console.error('Status fetch failed', err);
            }
        }

        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</body>
</html>
"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_status(self) -> None:
        with GLOBAL_STATE.lock:
            fix_dict = None
            age = None
            if GLOBAL_STATE.gps_fix:
                fix_dict = {
                    "latitude": GLOBAL_STATE.gps_fix.latitude,
                    "longitude": GLOBAL_STATE.gps_fix.longitude,
                    "speed_mps": GLOBAL_STATE.gps_fix.speed_mps,
                    "source": GLOBAL_STATE.gps_fix.source,
                }
                age = round(time.time() - GLOBAL_STATE.gps_fix.received_at, 2)

            payload = {
                "camera_ok": GLOBAL_STATE.camera_ok,
                "camera_source": GLOBAL_STATE.camera_source,
                "resolution": list(GLOBAL_STATE.resolution),
                "fps": round(GLOBAL_STATE.fps, 1),
                "frame_count": GLOBAL_STATE.frame_count,
                "camera_error": GLOBAL_STATE.camera_error,
                "gps_provider": GLOBAL_STATE.gps_provider,
                "gps_port": GLOBAL_STATE.gps_port,
                "gps_sentences_received": GLOBAL_STATE.gps_sentences_received,
                "gps_fix": fix_dict,
                "gps_fix_age": age,
                "last_nmea_line": GLOBAL_STATE.last_nmea_line,
                "gps_error": GLOBAL_STATE.gps_error,
            }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with GLOBAL_STATE.lock:
                    jpeg = GLOBAL_STATE.current_frame_jpeg
                if jpeg is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.066)  # ~15 FPS cap for web preview
        except (BrokenPipeError, ConnectionResetError):
            pass


def draw_hud(frame: Any, state: DiagnosticState) -> None:
    """Overlay diagnostic GPS and camera information directly on the frame."""
    try:
        import cv2

        h, w = frame.shape[:2]
        # Top banner background
        cv2.rectangle(frame, (0, 0), (w, 65), (20, 24, 33), -1)

        # Camera status
        cam_text = f"CAM: {state.camera_source} ({w}x{h}) | {state.fps:.1f} FPS"
        cv2.putText(frame, cam_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # GPS status
        if state.gps_fix:
            age = max(0.0, time.time() - state.gps_fix.received_at)
            gps_text = (
                f"GPS FIXED: Lat {state.gps_fix.latitude:.5f}, Lon {state.gps_fix.longitude:.5f} | "
                f"Speed: {state.gps_fix.speed_mps * 3.6:.1f} km/h (age: {age:.1f}s)"
            )
            color = (0, 255, 100)
        elif state.gps_sentences_received > 0:
            gps_text = f"GPS: RX {state.gps_sentences_received} pkts (Searching satellites / No Fix)"
            color = (0, 180, 255)
        else:
            gps_text = f"GPS: No signal ({state.gps_port})"
            color = (80, 80, 255)

        cv2.putText(frame, gps_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    except Exception:
        pass


def run_terminal_monitor(stop_event: threading.Event) -> None:
    """Print clean real-time status in terminal (SSH mode)."""
    while not stop_event.is_set():
        with GLOBAL_STATE.lock:
            fps = GLOBAL_STATE.fps
            res = GLOBAL_STATE.resolution
            cam_ok = GLOBAL_STATE.camera_ok
            src = GLOBAL_STATE.camera_source
            fix = GLOBAL_STATE.gps_fix
            pkts = GLOBAL_STATE.gps_sentences_received
            port = GLOBAL_STATE.gps_port
            raw_line = GLOBAL_STATE.last_nmea_line

        # ANSI clear screen & cursor home
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write("=====================================================\n")
        sys.stdout.write("       BEUM Raspberry Pi Hardware Live Inspector     \n")
        sys.stdout.write("=====================================================\n\n")

        sys.stdout.write(f"[CAMERA]\n")
        sys.stdout.write(f"  - Status    : {'OK (Capturing)' if cam_ok else 'FAILED / DISCONNECTED'}\n")
        sys.stdout.write(f"  - Source    : {src}\n")
        sys.stdout.write(f"  - Resolution: {res[0]} x {res[1]}\n")
        sys.stdout.write(f"  - Live FPS  : {fps:.1f}\n\n")

        sys.stdout.write(f"[GPS]\n")
        sys.stdout.write(f"  - Port/Type : {port}\n")
        sys.stdout.write(f"  - RX Packets: {pkts}\n")
        if fix:
            age = max(0.0, time.time() - fix.received_at)
            sys.stdout.write(f"  - Fix Status: FIXED (3D Lock)\n")
            sys.stdout.write(f"  - Latitude  : {fix.latitude:.6f}\n")
            sys.stdout.write(f"  - Longitude : {fix.longitude:.6f}\n")
            sys.stdout.write(f"  - Speed     : {fix.speed_mps * 3.6:.1f} km/h ({fix.speed_mps:.2f} m/s)\n")
            sys.stdout.write(f"  - Last Fix  : {age:.1f}s ago\n")
        elif pkts > 0:
            sys.stdout.write(f"  - Fix Status: SEARCHING (NMEA received, waiting for satellite fix)\n")
        else:
            sys.stdout.write(f"  - Fix Status: NO DATA / DISCONNECTED\n")

        if raw_line:
            sys.stdout.write(f"\n[LATEST NMEA SENTENCE]\n  {raw_line[:75]}\n")

        sys.stdout.write("\n-----------------------------------------------------\n")
        sys.stdout.write("Press Ctrl+C to stop.\n")
        sys.stdout.flush()
        time.sleep(1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BEUM Hardware Diagnostic & Live Preview Tool")
    parser.add_argument("--config", default="config.example.json", help="Path to BEUM config JSON")
    parser.add_argument("--source", help="Override camera source (e.g., 'picamera2', '0', or video path)")
    parser.add_argument("--gps-provider", choices=("none", "serial", "udp", "replay"), help="Override GPS provider")
    parser.add_argument("--gps-port", help="Override serial port (e.g., /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, help="Override serial baudrate (default 9600)")
    parser.add_argument("--display", action="store_true", help="Open local OpenCV GUI window (requires monitor/X11)")
    parser.add_argument("--web", action="store_true", help="Start web preview server (e.g. for smartphone/browser)")
    parser.add_argument("--web-port", type=int, default=8080, help="Web preview port (default: 8080)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        config = SystemConfig.from_json(config_path)
    else:
        config = SystemConfig()

    if args.source:
        config = SystemConfig(
            source=args.source,
            storage=config.storage,
            upload=config.upload,
            gps=config.gps,
            detector=config.detector,
            blockage=config.blockage,
            policy=config.policy,
        )
    if args.gps_provider:
        config.gps.provider = args.gps_provider
    if args.gps_port:
        config.gps.serial_port = args.gps_port
    if args.baudrate:
        config.gps.baudrate = args.baudrate

    GLOBAL_STATE.camera_source = str(config.source)

    # Start GPS Monitor
    gps_mon = DiagnosticGPSMonitor(config, GLOBAL_STATE)
    gps_mon.start()

    # Start Web Server if requested
    web_server = None
    if args.web:
        server_address = ("0.0.0.0", args.web_port)
        web_server = ThreadingHTTPServer(server_address, WebViewerHandler)
        threading.Thread(target=web_server.serve_forever, daemon=True).start()
        print(f"\n[WEB] Live inspector running at: http://0.0.0.0:{args.web_port}")
        print(f"      Access via smartphone or PC on same network (e.g. http://<Pi-IP>:{args.web_port})\n")

    # Start Terminal Status Monitor if not display mode
    term_stop = threading.Event()
    if not args.display:
        threading.Thread(target=run_terminal_monitor, args=(term_stop,), daemon=True).start()

    # Camera capture loop
    cam = CameraSource(source=config.source)
    try:
        import cv2
    except ImportError:
        cv2 = None

    try:
        cam.open()
        with GLOBAL_STATE.lock:
            GLOBAL_STATE.camera_ok = True
    except Exception as exc:
        with GLOBAL_STATE.lock:
            GLOBAL_STATE.camera_ok = False
            GLOBAL_STATE.camera_error = str(exc)
        print(f"[ERROR] Failed to open camera ({config.source}): {exc}")

    fps_count = 0
    fps_start = time.monotonic()

    try:
        while True:
            if GLOBAL_STATE.camera_ok:
                ok, frame = cam.read()
                if ok and frame is not None:
                    fps_count += 1
                    now = time.monotonic()
                    elapsed = now - fps_start
                    if elapsed >= 1.0:
                        with GLOBAL_STATE.lock:
                            GLOBAL_STATE.fps = fps_count / elapsed
                        fps_count = 0
                        fps_start = now

                    h, w = frame.shape[:2]
                    with GLOBAL_STATE.lock:
                        GLOBAL_STATE.resolution = (w, h)
                        GLOBAL_STATE.frame_count += 1

                    # Draw HUD
                    draw_hud(frame, GLOBAL_STATE)

                    # Encode JPEG for web streaming
                    if args.web and cv2:
                        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        with GLOBAL_STATE.lock:
                            GLOBAL_STATE.current_frame_jpeg = buf.tobytes()

                    # Display window if requested
                    if args.display and cv2:
                        cv2.imshow("BEUM Hardware Inspector", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                else:
                    time.sleep(0.05)
            else:
                time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        term_stop.set()
        gps_mon.stop()
        cam.release()
        if args.display and cv2:
            cv2.destroyAllWindows()
        if web_server:
            web_server.shutdown()
        print("\nHardware check finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
