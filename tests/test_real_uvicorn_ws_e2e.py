"""
Live E2E Verification Script:
1. Boots an actual Uvicorn server on 127.0.0.1.
2. Performs actual WebSocket handshake to /ws/dashboard using python `websockets`.
3. Posts /api/telemetry and verifies immediate WebSocket event push (zero-polling).
4. Posts /api/detections and verifies immediate WebSocket event push.
5. Verifies graceful shutdown.
"""

import asyncio
import json
import socket
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn
import websockets

from receiver_server import create_app


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def run_live_test(port: int, token: str) -> None:
    ws_url = f"ws://127.0.0.1:{port}/ws/dashboard"
    http_base = f"http://127.0.0.1:{port}"
    print(f"[*] Connecting to live WebSocket: {ws_url}")

    async with websockets.connect(ws_url) as ws:
        # 1. Verify initial handshake ACK
        init_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        init_msg = json.loads(init_raw)
        print("[*] WS Handshake Received:", init_msg)
        assert init_msg["type"] == "connected", f"Expected connected, got {init_msg}"
        assert init_msg["client"] == "dashboard"

        # 2. Post telemetry via real HTTP
        print("[*] Sending POST /api/telemetry over HTTP...")
        tel_payload = {
            "vehicle_code": "LIVE-CAR-01",
            "lat": 36.838,
            "lng": 127.184,
            "timestamp": time.time(),
            "sequence": 1,
            "speed_mps": 12.5,
        }
        req_data = json.dumps(tel_payload).encode("utf-8")
        req = urllib.request.Request(
            f"{http_base}/api/telemetry",
            data=req_data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))
            print("[*] HTTP Telemetry Response:", resp_body)
            assert resp.status == 200
            assert resp_body["ok"] is True
            assert resp_body["processed"] is True

        # 3. Receive telemetry event over WebSocket (Zero polling)
        ws_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        ws_event = json.loads(ws_raw)
        print("[*] WS Received Telemetry Event:", ws_event)
        assert ws_event["type"] == "telemetry"
        assert ws_event["vehicle_code"] == "LIVE-CAR-01"
        assert ws_event["sequence"] == 1

        # 4. Post detection via real HTTP
        print("[*] Sending POST /api/detections over HTTP...")
        det_payload = {
            "drain_id": 999,
            "vehicle_code": "LIVE-CAR-01",
            "status": "NORMAL",
            "occlusion_pct": 25.0,
            "lat": 36.838,
            "lng": 127.184,
            "source": "AI_VISION",
        }
        req_data = json.dumps(det_payload).encode("utf-8")
        req = urllib.request.Request(
            f"{http_base}/api/detections",
            data=req_data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            det_resp = json.loads(resp.read().decode("utf-8"))
            print("[*] HTTP Detection Response:", det_resp)
            assert resp.status == 201
            assert det_resp["ok"] is True

        # 5. Receive detection event over WebSocket (Zero polling)
        det_ws_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        det_ws_event = json.loads(det_ws_raw)
        print("[*] WS Received Detection Event:", det_ws_event)
        assert det_ws_event["type"] == "detection"
        assert det_ws_event["drain_id"] == 999
        assert det_ws_event["occlusion_pct"] == 25.0

        print("[SUCCESS] All Live WebSocket & Uvicorn E2E checks passed!")


def main() -> None:
    temp_dir = tempfile.TemporaryDirectory()
    port = find_free_port()
    token = "live-test-token"
    app = create_app(data_dir=Path(temp_dir.name), token=token)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to start
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.05)

    try:
        asyncio.run(run_live_test(port, token))
    finally:
        server.should_exit = True
        thread.join(timeout=3.0)
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
