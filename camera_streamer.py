"""
Live Camera & Video Object Detection Streamer for DrainSight Dashboard.
Provides real-time YOLO inference, visual overlay (segmentation + bbox),
MJPEG multipart video streaming, and single-frame detection API.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from pathlib import Path
from typing import Any, Generator

import cv2
import numpy as np

LOGGER = logging.getLogger("CameraStreamer")


class LiveCameraStreamer:
    """
    Manages video capture (video file replay or hardware camera),
    continuous YOLO segmentation / detection, visual annotation, and MJPEG streaming.
    """

    def __init__(
        self,
        video_source: str | int | None = None,
        model_path: str | Path | None = None,
        target_fps: float = 20.0,
        conf_thresh: float = 0.20,
    ) -> None:
        self.project_root = Path(__file__).resolve().parent
        self.target_fps = max(1.0, min(60.0, float(target_fps)))
        self.conf_thresh = float(conf_thresh)
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Find default video source if None
        if video_source is None:
            road_video = self.project_root / "data/2026-09-04_04-31-48/V20260904_131914000_79E42EB0-91A0-4F27-A3F0-BF6673616688.MOV"
            test_drain = self.project_root / "data/e2e_test_drain.mp4"
            if test_drain.exists():
                self.video_source = str(test_drain)
            elif road_video.exists():
                self.video_source = str(road_video)
            else:
                self.video_source = "0"
        else:
            self.video_source = str(video_source)

        # Find default model path if None
        if model_path is None:
            cand_paths = [
                self.project_root / "models/edge_exports/best-seg-2class_320.onnx",
                self.project_root / "models/best-seg-2class.pt",
                self.project_root / "models/deploy/best-seg-canonical.onnx",
                self.project_root / "yolov8n-seg.pt",
            ]
            self.model_path = next((str(p) for p in cand_paths if p.exists()), "")
        else:
            self.model_path = str(model_path)

        self.model: Any | None = None
        self._load_model()

        # Latest state
        self.latest_jpeg: bytes | None = None
        self.latest_metadata: dict[str, Any] = {
            "fps": 0.0,
            "latency_ms": 0.0,
            "frame_count": 0,
            "detections": [],
            "occlusion_pct": 0.0,
            "status": "normal",
            "source": self.video_source,
            "model": Path(self.model_path).name if self.model_path else "None",
        }

    def _load_model(self) -> None:
        if not self.model_path or not Path(self.model_path).exists():
            LOGGER.warning("No model found at %s. Running in overlay-only mode.", self.model_path)
            self.model = None
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            # Warm up
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=320, verbose=False)
            LOGGER.info("YOLO Model loaded successfully from %s", self.model_path)
        except Exception as exc:
            LOGGER.error("Failed to load YOLO model: %s", exc)
            self.model = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        LOGGER.info("LiveCameraStreamer started with source: %s", self.video_source)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        LOGGER.info("LiveCameraStreamer stopped")

    def set_source(self, source: str) -> None:
        source_str = str(source).strip()
        if (source_str.startswith("http://") or source_str.startswith("https://")) and ":8080" in source_str:
            if not source_str.endswith("/stream.mjpg"):
                source_str = source_str.rstrip("/") + "/stream.mjpg"

        with self._lock:
            self.video_source = source_str
            self.latest_metadata["source"] = self.video_source

    def set_confidence(self, conf: float) -> None:
        with self._lock:
            self.conf_thresh = max(0.05, min(0.95, float(conf)))

    @property
    def confidence_threshold(self) -> float:
        with self._lock:
            return self.conf_thresh

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            meta = dict(self.latest_metadata)
            meta["is_running"] = self._running
            return meta

    def _worker(self) -> None:
        cap = None
        current_source = None
        frame_idx = 0
        prev_time = time.time()
        fps_smooth = self.target_fps

        while self._running:
            with self._lock:
                src = self.video_source

            if cap is None or src != current_source:
                if cap is not None:
                    cap.release()
                current_source = src
                try:
                    src_val = int(current_source) if current_source.isdigit() else current_source
                    cap = cv2.VideoCapture(src_val)
                    if not cap.isOpened():
                        LOGGER.warning("Could not open video source %s, retrying...", current_source)
                        time.sleep(1.0)
                        continue
                except Exception as exc:
                    LOGGER.error("Error opening capture source %s: %s", current_source, exc)
                    time.sleep(1.0)
                    continue

            ret, frame = cap.read()
            if not ret or frame is None:
                # Loop video files automatically
                if isinstance(current_source, str) and Path(current_source).exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                    continue
                else:
                    # Release capture if network stream dropped to trigger clean reconnect
                    if isinstance(current_source, str) and (
                        current_source.startswith("http://")
                        or current_source.startswith("https://")
                        or current_source.startswith("rtsp://")
                    ):
                        if cap is not None:
                            cap.release()
                            cap = None
                        time.sleep(0.5)
                        continue
                    time.sleep(0.1)
                    continue

            # Downscale high-resolution frames (e.g. 1080x1920) to max 640px to eliminate latency & buffer bloat
            h, w = frame.shape[:2]
            max_dim = 640
            if max(h, w) > max_dim:
                scale = float(max_dim) / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            frame_idx += 1
            start_infer = time.time()

            # Perform YOLO inference & annotation
            annotated_frame, detections, occ_pct, status_str = self._process_frame(frame)
            infer_latency = (time.time() - start_infer) * 1000.0

            # Calculate FPS
            now = time.time()
            dt = max(0.001, now - prev_time)
            prev_time = now
            inst_fps = 1.0 / dt
            fps_smooth = 0.85 * fps_smooth + 0.15 * inst_fps

            # Draw HUD overlay (Top Bar)
            self._draw_hud(annotated_frame, fps_smooth, infer_latency, occ_pct, status_str)

            # Encode to JPEG (quality 65 for ultra-fast compression and low bandwidth)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
            _, buf = cv2.imencode(".jpg", annotated_frame, encode_param)
            jpeg_bytes = buf.tobytes()

            with self._lock:
                self.latest_jpeg = jpeg_bytes
                self.latest_metadata = {
                    "fps": round(fps_smooth, 1),
                    "latency_ms": round(infer_latency, 1),
                    "frame_count": frame_idx,
                    "detections": detections,
                    "occlusion_pct": round(occ_pct, 1),
                    "status": status_str,
                    "source": Path(current_source).name if Path(str(current_source)).exists() else str(current_source),
                    "model": Path(self.model_path).name if self.model_path else "None",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

            # Pace loop to target FPS
            target_period = 1.0 / self.target_fps
            elapsed = time.time() - start_infer
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

        if cap is not None:
            cap.release()

    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]], float, str]:
        annotated = frame.copy()
        h, w = frame.shape[:2]
        detections: list[dict[str, Any]] = []
        occ_pct = 0.0
        status_str = "normal"

        if self.model is None:
            return annotated, detections, occ_pct, status_str

        try:
            with self._lock:
                conf = self.conf_thresh

            results = self.model.predict(
                frame,
                imgsz=320,
                conf=conf,
                verbose=False,
            )
            if not results:
                return annotated, detections, occ_pct, status_str

            result = results[0]
            names = getattr(self.model, "names", {})

            # 1. Draw segmentation masks if available
            masks = getattr(result, "masks", None)
            if masks is not None and hasattr(masks, "xy"):
                overlay = annotated.copy()
                for idx, poly in enumerate(masks.xy):
                    if len(poly) == 0:
                        continue
                    box = result.boxes[idx]
                    cls_id = int(box.cls[0].item())
                    # color: drain_area (green: 0, 200, 0), drain_full/obstacle (red: 0, 0, 220)
                    color = (0, 0, 230) if cls_id == 1 or "full" in str(names.get(cls_id, "")).lower() else (0, 220, 50)
                    pts = np.asarray(poly, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

            # 2. Draw bounding boxes and collect detection objects
            for idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, f"class_{cls_id}")
                box_conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                is_blockage = cls_id == 1 or "full" in cls_name.lower()
                box_color = (0, 0, 240) if is_blockage else (0, 210, 50)

                # Draw rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                # Label tag
                label = f"{cls_name} {box_conf*100:.1f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), box_color, -1)
                cv2.putText(annotated, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(box_conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "is_blockage": is_blockage,
                })

            # Determine detection status
            full_boxes = [d for d in detections if d["is_blockage"]]
            area_boxes = [d for d in detections if not d["is_blockage"]]

            if full_boxes:
                status_str = "critical"
            elif area_boxes:
                status_str = "normal"
            else:
                status_str = "normal"

            occ_pct = 0.0

        except Exception as exc:
            LOGGER.error("Frame processing error: %s", exc)

        return annotated, detections, occ_pct, status_str

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        occ_pct: float,
        status_str: str,
    ) -> None:
        h, w = frame.shape[:2]
        hud_h = 34 if w < 550 else 42
        # Top banner background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, hud_h), (w, hud_h), (51, 65, 85), 1)

        # Status badge color
        stat_color = (0, 200, 50)
        stat_kor = "정상"
        if status_str == "critical":
            stat_color = (0, 0, 230)
            stat_kor = "이상 감지(심각)"
        elif status_str == "warning":
            stat_color = (0, 165, 255)
            stat_kor = "주의"

        # Text elements
        ts = time.strftime("%H:%M:%S")
        if w < 550:
            title = f"AI {fps:.0f}FPS | {latency_ms:.0f}ms | {stat_kor}"
            cv2.putText(frame, title, (8, int(hud_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 250, 252), 1, cv2.LINE_AA)
        else:
            title = f"BEUM AI Vision | {ts}"
            metrics = f"FPS: {fps:.1f} | Latency: {latency_ms:.0f}ms | 상태: {stat_kor}"
            cv2.putText(frame, title, (12, int(hud_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (248, 250, 252), 1, cv2.LINE_AA)
            (mw, _), _ = cv2.getTextSize(metrics, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.putText(frame, metrics, (max(w - mw - 12, 220), int(hud_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, stat_color, 1, cv2.LINE_AA)

    def get_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """Yields multipart MJPEG frame chunks with Content-Length to prevent socket buffer bloat."""
        # Ensure streamer is running
        if not self._running:
            self.start()

        blank_frame = None
        last_frame_count = -1
        while True:
            with self._lock:
                jpeg = self.latest_jpeg
                frame_count = self.latest_metadata.get("frame_count", 0)

            if jpeg is None:
                if blank_frame is None:
                    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(dummy, "Starting Camera Stream...", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    _, buf = cv2.imencode(".jpg", dummy)
                    blank_frame = buf.tobytes()
                jpeg = blank_frame

            # Skip if frame hasn't changed to prevent buffer bloat in network socket
            if frame_count == last_frame_count and jpeg != blank_frame:
                time.sleep(0.015)
                continue

            last_frame_count = frame_count

            header = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
            )
            yield header + jpeg + b"\r\n"
            time.sleep(1.0 / (self.target_fps * 1.5))

    def detect_single_image(self, image_bytes: bytes, conf: float | None = None) -> dict[str, Any]:
        """Runs YOLO detection on a single uploaded image or webcam frame."""
        t0 = time.time()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "error": "Failed to decode image"}

        h, w = frame.shape[:2]
        if max(h, w) > 960:
            scale = 960.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        if conf is not None:
            old_conf = self.conf_thresh
            self.conf_thresh = conf
            annotated, detections, occ_pct, status_str = self._process_frame(frame)
            self.conf_thresh = old_conf
        else:
            annotated, detections, occ_pct, status_str = self._process_frame(frame)

        latency_ms = (time.time() - t0) * 1000.0
        self._draw_hud(annotated, 1000.0 / max(1.0, latency_ms), latency_ms, occ_pct, status_str)

        _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        b64_img = base64.b64encode(buf).decode("utf-8")

        return {
            "ok": True,
            "latency_ms": round(latency_ms, 1),
            "detections": detections,
            "occlusion_pct": round(occ_pct, 1),
            "status": status_str,
            "annotated_image_base64": f"data:image/jpeg;base64,{b64_img}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
