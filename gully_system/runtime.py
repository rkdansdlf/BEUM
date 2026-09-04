"""End-to-end edge pipeline orchestrating camera, detector, blockage, and uploader."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from gully_system.blockage import BlockageAnalyzer, BlockageEventGate, BlockageMetrics
from gully_system.camera import CameraSource
from gully_system.config import SystemConfig
from gully_system.data_manager import StorageQueue
from gully_system.detector import YOLODetector
from gully_system.gps import (
    BaseGPSProvider,
    GPSFix,
    NullGPSProvider,
    ReplayGPSProvider,
    SerialGPSProvider,
    UDPGPSProvider,
)
from gully_system.policy import PolicyDecision, RuleBasedPolicy, SafePolicy, TablePolicy
from gully_system.roi import ROI
from gully_system.sensors import SensorProvider, SensorSnapshot, SystemSensorProvider
from gully_system.temporal_filter import TemporalFilter
from gully_system.types import Detection
from gully_system.uploader import HttpUploader, NullUploader, UploadWorker

LOGGER = logging.getLogger(__name__)


class GullyRuntime:
    def __init__(
        self,
        config: SystemConfig,
        sensor_provider: SensorProvider | None = None,
        detector: YOLODetector | None = None,
    ) -> None:
        self.config = config
        self.sensor_provider = sensor_provider or SystemSensorProvider()
        self.detector = detector or YOLODetector(config.detector)
        self.camera = CameraSource(
            source=config.source,
            realtime=config.realtime_source,
            buffer_size=config.camera_buffer_size,
        )
        self.roi = ROI(config.roi_points)
        self.expanded_roi = self.roi.scaled(config.roi_expanded_scale)
        self.temporal_filter = TemporalFilter(
            min_hits=config.temporal_hits,
            max_missed=config.temporal_max_missed,
            iou_threshold=config.temporal_iou,
        )
        self.analyzer = BlockageAnalyzer(
            gully_class_names=config.blockage.gully_class_names,
            obstacle_class_names=config.blockage.obstacle_class_names,
            warning_percent=config.blockage.warning_percent,
            critical_percent=config.blockage.critical_percent,
        )
        self.event_gate = BlockageEventGate(
            change_percent=config.blockage.event_change_percent,
            cooldown_s=config.blockage.event_cooldown_s,
        )
        self.queue = StorageQueue(
            base_dir=config.storage.spool_dir,
            max_bytes=int(config.storage.max_gb * 1024 * 1024 * 1024),
        )
        uploader = HttpUploader(
            url=config.upload.url,
            token=config.upload.token,
            timeout_s=config.upload.timeout_s,
        ) if config.upload.url else NullUploader()
        self.upload_worker = UploadWorker(
            queue=self.queue,
            uploader=uploader,
            base_backoff_s=config.upload.base_backoff_s,
            max_backoff_s=config.upload.max_backoff_s,
        )
        self.gps_provider = self._create_gps_provider()

        rule_policy = RuleBasedPolicy(config.policy)
        if config.policy.model_path and Path(config.policy.model_path).exists():
            base_policy = TablePolicy(config.policy.model_path, config.policy)
            self.policy = SafePolicy(base_policy, rule_policy, config.policy)
        else:
            self.policy = rule_policy

    def _create_gps_provider(self) -> BaseGPSProvider:
        provider = self.config.gps.provider.lower()
        if provider == "replay":
            return ReplayGPSProvider(
                csv_path=self.config.gps.csv_path,
                sample_period_s=self.config.gps.sample_period_s,
                replay_speed=self.config.gps.replay_speed,
            )
        if provider == "serial":
            return SerialGPSProvider(
                port=self.config.gps.serial_port,
                baudrate=self.config.gps.baudrate,
                timeout_s=self.config.gps.serial_timeout_s,
            )
        if provider == "udp":
            return UDPGPSProvider(
                host=self.config.gps.udp_host,
                port=self.config.gps.udp_port,
            )
        return NullGPSProvider()

    def _encode_jpeg(self, frame: Any, quality: int) -> bytes | None:
        try:
            import cv2
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def _draw_detections(self, frame: Any, detections: tuple[Detection, ...]) -> None:
        try:
            import cv2
            for d in detections:
                x1, y1, x2, y2 = [int(v) for v in d.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                label = f"{d.class_name} {d.confidence:.2f}"
                cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception:
            pass

    def _event_payload(
        self,
        detections: tuple[Detection, ...],
        decision: PolicyDecision,
        snapshot: SensorSnapshot,
        gps_fix: GPSFix | None,
        blockage: BlockageMetrics,
    ) -> dict[str, Any]:
        return {
            "event_type": "gully_blockage",
            "source": self.config.source,
            "detections": [d.to_dict() for d in detections],
            "blockage": blockage.to_dict(),
            "policy": {
                "mode": decision.mode,
                "roi_profile": decision.roi_profile,
                "reason": decision.reason,
            },
            "sensors": {
                "battery_pct": snapshot.battery_pct,
                "rain_level": snapshot.rain_level,
                "water_level": snapshot.water_level,
                "cpu_temp_c": snapshot.cpu_temp_c,
                "network_ok": snapshot.network_ok,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(snapshot.timestamp or time.time())),
            },
            "gps": gps_fix.to_dict() if gps_fix else None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def run(self, max_frames: int | None = None, display: bool = False) -> dict[str, int]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to run the camera pipeline") from exc

        limit = max_frames if max_frames is not None and max_frames > 0 else self.config.max_frames
        self.camera.open()
        self.gps_provider.start()

        writer = None
        if self.config.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.config.output_path, fourcc, max(10, int(self.camera.fps)), (1080, 1920))

        frame_count = 0
        inference_count = 0
        event_count = 0
        next_inference_at = 0.0
        next_policy_at = 0.0
        next_upload_at = 0.0
        active_roi = self.roi
        current_decision = PolicyDecision(mode="medium", inference_interval_s=1.0)
        current_snapshot = self.sensor_provider.read()

        try:
            while limit == 0 or frame_count < limit:
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    break
                now = time.monotonic()

                if now >= next_policy_at:
                    current_snapshot = self.sensor_provider.read()
                    current_decision = self.policy.decide(current_snapshot)
                    active_roi = self.expanded_roi if current_decision.roi_profile in ("wide", "expanded") else self.roi
                    next_policy_at = now + self.config.policy_interval_s

                raw_detections = []
                if now >= next_inference_at:
                    raw_detections = self.detector.predict(frame)
                    inference_count += 1
                    next_inference_at = now + current_decision.inference_interval_s

                in_roi = active_roi.filter(tuple(raw_detections)) if raw_detections else ()
                temporal = self.temporal_filter.update(list(in_roi))
                blockage = self.analyzer.analyze(frame, temporal.validated)

                if self.event_gate.should_emit(blockage, now=now):
                    gps_fix = self.gps_provider.latest()
                    event = self._event_payload(temporal.validated, current_decision, current_snapshot, gps_fix, blockage)
                    evidence_bytes = self._encode_jpeg(frame, self.config.storage.jpeg_quality) if self.config.storage.save_evidence else None
                    self.queue.enqueue(event, evidence_bytes=evidence_bytes)
                    event_count += 1

                if now >= next_upload_at:
                    self.upload_worker.flush_once(max_items=5)
                    next_upload_at = now + self.config.upload.flush_interval_s

                active_roi.draw(frame)
                self._draw_detections(frame, temporal.validated)
                self.analyzer.draw(frame, blockage)

                if writer:
                    writer.write(frame)

                if display:
                    cv2.imshow("gully-monitor", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
        finally:
            if writer:
                writer.release()
            self.camera.release()
            self.gps_provider.stop()
            if display:
                cv2.destroyAllWindows()
            try:
                self.upload_worker.flush_once(max_items=50)
            except Exception:
                pass

        LOGGER.info(
            "Finished: frames=%d inferences=%d events=%d pending=%d storage=%d bytes",
            frame_count,
            inference_count,
            event_count,
            self.queue.pending_count(),
            self.queue.usage_bytes(),
        )

        return {
            "frames": frame_count,
            "inferences": inference_count,
            "events": event_count,
            "pending": self.queue.pending_count(),
            "dropped_frames": self.camera.dropped_frames,
        }
