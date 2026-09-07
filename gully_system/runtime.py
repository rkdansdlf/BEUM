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
from gully_system.health import SystemHealthError, SystemHealthReport, run_health_check
from gully_system.policy import PolicyDecision, RuleBasedPolicy, SafePolicy, TablePolicy
from gully_system.roi import ROI
from gully_system.sensors import SensorProvider, SensorSnapshot, SystemSensorProvider
from gully_system.temporal_filter import TemporalFilter
from gully_system.triage import ActiveLearningTriager
from gully_system.types import Detection
from gully_system.uploader import HttpUploader, NullUploader, UploadWorker

LOGGER = logging.getLogger(__name__)


class GullyRuntime:
    def __init__(
        self,
        config: SystemConfig,
        sensor_provider: SensorProvider | None = None,
        detector: YOLODetector | None = None,
        triager: ActiveLearningTriager | None = None,
    ) -> None:
        self.config = config
        if sensor_provider:

            self.sensor_provider = sensor_provider
        elif hasattr(config, "sensors"):
            self.sensor_provider = SystemSensorProvider(
                battery_file=config.sensors.battery_file or None,
                rain_file=config.sensors.rain_file or None,
                water_file=config.sensors.water_file or None,
                rain_pin=config.sensors.rain_sensor_pin,
                water_pin=config.sensors.water_sensor_pin,
                network_check_host=config.sensors.network_check_host,
                network_check_port=config.sensors.network_check_port,
                network_check_timeout_s=config.sensors.network_check_timeout_s,
            )
        else:
            self.sensor_provider = SystemSensorProvider()

        self.detector = detector or YOLODetector(config.detector)
        self.triager = triager
        if self.triager is None and getattr(config, "triage", None) and config.triage.enabled:
            self.triager = ActiveLearningTriager(
                output_dir=config.triage.output_dir,
                uncertain_conf_range=config.triage.uncertain_conf_range,
                borderline_warning_range=config.triage.borderline_warning_range,
                borderline_critical_range=config.triage.borderline_critical_range,
                cooldown_s=config.triage.cooldown_s,
                max_candidates=config.triage.max_candidates,
                save_images=config.triage.save_images,
                jpeg_quality=config.triage.jpeg_quality,
            )

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

        # Synchronize blockage analyzer classes with model-config mapping if available
        analyzer_gullies = config.blockage.gully_class_names
        analyzer_obstacles = config.blockage.obstacle_class_names
        if hasattr(self.detector, "mapping_result") and self.detector.mapping_result:
            if self.detector.mapping_result.resolved_gully_classes:
                analyzer_gullies = self.detector.mapping_result.resolved_gully_classes
            if self.detector.mapping_result.resolved_obstacle_classes:
                analyzer_obstacles = self.detector.mapping_result.resolved_obstacle_classes

        self.analyzer = BlockageAnalyzer(
            gully_class_names=analyzer_gullies,
            obstacle_class_names=analyzer_obstacles,
            warning_percent=config.blockage.warning_percent,
            critical_percent=config.blockage.critical_percent,
            require_gully_presence=config.blockage.require_gully_presence,
            standalone_min_conf=config.blockage.standalone_min_conf,
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

        if config.run_preflight:
            report = self.get_health()
            if not report.can_start:
                failed_items = [
                    f"{k}: {v.message}"
                    for k, v in report.components.items()
                    if v.status.value == "unhealthy"
                ]
                raise SystemHealthError(
                    f"Pre-flight health check failed (status={report.status.value}): "
                    + "; ".join(failed_items)
                )

    def get_health(self, check_network: bool = False) -> SystemHealthReport:
        """Run system health check."""
        return run_health_check(self.config, check_network=check_network, sensor_provider=self.sensor_provider)

    def update_roi_for_resolution(self, width: int, height: int) -> None:
        """Update roi and expanded_roi based on target frame resolution."""
        base_roi = ROI(self.config.roi_points)
        self.roi = base_roi.scale_to_resolution(
            target_width=width,
            target_height=height,
            base_resolution=self.config.roi_base_resolution,
        )
        self.expanded_roi = self.roi.scaled(self.config.roi_expanded_scale)

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
            import numpy as np

            if not detections:
                return

            height, width = frame.shape[:2]

            class_colors = {
                "drain_area": (0, 220, 0),    # Clear drain grating (Green)
                "drain_full": (0, 0, 230),    # Blockage / clogged area (Red)
                "gully": (0, 220, 0),
                "debris": (0, 0, 230),
                "trash": (0, 0, 230),
                "leaf": (0, 140, 255),
            }
            default_color = (255, 120, 0)     # Blue/Orange

            # 1. Mutually exclusive mask segmentation (prevent muddy blended layers)
            obstacle_mask = np.zeros((height, width), dtype=bool)
            gully_mask = np.zeros((height, width), dtype=bool)
            other_masks: list[tuple[np.ndarray, tuple[int, int, int]]] = []

            for d in detections:
                if d.mask is not None and hasattr(d.mask, "shape") and d.mask.shape[:2] == (height, width):
                    m_bool = d.mask > 0
                    if not np.any(m_bool):
                        continue
                    c_name = d.class_name.lower()
                    if c_name in ("drain_full", "debris", "trash"):
                        obstacle_mask |= m_bool
                    elif c_name in ("drain_area", "gully"):
                        gully_mask |= m_bool
                    else:
                        other_masks.append((m_bool, class_colors.get(c_name, default_color)))

            # Mutually exclusive partition:
            # Blocked area is pure obstacle_mask (Red)
            # Clear grating area is gully_mask EXCLUDING obstacle_mask (Green)
            clear_gully_mask = gully_mask & (~obstacle_mask)

            overlay = frame.copy()
            has_overlay = False

            if np.any(clear_gully_mask):
                overlay[clear_gully_mask] = (0, 220, 0)
                has_overlay = True

            if np.any(obstacle_mask):
                overlay[obstacle_mask] = (0, 0, 230)
                has_overlay = True

            for m_bool, col in other_masks:
                valid_m = m_bool & (~obstacle_mask)
                if np.any(valid_m):
                    overlay[valid_m] = col
                    has_overlay = True

            if has_overlay:
                cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, dst=frame)
                # Sharp contour outlines for visual distinction
                if np.any(clear_gully_mask):
                    contours, _ = cv2.findContours(clear_gully_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, (0, 220, 0), 2)
                if np.any(obstacle_mask):
                    contours, _ = cv2.findContours(obstacle_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, (0, 0, 230), 2)

            # 2. Collision-free bounding box and label badge rendering
            occupied_badges: list[tuple[int, int, int, int]] = []
            sorted_dets = sorted(detections, key=lambda x: (x.class_name != "drain_full", x.bbox[1], x.bbox[0]))

            for d in sorted_dets:
                c_name = d.class_name.lower()
                color = class_colors.get(c_name, default_color)
                x1, y1, x2, y2 = [int(v) for v in d.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{d.class_name} {d.confidence:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                bw = tw + 8
                bh = th + baseline + 6

                # Candidate slots in order of preference
                candidates = [
                    (x1, y1 - bh, x1 + bw, y1),                             # 1. Above top-left
                    (x1, y1, x1 + bw, y1 + bh),                             # 2. Inside top-left
                    (max(0, x2 - bw), y1 - bh, max(0, x2 - bw) + bw, y1),   # 3. Above top-right
                    (max(0, x2 - bw), y1, max(0, x2 - bw) + bw, y1 + bh),   # 4. Inside top-right
                    (x1, y2, x1 + bw, y2 + bh),                             # 5. Below bottom-left
                    (x1, max(0, y1 - 2 * bh - 2), x1 + bw, max(0, y1 - bh - 2)), # 6. Stacked above
                ]

                chosen_slot = None
                for cx1, cy1, cx2, cy2 in candidates:
                    if cy1 < 0 or cy2 > height or cx1 < 0 or cx2 > width:
                        continue
                    collision = False
                    for ox1, oy1, ox2, oy2 in occupied_badges:
                        if not (cx2 + 2 <= ox1 or cx1 >= ox2 + 2 or cy2 + 2 <= oy1 or cy1 >= oy2 + 2):
                            collision = True
                            break
                    if not collision:
                        chosen_slot = (cx1, cy1, cx2, cy2)
                        break

                if chosen_slot is None:
                    chosen_slot = (
                        min(max(0, x1), width - bw),
                        min(max(0, y1), height - bh),
                        min(max(0, x1) + bw, width),
                        min(max(0, y1) + bh, height),
                    )

                bx1, by1, bx2, by2 = chosen_slot
                occupied_badges.append((bx1, by1, bx2, by2))

                # Draw solid badge with crisp 1px border
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    label,
                    (bx1 + 4, by1 + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
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
                "interval_s": decision.inference_interval_s,
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

        if self.camera.width and self.camera.height:
            self.update_roi_for_resolution(self.camera.width, self.camera.height)

        writer = None
        if self.config.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            frame_w = self.camera.width or 640
            frame_h = self.camera.height or 480
            writer = cv2.VideoWriter(self.config.output_path, fourcc, max(10, int(self.camera.fps)), (frame_w, frame_h))

        frame_count = 0
        inference_count = 0
        event_count = 0
        harvested_count = 0

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

                if frame_count == 0 and hasattr(frame, "shape") and len(frame.shape) >= 2:
                    h, w = int(frame.shape[0]), int(frame.shape[1])
                    if w != self.camera.width or h != self.camera.height:
                        self.camera.width = w
                        self.camera.height = h
                        self.update_roi_for_resolution(w, h)
                        active_roi = self.expanded_roi if current_decision.roi_profile in ("wide", "expanded") else self.roi
                        if self.config.output_path:
                            if writer:
                                writer.release()
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(self.config.output_path, fourcc, max(10, int(self.camera.fps)), (frame_w, frame_h))

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

                # Active Learning Auto-Triage on clean frame before rendering overlays
                if self.triager and raw_detections:
                    cand = self.triager.evaluate_frame(
                        frame=frame,
                        raw_detections=raw_detections,
                        blockage=blockage,
                        gps_fix=self.gps_provider.latest(),
                    )
                    if cand:
                        harvested_count += 1

                # Annotate frame with ROI, detection boxes/masks, and blockage status
                active_roi.draw(frame)
                self._draw_detections(frame, temporal.validated)
                self.analyzer.draw(frame, blockage)

                if self.event_gate.should_emit(blockage, now=now):
                    gps_fix = self.gps_provider.latest()
                    event = self._event_payload(temporal.validated, current_decision, current_snapshot, gps_fix, blockage)
                    evidence_bytes = self._encode_jpeg(frame, self.config.storage.jpeg_quality) if self.config.storage.save_evidence else None
                    self.queue.enqueue(event, evidence_bytes=evidence_bytes)
                    event_count += 1

                if now >= next_upload_at:
                    self.upload_worker.flush_once(max_items=5)
                    next_upload_at = now + self.config.upload.flush_interval_s

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
            "Finished: frames=%d inferences=%d events=%d harvested=%d pending=%d storage=%d bytes",
            frame_count,
            inference_count,
            event_count,
            harvested_count,
            self.queue.pending_count(),
            self.queue.usage_bytes(),
        )

        return {
            "frames": frame_count,
            "inferences": inference_count,
            "events": event_count,
            "harvested_candidates": harvested_count,
            "pending": self.queue.pending_count(),
            "dropped_frames": self.camera.dropped_frames,
        }
