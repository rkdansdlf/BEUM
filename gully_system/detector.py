"""Ultralytics YOLO wrapper for object detection and segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from gully_system.config import DetectorConfig
from gully_system.types import Detection
from gully_system.validator import (
    MappingResult,
    ModelMappingError,
    extract_model_classes,
    validate_and_resolve_mapping,
)


class YOLODetector:
    """Ultralytics adapter. The model is loaded only when the runtime starts."""

    def __init__(self, config: DetectorConfig) -> None:
        if not Path(config.model_path).exists():
            raise FileNotFoundError(f"Model file or export directory not found: {config.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the selected YOLO runtime before starting inference") from exc
        self.config = config
        self.model = YOLO(config.model_path)
        self.model_classes = extract_model_classes(self.model)

        names_attr = getattr(self.model, "names", {})
        if isinstance(names_attr, dict):
            self.names_map: dict[int, str] = {int(k): str(v) for k, v in names_attr.items()}
        elif isinstance(names_attr, (list, tuple)):
            self.names_map: dict[int, str] = {i: str(v) for i, v in enumerate(names_attr)}
        else:
            self.names_map: dict[int, str] = {}

        self.mapping_result: MappingResult = validate_and_resolve_mapping(
            model_classes=self.model_classes,
            config_class_names=config.class_names,
            mapping_mode=config.mapping_mode,
        )
        self.allowed_names = set(self.mapping_result.resolved_class_names)

    def predict(self, frame: object) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            imgsz=self.config.image_size,
            conf=self.config.confidence,
            iou=self.config.iou,
            agnostic_nms=self.config.agnostic_nms,
            device=self.config.device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        names = self.names_map

        detections: list[Detection] = []
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls[0].item())
            class_name = names.get(class_id, str(class_id))
            if self.allowed_names and class_name not in self.allowed_names:
                continue
            confidence = float(box.conf[0].item())
            bbox = tuple(float(v) for v in box.xyxy[0].tolist())
            mask = self._mask_for_box(result, index, frame)
            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    mask=mask,
                )
            )

        if self.config.agnostic_nms and len(detections) > 1:
            detections = self._apply_cross_class_nms(detections, iou_threshold=self.config.iou)

        return detections

    @staticmethod
    def _apply_cross_class_nms(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
        """Suppresses duplicate overlapping detections across different classes on the same physical object."""
        if len(detections) <= 1:
            return detections
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep: list[Detection] = []
        for det in sorted_dets:
            suppress = False
            for kept in keep:
                ix1 = max(det.bbox[0], kept.bbox[0])
                iy1 = max(det.bbox[1], kept.bbox[1])
                ix2 = min(det.bbox[2], kept.bbox[2])
                iy2 = min(det.bbox[3], kept.bbox[3])
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                if inter > 0.0:
                    a1 = max(0.0, det.bbox[2] - det.bbox[0]) * max(0.0, det.bbox[3] - det.bbox[1])
                    a2 = max(0.0, kept.bbox[2] - kept.bbox[0]) * max(0.0, kept.bbox[3] - kept.bbox[1])
                    union = a1 + a2 - inter
                    iou = (inter / union) if union > 0.0 else 0.0
                    if iou >= iou_threshold:
                        suppress = True
                        break
            if not suppress:
                keep.append(det)
        return keep

    @staticmethod
    def _mask_for_box(result: Any, index: int, frame: object) -> object | None:
        """Convert an Ultralytics polygon mask to a full-frame binary mask."""
        masks = getattr(result, "masks", None)
        if masks is None or not hasattr(masks, "xy"):
            return None
        try:
            import cv2
            import numpy as np
            height, width = frame.shape[:2]
            polygons = masks.xy
            if index >= len(polygons):
                return None
            polygon = np.asarray(polygons[index], dtype=np.int32)
            if polygon.size == 0:
                return None
            binary = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(binary, [polygon], 1)
            return binary
        except (AttributeError, ImportError, IndexError, TypeError, ValueError):
            return None
