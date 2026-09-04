"""Ultralytics YOLO wrapper for object detection and segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from gully_system.config import DetectorConfig
from gully_system.types import Detection


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
        self.allowed_names = set(config.class_names)

    def predict(self, frame: object) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            imgsz=self.config.image_size,
            conf=self.config.confidence,
            iou=self.config.iou,
            device=self.config.device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        names: dict[int, str] = {}
        model_names = getattr(self.model, "names", {})
        if isinstance(model_names, dict):
            names = {int(k): str(v) for k, v in model_names.items()}
        elif isinstance(model_names, list):
            names = {index: str(v) for index, v in enumerate(model_names)}

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
        return detections

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
