import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from gully_system.config import DetectorConfig
from gully_system.detector import YOLODetector


def test_exported_onnx_model_exists():
    onnx_path = Path("models/deploy/best-seg-canonical.onnx")
    assert onnx_path.is_file(), "ONNX model not found in models/deploy/"
    assert onnx_path.stat().st_size > 1_000_000, "ONNX model file is suspiciously small"


def test_config_pi_valid():
    cfg_path = Path("config.pi.json")
    assert cfg_path.is_file(), "config.pi.json not found"
    data = json.loads(cfg_path.read_text())
    assert data["detector"]["model_path"] in ("models/deploy/best-seg-canonical.onnx", "models/best-seg-2class_320.onnx")
    assert data["detector"]["device"] == "cpu"
    assert data["detector"]["confidence"] in (0.20, 0.45)


def test_detector_onnx_inference_parity():
    # Load both models
    cfg_pt = DetectorConfig(
        model_path="models/best-seg-canonical.pt",
        image_size=640,
        confidence=0.20,
        iou=0.45,
        device="cpu",
        class_names=["drain_area", "drain_full"]
    )
    cfg_onnx = DetectorConfig(
        model_path="models/deploy/best-seg-canonical.onnx",
        image_size=640,
        confidence=0.20,
        iou=0.45,
        device="cpu",
        class_names=["drain_area", "drain_full"]
    )

    det_pt = YOLODetector(cfg_pt)
    det_onnx = YOLODetector(cfg_onnx)

    sample_path = Path("dataset/canonical/images/val/images-1-_jpeg.rf.378aa7875e20dec918016b25ab3b97ad.jpg")
    img = cv2.imread(str(sample_path))
    assert img is not None

    dets_pt = det_pt.predict(img)
    dets_onnx = det_onnx.predict(img)

    # Class agreement check
    pt_classes = sorted([d.class_name for d in dets_pt])
    onnx_classes = sorted([d.class_name for d in dets_onnx])
    assert pt_classes == onnx_classes, f"Class mismatch: {pt_classes} vs {onnx_classes}"

    # Verify mask IoU parity for top detection
    assert len(dets_pt) > 0 and len(dets_onnx) > 0
    m_pt = dets_pt[0].mask
    m_onnx = dets_onnx[0].mask
    assert m_pt is not None and m_onnx is not None

    inter = np.logical_and(m_pt > 0, m_onnx > 0).sum()
    union = np.logical_or(m_pt > 0, m_onnx > 0).sum()
    mask_iou = inter / union if union > 0 else 0.0

    assert mask_iou >= 0.98, f"Mask IoU between PT and ONNX below tolerance: {mask_iou:.4f}"
