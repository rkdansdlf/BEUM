#!/usr/bin/env python
"""
Standard Canonical Segmentation Benchmark Evaluation.
Evaluates any YOLOv8 segmentation model on a dataset using the canonical contract
(overlap_mask=False, conf=0.001, iou=0.70) and outputs verified JSON metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from ultralytics import YOLO

from tools.segmentation_contract import (
    canonical_val_args,
    extract_class_metrics,
)


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def evaluate_model(
    model_path: str | Path,
    data_yaml: str | Path,
    split: str = "val",
    imgsz: int = 640,
    device: str = "cpu",
    output_json: str | Path | None = None,
) -> dict:
    model_path = Path(model_path)
    data_yaml = Path(data_yaml)

    print("=" * 80)
    print(f"CANONICAL EVALUATION: {model_path.name}")
    print(f"Data: {data_yaml} (split: {split}) | ImgSz: {imgsz} | Device: {device}")
    print("Contract: overlap_mask=False, conf=0.001, iou=0.70")
    print("=" * 80)

    model = YOLO(str(model_path))
    val_args = canonical_val_args()

    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        verbose=False,
        **val_args,
    )

    box_metrics = extract_class_metrics(results.box, model.names)
    mask_metrics = extract_class_metrics(results.seg, model.names)

    report = {
        "schema_version": 1,
        "model": {
            "path": str(model_path),
            "sha256": get_sha256(model_path) if model_path.is_file() else "unknown",
            "names": model.names,
        },
        "dataset": {
            "yaml": str(data_yaml),
            "sha256": get_sha256(data_yaml) if data_yaml.is_file() else "unknown",
            "split": split,
        },
        "validation_args": {
            "imgsz": imgsz,
            "device": device,
            **val_args,
        },
        "metrics": {
            "box": box_metrics,
            "mask": mask_metrics,
        },
    }

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved evaluation report to {out_p}")

    print("\nSummary Metrics:")
    print(f"  Box  mAP50: {box_metrics['overall'].get('map50')} | mAP50-95: {box_metrics['overall'].get('map50_95')}")
    print(f"  Mask mAP50: {mask_metrics['overall'].get('map50')} | mAP50-95: {mask_metrics['overall'].get('map50_95')}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO segmentation model on canonical split")
    parser.add_argument("--model", type=str, default="models/best-seg.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default="dataset/canonical/canonical-3class.yaml", help="Path to data YAML")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val/test)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/mps/0)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        output_json=args.output,
    )
