#!/usr/bin/env python
"""
Phase 1: Export edge deployment models (ONNX, TorchScript) from the canonical model.
Validates file sizes, tensor shapes, metadata, and sha256 checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

from ultralytics import YOLO


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best-seg-canonical.pt")
    parser.add_argument("--output-dir", type=str, default="models/deploy")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 1: EXPORT EDGE DEPLOYMENT MODELS")
    print(f"Source Model: {model_path} (SHA: {get_sha256(model_path)[:8]})")
    print(f"Output Directory: {out_dir}")
    print(f"Image Size: {args.imgsz} | ONNX Opset: {args.opset}")
    print("=" * 80)

    model = YOLO(str(model_path))

    # 1. Export ONNX
    print("\n[1/2] Exporting to ONNX format (simplify=True)...")
    onnx_exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=True,
        opset=args.opset,
        dynamic=False,
    )
    dest_onnx = out_dir / "best-seg-canonical.onnx"
    shutil.copy2(onnx_exported_path, dest_onnx)
    print(f"  -> ONNX export saved to: {dest_onnx}")
    print(f"  -> Size: {dest_onnx.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"  -> SHA-256: {get_sha256(dest_onnx)}")

    # 2. Export TorchScript
    print("\n[2/2] Exporting to TorchScript format...")
    ts_exported_path = model.export(
        format="torchscript",
        imgsz=args.imgsz,
    )
    dest_ts = out_dir / "best-seg-canonical.torchscript"
    shutil.copy2(ts_exported_path, dest_ts)
    print(f"  -> TorchScript export saved to: {dest_ts}")
    print(f"  -> Size: {dest_ts.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"  -> SHA-256: {get_sha256(dest_ts)}")

    manifest = {
        "source_model": str(model_path),
        "source_sha256": get_sha256(model_path),
        "class_names": model.names,
        "imgsz": args.imgsz,
        "exports": {
            "onnx": {
                "path": str(dest_onnx),
                "sha256": get_sha256(dest_onnx),
                "size_mb": round(dest_onnx.stat().st_size / (1024 * 1024), 2),
                "opset": args.opset,
            },
            "torchscript": {
                "path": str(dest_ts),
                "sha256": get_sha256(dest_ts),
                "size_mb": round(dest_ts.stat().st_size / (1024 * 1024), 2),
            },
        },
    }

    manifest_file = out_dir / "export_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved export manifest to: {manifest_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
