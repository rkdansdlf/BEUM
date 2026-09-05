#!/usr/bin/env python
"""
Exports trained YOLOv8 segmentation models to ONNX for edge deployment
(Raspberry Pi 5 / Hailo-8 NPU / OpenVINO / TensorRT).
Validates inference using ONNX Runtime.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO

import time
import shutil

def export_model(
    model_path: str,
    output_dir: str = "models/edge_exports",
    imgsz: int = 640,
    opset: int = 12,
    simplify: bool = True,
    canonical_alias: str | None = None,
) -> str:
    pt_path = Path(model_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"EXPORTING TO ONNX: {pt_path}")
    print(f"Image Size: {imgsz}x{imgsz} | Opset: {opset} | Simplify: {simplify}")
    print("=" * 80)

    model = YOLO(str(pt_path))
    
    # Run ultralytics export
    exported_onnx_file = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,
    )

    exported_path = Path(exported_onnx_file)
    target_filename = f"{pt_path.stem}_{imgsz}.onnx"
    target_path = out_dir / target_filename
    
    shutil.copy2(exported_path, target_path)
    size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"\nModel exported successfully to: {target_path} ({size_mb:.2f} MB)")

    # If canonical alias requested (e.g. best-seg-2class), copy alias
    if canonical_alias:
        alias_path = out_dir / f"{canonical_alias}_{imgsz}.onnx"
        shutil.copy2(target_path, alias_path)
        print(f"Canonical alias created: {alias_path}")

    # Validate with ONNX checker
    onnx_model = onnx.load(str(target_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure check: PASSED ✅")

    # Run inference test & latency profile with ONNX Runtime
    session = ort.InferenceSession(str(target_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"Input Name: {inputs[0].name}, Shape: {inputs[0].shape}, Type: {inputs[0].type}")
    for out in outputs:
        print(f"Output Name: {out.name}, Shape: {out.shape}, Type: {out.type}")

    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    # Warmup
    for _ in range(3):
        session.run(None, {inputs[0].name: dummy_input})

    # Benchmark 10 iterations
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        ort_outs = session.run(None, {inputs[0].name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = np.mean(latencies)
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0
    print(f"Inference latency: {mean_lat:.2f} ms ({fps:.1f} FPS on CPU)")
    print(f"Inference test run with dummy input: SUCCESS ✅ (Outputs count: {len(ort_outs)})")
    for i, out in enumerate(ort_outs):
        print(f"  Output [{i}]: shape={out.shape}, min={out.min():.4f}, max={out.max():.4f}")

    print("=" * 80)
    return str(target_path)

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 weights to ONNX")
    parser.add_argument("--weights", type=str, default="models/candidates/yolov8n_seg_2class_hn20_best_fp_suppression.pt")
    parser.add_argument("--output-dir", type=str, default="models/edge_exports")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 320])
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--canonical-alias", type=str, default="best-seg-2class")
    parser.add_argument("--set-default-pt", action="store_true", help="Copy source weights to models/best-seg-2class.pt")
    args = parser.parse_args()

    pt_source = Path(args.weights)
    if not pt_source.exists():
        print(f"Error: Weights file not found: {pt_source}", file=sys.stderr)
        sys.exit(1)

    if args.set_default_pt:
        canonical_pt = PROJECT_ROOT / "models" / f"{args.canonical_alias}.pt"
        shutil.copy2(pt_source, canonical_pt)
        print(f"Set canonical PT model: {canonical_pt} ({canonical_pt.stat().st_size / (1024*1024):.2f} MB)")

    for sz in args.imgsz:
        export_model(
            model_path=args.weights,
            output_dir=args.output_dir,
            imgsz=sz,
            opset=args.opset,
            canonical_alias=args.canonical_alias,
        )

if __name__ == "__main__":
    main()
