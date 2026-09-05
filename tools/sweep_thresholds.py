#!/usr/bin/env python
"""
Confidence Threshold Sweep Tool for BEUM YOLOv8 Segmentation.
Sweeps confidence thresholds from 0.05 to 0.70 to find class-wise optimal operating points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]


def sweep_thresholds(
    model_path: str | Path,
    data_yaml: str | Path,
    split: str = "val",
    imgsz: int = 640,
    iou: float = 0.70,
    device: str = "cpu",
) -> dict:
    model = YOLO(str(model_path))
    names = model.names
    print(f"Loaded model: {model_path} with classes: {names}")
    print(f"Sweeping thresholds on {split} split from {data_yaml} across {THRESHOLDS}...\n")

    sweep_history = []
    class_best_f1 = {cname: {"conf": 0.0, "f1": -1.0, "precision": 0.0, "recall": 0.0} for cname in names.values()}

    for conf in THRESHOLDS:
        print(f"--> Evaluating conf={conf:.2f}...", end=" ", flush=True)
        val_res = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )

        box_map50 = float(val_res.box.map50)
        mask_map50 = float(val_res.seg.map50) if hasattr(val_res, "seg") and val_res.seg is not None else 0.0
        mask_map50_95 = float(val_res.seg.map) if hasattr(val_res, "seg") and val_res.seg is not None else 0.0

        seg_p = val_res.seg.p if hasattr(val_res, "seg") and val_res.seg is not None else [0.0] * len(names)
        seg_r = val_res.seg.r if hasattr(val_res, "seg") and val_res.seg is not None else [0.0] * len(names)
        seg_f1 = val_res.seg.f1 if hasattr(val_res, "seg") and val_res.seg is not None else [0.0] * len(names)

        record = {
            "conf": conf,
            "mask_map50": round(mask_map50, 4),
            "mask_map50_95": round(mask_map50_95, 4),
            "box_map50": round(box_map50, 4),
            "classes": {},
        }

        for cid, cname in names.items():
            idx = int(cid)
            p = float(seg_p[idx]) if idx < len(seg_p) else 0.0
            r = float(seg_r[idx]) if idx < len(seg_r) else 0.0
            f1 = float(seg_f1[idx]) if idx < len(seg_f1) else 0.0

            record["classes"][cname] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            }

            if f1 > class_best_f1[cname]["f1"]:
                class_best_f1[cname] = {
                    "conf": conf,
                    "f1": round(f1, 4),
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                }

        print(f"Mask mAP50={mask_map50:.4f}")
        sweep_history.append(record)

    return {
        "model": str(model_path),
        "data_yaml": str(data_yaml),
        "split": split,
        "class_best_f1": class_best_f1,
        "sweep_history": sweep_history,
    }


def print_sweep_summary(results: dict):
    print("\n" + "=" * 80)
    print("                 CONFIDENCE THRESHOLD SWEEP SUMMARY                 ")
    print("=" * 80)
    print(f"Model: {results['model']}")
    print(f"Dataset: {results['data_yaml']} (Split: {results['split']})")
    print("\n[Optimal Operating Point per Class by Max F1-Score]:")
    print("-" * 80)
    print("Class".ljust(18), "Best Conf".rjust(12), "Precision".rjust(14), "Recall".rjust(14), "Max F1".rjust(14))
    print("-" * 80)
    for cname, best in results["class_best_f1"].items():
        print(
            cname.ljust(18),
            f"{best['conf']:.2f}".rjust(12),
            f"{best['precision']:.4f}".rjust(14),
            f"{best['recall']:.4f}".rjust(14),
            f"{best['f1']:.4f}".rjust(14),
        )

    print("\n[Threshold Progression Table]:")
    print("-" * 80)
    print("Conf".rjust(6), "mAP50(M)".rjust(10), "drain_area (P/R/F1)".rjust(24), "drain_full (P/R/F1)".rjust(24))
    print("-" * 80)
    for row in results["sweep_history"]:
        conf_str = f"{row['conf']:.2f}"
        map_str = f"{row['mask_map50']:.4f}"
        da = row["classes"].get("drain_area", {})
        df = row["classes"].get("drain_full", {})
        da_prf = f"{da.get('precision', 0):.2f}/{da.get('recall', 0):.2f}/{da.get('f1', 0):.2f}"
        df_prf = f"{df.get('precision', 0):.2f}/{df.get('recall', 0):.2f}/{df.get('f1', 0):.2f}"
        print(conf_str.rjust(6), map_str.rjust(10), da_prf.rjust(24), df_prf.rjust(24))
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds for YOLOv8 segmentation model.")
    parser.add_argument("--model", type=str, default="models/best-seg-3class.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate: val or test")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--iou", type=float, default=0.70, help="IoU threshold")
    parser.add_argument("--output", type=str, default="analysis/threshold_sweep_results.json", help="Path to save JSON")
    args = parser.parse_args()

    results = sweep_thresholds(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        iou=args.iou,
    )

    print_sweep_summary(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Sweep results saved to {out_path}")


if __name__ == "__main__":
    main()
