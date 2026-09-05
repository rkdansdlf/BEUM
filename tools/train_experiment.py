#!/usr/bin/env python
"""
Executes controlled training experiments for BEUM YOLOv8 segmentation models.
Supports Hard Negative ablation datasets, controlled augmentation,
and automated dual-track validation (Track A mAP @ 0.001 vs Track B operational @ 0.20)
plus driving video evaluation.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
import tools.evaluate_benchmark as eval_tool

import os
import torch

# Configure offline environment & local cache paths
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

def resolve_device(device: str = "auto") -> str:
    if device in ("auto", "", None):
        if torch.cuda.is_available():
            return "0"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device

def run_experiment(
    data_yaml: str,
    name: str,
    epochs: int = 100,
    patience: int = 20,
    batch: int = 8,
    imgsz: int = 640,
    base_model: str = "yolov8n-seg.pt",
    workers: int = 0,
    device: str = "auto",
    video_source: str = "analysis/new_data_frames",
):
    device = resolve_device(device)
    print("=" * 80)
    print(f"STARTING EXPERIMENT: {name}")
    print(f"Data: {data_yaml}")
    print(f"Base Model: {base_model} | Epochs: {epochs} | Patience: {patience} | Batch: {batch}")
    print(f"Device: {device} | ImgSz: {imgsz}")
    print("=" * 80)

    model = YOLO(base_model)

    from tools.segmentation_contract import canonical_train_args

    train_results = model.train(
        data=data_yaml,
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=imgsz,
        name=name,
        device=device,
        workers=workers,
        seed=42,
        deterministic=(device != "mps"),
        close_mosaic=10,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        copy_paste=0.0,
        fliplr=0.5,
        flipud=0.0,
        optimizer="AdamW",
        val=True,
        verbose=True,
        **canonical_train_args(),
    )

    save_dir = Path(train_results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        # Fallback to last.pt
        best_weights = save_dir / "weights" / "last.pt"

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE. Evaluating best weights: {best_weights}")
    print("=" * 80)

    # 1. Track A: Standard Model-Quality Validation (@ conf=0.001)
    print("\n>>> Running Track A: Standard Validation (@ conf=0.001, NMS iou=0.70)...")
    track_a = eval_tool.evaluate_dataset(
        model_path=best_weights,
        data_yaml=data_yaml,
        split="val",
        imgsz=imgsz,
        conf=0.001,
        iou=0.70,
        device=device,
    )

    # 2. Track B: Operational Validation (@ conf=0.20, NMS iou=0.70, GT matching iou=0.45)
    print("\n>>> Running Track B: Operational Validation (@ conf=0.20, NMS iou=0.70)...")
    track_b = eval_tool.evaluate_dataset(
        model_path=best_weights,
        data_yaml=data_yaml,
        split="val",
        imgsz=imgsz,
        conf=0.20,
        iou=0.70,
        device=device,
    )

    # 3. Operational Video Benchmark (@ conf=0.20)
    video_metrics = None
    if video_source and Path(video_source).exists():
        print(f"\n>>> Running Operational Video Benchmark on {video_source} (@ conf=0.20)...")
        video_metrics = eval_tool.evaluate_video_stream(
            model_path=best_weights,
            video_source=video_source,
            imgsz=imgsz,
            conf=0.20,
            iou=0.70,
        )

    summary = {
        "experiment_name": name,
        "best_weights": str(best_weights),
        "train_args": {
            "data_yaml": data_yaml,
            "epochs": epochs,
            "patience": patience,
            "batch": batch,
            "imgsz": imgsz,
            "base_model": base_model,
        },
        "track_a_standard_val": track_a,
        "track_b_operational_val": track_b,
        "operational_video_benchmark": video_metrics,
    }

    summary_file = save_dir / "experiment_evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT SUMMARY SAVED: {summary_file}")
    print(f"Track A Box mAP50:  {track_a['overall_metrics']['box_map50']:.4f}")
    print(f"Track A Mask mAP50: {track_a['overall_metrics']['mask_map50']:.4f}")
    print(f"Track B Background FP: {track_b['background_fp_count']}")
    if video_metrics:
        print(f"Video Detections/min: {video_metrics['detections_per_minute']}")
        print(f"Video Candidate Clusters/min: {video_metrics['candidate_clusters_per_minute']}")
    print("=" * 80)
    return summary

ABLATION_PRESETS = {
    "hn0": {
        "name": "exp_2class_hn0",
        "data": "dataset/canonical_2class_hn0/data.yaml",
        "description": "0% Hard Negatives (Baseline: Train 372 images)",
    },
    "hn5": {
        "name": "exp_2class_hn5",
        "data": "dataset/canonical_2class_hn5/data.yaml",
        "description": "5% Hard Negatives (Train 391 images = 372 + 19 neg)",
    },
    "hn10": {
        "name": "exp_2class_hn10",
        "data": "dataset/canonical_2class_hn10/data.yaml",
        "description": "10% Hard Negatives (Train 409 images = 372 + 37 neg)",
    },
    "hn20": {
        "name": "exp_2class_hn20",
        "data": "dataset/canonical_2class_hn20/data.yaml",
        "description": "20% Hard Negatives (Train 446 images = 372 + 74 neg)",
    },
}

def generate_comparison_markdown(all_summaries: dict) -> str:
    lines = [
        "# BEUM Hard Negative Ablation Study Comparison Report",
        "",
        "| Run | Ablation Ratio | Train Imgs | Box mAP50 (@0.001) | Mask mAP50 (@0.001) | drain_full Mask AP50 | Operational BG FP (@0.20) | Video Det/min | Video Clusters/min |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, summ in all_summaries.items():
        name = summ.get("experiment_name", key)
        tr_a = summ.get("track_a_standard_val", {})
        tr_b = summ.get("track_b_operational_val", {})
        vid = summ.get("operational_video_benchmark", {}) or {}
        
        box_map = tr_a.get("overall_metrics", {}).get("box_map50", 0.0)
        mask_map = tr_a.get("overall_metrics", {}).get("mask_map50", 0.0)
        df_mask_ap = tr_a.get("class_metrics", {}).get("drain_full", {}).get("mask_ap50", 0.0)
        bg_fp = tr_b.get("background_fp_count", "N/A")
        det_min = vid.get("detections_per_minute", "N/A")
        clu_min = vid.get("candidate_clusters_per_minute", "N/A")
        
        ratio_map = {"hn0": "0%", "hn5": "5%", "hn10": "10%", "hn20": "20%"}
        ratio = ratio_map.get(key, key)
        train_cnt_map = {"hn0": 372, "hn5": 391, "hn10": 409, "hn20": 446}
        tr_imgs = train_cnt_map.get(key, "-")
        
        lines.append(
            f"| {name} | {ratio} | {tr_imgs} | {box_map:.4f} | {mask_map:.4f} | {df_mask_ap:.4f} | {bg_fp} | {det_min} | {clu_min} |"
        )
    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="BEUM Controlled Experiment Runner")
    parser.add_argument("--ablation", type=str, default=None, help="Preset ablation experiment (hn0, hn5, hn10, hn20, all, remaining, or comma-separated)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (if not using preset)")
    parser.add_argument("--name", type=str, default=None, help="Experiment run name (if not using preset)")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--base-model", type=str, default="yolov8n-seg.pt", help="Pretrained base model")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, mps, cpu, 0)")
    parser.add_argument("--video-source", type=str, default="analysis/new_data_frames", help="Video frames dir")

    args = parser.parse_args()

    if args.ablation:
        if args.ablation == "all":
            runs_to_execute = ["hn0", "hn5", "hn10", "hn20"]
        elif args.ablation == "remaining":
            runs_to_execute = ["hn5", "hn10", "hn20"]
        elif "," in args.ablation:
            runs_to_execute = [r.strip() for r in args.ablation.split(",") if r.strip()]
        else:
            runs_to_execute = [args.ablation]

        # Load existing comparison if present
        comp_path = Path("analysis/hard_negative_ablation_comparison.json")
        all_summaries = {}
        if comp_path.exists():
            try:
                with open(comp_path, "r", encoding="utf-8") as f:
                    all_summaries = json.load(f)
            except Exception:
                all_summaries = {}

        # Also load hn0 summary if not yet in all_summaries
        hn0_summary_file = Path("runs/segment/exp_2class_hn0/experiment_evaluation_summary.json")
        if "hn0" not in all_summaries and hn0_summary_file.exists():
            try:
                with open(hn0_summary_file, "r", encoding="utf-8") as f:
                    all_summaries["hn0"] = json.load(f)
            except Exception:
                pass

        for key in runs_to_execute:
            if key not in ABLATION_PRESETS:
                print(f"Unknown ablation preset: {key}, skipping.")
                continue
            preset = ABLATION_PRESETS[key]
            print("\n" + "#" * 80)
            print(f"EXECUTING ABLATION PRESET [{key.upper()}]: {preset['description']}")
            print("#" * 80 + "\n")
            summary = run_experiment(
                data_yaml=preset["data"],
                name=preset["name"],
                epochs=args.epochs,
                patience=args.patience,
                batch=args.batch,
                imgsz=args.imgsz,
                base_model=args.base_model,
                workers=args.workers,
                device=args.device,
                video_source=args.video_source,
            )
            all_summaries[key] = summary

        # Save comparative summary JSON & Markdown
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nAll specified ablation runs complete. Comparative summary saved to: {comp_path}")

        md_content = generate_comparison_markdown(all_summaries)
        md_path = Path("analysis/hard_negative_ablation_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Comparative report markdown saved to: {md_path}")
    else:
        if not args.data or not args.name:
            parser.error("Either --ablation must be provided, or both --data and --name must be specified.")
        run_experiment(
            data_yaml=args.data,
            name=args.name,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            base_model=args.base_model,
            workers=args.workers,
            device=args.device,
            video_source=args.video_source,
        )

if __name__ == "__main__":
    main()
