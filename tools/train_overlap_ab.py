#!/usr/bin/env python
"""
P3 / P4: Controlled Overlap Mask Training A/B Runner.
Executes paired training runs under strictly identical hyperparameters
where the only independent variable is overlap_mask (True vs False).
Evaluates each trained checkpoint under both evaluation policies (True and False)
to construct the rigorous 2x2 evaluation matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import shutil
import torch
from ultralytics import YOLO

from tools.segmentation_contract import (
    extract_class_metrics,
    BENCHMARK_CONFIDENCE,
    BENCHMARK_IOU,
)


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def evaluate_checkpoint(model_path: Path, data_yaml: Path, overlap_mask: bool, device: str) -> dict:
    model = YOLO(str(model_path))
    val_res = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        conf=BENCHMARK_CONFIDENCE,
        iou=BENCHMARK_IOU,
        device=device,
        overlap_mask=overlap_mask,
        verbose=False,
        save=False,
        plots=False,
    )
    return {
        "box": extract_class_metrics(val_res.box, model.names),
        "mask": extract_class_metrics(val_res.seg, model.names),
    }


def run_paired_experiment(
    base_model: str | Path,
    data_yaml: str | Path,
    epochs: int,
    seed: int,
    batch: int = 8,
    imgsz: int = 640,
    device: str = "cpu",
    project: str = "runs/segment/overlap_mask_experiments",
    output_dir: str = "analysis/overlap_mask_experiments",
) -> dict:
    base_model = Path(base_model)
    data_yaml = Path(data_yaml)
    base_model_sha = get_sha256(base_model)
    data_yaml_sha = get_sha256(data_yaml)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_pair = {}

    for train_om in [True, False]:
        tag = f"train_om_{'true' if train_om else 'false'}_e{epochs}_s{seed}"
        run_name = f"ab_{tag}"
        print("\n" + "=" * 80)
        print(f"STARTING RUN: {run_name} (train overlap_mask={train_om})")
        print(f"Base Model: {base_model} | Epochs: {epochs} | Batch: {batch} | Seed: {seed}")
        print("=" * 80)

        # Fresh YOLO instance from base model
        model = YOLO(str(base_model))

        train_args = {
            "data": str(data_yaml),
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": device,
            "workers": 0,
            "seed": seed,
            "deterministic": True,
            "project": project,
            "name": run_name,
            "exist_ok": True,
            "save": True,
            "plots": True,
            "close_mosaic": 10,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "copy_paste": 0.0,
            "fliplr": 0.5,
            "flipud": 0.0,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "overlap_mask": train_om,
            "mask_ratio": 4,
            "verbose": True,
        }

        train_res = model.train(**train_args)
        save_dir = Path(train_res.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
        if not best_pt.is_file():
            best_pt = save_dir / "weights" / "last.pt"

        best_sha = get_sha256(best_pt)

        # 2x2 Evaluation on best weights:
        print(f"\nEvaluating {best_pt.name} with eval overlap_mask=True...")
        eval_true = evaluate_checkpoint(best_pt, data_yaml, overlap_mask=True, device=device)

        print(f"Evaluating {best_pt.name} with eval overlap_mask=False...")
        eval_false = evaluate_checkpoint(best_pt, data_yaml, overlap_mask=False, device=device)

        run_record = {
            "run_name": run_name,
            "tag": tag,
            "train_overlap_mask": train_om,
            "seed": seed,
            "epoch": epochs,
            "base_model": str(base_model),
            "base_model_sha256": base_model_sha,
            "best_weights": str(best_pt),
            "best_weights_sha256": best_sha,
            "eval_overlap_true": eval_true,
            "eval_overlap_false": eval_false,
        }

        # Save individual run metrics
        run_out_dir = out_dir / tag
        run_out_dir.mkdir(parents=True, exist_ok=True)
        (run_out_dir / "metrics.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")

        results_pair[str(train_om)] = run_record

    return results_pair


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run controlled overlap_mask A/B training")
    parser.add_argument("--base-model", type=str, default="models/best-seg.pt")
    parser.add_argument("--data", type=str, default="dataset/canonical/canonical-3class.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--project", type=str, default="runs/segment/overlap_mask_experiments")
    parser.add_argument("--output-dir", type=str, default="analysis/overlap_mask_experiments")
    args = parser.parse_args()

    all_pairs = {}
    for s in args.seeds:
        print(f"\n>>> Running Paired Experiment for Seed {s} (Epochs={args.epochs})")
        pair = run_paired_experiment(
            base_model=args.base_model,
            data_yaml=args.data,
            epochs=args.epochs,
            seed=s,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            output_dir=args.output_dir,
        )
        all_pairs[str(s)] = pair

    summary_file = Path(args.output_dir) / f"experiment_summary_e{args.epochs}.json"
    summary_file.write_text(json.dumps(all_pairs, indent=2), encoding="utf-8")
    print(f"\nExperiment complete! Summary written to: {summary_file}")
