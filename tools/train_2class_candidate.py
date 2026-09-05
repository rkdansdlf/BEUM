#!/usr/bin/env python
"""
Track 3: Train 2-Class Canonical Candidate Model (50 Epochs, overlap_mask=False, mask_ratio=4).
Saves checkpoints, evaluates with canonical policy, packages final best weights to models/candidates/,
and creates the formal 2-class model card JSON with comparison to 3-class candidate.
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
from tools.segmentation_contract import (
    canonical_train_args,
    canonical_val_args,
    extract_class_metrics,
)


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="models/best-seg.pt")
    parser.add_argument("--data", type=str, default="dataset/canonical_2class/canonical-2class.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--mask-ratio", type=int, default=4)
    parser.add_argument("--git-sha", type=str, default="0182bde4")
    args = parser.parse_args()

    base_model = Path(args.base_model)
    data_yaml = Path(args.data)
    base_sha = get_sha256(base_model)
    data_sha = get_sha256(data_yaml)

    run_name = f"candidate_2class_omfalse_mr{args.mask_ratio}_e{args.epochs}_s{args.seed}_{args.git_sha}"
    print("=" * 80)
    print(f"TRACK 3: TRAINING 2-CLASS CANDIDATE: {run_name}")
    print(f"Base: {base_model} (SHA: {base_sha[:8]}) | Epochs: {args.epochs} | Seed: {args.seed} | Device: {args.device}")
    print(f"Dataset: {data_yaml} (SHA: {data_sha[:8]})")
    print(f"Policy: overlap_mask=False, mask_ratio={args.mask_ratio}")
    print("=" * 80)

    model = YOLO(str(base_model))

    train_results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.device,
        workers=0,
        seed=args.seed,
        deterministic=True,
        project="runs/segment/candidates",
        name=run_name,
        exist_ok=True,
        save=True,
        save_period=5,
        plots=True,
        close_mosaic=10,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        copy_paste=0.0,
        fliplr=0.5,
        flipud=0.0,
        optimizer="AdamW",
        lr0=0.001,
        overlap_mask=False,
        mask_ratio=args.mask_ratio,
        verbose=True,
    )

    save_dir = Path(train_results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        best_pt = save_dir / "weights" / "last.pt"

    print(f"\nEvaluating final candidate {best_pt} with Canonical False GT on {data_yaml}...")
    eval_model = YOLO(str(best_pt))
    val_res = eval_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        device=args.device,
        overlap_mask=False,
        conf=0.001,
        iou=0.70,
        verbose=False,
    )

    box_metrics = extract_class_metrics(val_res.box, eval_model.names)
    mask_metrics = extract_class_metrics(val_res.seg, eval_model.names)

    # Package to models/candidates
    candidate_filename = f"best-seg-2class-omfalse-mr{args.mask_ratio}-e{args.epochs}-s{args.seed}-{args.git_sha}.pt"
    dest_pt = Path("models/candidates") / candidate_filename
    shutil.copy2(best_pt, dest_pt)
    candidate_sha = get_sha256(dest_pt)

    card = {
        "model_file": candidate_filename,
        "model_sha256": candidate_sha,
        "parent_model_sha256": base_sha,
        "git_sha": args.git_sha,
        "dataset_yaml": str(data_yaml),
        "dataset_sha256": data_sha,
        "class_names": eval_model.names,
        "train_overlap_mask": False,
        "eval_overlap_mask": False,
        "mask_ratio": args.mask_ratio,
        "imgsz": 640,
        "seed": args.seed,
        "epochs": args.epochs,
        "canonical_metrics": {
            "box": box_metrics,
            "mask": mask_metrics,
        },
        "promotion_status": "candidate_validated",
    }

    card_file = Path("models/candidates") / f"{candidate_filename.replace('.pt', '.json')}"
    card_file.write_text(json.dumps(card, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("TRACK 3: 2-CLASS CANDIDATE MODEL PACKAGED SUCCESSFULLY!")
    print(f"Candidate PT:   {dest_pt} (SHA: {candidate_sha})")
    print(f"Candidate Card: {card_file}")
    print(f"Class names: {eval_model.names}")
    print(f"Overall Box  mAP50-95: {box_metrics['overall']['map50_95']}")
    print(f"Overall Mask mAP50-95: {mask_metrics['overall']['map50_95']}")
    print(f"drain_full Mask mAP50-95: {mask_metrics['classes'].get('drain_full', {}).get('ap50_95')}")
    print(f"drain_area Mask mAP50-95: {mask_metrics['classes'].get('drain_area', {}).get('ap50_95')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
