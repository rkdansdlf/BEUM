#!/usr/bin/env python
"""
P1: Evaluation-only A/B on the frozen baseline model (models/best-seg.pt).
Strictly measures the evaluation-target effect by comparing overlap_mask=True vs False
while verifying that box predictions and box metrics remain completely invariant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = Path("models/best-seg.pt")
DATA_YAML = Path("dataset/canonical/canonical-3class.yaml")
OUTPUT_JSON = Path("analysis/overlap_mask_audit/evaluation_ab_baseline.json")


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def extract_metrics(val_res, names: dict[int, str]) -> dict:
    def _extract_metric(m):
        if m is None:
            return {"overall": {}, "classes": {}}
        overall = {
            "mean_precision": round(float(m.mp), 4),
            "mean_recall": round(float(m.mr), 4),
            "map50": round(float(m.map50), 4),
            "map50_95": round(float(m.map), 4),
        }
        classes = {}
        for metric_idx, cid in enumerate(m.ap_class_index):
            cid = int(cid)
            cname = names.get(cid, str(cid))
            classes[cname] = {
                "class_id": cid,
                "precision_at_best_f1": round(float(m.p[metric_idx]), 4),
                "recall_at_best_f1": round(float(m.r[metric_idx]), 4),
                "f1_at_best_f1": round(float(m.f1[metric_idx]), 4) if hasattr(m, "f1") else round(
                    float(2 * m.p[metric_idx] * m.r[metric_idx] / (m.p[metric_idx] + m.r[metric_idx] + 1e-16)), 4
                ),
                "ap50": round(float(m.ap50[metric_idx]), 4),
                "ap50_95": round(float(m.ap[metric_idx]), 4),
            }
        return {"overall": overall, "classes": classes}

    return {
        "box": _extract_metric(val_res.box),
        "mask": _extract_metric(val_res.seg),
    }


def main():
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    model_sha = get_sha256(MODEL_PATH)
    yaml_sha = get_sha256(DATA_YAML)

    print("=" * 80)
    print("P1: EXISTING MODEL EVALUATION-ONLY A/B")
    print(f"Model: {MODEL_PATH} (SHA: {model_sha})")
    print(f"Data: {DATA_YAML} (SHA: {yaml_sha})")
    print("Device: cpu (to ensure zero GPU non-determinism)")
    print("=" * 80)

    model = YOLO(str(MODEL_PATH))
    names = model.names

    common_val_args = {
        "data": str(DATA_YAML),
        "split": "val",
        "imgsz": 640,
        "conf": 0.001,
        "iou": 0.70,
        "device": "cpu",
        "workers": 0,
        "verbose": False,
        "save": False,
        "plots": False,
    }

    # 1. Condition A: overlap_mask=True (Legacy GT)
    print("\n[Condition A] Evaluating with overlap_mask=True...")
    res_a = model.val(**common_val_args, overlap_mask=True)
    metrics_a = extract_metrics(res_a, names)

    # 2. Condition B: overlap_mask=False (Canonical GT)
    print("\n[Condition B] Evaluating with overlap_mask=False...")
    res_b = model.val(**common_val_args, overlap_mask=False)
    metrics_b = extract_metrics(res_b, names)

    # 3. Invariance check on Box metrics
    box_map50_diff = abs(metrics_a["box"]["overall"]["map50"] - metrics_b["box"]["overall"]["map50"])
    box_map50_95_diff = abs(metrics_a["box"]["overall"]["map50_95"] - metrics_b["box"]["overall"]["map50_95"])

    print("\n" + "=" * 80)
    print("INVARIANCE CHECKS (Box Metrics):")
    print(f"Condition A Box mAP50:    {metrics_a['box']['overall']['map50']}")
    print(f"Condition B Box mAP50:    {metrics_b['box']['overall']['map50']} (diff: {box_map50_diff:.6f})")
    print(f"Condition A Box mAP50-95: {metrics_a['box']['overall']['map50_95']}")
    print(f"Condition B Box mAP50-95: {metrics_b['box']['overall']['map50_95']} (diff: {box_map50_95_diff:.6f})")
    print("=" * 80)

    assert box_map50_diff <= 0.0001, f"Box mAP50 changed! diff={box_map50_diff}"
    assert box_map50_95_diff <= 0.0001, f"Box mAP50-95 changed! diff={box_map50_95_diff}"

    # 4. Evaluation-Target Effect (delta = False - True)
    delta_mask_map50 = round(metrics_b["mask"]["overall"]["map50"] - metrics_a["mask"]["overall"]["map50"], 4)
    delta_mask_map = round(metrics_b["mask"]["overall"]["map50_95"] - metrics_a["mask"]["overall"]["map50_95"], 4)
    delta_mask_recall = round(metrics_b["mask"]["overall"]["mean_recall"] - metrics_a["mask"]["overall"]["mean_recall"], 4)

    class_deltas = {}
    for cname in ["drain_area", "drain_full"]:
        if cname in metrics_a["mask"]["classes"] and cname in metrics_b["mask"]["classes"]:
            class_deltas[cname] = {
                "ap50_A": metrics_a["mask"]["classes"][cname]["ap50"],
                "ap50_B": metrics_b["mask"]["classes"][cname]["ap50"],
                "delta_ap50": round(metrics_b["mask"]["classes"][cname]["ap50"] - metrics_a["mask"]["classes"][cname]["ap50"], 4),
                "ap50_95_A": metrics_a["mask"]["classes"][cname]["ap50_95"],
                "ap50_95_B": metrics_b["mask"]["classes"][cname]["ap50_95"],
                "delta_ap50_95": round(metrics_b["mask"]["classes"][cname]["ap50_95"] - metrics_a["mask"]["classes"][cname]["ap50_95"], 4),
                "recall_A": metrics_a["mask"]["classes"][cname]["recall_at_best_f1"],
                "recall_B": metrics_b["mask"]["classes"][cname]["recall_at_best_f1"],
                "delta_recall": round(metrics_b["mask"]["classes"][cname]["recall_at_best_f1"] - metrics_a["mask"]["classes"][cname]["recall_at_best_f1"], 4),
            }

    print("\nEVALUATION-TARGET EFFECT (False - True on existing frozen model):")
    print(f"Overall Mask mAP50:    {metrics_a['mask']['overall']['map50']} -> {metrics_b['mask']['overall']['map50']} (Δ = {delta_mask_map50:+0.4f})")
    print(f"Overall Mask mAP50-95: {metrics_a['mask']['overall']['map50_95']} -> {metrics_b['mask']['overall']['map50_95']} (Δ = {delta_mask_map:+0.4f})")
    print(f"Overall Mask Recall:   {metrics_a['mask']['overall']['mean_recall']} -> {metrics_b['mask']['overall']['mean_recall']} (Δ = {delta_mask_recall:+0.4f})")
    for cname, cd in class_deltas.items():
        print(f"[{cname}] Mask AP50: {cd['ap50_A']} -> {cd['ap50_B']} (Δ = {cd['delta_ap50']:+0.4f}) | AP50-95: {cd['ap50_95_A']} -> {cd['ap50_95_B']} (Δ = {cd['delta_ap50_95']:+0.4f}) | Recall: {cd['recall_A']} -> {cd['recall_B']} (Δ = {cd['delta_recall']:+0.4f})")

    report = {
        "schema_version": 1,
        "purpose": "P1 Evaluation-Target Effect Baseline (Evaluation-only A/B on frozen weights)",
        "model": {
            "path": str(MODEL_PATH),
            "sha256": model_sha,
            "names": names
        },
        "dataset": {
            "yaml": str(DATA_YAML),
            "sha256": yaml_sha
        },
        "common_args": common_val_args,
        "box_invariance_verified": True,
        "condition_A_overlap_true": metrics_a,
        "condition_B_overlap_false": metrics_b,
        "evaluation_target_effect": {
            "overall_mask_map50_delta": delta_mask_map50,
            "overall_mask_map50_95_delta": delta_mask_map,
            "overall_mask_recall_delta": delta_mask_recall,
            "class_deltas": class_deltas
        }
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {OUTPUT_JSON}")
    print("PASS — LEGACY_AND_CANONICAL_GT_EVALUATION_BASELINES_ESTABLISHED")


if __name__ == "__main__":
    main()
