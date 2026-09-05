#!/usr/bin/env python
"""
Model Comparison Table Generator.
Reads structured evaluation JSON files from analysis/ and outputs a verified comparison table
with explicit train & eval overlap policies, SHA-256 identifiers, and AP metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

ANALYSIS_DIR = Path("analysis")


def discover_evaluation_reports() -> list[dict]:
    reports = []

    # 1. Baseline A/B report
    p1_file = ANALYSIS_DIR / "overlap_mask_audit" / "evaluation_ab_baseline.json"
    if p1_file.is_file():
        try:
            d = json.loads(p1_file.read_text(encoding="utf-8"))
            m_a = d["condition_A_overlap_true"]
            m_b = d["condition_B_overlap_false"]

            reports.append({
                "model_name": "best-seg.pt (Legacy Eval)",
                "model_sha": d["model"]["sha256"][:8],
                "dataset_sha": d["dataset"]["sha256"][:8],
                "train_overlap": "True",
                "eval_overlap": "True",
                "mask_ap50": m_a["mask"]["overall"].get("map50", 0.0),
                "mask_ap50_95": m_a["mask"]["overall"].get("map50_95", 0.0),
                "box_ap50_95": m_a["box"]["overall"].get("map50_95", 0.0),
                "df_mask_ap50": m_a["mask"]["classes"].get("drain_full", {}).get("ap50", 0.0),
                "df_mask_ap50_95": m_a["mask"]["classes"].get("drain_full", {}).get("ap50_95", 0.0),
                "seed": "-",
                "epoch": "-",
            })

            reports.append({
                "model_name": "best-seg.pt (Canonical Eval)",
                "model_sha": d["model"]["sha256"][:8],
                "dataset_sha": d["dataset"]["sha256"][:8],
                "train_overlap": "True",
                "eval_overlap": "False",
                "mask_ap50": m_b["mask"]["overall"].get("map50", 0.0),
                "mask_ap50_95": m_b["mask"]["overall"].get("map50_95", 0.0),
                "box_ap50_95": m_b["box"]["overall"].get("map50_95", 0.0),
                "df_mask_ap50": m_b["mask"]["classes"].get("drain_full", {}).get("ap50", 0.0),
                "df_mask_ap50_95": m_b["mask"]["classes"].get("drain_full", {}).get("ap50_95", 0.0),
                "seed": "-",
                "epoch": "-",
            })
        except Exception as e:
            print(f"Warning: could not parse {p1_file}: {e}")

    # 2. Experiment reports from analysis/overlap_mask_experiments/*/metrics.json
    exp_dir = ANALYSIS_DIR / "overlap_mask_experiments"
    if exp_dir.is_dir():
        for m_file in sorted(exp_dir.glob("*/metrics.json")):
            try:
                data = json.loads(m_file.read_text(encoding="utf-8"))
                reports.append({
                    "model_name": m_file.parent.name,
                    "model_sha": data.get("model_sha256", "unknown")[:8],
                    "dataset_sha": data.get("dataset_sha256", "unknown")[:8],
                    "train_overlap": str(data.get("train_overlap_mask", "unknown")),
                    "eval_overlap": str(data.get("eval_overlap_mask", "False")),
                    "mask_ap50": data.get("metrics", {}).get("mask", {}).get("overall", {}).get("map50", 0.0),
                    "mask_ap50_95": data.get("metrics", {}).get("mask", {}).get("overall", {}).get("map50_95", 0.0),
                    "box_ap50_95": data.get("metrics", {}).get("box", {}).get("overall", {}).get("map50_95", 0.0),
                    "df_mask_ap50": data.get("metrics", {}).get("mask", {}).get("classes", {}).get("drain_full", {}).get("ap50", 0.0),
                    "df_mask_ap50_95": data.get("metrics", {}).get("mask", {}).get("classes", {}).get("drain_full", {}).get("ap50_95", 0.0),
                    "seed": str(data.get("seed", "-")),
                    "epoch": str(data.get("epoch", "-")),
                })
            except Exception as e:
                print(f"Warning: could not parse {m_file}: {e}")

    return reports


def print_comparison_table():
    reports = discover_evaluation_reports()
    if not reports:
        print("No evaluation reports found in analysis/.")
        return

    print("=" * 125)
    print("BEUM CANONICAL SEGMENTATION MODEL COMPARISON TABLE")
    print("=" * 125)
    header = f"{'Model':<30} | {'M-SHA':<8} | {'Tr-OM':<5} | {'Ev-OM':<5} | {'mAP50':<7} | {'mAP50-95':<8} | {'BoxAP':<7} | {'DF-mAP50':<8} | {'DF-AP':<7} | {'Seed':<4} | {'Ep':<3}"
    print(header)
    print("-" * 125)

    for r in reports:
        line = (
            f"{r['model_name']:<30} | "
            f"{r['model_sha']:<8} | "
            f"{r['train_overlap']:<5} | "
            f"{r['eval_overlap']:<5} | "
            f"{r['mask_ap50']:<7.4f} | "
            f"{r['mask_ap50_95']:<8.4f} | "
            f"{r['box_ap50_95']:<7.4f} | "
            f"{r['df_mask_ap50']:<8.4f} | "
            f"{r['df_mask_ap50_95']:<7.4f} | "
            f"{r['seed']:<4} | "
            f"{r['epoch']:<3}"
        )
        print(line)
    print("=" * 125)


if __name__ == "__main__":
    print_comparison_table()
