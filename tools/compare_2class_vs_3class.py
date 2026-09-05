#!/usr/bin/env python
"""
Compare 3-Class vs 2-Class Canonical Candidates.
Extracts metrics from both model cards and outputs a detailed Markdown table and analysis.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_card(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    card_3class_path = Path("models/candidates/best-seg-3class-omfalse-mr4-e50-s44-0182bde4.json")
    card_2class_path = Path("models/candidates/best-seg-2class-omfalse-mr4-e50-s44-0182bde4.json")

    if not card_3class_path.exists():
        print(f"Error: {card_3class_path} not found.")
        return
    if not card_2class_path.exists():
        print(f"Error: {card_2class_path} not found.")
        return

    c3 = load_card(card_3class_path)
    c2 = load_card(card_2class_path)

    m3_mask = c3["canonical_metrics"]["mask"]
    m2_mask = c2["canonical_metrics"]["mask"]
    m3_box = c3["canonical_metrics"]["box"]
    m2_box = c2["canonical_metrics"]["box"]

    print("=" * 90)
    print("BEUM CANONICAL CANDIDATE COMPARISON: 3-CLASS vs 2-CLASS (50 Epochs, seed=44)")
    print("=" * 90)

    rows = [
        ("Overall Box mAP50", m3_box["overall"]["map50"], m2_box["overall"]["map50"]),
        ("Overall Box mAP50-95", m3_box["overall"]["map50_95"], m2_box["overall"]["map50_95"]),
        ("Overall Mask mAP50", m3_mask["overall"]["map50"], m2_mask["overall"]["map50"]),
        ("Overall Mask mAP50-95", m3_mask["overall"]["map50_95"], m2_mask["overall"]["map50_95"]),
        ("drain_full Mask AP50", m3_mask["classes"]["drain_full"]["ap50"], m2_mask["classes"]["drain_full"]["ap50"]),
        ("drain_full Mask AP50-95", m3_mask["classes"]["drain_full"]["ap50_95"], m2_mask["classes"]["drain_full"]["ap50_95"]),
        ("drain_full Recall", m3_mask["classes"]["drain_full"]["recall_at_best_f1"], m2_mask["classes"]["drain_full"]["recall_at_best_f1"]),
        ("drain_area Mask AP50", m3_mask["classes"]["drain_area"]["ap50"], m2_mask["classes"]["drain_area"]["ap50"]),
        ("drain_area Mask AP50-95", m3_mask["classes"]["drain_area"]["ap50_95"], m2_mask["classes"]["drain_area"]["ap50_95"]),
        ("drain_area Recall", m3_mask["classes"]["drain_area"]["recall_at_best_f1"], m2_mask["classes"]["drain_area"]["recall_at_best_f1"]),
    ]

    print(f"{'지표 (Metric)':<30} | {'3-Class 모델':<18} | {'2-Class 정규화 모델':<18} | {'변화량 (Δ)':<12}")
    print("-" * 90)
    for name, v3, v2 in rows:
        diff = v2 - v3
        sign = "+" if diff >= 0 else ""
        print(f"{name:<30} | {v3:<18.4f} | {v2:<18.4f} | {sign}{diff:<10.4f}")
    print("=" * 90)

    # Save summary report
    report_data = {
        "candidate_3class": c3["model_file"],
        "candidate_2class": c2["model_file"],
        "metrics_comparison": {
            name: {"3class": v3, "2class": v2, "delta": round(v2 - v3, 4)}
            for name, v3, v2 in rows
        }
    }
    out_file = Path("analysis/canonical_2class_vs_3class_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"Detailed comparison saved to: {out_file}")


if __name__ == "__main__":
    main()
