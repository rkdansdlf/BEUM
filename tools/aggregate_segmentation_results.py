#!/usr/bin/env python
"""
Aggregate Segmentation Results & Compute 2x2 Evaluation Matrix.
Parses experimental outputs from analysis/overlap_mask_experiments/
and generates comparative reports with statistical analysis (Bootstrap 95% CI).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

EXP_DIR = Path("analysis/overlap_mask_experiments")


def compute_bootstrap_ci(deltas: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Computes bootstrap confidence interval for the delta."""
    if len(deltas) == 0:
        return 0.0, 0.0
    if len(deltas) == 1:
        return deltas[0], deltas[0]

    boot_means = []
    rng = np.random.default_rng(42)
    arr = np.array(deltas)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(sample.mean())

    lower_p = ((1.0 - ci) / 2.0) * 100
    upper_p = (ci + (1.0 - ci) / 2.0) * 100
    return float(np.percentile(boot_means, lower_p)), float(np.percentile(boot_means, upper_p))


def aggregate_paired_experiments(epochs: int) -> dict:
    summary_file = EXP_DIR / f"experiment_summary_e{epochs}.json"
    if not summary_file.is_file():
        print(f"Summary file not found: {summary_file}")
        return {}

    data = json.loads(summary_file.read_text(encoding="utf-8"))

    matrix_rows = []
    train_effects_overall = []
    train_effects_df = []

    print("=" * 110)
    print(f"2x2 EVALUATION MATRIX & TRAINING EFFECT SUMMARY (Epochs={epochs})")
    print("=" * 110)

    for seed, pair in data.items():
        run_true = pair.get("True")
        run_false = pair.get("False")

        if not run_true or not run_false:
            continue

        # Extract metrics
        # TT: Train True, Eval True
        tt_box = run_true["eval_overlap_true"]["box"]["overall"]["map50_95"]
        tt_mask = run_true["eval_overlap_true"]["mask"]["overall"]["map50_95"]
        tt_df_mask = run_true["eval_overlap_true"]["mask"]["classes"].get("drain_full", {}).get("ap50_95", 0.0)

        # TF: Train True, Eval False
        tf_box = run_true["eval_overlap_false"]["box"]["overall"]["map50_95"]
        tf_mask = run_true["eval_overlap_false"]["mask"]["overall"]["map50_95"]
        tf_df_mask = run_true["eval_overlap_false"]["mask"]["classes"].get("drain_full", {}).get("ap50_95", 0.0)

        # FT: Train False, Eval True
        ft_box = run_false["eval_overlap_true"]["box"]["overall"]["map50_95"]
        ft_mask = run_false["eval_overlap_true"]["mask"]["overall"]["map50_95"]
        ft_df_mask = run_false["eval_overlap_true"]["mask"]["classes"].get("drain_full", {}).get("ap50_95", 0.0)

        # FF: Train False, Eval False (Canonical Candidate)
        ff_box = run_false["eval_overlap_false"]["box"]["overall"]["map50_95"]
        ff_mask = run_false["eval_overlap_false"]["mask"]["overall"]["map50_95"]
        ff_df_mask = run_false["eval_overlap_false"]["mask"]["classes"].get("drain_full", {}).get("ap50_95", 0.0)

        # Pure Training Effect: (False, False) - (True, False)
        train_effect_overall = ff_mask - tf_mask
        train_effect_df = ff_df_mask - tf_df_mask

        train_effects_overall.append(train_effect_overall)
        train_effects_df.append(train_effect_df)

        print(f"\n--- Seed: {seed} ---")
        print(f"  [TT] Train=True,  Eval=True  -> Overall Mask: {tt_mask:.4f} | DF Mask: {tt_df_mask:.4f} | Box: {tt_box:.4f}")
        print(f"  [TF] Train=True,  Eval=False -> Overall Mask: {tf_mask:.4f} | DF Mask: {tf_df_mask:.4f} | Box: {tf_box:.4f}")
        print(f"  [FT] Train=False, Eval=True  -> Overall Mask: {ft_mask:.4f} | DF Mask: {ft_df_mask:.4f} | Box: {ft_box:.4f}")
        print(f"  [FF] Train=False, Eval=False -> Overall Mask: {ff_mask:.4f} | DF Mask: {ff_df_mask:.4f} | Box: {ff_box:.4f}")
        print(f"  --> PURE TRAINING EFFECT [FF - TF]: Overall: {train_effect_overall:+.4f} | drain_full: {train_effect_df:+.4f}")

    if train_effects_overall:
        mean_eff_overall = float(np.mean(train_effects_overall))
        med_eff_df = float(np.median(train_effects_df))
        ci_lower, ci_upper = compute_bootstrap_ci(train_effects_df)

        print("\n" + "=" * 110)
        print("STATISTICAL GATE EVALUATION:")
        print(f"Number of Seeds: {len(train_effects_overall)}")
        print(f"Overall Mask AP50-95 Mean Δ: {mean_eff_overall:+.4f}")
        print(f"drain_full Mask AP50-95 Median Δ: {med_eff_df:+.4f}")
        print(f"drain_full 95% Bootstrap CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
        print("=" * 110)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    aggregate_paired_experiments(epochs=args.epochs)
