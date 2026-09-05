import re
from pathlib import Path
import pytest
from tools.segmentation_contract import (
    CANONICAL_OVERLAP_MASK,
    CANONICAL_MASK_RATIO,
    BENCHMARK_CONFIDENCE,
    canonical_train_args,
    canonical_val_args,
)

def test_contract_constants():
    assert CANONICAL_OVERLAP_MASK is False
    assert CANONICAL_MASK_RATIO == 4
    assert BENCHMARK_CONFIDENCE == 0.001

    train_args = canonical_train_args()
    assert train_args["overlap_mask"] is False
    assert train_args["mask_ratio"] == 4

    val_args = canonical_val_args()
    assert val_args["overlap_mask"] is False
    assert val_args["conf"] == 0.001


def test_no_hardcoded_windows_paths_in_scripts():
    scripts = [
        Path("train_phase3.py"),
        Path("finetune_roboflow.py"),
        Path("retrain_clean.py"),
        Path("retrain_fast.py"),
        Path("evaluate_clean.py"),
        Path("tools/train_experiment.py"),
        Path("tools/evaluate_benchmark.py"),
    ]
    for s in scripts:
        if s.exists():
            content = s.read_text(encoding="utf-8")
            assert "C:/Project" not in content, f"Hardcoded Windows path found in {s}"


def test_scripts_specify_overlap_mask():
    scripts = [
        Path("train_phase3.py"),
        Path("finetune_roboflow.py"),
        Path("retrain_clean.py"),
        Path("retrain_fast.py"),
        Path("evaluate_clean.py"),
        Path("tools/train_experiment.py"),
        Path("tools/evaluate_benchmark.py"),
    ]
    for s in scripts:
        if s.exists():
            content = s.read_text(encoding="utf-8")
            assert (
                "overlap_mask" in content or "canonical_train_args" in content or "canonical_val_args" in content
            ), f"Script {s} does not specify overlap_mask or import canonical contract"
