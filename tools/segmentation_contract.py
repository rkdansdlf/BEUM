"""
BEUM Canonical Segmentation Contract.

Defines project-wide standards for training and evaluating YOLOv8/YOLO11
segmentation models to eliminate overlap-mask donut degradation and ensure
consistent metrics across all experiment runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np

# Canonical Hyperparameter Standards
CANONICAL_OVERLAP_MASK: bool = False
CANONICAL_MASK_RATIO: int = 4
BENCHMARK_CONFIDENCE: float = 0.001
BENCHMARK_IOU: float = 0.70
OPERATIONAL_CONFIDENCE: float = 0.20
OPERATIONAL_MATCHING_IOU: float = 0.45


def canonical_train_args() -> Dict[str, Any]:
    """Returns canonical hyperparameters for model.train()."""
    return {
        "overlap_mask": CANONICAL_OVERLAP_MASK,
        "mask_ratio": CANONICAL_MASK_RATIO,
    }


def canonical_val_args() -> Dict[str, Any]:
    """Returns canonical arguments for model.val() benchmarking."""
    return {
        "overlap_mask": CANONICAL_OVERLAP_MASK,
        "conf": BENCHMARK_CONFIDENCE,
        "iou": BENCHMARK_IOU,
    }


def legacy_val_args() -> Dict[str, Any]:
    """Returns legacy arguments for reproducing historical metrics only."""
    return {
        "overlap_mask": True,
        "conf": BENCHMARK_CONFIDENCE,
        "iou": BENCHMARK_IOU,
    }


def extract_class_metrics(metric: Any, names: Dict[int, str]) -> Dict[str, Any]:
    """
    Extracts strictly verified segmentation/detection metrics.
    Ensures that AP50 and AP50-95 are strictly separated and mapped via ap_class_index.
    """
    if metric is None:
        return {"overall": {}, "classes": {}}

    overall = {
        "mean_precision": round(float(metric.mp), 4),
        "mean_recall": round(float(metric.mr), 4),
        "map50": round(float(metric.map50), 4),
        "map50_95": round(float(metric.map), 4),
    }

    classes = {}
    for metric_index, class_id in enumerate(metric.ap_class_index):
        class_id = int(class_id)
        class_name = names.get(class_id, f"class_{class_id}")

        p = float(metric.p[metric_index])
        r = float(metric.r[metric_index])
        f1 = float(metric.f1[metric_index]) if hasattr(metric, "f1") else (
            2 * p * r / (p + r + 1e-16)
        )

        classes[class_name] = {
            "class_id": class_id,
            "precision_at_best_f1": round(p, 4),
            "recall_at_best_f1": round(r, 4),
            "f1_at_best_f1": round(f1, 4),
            "ap50": round(float(metric.ap50[metric_index]), 4),
            "ap50_95": round(float(metric.ap[metric_index]), 4),
        }

    return {"overall": overall, "classes": classes}
