import numpy as np
import pytest
from types import SimpleNamespace

def extract_metrics(box_metric, seg_metric, names):
    """
    Standard BEUM metric extraction helper function.
    Strictly separates:
    - ap50 (AP@0.50) from ap (AP@0.50:0.95)
    - overall mean precision/recall (mp, mr) from class-specific p[i], r[i]
    """
    def _extract_single(m):
        if m is None:
            return {"overall": {}, "classes": {}}
        
        overall = {
            "mean_precision": float(m.mp),
            "mean_recall": float(m.mr),
            "map50": float(m.map50),
            "map50_95": float(m.map),
        }
        
        classes = {}
        for metric_idx, cid in enumerate(m.ap_class_index):
            cid = int(cid)
            cname = names.get(cid, str(cid))
            classes[cname] = {
                "class_id": cid,
                "precision_at_best_f1": float(m.p[metric_idx]),
                "recall_at_best_f1": float(m.r[metric_idx]),
                "f1_at_best_f1": float(m.f1[metric_idx]) if hasattr(m, "f1") else float(
                    2 * m.p[metric_idx] * m.r[metric_idx] / (m.p[metric_idx] + m.r[metric_idx] + 1e-16)
                ),
                "ap50": float(m.ap50[metric_idx]),
                "ap50_95": float(m.ap[metric_idx]),
            }
        return {"overall": overall, "classes": classes}

    return {
        "box": _extract_single(box_metric),
        "mask": _extract_single(seg_metric)
    }


def test_metric_extraction_semantics():
    names = {0: "object", 1: "drain_area", 2: "drain_full"}
    
    # Mock Ultralytics Metric object
    mock_box = SimpleNamespace(
        mp=0.75,
        mr=0.80,
        map50=0.85,
        map=0.60,
        ap_class_index=np.array([1, 2]),
        p=np.array([0.70, 0.80]),
        r=np.array([0.75, 0.85]),
        f1=np.array([0.724, 0.824]),
        ap50=np.array([0.82, 0.88]),
        ap=np.array([0.55, 0.65]),
    )
    
    mock_seg = SimpleNamespace(
        mp=0.65,
        mr=0.70,
        map50=0.72,
        map=0.48,
        ap_class_index=np.array([1, 2]),
        p=np.array([0.60, 0.70]),
        r=np.array([0.65, 0.75]),
        f1=np.array([0.624, 0.724]),
        ap50=np.array([0.71, 0.73]),
        ap=np.array([0.45, 0.51]),
    )
    
    res = extract_metrics(mock_box, mock_seg, names)
    
    # Check overall
    assert res["box"]["overall"]["map50"] == 0.85
    assert res["box"]["overall"]["map50_95"] == 0.60
    assert res["box"]["overall"]["mean_precision"] == 0.75
    assert res["box"]["overall"]["mean_recall"] == 0.80

    # Check class-specific
    da_mask = res["mask"]["classes"]["drain_area"]
    df_mask = res["mask"]["classes"]["drain_full"]
    
    assert da_mask["ap50"] == 0.71
    assert da_mask["ap50_95"] == 0.45
    assert df_mask["ap50"] == 0.73
    assert df_mask["ap50_95"] == 0.51
    
    # Critical verification: ap50 != ap50_95
    assert da_mask["ap50"] != da_mask["ap50_95"]
    assert df_mask["ap50"] != df_mask["ap50_95"]
