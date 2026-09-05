#!/usr/bin/env python
"""
Review and Temporally Cluster FP Candidates.
Clusters candidate detections based on elapsed time difference (delta_t <= T seconds)
and spatial bounding box proximity into 'Temporal Candidate Clusters'.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


TIMESTAMP_PATTERN = re.compile(r"_t(\d+\.\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_manifest(manifest_csv: Path) -> list[dict]:
    rows = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["frame_id"] = int(r["frame_id"])
            # Re-verify timestamp from raw_file name if needed
            m = TIMESTAMP_PATTERN.search(r["raw_file"])
            if m:
                r["timestamp_s"] = float(m.group(1))
            else:
                r["timestamp_s"] = float(r["timestamp_s"])
            confs = [float(c) for c in r["confidences"].split(";") if c.strip()]
            r["max_conf"] = max(confs) if confs else 0.0
            r["all_confs"] = confs
            r["classes_list"] = r["classes"].split(";")
            rows.append(r)
    return sorted(rows, key=lambda x: x["timestamp_s"])


def cluster_temporal_candidates(candidates: list[dict], time_window_seconds: float = 6.0) -> list[dict]:
    """
    Groups frames that occur within time_window_seconds into a single Temporal Candidate Cluster.
    Selects the frame with the maximum confidence as the representative keyframe.
    """
    if not candidates:
        return []

    clusters = []
    current_cluster = [candidates[0]]

    for cand in candidates[1:]:
        prev_cand = current_cluster[-1]
        time_diff = cand["timestamp_s"] - prev_cand["timestamp_s"]
        if time_diff <= time_window_seconds:
            current_cluster.append(cand)
        else:
            clusters.append(current_cluster)
            current_cluster = [cand]
    if current_cluster:
        clusters.append(current_cluster)

    cluster_summaries = []
    for cluster_id, cluster in enumerate(clusters, start=1):
        keyframe = max(cluster, key=lambda x: x["max_conf"])
        frame_ids = [c["frame_id"] for c in cluster]
        start_t = cluster[0]["timestamp_s"]
        end_t = cluster[-1]["timestamp_s"]
        all_classes = set()
        for c in cluster:
            all_classes.update(c["classes_list"])

        cluster_summaries.append({
            "cluster_id": cluster_id,
            "frame_count": len(cluster),
            "frame_range": f"{frame_ids[0]} ~ {frame_ids[-1]}",
            "time_range_s": f"{start_t:.2f}s ~ {end_t:.2f}s",
            "duration_s": round(end_t - start_t, 2),
            "keyframe_id": keyframe["frame_id"],
            "keyframe_timestamp_s": round(keyframe["timestamp_s"], 2),
            "keyframe_raw_file": keyframe["raw_file"],
            "keyframe_max_conf": round(keyframe["max_conf"], 3),
            "detected_classes": sorted(list(all_classes)),
            "cluster_frame_ids": frame_ids,
            "review_status": "pending",  # pending, confirmed_fp, actual_drain, ambiguous
            "notes": "",
        })

    return cluster_summaries


def main():
    parser = argparse.ArgumentParser(description="Temporally cluster FP candidates.")
    parser.add_argument("--manifest", type=str, default="data/FP_mined/candidates_manifest.csv")
    parser.add_argument("--output-json", type=str, default="data/FP_mined/deduped_events.json")
    parser.add_argument("--output-csv", type=str, default="data/FP_mined/deduped_events.csv")
    parser.add_argument("--time-window", type=float, default=6.0, help="Max time window (seconds) to cluster")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}")
        return

    candidates = parse_manifest(manifest_path)
    clusters = cluster_temporal_candidates(candidates, time_window_seconds=args.time_window)

    out_json = Path(args.output_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    out_csv = Path(args.output_csv)
    fieldnames = [
        "cluster_id", "frame_count", "frame_range", "time_range_s", "duration_s",
        "keyframe_id", "keyframe_timestamp_s", "keyframe_max_conf", "detected_classes", "review_status", "notes"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cl in clusters:
            row = {k: cl[k] for k in fieldnames if k in cl}
            row["detected_classes"] = ";".join(row["detected_classes"])
            writer.writerow(row)

    print("=" * 80)
    print("                 TEMPORAL CANDIDATE CLUSTERING REPORT                 ")
    print("=" * 80)
    print(f"Total raw candidate frames: {len(candidates)}")
    print(f"Temporal Candidate Clusters: {len(clusters)} (Time delta <= {args.time_window}s)")
    print("-" * 80)
    print("Cluster ID".ljust(12), "Frames".rjust(8), "Frame Range".rjust(16), "Time Range".rjust(20), "Keyframe Conf".rjust(16))
    print("-" * 80)
    for cl in clusters:
        print(
            str(cl["cluster_id"]).ljust(12),
            str(cl["frame_count"]).rjust(8),
            str(cl["frame_range"]).rjust(16),
            cl["time_range_s"].rjust(20),
            f"{cl['keyframe_max_conf']:.3f} (t={cl['keyframe_timestamp_s']}s)".rjust(16),
        )
    print("=" * 80)
    print(f"Saved temporal candidate clusters to {out_json} and {out_csv}\n")


if __name__ == "__main__":
    main()
