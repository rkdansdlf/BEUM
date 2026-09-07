#!/usr/bin/env python3
"""
Active Learning Dataset Ingestion Pipeline.
Merges approved positives and approved hard negatives into a versioned
canonical dataset (e.g. dataset/canonical_2class_al_v1) for continuous retraining.
Keeps validation and test splits pristine to prevent data leakage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Ingest Active Learning Samples into Canonical Dataset")
    parser.add_argument("--base-dataset", type=str, default="dataset/canonical_2class",
                        help="Path to base 2-class dataset")
    parser.add_argument("--al-dir", type=str, default="data/active_learning",
                        help="Active learning review root directory")
    parser.add_argument("--version", type=str, default="v1",
                        help="Dataset version suffix (e.g. v1, v2)")
    args = parser.parse_args()

    base_ds = Path(args.base_dataset)
    al_dir = Path(args.al_dir)
    dst_ds = Path(f"dataset/canonical_2class_al_{args.version}")

    pos_img_dir = al_dir / "approved_positives" / "images"
    pos_lbl_dir = al_dir / "approved_positives" / "labels"
    neg_img_dir = al_dir / "approved_negatives" / "images"
    neg_lbl_dir = al_dir / "approved_negatives" / "labels"

    pos_imgs = list(pos_img_dir.glob("*.*")) if pos_img_dir.exists() else []
    neg_imgs = list(neg_img_dir.glob("*.*")) if neg_img_dir.exists() else []

    print("=" * 80)
    print(f"ACTIVE LEARNING DATASET INGESTION: {dst_ds}")
    print(f"Base Dataset:       {base_ds}")
    print(f"Approved Positives: {len(pos_imgs)} images")
    print(f"Approved Negatives: {len(neg_imgs)} images")
    print("=" * 80)

    if dst_ds.exists():
        shutil.rmtree(dst_ds)
    dst_ds.mkdir(parents=True, exist_ok=True)

    # 1. Copy pristine Val and Test splits
    for split in ["val", "test"]:
        shutil.copytree(base_ds / "images" / split, dst_ds / "images" / split)
        shutil.copytree(base_ds / "labels" / split, dst_ds / "labels" / split)

    # 2. Copy base Train split
    dst_train_img = dst_ds / "images" / "train"
    dst_train_lbl = dst_ds / "labels" / "train"
    shutil.copytree(base_ds / "images" / "train", dst_train_img)
    shutil.copytree(base_ds / "labels" / "train", dst_train_lbl)

    base_train_count = len(list(dst_train_img.glob("*.*")))

    # 3. Ingest Approved Positives
    pos_ingested = 0
    for img_path in pos_imgs:
        lbl_path = pos_lbl_dir / f"{img_path.stem}.txt"
        dest_img = dst_train_img / f"al_pos_{img_path.name}"
        dest_lbl = dst_train_lbl / f"al_pos_{img_path.stem}.txt"

        shutil.copy2(img_path, dest_img)
        if lbl_path.exists():
            shutil.copy2(lbl_path, dest_lbl)
        else:
            dest_lbl.write_text("", encoding="utf-8")
        pos_ingested += 1

    # 4. Ingest Approved Negatives (Hard Negatives)
    neg_ingested = 0
    for img_path in neg_imgs:
        dest_img = dst_train_img / f"al_neg_{img_path.name}"
        dest_lbl = dst_train_lbl / f"al_neg_{img_path.stem}.txt"

        shutil.copy2(img_path, dest_img)
        dest_lbl.write_text("", encoding="utf-8")  # 0-byte background
        neg_ingested += 1

    total_train_count = len(list(dst_train_img.glob("*.*")))

    # 5. Write data.yaml
    yaml_content = {
        "path": str(dst_ds.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "drain_area",
            1: "drain_full"
        }
    }
    yaml_file = dst_ds / "data.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    # 6. Audit class counts
    class_counts = Counter()
    for lbl in dst_train_lbl.glob("*.txt"):
        for line in lbl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cid = int(line.split()[0])
                class_counts[cid] += 1

    manifest = {
        "dataset_name": dst_ds.name,
        "base_dataset": str(base_ds),
        "version": args.version,
        "train_images": total_train_count,
        "base_train_images": base_train_count,
        "ingested_positives": pos_ingested,
        "ingested_negatives": neg_ingested,
        "val_images": len(list((dst_ds / "images/val").glob("*.*"))),
        "test_images": len(list((dst_ds / "images/test").glob("*.*"))),
        "train_class_instances": {
            "drain_area (0)": class_counts[0],
            "drain_full (1)": class_counts[1],
        },
        "yaml_path": str(yaml_file),
    }

    manifest_file = dst_ds / "ingestion_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"INGESTION COMPLETE: {dst_ds.name}")
    print(f"Total Train Images : {total_train_count} (Base: {base_train_count}, +Pos: {pos_ingested}, +Neg: {neg_ingested})")
    print(f"Train Instances    : drain_area={class_counts[0]}, drain_full={class_counts[1]}")
    print(f"YAML Config Path   : {yaml_file}")
    print(f"Manifest Path      : {manifest_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
