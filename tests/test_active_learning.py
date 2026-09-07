"""Unit tests for Active Learning triage, review, and dataset ingestion."""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gully_system.blockage import BlockageMetrics
from gully_system.gps import GPSFix
from gully_system.triage import ActiveLearningTriager, TriageReason
from gully_system.types import Detection
from tools.review_active_learning import process_candidate, setup_directories


def test_active_learning_triager_trigger_reasons():
    temp_dir = Path(tempfile.mkdtemp(prefix="beum_al_test_"))
    try:
        triager = ActiveLearningTriager(
            output_dir=temp_dir / "triage",
            uncertain_conf_range=(0.15, 0.28),
            borderline_warning_range=(15.0, 28.0),
            cooldown_s=0.01,
            save_images=True,
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Test Case 1: High confidence, clean road (no reason -> should NOT harvest)
        det_high = Detection(
            bbox=(100.0, 100.0, 200.0, 200.0),
            confidence=0.85,
            class_id=0,
            class_name="drain_area",
        )
        cand_clean = triager.evaluate_frame(
            frame=dummy_frame,
            raw_detections=[det_high],
            blockage=BlockageMetrics(
                status="clear",
                coverage_percent=0.0,
                blocked_area_px=0,
                gully_area_px=1000,
                gully_count=1,
                obstacle_count=0,
                method="mask",
                confidence=0.85,
            ),
        )
        assert cand_clean is None

        # Test Case 2: Low confidence detection (uncertainty -> should harvest)
        det_low = Detection(
            bbox=(100.0, 100.0, 200.0, 200.0),
            confidence=0.22,
            class_id=0,
            class_name="drain_area",
        )
        cand_uncertain = triager.evaluate_frame(
            frame=dummy_frame,
            raw_detections=[det_low],
            blockage=BlockageMetrics(
                status="clear",
                coverage_percent=0.0,
                blocked_area_px=0,
                gully_area_px=1000,
                gully_count=1,
                obstacle_count=0,
                method="mask",
                confidence=0.22,
            ),
        )
        assert cand_uncertain is not None
        assert TriageReason.UNCERTAINTY in cand_uncertain.reasons
        assert (temp_dir / "triage" / cand_uncertain.image_filename).exists()
        assert (temp_dir / "triage" / f"{cand_uncertain.candidate_id}.json").exists()

        # Test Case 3: Borderline blockage warning (coverage=22.5%)
        triager._last_harvest_time = 0.0
        cand_borderline = triager.evaluate_frame(
            frame=dummy_frame,
            raw_detections=[det_high],
            blockage=BlockageMetrics(
                status="warning",
                coverage_percent=22.5,
                blocked_area_px=225,
                gully_area_px=1000,
                gully_count=1,
                obstacle_count=1,
                method="mask",
                confidence=0.85,
            ),
        )
        assert cand_borderline is not None
        assert TriageReason.BORDERLINE_WARNING in cand_borderline.reasons

        # Test Case 4: Temporal flicker flag
        triager._last_harvest_time = 0.0
        cand_flicker = triager.evaluate_frame(
            frame=dummy_frame,
            raw_detections=[det_high],
            is_flicker=True,
        )
        assert cand_flicker is not None
        assert TriageReason.TEMPORAL_FLICKER in cand_flicker.reasons

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_active_learning_review_actions():
    temp_dir = Path(tempfile.mkdtemp(prefix="beum_review_test_"))
    try:
        dirs = setup_directories(temp_dir)
        class_mapping = {"drain_area": 0, "drain_full": 1}

        # 1. Create a dummy positive candidate
        cid_pos = "al_test_pos_001"
        img_pos = dirs["triage"] / f"{cid_pos}.jpg"
        img_pos.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20 + b"\xff\xd9")
        json_pos = dirs["triage"] / f"{cid_pos}.json"
        json_pos.write_text(json.dumps({
            "candidate_id": cid_pos,
            "image_filename": f"{cid_pos}.jpg",
            "detections": [{
                "class_name": "drain_area",
                "confidence": 0.25,
                "bbox": [0.1, 0.1, 0.5, 0.5]
            }]
        }))

        res_pos = process_candidate(json_pos, dirs, "positive", class_mapping)
        assert res_pos["status"] == "approved_positive"
        assert (dirs["pos_img"] / f"{cid_pos}.jpg").exists()
        assert (dirs["pos_lbl"] / f"{cid_pos}.txt").exists()
        lbl_content = (dirs["pos_lbl"] / f"{cid_pos}.txt").read_text()
        assert lbl_content.startswith("0 0.30000 0.30000 0.40000 0.40000")

        # 2. Create a dummy negative candidate (Hard Negative)
        cid_neg = "al_test_neg_002"
        img_neg = dirs["triage"] / f"{cid_neg}.jpg"
        img_neg.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20 + b"\xff\xd9")
        json_neg = dirs["triage"] / f"{cid_neg}.json"
        json_neg.write_text(json.dumps({
            "candidate_id": cid_neg,
            "image_filename": f"{cid_neg}.jpg",
            "detections": []
        }))

        res_neg = process_candidate(json_neg, dirs, "negative", class_mapping)
        assert res_neg["status"] == "approved_negative"
        assert (dirs["neg_img"] / f"{cid_neg}.jpg").exists()
        assert (dirs["neg_lbl"] / f"{cid_neg}.txt").exists()
        assert (dirs["neg_lbl"] / f"{cid_neg}.txt").stat().st_size == 0  # 0-byte background

        # 3. Create a dummy discard candidate
        cid_disc = "al_test_disc_003"
        img_disc = dirs["triage"] / f"{cid_disc}.jpg"
        img_disc.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20 + b"\xff\xd9")
        json_disc = dirs["triage"] / f"{cid_disc}.json"
        json_disc.write_text(json.dumps({
            "candidate_id": cid_disc,
            "image_filename": f"{cid_disc}.jpg",
        }))

        res_disc = process_candidate(json_disc, dirs, "discard", class_mapping)
        assert res_disc["status"] == "discarded"
        assert (dirs["discard"] / f"{cid_disc}.json").exists()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_active_learning_dataset_ingestion():
    temp_dir = Path(tempfile.mkdtemp(prefix="beum_ingest_test_"))
    try:
        # Mock base dataset
        base_dir = temp_dir / "base_2class"
        for split in ["train", "val", "test"]:
            (base_dir / "images" / split).mkdir(parents=True)
            (base_dir / "labels" / split).mkdir(parents=True)

        (base_dir / "images" / "train" / "base_001.jpg").write_bytes(b"dummy")
        (base_dir / "labels" / "train" / "base_001.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (base_dir / "images" / "val" / "val_001.jpg").write_bytes(b"dummy")
        (base_dir / "labels" / "val" / "val_001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        # Mock AL review directory
        al_dir = temp_dir / "al_review"
        dirs = setup_directories(al_dir)

        # 1 Approved positive
        (dirs["pos_img"] / "al_pos_001.jpg").write_bytes(b"dummy")
        (dirs["pos_lbl"] / "al_pos_001.txt").write_text("1 0.4 0.4 0.3 0.3\n")

        # 1 Approved negative
        (dirs["neg_img"] / "al_neg_001.jpg").write_bytes(b"dummy")
        (dirs["neg_lbl"] / "al_neg_001.txt").write_text("")

        # Run ingestion subprocess / script function
        import subprocess
        dst_dir = temp_dir / "dataset_canonical_2class_al_test"
        cmd = [
            sys.executable,
            "tools/ingest_active_learning.py",
            "--base-dataset", str(base_dir),
            "--al-dir", str(al_dir),
            "--version", "test_v1"
        ]
        # In the script, dst_ds is Path(f"dataset/canonical_2class_al_{args.version}")
        # Let's test direct import or call
        from tools.ingest_active_learning import main as ingest_main
        sys.argv = [
            "ingest_active_learning.py",
            "--base-dataset", str(base_dir),
            "--al-dir", str(al_dir),
            "--version", "test_mock"
        ]
        ingest_main()

        expected_dst = Path("dataset/canonical_2class_al_test_mock")
        assert expected_dst.exists()
        assert (expected_dst / "data.yaml").exists()
        assert (expected_dst / "ingestion_manifest.json").exists()

        # Check train images count (1 base + 1 pos + 1 neg = 3)
        train_imgs = list((expected_dst / "images" / "train").glob("*.*"))
        assert len(train_imgs) == 3

        # Clean up created test dataset
        shutil.rmtree(expected_dst, ignore_errors=True)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
