"""
Builds Hard Negative Ablation Datasets (0%, 5%, 10%, 20%)
Based on Canonical 2-Class Dataset and human-confirmed FP candidate clusters.
Strictly excludes Cluster 7 (frame 155) where a true metal grate drain exists.
"""

import json
import shutil
from pathlib import Path
import yaml

BASE_2CLASS = Path("dataset/canonical_2class")
FP_EVENTS_JSON = Path("data/FP_mined/deduped_events.json")
ALL_VIDEO_FRAMES_DIR = Path("analysis/new_data_frames")
OUTPUT_BASE = Path("dataset")

EXCLUDED_FRAME_RANGE = (148, 160)  # Contains real drain at frame 155

def get_confirmed_fp_frames():
    with open(FP_EVENTS_JSON, "r", encoding="utf-8") as f:
        events = json.load(f)
    
    keyframes = []
    all_cluster_frames = []
    
    for ev in events:
        if ev["review_status"] == "confirmed_fp":
            kf_id = ev["keyframe_id"]
            keyframes.append(kf_id)
            for fid in ev["cluster_frame_ids"]:
                if fid not in all_cluster_frames:
                    all_cluster_frames.append(fid)
    
    # Sort
    keyframes.sort()
    all_cluster_frames.sort()
    return keyframes, all_cluster_frames

def get_clean_background_frames(needed_count, existing_frames):
    all_frames = sorted(list(ALL_VIDEO_FRAMES_DIR.glob("frame_*.jpg")))
    available = []
    for p in all_frames:
        # Extract frame id: frame_00123_t0244.10.jpg
        stem = p.stem
        parts = stem.split("_")
        fid = int(parts[1])
        if fid in existing_frames:
            continue
        if EXCLUDED_FRAME_RANGE[0] <= fid <= EXCLUDED_FRAME_RANGE[1]:
            continue
        available.append(p)
    
    # Sample evenly
    if len(available) <= needed_count:
        return available
    
    step = len(available) / needed_count
    sampled = [available[int(i * step)] for i in range(needed_count)]
    return sampled

def frame_id_to_file(fid):
    # Find matching file in ALL_VIDEO_FRAMES_DIR
    pattern = f"frame_{fid:05d}_*.jpg"
    matches = list(ALL_VIDEO_FRAMES_DIR.glob(pattern))
    if matches:
        return matches[0]
    return None

def build_ablation_dataset(name, negative_files):
    dst_dir = OUTPUT_BASE / name
    print(f"\n==================================================")
    print(f"Building {name} with {len(negative_files)} negative images...")
    
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy val and test directly
    for split in ["val", "test"]:
        shutil.copytree(BASE_2CLASS / "images" / split, dst_dir / "images" / split)
        shutil.copytree(BASE_2CLASS / "labels" / split, dst_dir / "labels" / split)
    
    # Copy base train
    shutil.copytree(BASE_2CLASS / "images" / "train", dst_dir / "images" / "train")
    shutil.copytree(BASE_2CLASS / "labels" / "train", dst_dir / "labels" / "train")
    
    # Inject negatives into train
    dst_img_train = dst_dir / "images" / "train"
    dst_lbl_train = dst_dir / "labels" / "train"
    
    for neg_path in negative_files:
        dst_img = dst_img_train / f"hn_{neg_path.name}"
        dst_lbl = dst_lbl_train / f"hn_{neg_path.stem}.txt"
        shutil.copy2(neg_path, dst_img)
        # Create empty 0-byte label file
        dst_lbl.write_text("", encoding="utf-8")
    
    # Write yaml
    yaml_content = {
        "path": str(dst_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "drain_area",
            1: "drain_full"
        }
    }
    
    yaml_path = dst_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    train_imgs = len(list(dst_img_train.glob("*.*")))
    neg_ratio = len(negative_files) / train_imgs * 100
    print(f"Generated {name}: Total Train Images = {train_imgs} (Base: 372, Negatives: {len(negative_files)}, Ratio: {neg_ratio:.1f}%)")
    print(f"YAML: {yaml_path}")
    return yaml_path

def main():
    keyframes, all_cluster_frames = get_confirmed_fp_frames()
    print(f"Confirmed FP Keyframes ({len(keyframes)}): {keyframes}")
    print(f"Confirmed FP Clustered Frames ({len(all_cluster_frames)}): {all_cluster_frames}")
    
    # 0% Hard Negative (0 images)
    hn0_files = []
    
    # 5% Hard Negative (target 19 images ~ 5%)
    # 8 keyframes + 11 other cluster frames
    keyframe_files = [frame_id_to_file(fid) for fid in keyframes if frame_id_to_file(fid)]
    other_cluster_files = [frame_id_to_file(fid) for fid in all_cluster_frames if fid not in keyframes and frame_id_to_file(fid)]
    hn5_files = keyframe_files + other_cluster_files[:19 - len(keyframe_files)]
    
    # 10% Hard Negative (target 37 images ~ 10%)
    # All 22 cluster frames + 15 clean background frames
    all_cluster_file_list = [frame_id_to_file(fid) for fid in all_cluster_frames if frame_id_to_file(fid)]
    extra_15 = get_clean_background_frames(15, all_cluster_frames)
    hn10_files = all_cluster_file_list + extra_15
    
    # 20% Hard Negative (target 74 images ~ 20%)
    # All 22 cluster frames + 52 clean background frames
    extra_52 = get_clean_background_frames(52, all_cluster_frames)
    hn20_files = all_cluster_file_list + extra_52
    
    build_ablation_dataset("canonical_2class_hn0", hn0_files)
    build_ablation_dataset("canonical_2class_hn5", hn5_files)
    build_ablation_dataset("canonical_2class_hn10", hn10_files)
    build_ablation_dataset("canonical_2class_hn20", hn20_files)
    
    print("\nAll 4 Hard Negative Ablation Datasets built successfully!")

if __name__ == "__main__":
    main()
