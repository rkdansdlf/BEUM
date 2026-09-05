"""
Builds a pristine 2-class canonical dataset from dataset/canonical.
Class 1 (drain_area) -> Class 0 (drain_area)
Class 2 (drain_full) -> Class 1 (drain_full)
"""

import shutil
from pathlib import Path
import yaml

SRC_DIR = Path("dataset/canonical")
DST_DIR = Path("dataset/canonical_2class")

SPLITS = ["train", "val", "test"]

def main():
    print(f"Building Canonical 2-Class Dataset in {DST_DIR}...")
    
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    for split in SPLITS:
        src_img_dir = SRC_DIR / "images" / split
        src_lbl_dir = SRC_DIR / "labels" / split
        
        dst_img_dir = DST_DIR / "images" / split
        dst_lbl_dir = DST_DIR / "labels" / split
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        split_counts = {"drain_area": 0, "drain_full": 0, "images": 0}
        
        img_files = list(src_img_dir.glob("*.*"))
        for img_path in img_files:
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            
            # Copy image
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            split_counts["images"] += 1
            
            # Process label
            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
            dst_lbl_path = dst_lbl_dir / f"{img_path.stem}.txt"
            
            if not lbl_path.exists():
                # Empty label file for background
                dst_lbl_path.write_text("", encoding="utf-8")
                continue
            
            new_lines = []
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                coords = parts[1:]
                
                if cid == 1:  # drain_area -> 0
                    new_lines.append(f"0 {' '.join(coords)}")
                    split_counts["drain_area"] += 1
                elif cid == 2:  # drain_full -> 1
                    new_lines.append(f"1 {' '.join(coords)}")
                    split_counts["drain_full"] += 1
                else:
                    print(f"Warning: Unexpected class {cid} in {lbl_path}")
            
            dst_lbl_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        
        stats[split] = split_counts
        print(f"[{split.upper()}] Images: {split_counts['images']}, drain_area (0): {split_counts['drain_area']}, drain_full (1): {split_counts['drain_full']}")
    
    # Write dataset yaml
    yaml_content = {
        "path": str(DST_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "drain_area",
            1: "drain_full"
        }
    }
    
    yaml_path = DST_DIR / "canonical-2class.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print(f"\nCanonical 2-class dataset generated at {DST_DIR}")
    print(f"YAML configuration written to: {yaml_path}")

if __name__ == "__main__":
    main()
