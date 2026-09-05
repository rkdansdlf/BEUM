#!/usr/bin/env python3
"""
Convert dataset labels from old class system (object, drain_area, drain_full)
to new domain-specific class system (grating, device, blockage).

Mapping:
  old 0 (object) -> new 1 (device)
  old 1 (drain_area) -> new 0 (grating)
  old 2 (drain_full) -> new 2 (blockage)

This script processes the canonical dataset and creates new labeled files.
"""

import os
import shutil
from pathlib import Path

# Class ID mapping: old_id -> new_id
CLASS_MAP = {
    0: 1,  # object -> device
    1: 0,  # drain_area -> grating
    2: 2,  # drain_full -> blockage
}

# Paths - canonical dataset structure: images/ and labels/ at root, with train/val/test subdirs
CANONICAL_DIR = Path(r"dataset/canonical")
OUTPUT_DIR = Path(r"dataset/grating_device_blockage")

# Create output directory structure
for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# Note: canonical dataset has labels at CANONICAL_DIR / "labels" / split
#        and images at CANONICAL_DIR / "images" / split
# Process each split
for split in ["train", "val", "test"]:
    label_dir = CANONICAL_DIR / "labels" / split
    img_dir = CANONICAL_DIR / "images" / split
    out_label_dir = OUTPUT_DIR / split / "labels"
    out_img_dir = OUTPUT_DIR / split / "images"

    if not label_dir.exists():
        print(f"  {split}: 라벨 디렉터리 없음, 건너뜀")
        continue

    txt_files = os.listdir(label_dir)
    print(f"  {split}: {len(txt_files)} 파일 처리")

    for txt_file in txt_files:
        txt_path = label_dir / txt_file
        img_name = txt_file.replace(".txt", "")
        img_path = img_dir / img_name

        # 이미지가 없으면 건너뜀
        if not img_path.exists():
            print(f"    경고: {img_name} 해당 이미지 없음, 건너뜀")
            continue

        # 새 이미지 파일 복사
        out_img = out_img_dir / img_name
        shutil.copy2(str(img_path), str(out_img))

        # 라벨 변환
        new_lines = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 9:  # class_id + 8 coordinates (4 points)
                    print(f"    경고: {txt_file} 형식이 올바르지 않음, 건너뜀: {line}")
                    continue

                old_class_id = int(parts[0])
                new_class_id = CLASS_MAP.get(old_class_id, -1)

                if new_class_id == -1:
                    print(f"    경고: 알 수 없는 클래스 ID {old_class_id}, 건너뜀")
                    continue

                # 새 라인 형식: new_class_id + 좌표들 그대로 유지
                new_line = f"{new_class_id} " + " ".join(parts[1:])
                new_lines.append(new_line)

        # 변환된 라벨 저장
        out_label_path = out_label_dir / txt_file
        with open(out_label_path, 'w', encoding='utf-8') as f:
            for nl in new_lines:
                f.write(nl + "\n")

    print(f"  {split}: 완료")

# 새 데이터셋 YAML 생성
yaml_path = OUTPUT_DIR / "gully-seg-3class.yaml"
NEW_YAML = """path: dataset/grating_device_blockage

train: images/train
val: images/val
test: images/test

nc: 3
names: ['grating', 'device', 'blockage']
"""
yaml_path.write_text(NEW_YAML)
print(f"\n새 데이터셋 YAML 생성: {yaml_path}")

# 클래스 이름 출력
print("\n클래스 매핑 요약:")
print("  old 0 (object) -> new 1 (device)")
print("  old 1 (drain_area) -> new 0 (grating)")
print("  old 2 (drain_full) -> new 2 (blockage)")
print("\n변환 완료!")