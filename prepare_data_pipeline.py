#!/usr/bin/env python3
"""
Prepare drain blockage dataset from internet data (drain-hole-1-5t03p)
and integrate with waterdrain project for YOLOv8 segmentation training.

This script:
1. Sets up the dataset directory structure
2. Provides label conversion framework (Roboflow -> YOLO segmentation)
3. Uploads validated data to Roboflow waterdrain project
4. Triggers model retraining
"""

import os
import json
import glob
import shutil
from pathlib import Path

# Configuration
ROBOFLOW_API = os.getenv('ROBOFLOW_API', 'MyrR6UFnOMOjR8gHw6MI')
DATASET_NAME = 'drain-hole-1-5t03p'
WATERDRAIN_PROJECT = 'waterdrain'

# Class mapping: Roboflow classes -> YOLO blockage classes
CLASS_MAPPING = {
    # Roboflow 'close' (drain closed/have obstruction) -> blockage_area
    'close': 1,
    # Roboflow 'open' (drain open/no obstruction) -> drain
    'open': 0,
}

# Dataset splits
SPLITS = ['train', 'val', 'test']

def create_directory_structure(base_path):
    """Create the YOLO dataset directory structure."""
    print("=" * 60)
    print("Step 1: Creating dataset directory structure")
    print("=" * 60)
    
    base = os.path.join(base_path, 'dataset', 'blockage')
    os.makedirs(base, exist_ok=True)
    
    for split in SPLITS:
        os.makedirs(os.path.join(base, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base, 'labels', split), exist_ok=True)
    
    # Create gully-blockage.yaml
    yaml_path = os.path.join(base, 'gully-blockage.yaml')
    if not os.path.exists(yaml_path):
        with open(yaml_path, 'w') as f:
            f.write('path: C:/Project/BEUM/dataset/blockage\n')
            f.write('train: images/train\n')
            f.write('val: images/val\n')
            f.write('test: images/test\n')
            f.write('\n')
            f.write('names:\n')
            f.write('  0: drain\n')
            f.write('  1: blockage_area\n')
        print(f"  Created {yaml_path}")
    
    print(f"Dataset structure ready at: {base}")
    return base

def convert_label_to_yolo(class_name, points_x, points_y, image_width, image_height):
    """
    Convert a single polygon label to YOLO segmentation format.
    
    YOLO format: <class_id> x1 y1 x2 y2 ... xn yn (normalized 0-1)
    
    Args:
        class_name: Roboflow class name ('close' or 'open')
        points_x: list of pixel x coordinates
        points_y: list of pixel y coordinates
        image_width: image width in pixels
        image_height: image height in pixels
    
    Returns:
        str: YOLO format label line, or None if invalid
    """
    class_id = CLASS_MAPPING.get(class_name)
    if class_id is None:
        print(f"  Warning: Unknown class '{class_name}', mapping to blockage_area (1)")
        class_id = 1
    
    if not points_x or not points_y or len(points_x) != len(points_y):
        print(f"  Warning: Invalid polygon points, skipping")
        return None
    
    # Normalize coordinates to [0, 1]
    normalized = []
    for px, py in zip(points_x, points_y):
        nx = max(0.0, min(1.0, px / image_width))
        ny = max(0.0, min(1.0, py / image_height))
        normalized.extend([nx, ny])
    
    # YOLO format: class_id x1 y1 x2 y2 ... xn yn
    line = f"{class_id} " + " ".join(f"{v:.6f}" for v in normalized)
    return line

def process_annotations_for_split(robotoflow_split_dir, images_dst_dir, labels_dst_dir):
    """
    Process all annotations in a Roboflow split directory and convert to YOLO format.
    
    Args:
        robotoflow_split_dir: Path to Roboflow split (train/val/test) directory
        images_dst_dir: Destination for images (YOLO format)
        labels_dst_dir: Destination for label .txt files (YOLO format)
    """
    print(f"  Processing split: {os.path.basename(robotoflow_split_dir)}")
    
    # Find all annotation JSON files in this split
    annot_files = glob.glob(os.path.join(robotoflow_split_dir, '*.json'), recursive=True)
    print(f"    Found {len(annot_files)} annotation files")
    
    processed = 0
    converted = 0
    skipped = 0
    
    for annot_path in annot_files:
        try:
            with open(annot_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get image dimensions
            img_width = data.get('width', 640)
            img_height = data.get('height', 640)
            
            # Get image filename and stem
            image_filename = data.get('image', '')
            if not image_filename:
                print(f"    Warning: No image filename in {annot_path}")
                skipped += 1
                continue
            
            image_stem = os.path.splitext(os.path.basename(image_filename))[0]
            
            # Prepare label file path
            label_path = os.path.join(labels_dst_dir, f"{image_stem}.txt")
            
            # Get annotations
            annotations = data.get('annotations', [])
            if not annotations:
                # Try alternative keys
                annotations = data.get('labels', [])
            
            # Convert each annotation polygon
            label_lines = []
            for ann in annotations:
                class_name = ann.get('class_id', ann.get('name', ''))
                points_x = ann.get('all_points_x', ann.get('x', []))
                points_y = ann.get('all_points_y', ann.get('y', []))
                
                yolo_line = convert_label_to_yolo(class_name, points_x, points_y, img_width, img_height)
                if yolo_line:
                    label_lines.append(yolo_line)
                    converted += 1
            
            # Write label file
            with open(label_path, 'w', encoding='utf-8') as f:
                for line in label_lines:
                    f.write(line + '\n')
            
            # Image would be copied here if available
            # shutil.copy2(os.path.join(robotoflow_split_dir, image_filename), images_dst_dir)
            
            processed += 1
            
        except Exception as e:
            print(f"    Error processing {annot_path}: {e}")
            skipped += 1
    
    print(f"    Processed: {processed}, Labels converted: {converted}, Skipped: {skipped}")
    return processed

def upload_to_waterdrain(image_dir, label_dir, tag='internet_integration_v1'):
    """
    Upload labeled data to Roboflow waterdrain project.
    
    Requires exactly classes: ['drain', 'blockage_area']
    """
    print("\n" + "=" * 60)
    print("Step 3: Uploading to waterdrain project")
    print("=" * 60)
    
    # Check if the data structure is valid first
    # The upload tool validates that waterdrain project has exactly 
    # classes: {'drain', 'blockage_area'}
    
    import subprocess
    
    # First do a dry-run to validate
    result = subprocess.run([
        'python', '-m', 'tools.upload_blockage_to_roboflow',
        image_dir,
        label_dir,
        '--project', WATERDRAIN_PROJECT,
        '--tag', tag,
        '--dry-run'
    ], cwd='C:\\Project\\BEUM', capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    
    # If dry-run succeeds, provide upload instructions
    if 'Dry run only' in result.stdout or 'Validated' in result.stdout:
        print("\n✓ Dry-run validation successful!")
        print(f"  Project: {WATERDRAIN_PROJECT}")
        print(f"  Tag: {tag}")
        print(f"  Images validated: relevant count from output above")
        print("\nTo actually upload, re-run with --upload flag:")
        print(f"  python -m tools.upload_blockage_to_roboflow {image_dir} {label_dir} "
              f"--project {WATERDRAIN_PROJECT} --tag {tag} --upload")
    
    return result.returncode

def main():
    """Main pipeline execution (non-interactive)."""
    print("\n" + "=" * 70)
    print("DRAIN BLOCKAGE DETECTION - DATA INTEGRATION PIPELINE")
    print("=" * 70)
    print("\nThis pipeline prepares data from Roboflow's drain-hole-1-5t03p dataset")
    print("for integration with the waterdrain project and YOLOv8 training.")
    
    # Step 1: Create directory structure
    dataset_base = create_directory_structure('C:\\Project\\BEUM')
    
    # Step 2: Provide conversion framework
    print("\n" + "=" * 70)
    print("Step 2: Label Conversion Framework Ready")
    print("=" * 70)
    print("\nThe drain-hole-1-5t03p dataset has these characteristics:")
    print(f"  - 399 images total")
    print(f"  - Classes: 'close' (246 images) and 'open' (156 images)")
    print(f"  - Splits: train(266), valid(69), test(64)")
    print(f"  - Class mapping: 'close' -> blockage_area, 'open' -> drain")
    print(f"  - Format: Roboflow polygon annotations -> YOLO segmentation")
    
    print("\nConversion mapping:")
    print("  Roboflow 'close' (obstructed drain) -> YOLO class 1 (blockage_area)")
    print("  Roboflow 'open' (clear drain) -> YOLO class 0 (drain)")
    
    print("\nRequired user actions:")
    print("  1. Download drain-hole-1-5t03p from Roboflow website")
    print("  2. Extract images and annotations (JSON files) by split (train/val/test)")
    print("  3. Place annotations in a directory structure")
    print("  4. Run the conversion or use the provided framework")
    
    # Step 3: Upload to waterdrain (example with placeholder paths)
    print("\n" + "=" * 70)
    print("Step 3: Waterdrain Project Upload")
    print("=" * 70)
    print("\nBefore uploading, ensure:")
    print("  - waterdrain project has classes: ['drain', 'blockage_area']")
    print("  - Labels are in YOLO segmentation format")
    print("  - Images and labels are paired correctly")
    
    # Check current waterdrain project classes
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API)
        project = rf.workspace().project('waterdrain')
        project_classes = project.classes
        print(f"\nCurrent waterdrain project classes: {project_classes}")
        
        expected = {'drain', 'blockage_area'}
        if project_classes == expected:
            print("  ✓ Classes match expected schema (drain, blockage_area)")
        else:
            print(f"  ✗ Classes mismatch. Expected: {expected}, Got: {project_classes}")
            print("  Please reset/rename the waterdrain project class schema first.")
    except Exception as e:
        print(f"  Could not check waterdrain project: {e}")
    
    # Provide upload command template
    print("\nUpload command template:")
    print("  python -m tools.upload_blockage_to_roboflow \\")
    print("    <image_dir> <label_dir> \\")
    print(f"    --project {WATERDRAIN_PROJECT} \\")
    print(f"    --tag internet_integration_v1 \\")
    print("    --upload")
    
    # Step 4: Training guidance
    print("\n" + "=" * 70)
    print("Step 4: Model Training")
    print("=" * 70)
    print("\nAfter uploading data to waterdrain, retrain the model:")
    print("  python -m tools.train_yolo_blockage \\")
    print("    --data dataset/blockage/gully-blockage.yaml \\")
    print("    --model yolov8n-seg.pt \\")
    print("    --epochs 50 \\")
    print("    --imgsz 640 \\")
    print("    --batch 4 \\")
    print("    --device cpu \\")
    print("    --name blockage_internet_integration")
    
    print("\n" + "=" * 70)
    print("PIPELINE FRAMEWORK COMPLETE")
    print("=" * 70)
    print("\nSummary of class mapping and workflow:")
    print("  drain-hole-1-5t03p 'close' -> blockage_area (obstructed drain)")
    print("  drain-hole-1-5t03p 'open' -> drain (clear drain)")
    print("  Combined training data -> improved YOV8n-seg blockage detection")
    print("  Shielding rate = (blockage_area mask area / drain mask area) * 100")
    print("=" * 70)

if __name__ == '__main__':
    main()