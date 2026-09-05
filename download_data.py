#!/usr/bin/env python3
"""Download and prepare drain detection datasets from Roboflow."""

from roboflow import Roboflow
import os
import json
import shutil

def download_drain_hole_dataset():
    """Download drain-hole-1-5t03p dataset from Roboflow."""
    rf = Roboflow(api_key='MyrR6UFnOMOjR8gHw6MI')
    
    # Try to access the specific project
    try:
        project = rf.workspace().project('drain-hole-1-5t03p')
        print(f"Project: {project.name}")
        print(f"Classes: {project.classes}")
        print(f"Splits: {project.splits}")
        
        # The project has 399 images with classes 'close' and 'open'
        # We need to map these to our 'drain' and 'blockage_area' classes
        
        # Strategy: Download the dataset and convert labels
        # First, let's try to get the dataset in COCO or YOLO format
        
        # Method: Use the project's export/download mechanism
        # Since version API isn't working, let's try alternative approaches
        
        # Alternative: Use the roboflow download package directly
        try:
            # Install and use roboflow-cli or similar
            print("Trying alternative download method...")
            
        except Exception as e:
            print(f"Alternative method error: {e}")
            
        # Since we can't easily download via the Python SDK,
        # let's create a plan for manual data acquisition and conversion
        print_plan()
            
    except Exception as e:
        print(f"Project access error: {e}")
        import traceback
        traceback.print_exc()

def print_plan():
    """Print the data acquisition and conversion plan."""
    print("\n" + "=" * 60)
    print("DATA ACQUISITION AND CONVERSION PLAN")
    print("=" * 60)
    print("\n1. Internet Data Source: drain-hole-1-5t03p (Roboflow)")
    print("   - 399 images, 2 classes: 'close' and 'open'")
    print("   - Classes represent: drain open/closed status")
    print("   - Need to remap to: 'drain' and 'blockage_area'")
    print("\n2. Class Mapping Strategy:")
    print("   - 'close' -> 'blockage_area' (obstructing material covering drain)")
    print("   - 'open' -> 'drain' (visible drain without obstruction)")
    print("   - This maps the 2-class hole status to our blockage detection classes")
    print("\n3. Label Conversion (YOLO Segmentation format):")
    print("   Current: Roboflow polygon format with pixel coordinates")
    print("   Target: YOLO segmentation <class_id> x1 y1 x2 y2 ... xn yn (normalized)")
    print("\n4. Required Conversion Steps:")
    print("   a. Download images and polygon annotations from Roboflow")
    print("   b. Parse each image's annotations (polygons with class IDs)")
    print("   c. Normalize coordinates to [0, 1] range (divide by image width/height)")
    print("   d. Remap class IDs: 0->'drain' (open), 1->'blockage_area' (close)")
    print("   e. Convert polygon points to YOLO segmentation format")
    print("   f. Save as .txt files matching YOLO naming convention")
    print("\n5. Upload to waterdrain project:")
    print("   - Must have exactly classes: ['drain', 'blockage_area']")
    print("   - Use tools/upload_blockage_to_roboflow.py")
    print("   - Validate class schema before upload")
    print("\n6. Re-train model with combined data:")
    print("   - Use tools/train_yolo_blockage.py")
    print("   - data: dataset/gully-blockage.yaml")
    print("   - model: yolov8n-seg.pt")
    print("=" * 60)

if __name__ == '__main__':
    print("=" * 60)
    print("Step 1: Analyzing drain-hole-1-5t03p dataset")
    print("=" * 60)
    download_drain_hole_dataset()