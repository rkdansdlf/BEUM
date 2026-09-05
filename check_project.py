#!/usr/bin/env python3
"""Check waterdrain project and provide reset guidance."""

from roboflow import Roboflow
import os

rf = Roboflow(api_key='MyrR6UFnOMOjR8gHw6MI')
project = rf.workspace().project('waterdrain')

print("=" * 60)
print("Waterdrain Project Analysis")
print("=" * 60)
print(f"\nProject name: {project.name}")
print(f"Project type: {project.type}")
print(f"Current classes: {project.classes}")
print(f"Classes as set: {set(project.classes)}")

expected = {"drain", "blockage_area"}
print(f"\nExpected classes (for upload): {expected}")
print(f"Match: {set(project.classes) == expected}")

print("\n" + "=" * 60)
print("Required Action: Reset Project Schema")
print("=" * 60)
print("\nThe upload tool requires exactly 2 classes: 'drain' and 'blockage_area'.")
print("Current waterdrain project has 4 classes: object, p, drain_area, drain_full")
print("\nTo fix this, you need to:")
print("1. Go to Roboflow web interface: https://roboflow.com/")
print("2. Navigate to your waterdrain project")
print("3. Open Project Settings → Classes")
print("4. Remove existing classes and add exactly:")
print("   - 'drain' (class ID 0)")
print("   - 'blockage_area' (class ID 1)")
print("5. Save the changes")
print("6. Re-run the upload script")

print("\n" + "=" * 60)
print("Alternative: Create new project")
print("=" * 60)
print("If resetting is too complex, you can:")
print("1. Create a new Roboflow project")
print("2. Upload your converted data with the correct 2-class schema")
print("3. Use the new project name in the upload command")

print("\n" + "=" * 60)
print("Upload Command After Schema Fix")
print("=" * 60)
print("""
python -m tools.upload_blockage_to_roboflow \\
  <image_dir> <label_dir> \\
  --project waterdrain \\
  --tag internet_integration_v1 \\
  --upload
""")

print("\nAfter schema is fixed and data uploaded, proceed with:")
print("  python -m tools.train_yolo_blockage --data dataset/blockage/gully-blockage.yaml")
print("  --model yolov8n-seg.pt --epochs 50 --imgsz 640 --batch 4 --device cpu")