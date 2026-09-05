"""Fine-tune YOLOv8 segmentation model with Roboflow data.

This script fine-tunes a pre-trained model on the Roboflow fine-tune dataset,
which contains 165 train images with drain_area segmentation masks.

The data is in dataset/roboflow_finetune/ created by convert_roboflow_to_yolo.py
"""

from pathlib import Path
import sys

sys.path.insert(0, '.')

from ultralytics import YOLO

# Configuration for Roboflow fine-tuning
# Based on the converted data: 1 class (drain_area) mapped to class_id 0
CONFIG = dict(
    data='dataset/roboflow_finetune/gully-seg-roboflow.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    device='cpu',
    name='finetune_roboflow_30ep',
    copy_paste=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    fliplr=0.5,
    close_mosaic=10,
    lr0=0.001,
    optimizer='AdamW',
    seed=42,
    overlap_mask=False,
    mask_ratio=4,
)

# Use the 3-class model pre-trained on clean data as base
# This already has the class architecture that can adapt to new data
base_model = 'models/best-seg-3class.pt'

print(f"Starting fine-tuning on Roboflow data...")
print(f"Base model: {base_model}")
print(f"Dataset: dataset/roboflow_finetune/ (165 train, 8 test)")
print(f"Data classes: drain_area (class_id 0)")
print(f"Config: {CONFIG}")

model = YOLO(base_model)

print("\nStarting fine-tuning...")
results = model.train(**CONFIG)

print("\nTraining complete.")
print(f"Best model: runs/segment/finetune_roboflow_30ep/weights/best.pt")
print(f"Last model: runs/segment/finetune_roboflow_30ep/weights/last.pt")