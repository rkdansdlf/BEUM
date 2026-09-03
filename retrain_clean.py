"""Fine-tune best-seg-3class model on clean dataset (no pothole contamination)."""
import sys
from pathlib import Path
sys.path.insert(0, '.')

from ultralytics import YOLO
import yaml

model = YOLO('models/best-seg-3class.pt')

config_params = dict(
    data='C:/Project/BEUM/dataset/clean/gully-seg-clean.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    device='cpu',
    name='retrain_clean_50ep',
    copy_paste=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    fliplr=0.5,
    close_mosaic=10,
    lr0=0.001,
    optimizer='AdamW',
    seed=42
)

print("Starting fine-tuning on clean dataset...")
print(f"Base model: models/best-seg-3class.pt")
print(f"Dataset: dataset/clean/gully-seg-clean.yaml (372 train, 31 val)")
results = model.train(**config_params)
print("Training complete.")
print(f"Best model: runs/segment/retrain_clean_50ep/weights/best.pt")
print(f"Last model: runs/segment/retrain_clean_50ep/weights/last.pt")