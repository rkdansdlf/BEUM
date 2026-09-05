"""Fine-tune on clean dataset - 20 epochs for faster results."""
from pathlib import Path
import sys
sys.path.insert(0, '.')
from ultralytics import YOLO

# Use the original best-seg-3class.pt as base (6.5 MB - this is the pre-trained model)
base_model = 'models/best-seg-3class.pt'
if not Path(base_model).exists():
    base_model = 'models/best-seg.pt'

print(f"Loading base model: {base_model}")
model = YOLO(base_model)

config_params = dict(
    data='dataset/clean/gully-seg-clean.yaml',
    epochs=20,
    imgsz=640,
    batch=8,
    device='cpu',
    name='retrain_clean_20ep',
    copy_paste=0.0,
    degrees=5.0,
    translate=0.05,
    scale=0.1,
    fliplr=0.5,
    close_mosaic=5,
    lr0=0.001,
    optimizer='AdamW',
    seed=42,
    patience=5,
    pretrained=True,
    overlap_mask=False,
    mask_ratio=4,
)

print("Starting fine-tuning on clean dataset (20 epochs)...")
print(f"Base model: {base_model}")
print(f"Dataset: dataset/clean/gully-seg-clean.yaml (372 train, 31 val)")
results = model.train(**config_params)
print("\nTraining complete.")
print(f"Best model: runs/segment/retrain_clean_20ep/weights/best.pt")
print(f"Last model: runs/segment/retrain_clean_20ep/weights/last.pt")