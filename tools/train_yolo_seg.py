import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.validate_yolo_seg import validate_split


DATASET_ROOT = Path(r'C:\Project\BEUM\dataset\five_class')


def ensure_dataset_ready() -> None:
    for split in ('train', 'val'):
        result = validate_split(DATASET_ROOT, split, class_count=5)
        if result['missing'] or result['extra'] or result['invalid']:
            raise RuntimeError(
                f"Five-class {split} split is incomplete or invalid: "
                f"missing={result['missing']} extra={result['extra']} "
                f"invalid={len(result['invalid'])}"
            )
        if not result['classes']:
            raise RuntimeError(f"Five-class {split} split has no annotated polygons")

os.environ["YOLO_VERBOSE"] = "False"

def main():
    device = 'cpu'
    ensure_dataset_ready()

    model = YOLO('yolov8n-seg.pt')
    print(f"Using device: {device}")

    results = model.train(
        data=r'C:\Project\BEUM\dataset\gully-seg.yaml',
        epochs=100,
        imgsz=640,
        batch=4,
        device=device,
        project=r'C:\Project\BEUM\models',
        name='yolo_seg_5class',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        patience=20,
        verbose=True,
        workers=2,
        cache=False,
        plots=False,
        save=True,
        save_period=5,
    )

    print(f"\nTraining complete!")
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    last_model_path = os.path.join(results.save_dir, 'weights', 'last.pt')

    if os.path.exists(best_model_path):
        final_path = r'C:\Project\BEUM\models\best-seg-5class.pt'
        shutil.copy2(best_model_path, final_path)
        print(f"Best model copied to: {final_path}")
    elif os.path.exists(last_model_path):
        final_path = r'C:\Project\BEUM\models\best-seg-5class.pt'
        shutil.copy2(last_model_path, final_path)
        print(f"Last model copied to: {final_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
