"""Run YOLO inference on pre-extracted frames and save results."""

import json
import cv2
from pathlib import Path


def main():
    model_path = Path("models/best-seg-3class.pt")
    frames_dir = Path("analysis/new_data_frames")
    output_json = Path("analysis/new_data_detections.json")

    model = None
    import ultralytics
    model = ultralytics.YOLO(str(model_path))

    frame_files = sorted(frames_dir.glob("*.jpg"))
    print(f"Processing {len(frame_files)} frames...")

    all_results = []
    for i, img_file in enumerate(frame_files):
        frame = cv2.imread(str(img_file))
        if frame is None:
            continue

        results = model.predict(source=frame, imgsz=640, conf=0.15, iou=0.45, device="cpu", verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0].item())
            bbox = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            detections.append({
                "class": cls_name,
                "confidence": round(confidence, 4),
                "bbox": bbox,
                "class_id": cls_id
            })

        time_str = img_file.stem.split('_t')[-1]
        try:
            timestamp = float(time_str)
        except ValueError:
            timestamp = round(i * 2.0, 2)

        all_results.append({
            "image": img_file.name,
            "timestamp": timestamp,
            "detections": detections
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(frame_files)} frames")

    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    total_det = sum(len(r['detections']) for r in all_results)
    classes = {}
    for r in all_results:
        for d in r['detections']:
            classes[d['class']] = classes.get(d['class'], 0) + 1
    print(f"\n=== Summary ===")
    print(f"Frames: {len(all_results)}")
    print(f"Total detections: {total_det}")
    print(f"Class counts: {classes}")
    print(f"Results saved to: {output_json}")


if __name__ == "__main__":
    main()