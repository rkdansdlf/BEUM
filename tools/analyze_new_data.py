"""Quick frame extraction and inference for new data video"""

from pathlib import Path
import json
import cv2
import numpy as np
from ultralytics import YOLO


def extract_frames(video_path: Path, output_dir: Path, interval_s: float = 2.0, max_frames: int = 0):
    """Extract frames from video at given interval"""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"Video: {video_path.name}, FPS: {fps:.1f}, Frames: {total_frames}, Duration: {duration:.1f}s")
    
    frame_interval = int(fps * interval_s)
    if frame_interval <= 0:
        frame_interval = 1
    
    saved = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            time_s = frame_idx / fps
            filename = f"frame_{saved:05d}_t{time_s:07.2f}.jpg"
            out_path = output_dir / filename
            cv2.imwrite(str(out_path), frame)
            saved += 1
            
            if max_frames > 0 and saved >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {saved} frames to {output_dir}")
    return saved


def run_inference(model_path: Path, frames_dir: Path, output_json: Path, conf: float = 0.35, iou: float = 0.45):
    """Run YOLO inference on extracted frames"""
    model = YOLO(str(model_path))
    class_names = ["gully", "debris", "sediment"]
    
    image_files = sorted(frames_dir.glob("*.jpg"))
    print(f"Running inference on {len(image_files)} frames...")
    
    all_results = []
    for i, img_file in enumerate(image_files):
        frame = cv2.imread(str(img_file))
        if frame is None:
            continue
        
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=conf,
            iou=iou,
            device="cpu",
            verbose=False
        )
        
        if not results:
            continue
        
        result = results[0]
        names = result.names if hasattr(result, 'names') else {}
        
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = names.get(class_id, str(class_id))
            if class_name not in class_names:
                continue
            confidence = float(box.conf[0].item())
            bbox = tuple(float(v) for v in box.xyxy[0].tolist())
            
            detections.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "class_id": class_id
            })
        
        # Extract timestamp from filename
        time_str = img_file.stem.split('_t')[-1]
        try:
            timestamp = float(time_str)
        except:
            timestamp = i * 2.0  # fallback
        
        all_results.append({
            "image": img_file.name,
            "timestamp": timestamp,
            "detections": detections
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} frames")
    
    # Save results
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    total_detections = sum(len(r['detections']) for r in all_results)
    classes = {}
    for r in all_results:
        for d in r['detections']:
            classes[d['class']] = classes.get(d['class'], 0) + 1
    
    print(f"\n=== Inference Summary ===")
    print(f"Frames processed: {len(all_results)}")
    print(f"Total detections: {total_detections}")
    print(f"Class breakdown: {classes}")
    
    # Blockage analysis
    gully_count = sum(1 for r in all_results for d in r['detections'] if d['class'] == 'gully')
    debris_count = sum(1 for r in all_results for d in r['detections'] if d['class'] == 'debris')
    sediment_count = sum(1 for r in all_results for d in r['detections'] if d['class'] == 'sediment')
    
    print(f"\n=== Blockage Analysis ===")
    print(f"Gully detections: {gully_count}")
    print(f"Debris detections: {debris_count}")
    print(f"Sediment detections: {sediment_count}")
    
    return all_results


def main():
    video_path = Path("data/_1-15-2026-09-03_02-42-46/V20260903_114255000_6A562839-3B18-4C50-BC24-38C8DC496D65.MOV")
    frames_dir = Path("analysis/new_data_frames")
    model_path = Path("models/best-seg-3class.pt")
    output_json = Path("analysis/new_data_detections.json")
    
    print("=== Step 1: Extract frames ===")
    extract_frames(video_path, frames_dir, interval_s=2.0, max_frames=0)
    
    print("\n=== Step 2: Run YOLO inference ===")
    results = run_inference(model_path, frames_dir, output_json)
    
    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()