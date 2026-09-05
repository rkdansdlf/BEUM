from ultralytics import YOLO

model = YOLO('runs/segment/finetune_roboflow_30ep-3/weights/best.pt')
results = model.val(data='dataset/roboflow_finetune/gully-seg-roboflow.yaml', imgsz=640, conf=0.10, verbose=False)
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map50_95:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")