#!/usr/bin/env python3
"""Phase 3: Model Retraining & Fine-tuning Script

Target: 3-class segmentation (grating, device, blockage) with hard negatives.
Trains on canonical dataset + hard negatives for 15-20 epochs.
"""

from pathlib import Path
from ultralytics import YOLO
import os

def main():
  # 1. 이전 Phase 검증 완료 모델 가중치
  # best.pt가 없으면 yolo11n-seg 로 시작 (프리트레인된 백본)
  pretrained_weights = 'models/best-seg-3class.pt'
  if not Path(pretrained_weights).exists():
    print(f"⚠️  Weight file not found: {pretrained_weights}")
    print("   Using YOLO11n-seg pretrained model instead...")
    pretrained_weights = 'yolo11n-seg'  # ultralytics Hub model name
  
  data_yaml = 'dataset/grating_device_blockage/gully-seg-3class.yaml'

  print(f'🚀 Starting Phase 3 Fine-tuning...')
  print(f'   Weights: {pretrained_weights}')
  print(f'   Data: {data_yaml}')
  print(f'   Hard negatives: dataset/hard_negatives/ (60 images)')

  # 2. 모델 로드
  print("📥 Loading model...")
  model = YOLO(pretrained_weights)

  # 3. 파인튜닝 하이퍼파라미터 설정
  print("🔧 Starting training...")
  results = model.train(
      data='datasets/grating_device_blockage/gully-seg-3class.yaml',
      epochs=20,  # 15~20 에포크 파인튜닝
      imgsz=640,  # 라즈베리 파이 배포를 고려하여 640 유지
      batch=16,  # GPU 메모리 상황에 따라 8 또는 16 설정
      patience=5,  # 조기 종료(Early stopping) 기준
      # 파인튜닝을 위한 학습률 조정
      lr0=0.001,  # 초기 학습률을 낮춰 기존 피처 보존
      lrf=0.01,  # 최종 학습률 비율 (lr0 * lrf)
      warmup_epochs=2,
      # 손실 가중치 (seg 파라미터는 Ultralytics YOLO에서 자동 처리)
      box=7.5,
      cls=0.5,
      # 증강 옵션 (현장 환경 대응)
      mosaic=0.5,  # 모자이크 증강 비중 적절히 유지
      mixup=0.1,  # 클래스 중첩 대응
      degrees=10.0,  # 미세 회전 대응
      perspective=0.0005,  # 카메라 각도 왜곡 대응
      # 저장 설정
      project='runs/train',
      name='phase3_3class_hard_neg',
      exist_ok=True,
      save=True,
      plots=True,
      save_txt=True,
      # Canonical segmentation contract (overlap_mask=False, mask_ratio=4)
      overlap_mask=False,
      mask_ratio=4,
  )

  print('\n✅ Phase 3 Training Complete!')
  print('📊 Results & Plots saved at: runs/train/phase3_3class_hard_neg/weights/')
  print('   - best.pt: 최적 가중치')
  print('   - last.pt: 마지막 에포크 가중치')
  print('   - labels.csv: 라벨 통계')
  print('   - results.png: 훈련 손실/정도 플롯')
  print('   - confusion_matrix.png: 혼동 행렬')

  
  # 4. 간단 성능 요약 출력
  try:
    # 훈련 이력 확인
    import pandas as pd
    results_log = Path(f'runs/train/phase3_3class_hard_neg/results.csv')
    if results_log.exists():
      df = pd.read_csv(results_log)
      final_metrics = df.iloc[-1]
      print(f'\n📈 Final Training Metrics:')
      print(f'   Box Loss: {df["box_loss"][-1]:.4f}')
      print(f'   Seg Loss: {df["seg_loss"][-1]:.4f}')
      print(f'   CLoss: {df["cls_loss"][-1]:.4f}')
      print(f'   LR: {df["lr"][-1]:.6f}')
  except Exception as e:
    print(f'   (Metrics summary available in plots/)')
    print(f'   (Check runs/train/phase3_3class_hard_neg/ for details)')

if __name__ == '__main__':
  main()