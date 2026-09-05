#!/usr/bin/env python3
"""
Visual QA & Inference Verification Script for YOLO Segmentation
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def run_visual_qa(
    weights_path: str = "runs/segment/finetune_roboflow_30ep-3/weights/best.pt",
    source_dir: str = "dataset/roboflow_finetune/test/images",
    project_dir: str = "runs/segment",
    name_dir: str = "visual_qa_test",
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.45,
    device: str = ""
):
    model_file = Path(weights_path)
    source_path = Path(source_dir)
    output_path = Path(project_dir) / name_dir

    # 1. 파일 및 디렉터리 검증
    if not model_file.exists():
        print(f"[Error] 모델 가중치를 찾을 수 없습니다: {model_file.resolve()}")
        sys.exit(1)

    if not source_path.exists():
        print(f"[Error] 테스트 이미지 경로를 찾을 수 없습니다: {source_path.resolve()}")
        sys.exit(1)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in source_path.iterdir() if p.suffix.lower() in valid_extensions]
    if not images:
        print(f"[Error] 테스트 경로 내 이미지 파일이 없습니다: {source_path.resolve()}")
        sys.exit(1)

    # 결과 디렉터리 미리 정리/생성 (중복 경로 방지)
    # project_dir가 절대경로이든 상대경로이든 normalizer
    project_dir_norm = Path(project_dir).resolve()
    output_path = project_dir_norm / name_dir
    # 상위 프로젝트 폴더가 없으면 생성
    project_dir_norm.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" [Visual QA] 세그멘테이션 모델 추론 및 결과 분석 시작")
    print("=" * 70)
    print(f" * 가중치 경로     : {model_file}")
    print(f" * 대상 이미지 수   : {len(images)}장")
    print(f" * 신뢰도(Conf)     : {conf_thresh}")
    print(f" * NMS IoU         : {iou_thresh}")
    print(f" * 결과 저장 위치   : {output_path}")
    print("-" * 70)

    # 2. 모델 로드
    model = YOLO(str(model_file))

    # 3. 일괄 추론 실행
    start_time = time.time()
    # save=True 시 runs/segment/... 를 생성하므로,이미 존재하는 경우 깨끗이 정리 후 진행
    import shutil
    if output_path.exists():
        shutil.rmtree(output_path)
    results = model.predict(
        source=str(source_path),
        conf=conf_thresh,
        iou=iou_thresh,
        retina_masks=True,
        save=True,
        project=str(project_dir_norm),
        name=name_dir,
        device=device,
        exist_ok=True,
        verbose=False
    )
    total_infer_time = time.time() - start_time

    # 4. 추론 결과 통계 수집
    records = []
    class_counter = {}
    images_with_detections = 0
    zero_detection_images = []

    for r in results:
        img_name = Path(r.path).name
        orig_shape = r.orig_shape  # (H, W)
        total_pixels = orig_shape[0] * orig_shape[1]

        boxes = r.boxes
        masks = r.masks

        num_det = len(boxes) if boxes is not None else 0
        if num_det > 0:
            images_with_detections += 1
        else:
            zero_detection_images.append(img_name)

        det_info = []
        if num_det > 0 and masks is not None:
            mask_data = masks.data.cpu().numpy()  # (N, H, W)
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for i in range(num_det):
                cls_name = model.names[cls_ids[i]]
                conf = float(confs[i])
                class_counter[cls_name] = class_counter.get(cls_name, 0) + 1

                # 마스크 픽셀 면적 계산
                binary_mask = (mask_data[i] > 0.5).astype(np.uint8)
                pixel_area = int(np.sum(binary_mask))
                coverage_pct = (pixel_area / total_pixels) * 100.0

                det_info.append(f"{cls_name}({conf:.2f}, {coverage_pct:.1f}%)")

        det_str = ", ".join(det_info) if det_info else "미검출"
        records.append({
            "image": img_name,
            "detections": num_det,
            "details": det_str
        })

    # 5. 요약 리포트 파일 작성 (Markdown 형식)
    report_file = output_path / "qa_summary_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Visual QA 추론 검증 리포트\n\n")
        f.write(f"- **실행 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **사용 모델**: `{model_file.name}`\n")
        f.write(f"- **총 이미지 수**: {len(images)}장\n")
        f.write(f"- **검출 성공**: {images_with_detections}장 (미검출: {len(zero_detection_images)}장)\n")
        f.write(f"- **총 추론 소요 시간**: {total_infer_time:.2f}초 (평균 {total_infer_time / len(images) * 1000:.1f} ms/img)\n\n")

        f.write("### 클래스별 검출 객체 수\n")
        if class_counter:
            for cls_name, count in sorted(class_counter.items()):
                f.write(f"- `{cls_name}`: {count}건\n")
        else:
            f.write("- 검출된 객체 없음\n")

        if zero_detection_images:
            f.write("\n### 미검출 이미지 목록\n")
            for z_img in zero_detection_images:
                f.write(f"- {z_img}\n")

        f.write("\n### 이미지별 세부 검출 내역\n")
        f.write("| 파일명 | 검출 수 | 세부 내역 (클래스, 신뢰도, 면적 점유율) |\n")
        f.write("|---|:---:|---|\n")
        for rec in records:
            f.write(f"| {rec['image']} | {rec['detections']} | {rec['details']} |\n")

    # 6. 콘솔 요약 출력
    print("=" * 70)
    print(" [QA 분석 완료 결과]")
    print(f" * 처리 완료 이미지 : {len(images)}장")
    print(f" * 객체 검출 성공   : {images_with_detections}장 / {len(images)}장 ({images_with_detections / len(images) * 100:.1f}%)")
    print(f" * 미검출 이미지     : {len(zero_detection_images)}장")
    print(f" * 클래스별 카운트   : {dict(class_counter)}")
    print(f" * 소요 시간        : {total_infer_time:.2f}초 ({total_infer_time / len(images) * 1000:.1f} ms/장)")
    print(f" * 결과 이미지 위치 : {output_path.resolve()}")
    print(f" * 상세 리포트 위치 : {report_file.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO Segment Visual QA runner")
    parser.add_argument("--weights", default="runs/segment/finetune_roboflow_30ep-3/weights/best.pt", help="가중치 파일 경로")
    parser.add_argument("--source", default="dataset/roboflow_finetune/test/images", help="테스트 이미지 디렉터리")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (기본값: 0.5)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (기본값: 0.45)")
    parser.add_argument("--device", default="", help="CPU or CUDA device id (예: 'cpu', '0')")
    parser.add_argument("--name", default="visual_qa_test", help="출력 폴더 이름")

    args = parser.parse_args()
    run_visual_qa(
        weights_path=args.weights,
        source_dir=args.source,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device=args.device,
        name_dir=args.name
    )