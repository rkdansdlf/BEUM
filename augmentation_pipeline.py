#!/usr/bin/env python3
"""
Albumentations-based Segmentation Augmentation Pipeline for Gutter Detection

This pipeline is designed for:
- Railway/Edge deployment (Raspberry Pi 5 compatible)
- Train/Val/Test split consistency
- Hard negative mining support
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from pathlib import Path

def create_augmentation_pipelines(
    img_size: int = 640,
    train_augmentation: bool = True,
) -> dict:
    """
    Create train and validation augmentation pipelines.
    
    Args:
        img_size: Target image size for resizing
        train_augmentation: Whether to apply training augmentations
    
    Returns:
        Dictionary with 'train' and 'val' transform keys
    """
    
    if train_augmentation:
        # Strong training augmentations for overfitting prevention
        train_transform = A.Compose([
            # 1. 조도 및 색상 변환 (강도 0.5)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, 
                    sat_shift_limit=15, 
                    val_shift_limit=10, 
                    p=1.0
                ),
            ], p=0.5),
            
            # 2. 노이즈 및 블러 (강도 0.3) - 모션 블러는 엣지 보존에 도움
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.3),
            
            # 3. 기하학적 변환 (강도 0.5)
            # - ShiftScaleRotate: 카메라 위치 변화 시뮬레이션
            # - Perspective: 원근 변화 시뮬레이션 (배수구 가장자리 효과)
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05, 
                rotate_limit=15, 
                p=1.0, 
                border_mode=cv2.BORDER_REFLECT_101
            ),
            A.Perspective(
                scale=(0.02, 0.05), 
                p=0.3, 
                border_mode=cv2.BORDER_REFLECT_101
            ),
            
            # 4. 정규화 ( 항상 적용)
            A.Normalize(
                mean=[0.0, 0.0, 0.0], 
                std=[1.0, 1.0, 1.0], 
                max_pixel_value=255.0
            ),
            # 5. Tensor 변환 (PyTorch 호환)
            ToTensorV2(),
        ], additional_targets={'mask': 'image'})
        
        print(f"✅ Train augmentation pipeline created ({len(train_transform.transforms)} ops)")
        
    else:
        # Minimal validation augmentation
        train_transform = None
    
    # Validation: minimal augmentation, just resize + normalize
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.0, 0.0, 0.0], 
            std=[1.0, 1.0, 1.0], 
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ], additional_targets={'mask': 'image'})
    
    print(f"✅ Val augmentation pipeline created ({len(val_transform.transforms)} ops)")
    
    return {
        "train": train_transform,
        "val": val_transform
    }


def create_hard_negative_augmentation(
    source_dir: Path,
    output_dir: Path,
    num_augmented: int = 20,
    seed: int = 42,
) -> None:
    """
    Generate hard negative augmented samples from source images.
    
    This is specifically for creating hard negatives:
    - Uninstalled grating images (no device)
    - Background simulation images
    - Blockage variant images
    
    Args:
        source_dir: Source images directory (should have .jpg/.png files)
        output_dir: Output directory for augmented samples
        num_augmented: Number of augmented samples per source image
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Get all source images
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    source_imgs = [
        f for f in os.listdir(source_dir) 
        if Path(f).suffix.lower() in img_extensions
    ]
    
    print(f"📁 Source images found: {len(source_imgs)}")
    
    # Create output subdirectories
    output_subdirs = [
        "no_device_grating",
        "background_simulation", 
        "blockage_variants"
    ]
    for subdir in output_subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    train_transform, val_transform = create_augmentation_pipelines(train_augmentation=True)
    
    augmented_count = 0
    
    for img_name in source_imgs:
        img_path = source_dir / img_name
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is None:
            print(f"⚠️  Failed to read: {img_name}")
            continue
        
        # Generate augmented versions
        for i in range(num_augmented):
            # Randomly choose which augmentation pipeline to apply
            # Hard negatives benefit from stronger augmentations
            if np.random.random() > 0.5:
                transformed = train_transform(image=img, mask=np.random.randint(0, 255, img.shape[:2]))
            else:
                transformed = val_transform(image=img, mask=np.random.randint(0, 255, img.shape[:2]))
            
            augmented_img = transformed['image']
            
            # Save augmented image
            aug_name = f"{Path(img_name).stem}_aug_{i:03d}{Path(img_name).suffix}"
            
            # Distribute into hard negative categories
            category = np.random.choice([
                "no_device_grating",     # 50%
                "background_simulation", # 30%  
                "blockage_variants"      # 20%
            ])
            
            save_path = output_dir / category / aug_name
            # Convert RGB back to BGR for OpenCV imwrite
            aug_img_bgr = cv2.cvtColor(augmented_img.numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), aug_img_bgr)
            
            augmented_count += 1
    
    print(f"✅ Hard negative augmentation complete: {augmented_count} samples generated")
    print(f"   - no_device_grating: {len(os.listdir(output_dir / 'no_device_grating'))} samples")
    print(f"   - background_simulation: {len(os.listdir(output_dir / 'background_simulation'))} samples")
    print(f"   - blockage_variants: {len(os.listdir(output_dir / 'blockage_variants'))} samples")


if __name__ == "__main__":
    # Create augmentation pipelines
    pipelines = create_augmentation_pipelines(img_size=640, train_augmentation=True)
    
    # Setup output directories
    output_base = Path("dataset/hard_negatives")
    
    # Generate hard negative samples (example: 20 augmented per source image)
    # NOTE: Replace 'dataset/hard_negatives_source' with actual source directory
    create_hard_negative_augmentation(
        source_dir=Path("dataset/hard_negatives_source"),
        output_dir=Path("dataset/hard_negatives"),
        num_augmented=20,
        seed=42
    )
    
    print("\n🚀 Augmentation pipeline ready for Phase 2 deployment")
    print("   - Train transforms: Strong augmentations for overfitting prevention")
    print("   - Val transforms: Minimal augmentation for reliable evaluation")
    print("   - Hard negative categories: no_device_grating, background_simulation, blockage_variants")