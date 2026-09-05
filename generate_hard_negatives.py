#!/usr/bin/env python3
import shutil, os, random
from pathlib import Path
import cv2
import numpy as np

src_dir = Path('dataset/canonical/images/train')
output_base = Path('dataset/hard_negatives')

categories = ['no_device_grating', 'background_simulation', 'blockage_variants']
np.random.seed(42)

for cat in categories:
    cat_dir = output_base / cat
    cat_dir.mkdir(parents=True, exist_ok=True)
    # Generate 20 samples per category
    for i in range(20):
        # Randomly pick a source image
        img_name = random.choice(list(src_dir.glob('*.jpg')))
        img = cv2.imread(str(img_name))
        if img is None:
            continue
        # Random transform
        transform_type = random.choice(['brightness', 'noise', 'perspective', 'none'])
        if transform_type == 'brightness':
            beta = random.uniform(-30, 30)
            augmented = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
        elif transform_type == 'noise':
            row, col, ch = img.shape
            mean = 0
            sigma = random.uniform(10, 50)
            gaussian_noise = np.random.normal(mean, sigma, (row, col, ch)).astype('int16')
            augmented = cv2.convertScaleAbs(img.astype('int16') + gaussian_noise)
        elif transform_type == 'perspective':
            h, w = img.shape[:2]
            pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
            pts2 = np.float32([[random.uniform(-20,20), random.uniform(-20,20)],
                           [w+random.uniform(-20,20), random.uniform(-20,20)],
                           [random.uniform(-20,20), h+random.uniform(-20,20)],
                           [w+random.uniform(-20,20), h+random.uniform(-20,20)]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            augmented = cv2.warpPerspective(img, matrix, (w,h))
        else:
            augmented = img.copy()
        
        # Save with random name
        aug_name = f'aug_{random.randint(10000,99999):05d}.jpg'
        save_path = cat_dir / aug_name
        cv2.imwrite(str(save_path), augmented)

print('Hard negative generation complete')
print('Checking counts...')
for cat in categories:
    count = len(list((output_base / cat).glob('*.jpg')))
    print(f'  {cat}: {count} images')