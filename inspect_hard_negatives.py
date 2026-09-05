#!/usr/bin/env python3
"""
Hard Negative Sample Visual Inspection Script (Enhanced)
Supports both YOLO segmentation (.txt polygon) and binary mask (.png).
"""

import os
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CATEGORIES = {
    'no_device_grating': 'grating only (no device)',
    'background_simulation': 'background only (no labels)',
    'blockage_variants': 'blockage (leaves/trash/mud)',
}

# BGR colors for OpenCV overlay
CATEGORY_COLORS = {
    'no_device_grating': (0, 255, 0),  # Green for grating
    'background_simulation': (255, 0, 0),  # Blue
    'blockage_variants': (0, 0, 255),  # Red for blockage
}

FIG_SIZE = (16, 12)


def load_mask(img_path, img_shape, cat_name):
  """Loads mask from .png or parses YOLO polygon from .txt."""
  h, w = img_shape[:2]
  mask = np.zeros((h, w), dtype=np.uint8)

  # 1. Check for .png mask
  png_path = img_path.with_suffix('.png')
  if png_path.exists():
    loaded = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if loaded is not None:
      return loaded

  # 2. Check for YOLO .txt label (class x1 y1 x2 y2 ...)
  txt_path = img_path.with_suffix('.txt')
  # Also check labels/ directory if images are in images/
  if img_path.parent.name == 'images':
    txt_alt = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
    if txt_alt.exists():
      txt_path = txt_alt
  
  if txt_path.exists():
    with open(txt_path, 'r') as f:
      for line in f:
        parts = line.strip().split()
        if len(parts) >= 5:
          coords = np.array(
              [float(x) for x in parts[1:]], dtype=np.float32
          ).reshape(-1, 2)
          pts = (coords * [w, h]).astype(np.int32)
          cv2.fillPoly(mask, [pts], 255)

  return mask


def render_mask_overlay(image, mask, cat_name, alpha=0.4):
  color = CATEGORY_COLORS.get(cat_name, (0, 255, 0))
  colored_mask = np.zeros_like(image)
  colored_mask[mask > 0] = color
  return cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)


def main():
  base_dir = Path('dataset/hard_negatives')

  for cat_name, cat_desc in CATEGORIES.items():
    cat_dir = base_dir / cat_name
    img_dir = cat_dir / 'images' if (cat_dir / 'images').exists() else cat_dir

    if not img_dir.exists():
      continue

    img_files = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg']]
    )
    if not img_files:
      print(f'⚠️ No images found in {cat_name}')
      continue

    print(f'Rendering {len(img_files)} images for: {cat_name}')
    n_images = len(img_files)
    n_rows = (n_images + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=FIG_SIZE)
    fig.suptitle(
        f'Hard Negative Inspection: {cat_name} ({cat_desc})', fontsize=16
    )

    axes_flat = axes.flatten() if n_rows > 1 else np.array(axes).flatten()

    for idx, img_path in enumerate(img_files):
      idx = min(idx, len(axes_flat) - 1)  # Safety bound
      img = cv2.imread(str(img_path))
      if img is None:
        continue
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      mask = load_mask(img_path, img.shape, cat_name)
      overlay = render_mask_overlay(img_rgb, mask, cat_name)

      ax = axes_flat[idx]
      ax.imshow(overlay)
      ax.set_title(f'{idx+1}: {img_path.name[:20]}...', fontsize=8)
      ax.axis('off')

    # Turn off unused subplots
    for ax in axes_flat[n_images:]:
      ax.axis('off')

    plt.tight_layout()
    output_path = Path(f'visual_inspection_{cat_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {output_path.name}')

  print('\nVisual inspection grid creation complete!')


if __name__ == '__main__':
  main()