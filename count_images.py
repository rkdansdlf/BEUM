import os
import glob

# Count canonical images
canonical_dir = 'dataset/canonical'
for split in ['train', 'val', 'test']:
    img_dir = os.path.join(canonical_dir, split, 'images')
    if os.path.exists(img_dir):
        jpgs = glob.glob(os.path.join(img_dir, '*.jpg'))
        pngs = glob.glob(os.path.join(img_dir, '*.png'))
        print(f'canonical/{split}/images: {len(jpgs) + len(pngs)} files')
    else:
        print(f'canonical/{split}/images: NOT FOUND')

# Count in grating_device_blockage
blockage_dir = 'dataset/grating_device_blockage'
for split in ['train', 'val', 'test']:
    img_dir = os.path.join(blockage_dir, split, 'images')
    lbl_dir = os.path.join(blockage_dir, split, 'labels')
    if os.path.exists(img_dir):
        jpgs = glob.glob(os.path.join(img_dir, '*.jpg'))
        pngs = glob.glob(os.path.join(img_dir, '*.png'))
        print(f'blockage/{split}/images: {len(jpgs) + len(pngs)} files')
        lbls = glob.glob(os.path.join(lbl_dir, '*.txt')) if os.path.exists(lbl_dir) else []
        print(f'blockage/{split}/labels: {len(lbls)} files')
    else:
        print(f'blockage/{split}/images: NOT FOUND')

# Hard negatives
hn_dir = 'dataset/hard_negatives'
if os.path.exists(hn_dir):
    for cat in sorted(os.listdir(hn_dir)):
        cat_path = os.path.join(hn_dir, cat)
        if os.path.isdir(cat_path):
            imgs = [f for f in os.listdir(cat_path) if f.endswith('.jpg') or f.endswith('.png')]
            print(f'hard_negatives/{cat}/: {len(imgs)} images')