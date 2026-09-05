import os
import shutil

src = 'dataset/grating_device_blockage'
dst = 'datasets/grating_device_blockage'

# Create dir structure
for split in ['train', 'val', 'test']:
    for sub in ['images', 'labels']:
        os.makedirs(os.path.join(dst, split, sub), exist_ok=True)

# Copy files
for split in ['train', 'val', 'test']:
    for sub in ['images', 'labels']:
        src_dir = os.path.join(src, split, sub)
        dst_dir = os.path.join(dst, split, sub)
        if os.path.exists(src_dir):
            for f in os.listdir(src_dir):
                shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            print(f'Copied {len(os.listdir(src_dir))} files from {src_dir} to {dst_dir}')

# Also copy hard negatives
src_hn = 'dataset/hard_negatives'
dst_hn = 'datasets/hard_negatives'
os.makedirs(dst_hn, exist_ok=True)
for cat in os.listdir(src_hn):
    src_cat = os.path.join(src_hn, cat)
    dst_cat = os.path.join(dst_hn, cat)
    if os.path.isdir(src_cat):
        os.makedirs(dst_cat, exist_ok=True)
        for f in os.listdir(src_cat):
            shutil.copy2(os.path.join(src_cat, f), os.path.join(dst_cat, f))
        print(f'Copied hard negative category: {cat}')

print('\nDone copying to datasets directory')