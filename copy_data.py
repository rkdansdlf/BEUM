import os
import shutil
import glob

canonical_dir = 'dataset/canonical'
blockage_dir = 'dataset/grating_device_blockage'

# Ensure output dirs exist
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(blockage_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(blockage_dir, split, 'labels'), exist_ok=True)

# Copy images and convert labels
# Actual structure: dataset/canonical/images/train, dataset/canonical/images/val, etc.
for split in ['train', 'val', 'test']:
    src_img = os.path.join(canonical_dir, 'images', split)
    src_lbl = os.path.join(canonical_dir, 'labels', split)
    dst_img = os.path.join(blockage_dir, split, 'images')
    dst_lbl = os.path.join(blockage_dir, split, 'labels')
    
    # Copy images
    if os.path.exists(src_img):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for f in glob.glob(os.path.join(src_img, ext)):
                shutil.copy2(f, os.path.join(dst_img, os.path.basename(f)))
        img_count = len(os.listdir(dst_img))
        print(f'Copied {img_count} images to blockage/{split}/images')
    else:
        print(f'Source images not found: {src_img}')
    
    # Convert and copy labels (old 2-class → new 3-class)
    if os.path.exists(src_lbl):
        converted = 0
        for f in os.listdir(src_lbl):
            src_path = os.path.join(src_lbl, f)
            dst_path = os.path.join(dst_lbl, f)
            with open(src_path, 'r') as sf:
                content = sf.read().strip()
            if not content:
                continue
            parts = content.split()
            old_class = int(parts[0])
            # Map: old 0 (drain_area→grating) → new 0 (grating)
            #      old 1 (object→device) → new 1 (device)
            #      old 2 (drain_full→blockage) → new 2 (blockage)
            # Class ID stays same numerically since names reorder matches
            new_class = old_class
            new_content = f'{new_class} ' + ' '.join(parts[1:])
            with open(dst_path, 'w') as df:
                df.write(new_content)
            converted += 1
        print(f'Converted {converted} labels to blockage/{split}/labels')

# Verify hard negatives structure
print('\nHard negatives:')
hn_dir = 'dataset/hard_negatives'
for cat in sorted(os.listdir(hn_dir)):
    cat_path = os.path.join(hn_dir, cat)
    if os.path.isdir(cat_path):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.txt']
        img_count = sum(len(glob.glob(os.path.join(cat_path, ext))) for ext in exts)
        # Count only image files (not .txt)
        img_only = len([f for f in os.listdir(cat_path) if f.endswith('.jpg') or f.endswith('.png')])
        print(f'  {cat}/: {img_only} images, total files: {img_count}')