import os
import shutil

# Check canonical structure
canonical_d = 'dataset/canonical'
blockage_d = 'dataset/grating_device_blockage'

for split in ['train', 'val', 'test']:
    src_img = os.path.join(canonical_d, split, 'images')
    src_lbl = os.path.join(canonical_d, split, 'labels')
    dst_img = os.path.join(blockage_d, split, 'images')
    dst_lbl = os.path.join(blockage_d, split, 'labels')
    
    # Create dirs
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)
    
    # Copy images
    if os.path.exists(src_img):
        copied = 0
        for f in os.listdir(src_img):
            shutil.copy2(os.path.join(src_img, f), os.path.join(dst_img, f))
            copied += 1
        print(f'Copied {copied} images from canonical/{split}/images')
    
    # Copy/converting labels
    if os.path.exists(src_lbl):
        converted = 0
        for f in os.listdir(src_lbl):
            # Read old label
            with open(os.path.join(src_lbl, f)) as fh:
                content = fh.read().strip()
            if not content:
                continue
            parts = content.split()
            old_class = int(parts[0])
            # Map old classes to new:
            # old 0 (drain_area→grating) → new 0 (grating)
            # old 1 (object→device) → new 1 (device)
            # old 2 (drain_full→blockage) → new 2 (blockage)
            # Class IDs stay numerically the same since names reorder matches
            new_class = old_class
            
            new_content = f'{new_class} ' + ' '.join(parts[1:])
            with open(os.path.join(dst_lbl, f), 'w') as fh:
                fh.write(new_content)
            converted += 1
        print(f'Processed {converted} labels from canonical/{split}/labels')

print('\nDone copying/converting canonical data to grating_device_blockage')