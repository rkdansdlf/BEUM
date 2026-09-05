import shutil, os

# Copy test images to val directory
src_val = 'dataset/roboflow_finetune/test/images'
dst_val = 'dataset/roboflow_finetune/val/images'
src_lbl = 'dataset/roboflow_finetune/test/labels'
dst_lbl = 'dataset/roboflow_finetune/val/labels'

os.makedirs(dst_val, exist_ok=True)
os.makedirs(dst_lbl, exist_ok=True)

# Copy images
count = 0
for f in os.listdir(src_val):
    if f.endswith('.jpg'):
        shutil.copy2(os.path.join(src_val, f), os.path.join(dst_val, f))
        count += 1
print(f'Copied {count} images')

# Copy labels
count = 0
for f in os.listdir(src_lbl):
    if f.endswith('.txt'):
        shutil.copy2(os.path.join(src_lbl, f), os.path.join(dst_lbl, f))
        count += 1
print(f'Copied {count} labels')