import os

d = 'dataset/grating_device_blockage'
print(f'Checking: {d}\n')

for split in ['train', 'val', 'test']:
    img_dir = os.path.join(d, split, 'images')
    lbl_dir = os.path.join(d, split, 'labels')
    imgs = os.listdir(img_dir) if os.path.exists(img_dir) else []
    lbls = os.listdir(lbl_dir) if os.path.exists(lbl_dir) else []
    print(f'{split}/images: {len(imgs)} files')
    print(f'{split}/labels: {len(lbls)} files')
    if imgs:
        print(f'  sample: {imgs[0]}')
    if lbls:
        with open(os.path.join(lbl_dir, lbls[0])) as f:
            print(f'  first label: {f.read().strip()[:80]}')

# Also check hard_negatives
print('\nHard negatives:')
hn_dir = 'dataset/hard_negatives'
if os.path.exists(hn_dir):
    for cat in os.listdir(hn_dir):
        cat_path = os.path.join(hn_dir, cat)
        if os.path.isdir(cat_path):
            imgs = os.listdir(cat_path)
            print(f'  {cat}/: {len(imgs)} images')
else:
    print('  NOT FOUND')