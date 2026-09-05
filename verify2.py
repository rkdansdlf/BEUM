import os

d = 'dataset/grating_device_blockage'
for split in ['train', 'val', 'test']:
    img_dir = os.path.join(d, split, 'images')
    lbl_dir = os.path.join(d, split, 'labels')
    imgs = os.listdir(img_dir) if os.path.exists(img_dir) else []
    lbls = os.listdir(lbl_dir) if os.path.exists(lbl_dir) else []
    print(f'{split}/images: {len(imgs)} files')
    print(f'{split}/labels: {len(lbls)} files')
    # Show first label example
    if lbls:
        with open(os.path.join(lbl_dir, lbls[0])) as f:
            print(f'  First label: {f.read().strip()[:80]}')

print('\nHard negatives:')
hn_dir = 'dataset/hard_negatives'
for cat in sorted(os.listdir(hn_dir)):
    cat_path = os.path.join(hn_dir, cat)
    if os.path.isdir(cat_path):
        imgs = [f for f in os.listdir(cat_path) if f.endswith('.jpg') or f.endswith('.png')]
        print(f'  {cat}/: {len(imgs)} images')