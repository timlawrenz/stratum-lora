#!/usr/bin/env python3
import os

data_dir = '/mnt/nas-ai-models/training-data/loras/scarl3tt'
caption = 'A photo of scarl3tt person'
count = 0

for fname in os.listdir(data_dir):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(data_dir, base + '.txt')
        with open(txt_path, 'w') as f:
            f.write(caption)
        count += 1

print(f'Created {count} caption files')
