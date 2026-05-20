#!/usr/bin/env python3
"""
Compute DINOv3 CLS tokens for all training images and save the averaged
identity anchor. DINOv3 CLS tokens are sensitive to lighting and pose,
so averaging across the full dataset produces a more robust representation.

Usage:
  python tools/compute_dinov3_token.py \
    --image_dir /path/to/images \
    --output /path/to/dinov3_cls.npy
"""

import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Compute averaged DINOv3 CLS token for identity conditioning"
    )
    parser.add_argument(
        "--image_dir", required=True,
        help="Directory containing training images"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .npy file for the averaged CLS token"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for inference (cuda or cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for DINOv3 inference"
    )
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from library.dinov3_utils import DINOv3Wrapper

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dinov3 = DINOv3Wrapper(device=device)

    # DINOv3 expects 518x518 with ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    image_dir = Path(args.image_dir)
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_exts
    ])
    print(f"Found {len(images)} images in {image_dir}")

    if not images:
        print("No images found!")
        sys.exit(1)

    all_tokens = []
    skipped = 0

    for i in tqdm(range(0, len(images), args.batch_size), desc="Computing CLS tokens"):
        batch_paths = images[i:i + args.batch_size]
        batch_tensors = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_tensors.append(transform(img))
            except Exception as e:
                print(f"  Skipping {path.name}: {e}")
                skipped += 1

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)
        tokens = dinov3(batch)  # (B, 1024)
        all_tokens.append(tokens.cpu().numpy())

    if not all_tokens:
        print("No tokens computed!")
        sys.exit(1)

    all_tokens = np.concatenate(all_tokens, axis=0)  # (N, 1024)
    mean_token = all_tokens.mean(axis=0, keepdims=True)  # (1, 1024)

    # Ensure output dir exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), mean_token)

    print(f"Saved averaged CLS token ({mean_token.shape})")
    print(f"  From {len(all_tokens)} images ({skipped} skipped)")
    print(f"  To: {output_path}")


if __name__ == "__main__":
    main()
