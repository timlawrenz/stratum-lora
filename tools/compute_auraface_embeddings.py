#!/usr/bin/env python3
"""
Precompute AuraFace identity embeddings for training images.

For each training image, compute the AuraFace embedding. The mean embedding
across all images serves as the target for identity loss during training.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def preprocess_face(image_path: str, bbox: tuple, target_size: int = 112) -> np.ndarray:
    """
    Load image, crop to face bounding box, resize to target_size x target_size.

    Args:
        image_path: Path to image file
        bbox: (y1, y2, x1, x2) in pixel coordinates
        target_size: Output size (AuraFace expects 112)

    Returns:
        RGB image as numpy array of shape (target_size, target_size, 3), float32 [0,1]
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    y1, y2, x1, x2 = bbox
    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return face.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(
        description="Precompute AuraFace embeddings for training images"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing training images"
    )
    parser.add_argument(
        "--bbox_file", type=str, required=True,
        help="Path to face_bboxes.json"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .npz file. Default: <image_dir>/auraface_embeddings.npz"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (cuda or cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Load bounding boxes
    image_dir = Path(args.image_dir).expanduser().resolve()
    bbox_path = Path(args.bbox_file).expanduser().resolve()

    with open(bbox_path) as f:
        bboxes = json.load(f)
    logger.info(f"Loaded {len(bboxes)} bounding boxes from {bbox_path}")

    # Load AuraFace
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from library.auraface_utils import AuraFaceWrapper
        auraface = AuraFaceWrapper(device=device)
        logger.info("AuraFace loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load AuraFace: {e}")
        logger.warning("Falling back to dummy embeddings (random vectors) for testing")
        auraface = None

    # Process images in batches
    filenames = list(bboxes.keys())
    all_embeddings = []

    for i in tqdm(range(0, len(filenames), args.batch_size), desc="Computing embeddings"):
        batch_names = filenames[i:i + args.batch_size]
        batch_faces = []

        for name in batch_names:
            img_path = image_dir / name
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}, skipping")
                continue

            try:
                face = preprocess_face(str(img_path), tuple(bboxes[name]))
                batch_faces.append(face)
            except Exception as e:
                logger.warning(f"Failed to process {name}: {e}")
                continue

        if not batch_faces:
            continue

        batch_tensor = torch.from_numpy(np.stack(batch_faces))  # (B, H, W, C)
        batch_tensor = batch_tensor.permute(0, 3, 1, 2).to(device)  # (B, C, H, W)

        if auraface is not None:
            with torch.no_grad():
                emb = auraface(batch_tensor)
            all_embeddings.append(emb.cpu().numpy())
        else:
            # Dummy fallback: random normalized vectors
            dummy = np.random.randn(len(batch_faces), 512).astype(np.float32)
            dummy = dummy / np.linalg.norm(dummy, axis=1, keepdims=True)
            all_embeddings.append(dummy)

    if not all_embeddings:
        logger.error("No embeddings computed!")
        sys.exit(1)

    # Stack all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)  # (N, D)

    # Compute mean embedding (target for identity loss)
    mean_embedding = all_embeddings.mean(axis=0, keepdims=True)
    # Re-normalize
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding, axis=1, keepdims=True)

    # Save
    output_path = Path(args.output) if args.output else image_dir / "auraface_embeddings.npz"
    output_path = output_path.with_suffix('.npz')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        embeddings=all_embeddings,
        mean_embedding=mean_embedding,
        filenames=np.array(filenames[:len(all_embeddings)]),
    )

    logger.info(f"Saved {len(all_embeddings)} embeddings to {output_path}")
    logger.info(f"Mean embedding shape: {mean_embedding.shape}")


if __name__ == "__main__":
    main()
