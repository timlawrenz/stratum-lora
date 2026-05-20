#!/usr/bin/env python3
"""
Precompute face bounding boxes for training images.

Uses a face detector (insightface, then OpenCV DNN fallback) to locate faces,
then saves bounding boxes as .json and .npz files alongside images.

The saved bbox is in pixel coordinates [y1, y2, x1, x2] for the
original image resolution, so we can scale it to latent space during training.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def collect_images(image_dir: Path) -> list:
    """Collect all image files from a directory."""
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_exts
    ])
    return images


def expand_bbox(y1: int, y2: int, x1: int, x2: int,
                img_h: int, img_w: int, margin: float = 0.2) -> tuple:
    """Expand bounding box by a margin fraction, clamped to image bounds."""
    margin_y = int((y2 - y1) * margin)
    margin_x = int((x2 - x1) * margin)
    return (
        max(0, y1 - margin_y),
        min(img_h, y2 + margin_y),
        max(0, x1 - margin_x),
        min(img_w, x2 + margin_x),
    )


def detect_with_insightface(image_path: str):
    """Detect face using InsightFace. Returns (y1, y2, x1, x2) or None."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        return None

    try:
        import cv2
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        img = cv2.imread(image_path)
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        # Take the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        return (y1, y2, x1, x2)
    except Exception as e:
        logger.debug(f"InsightFace detection failed for {image_path}: {e}")
        return None


def detect_with_opencv(image_path: str) -> tuple | None:
    """Fallback face detector using OpenCV DNN.

    Requires opencv_face_detector_uint8.pb and opencv_face_detector.pbtxt
    in the same directory as this script.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for face detection")
        return None

    script_dir = Path(__file__).parent
    model_file = script_dir / "opencv_face_detector_uint8.pb"
    config_file = script_dir / "opencv_face_detector.pbtxt"

    if not model_file.exists() or not config_file.exists():
        logger.debug(
            "OpenCV face detector model files not found. "
            "Download them from: "
            "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector"
        )
        return None

    try:
        net = cv2.dnn.readNetFromTensorflow(str(model_file), str(config_file))
    except Exception as e:
        logger.warning(f"Failed to load OpenCV DNN model: {e}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()

    best_conf = 0.5
    best_bbox = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > best_conf:
            best_conf = confidence
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            best_bbox = expand_bbox(y1, y2, x1, x2, h, w, margin=0.2)

    return best_bbox


def detect_face_bbox(image_path: str) -> tuple | None:
    """Detect a single face and return (y1, y2, x1, x2) or None."""
    # Try InsightFace first
    bbox = detect_with_insightface(image_path)
    if bbox is not None:
        return bbox
    # Fall back to OpenCV DNN
    return detect_with_opencv(image_path)


def save_results(bboxes: dict, image_dir: Path, output_base: Path):
    """Save bounding boxes as .json and .npz."""
    # JSON
    json_path = output_base.with_suffix('.json')
    # Convert numpy ints to Python ints for JSON serialization
    json_bboxes = {k: [int(x) for x in v] for k, v in bboxes.items()}
    with open(json_path, 'w') as f:
        json.dump(json_bboxes, f, indent=2)

    # NPZ
    npz_path = output_base.with_suffix('.npz')
    bbox_array = np.array([v for v in bboxes.values()], dtype=np.int32)
    filenames = np.array(list(bboxes.keys()))
    np.savez(npz_path, bboxes=bbox_array, filenames=filenames)

    logger.info(f"Saved {len(bboxes)} bounding boxes to {json_path} and {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute face bounding boxes for training images"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing training images"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for bboxes (without extension). Default: <image_dir>/face_bboxes"
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

    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        logger.error(f"Directory not found: {image_dir}")
        sys.exit(1)

    images = collect_images(image_dir)
    logger.info(f"Found {len(images)} images in {image_dir}")

    if not images:
        logger.warning("No images found!")
        return

    output_base = Path(args.output) if args.output else image_dir / "face_bboxes"
    output_base = output_base.with_suffix('')  # Strip extension, will add .json/.npz

    bboxes = {}
    for img_path in tqdm(images, desc="Detecting faces"):
        bbox = detect_face_bbox(str(img_path))
        if bbox is not None:
            bboxes[img_path.name] = list(bbox)
        else:
            logger.warning(f"No face found in {img_path.name}")

    if not bboxes:
        logger.error("No faces detected in any images!")
        sys.exit(1)

    save_results(bboxes, image_dir, output_base)


if __name__ == "__main__":
    main()
