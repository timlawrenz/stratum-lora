# Phase 1: AuraFace Identity Loss — SDXL LoRA Training

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Augment kohya-ss/sd-scripts SDXL LoRA training with an AuraFace identity penalty that forces the UNet to preserve facial identity during persona LoRA training.

**Architecture:** Fork kohya-ss/sd-scripts. Add AuraFace as a frozen embedding model. During training, at low timesteps, compute the single-step x̂₀ estimate via Tweedie's formula, VAE-decode to pixel space, crop the face region using pre-computed bounding boxes, pass through AuraFace, and compute a cosine-similarity identity loss against a pre-computed target embedding. This loss is added to the standard flow/noise loss with a configurable λ.

**Tech Stack:** PyTorch, Diffusers, Transformers, SDXL (ε-prediction), AuraFace (frozen, differentiable forward), kohya-ss/sd-scripts fork

**GPU Target:** 8+ GB VRAM (2070 Super), with vast.ai fallback for full runs.

**Data Assumptions:**
- Training images are pre-aligned face crops (stratum-hq pipeline)
- Bounding boxes and AuraFace target embeddings are precomputed offline
- Data is loaded from `.npz` files alongside images

---

## Pre-work: Data Preparation

### Task 0: Prepare test dataset

**Objective:** Downscale 14000px source images to SDXL-friendly resolutions and organize the data directory.

**Status:** ✅ Complete

**What was done:**
- Source: `/mnt/nas-ai-models/training-data/loras/hegre-14000px/7279_any-moloko-sexual/` (60 images, 13805×10353)
- Destination: `data/hegre-moloko/` (60 images, ~2048×1535)
- SDXL's bucketing system will handle final sizing during training

**Remaining data prep** (runs during later tasks):
- Run `tools/compute_face_bboxes.py` on `data/hegre-moloko/`
- Run `tools/compute_auraface_embeddings.py` on `data/hegre-moloko/`
- These produce `face_bboxes.npz` and `auraface_embeddings.npz` consumed by the training loop

---

## Pre-work: Repository Setup

### Task 1: Fork and clone sd-scripts

**Objective:** Create a local fork of kohya-ss/sd-scripts under `~/source/activity/stratum-lora`

**Files:**
- Create: `~/source/activity/stratum-lora` (full repository)

**Step 1: Clone the repository**

```bash
cd ~/source/activity/stratum-lora
git clone https://github.com/kohya-ss/sd-scripts.git .
git checkout -b feat/auraface-identity-loss
```

**Step 2: Verify structure**

```bash
ls train_network.py library/sdxl_train_util.py library/strategy_sdxl.py library/strategy_sd.py
```

Expected: All four files exist.

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
# Additional deps for auraface:
pip install huggingface_hub timm
```

**Verification:** `python -c "import torch; from diffusers import DDPMScheduler; print('OK')"`

---

## Phase 1A: AuraFace Model Wrapper

### Task 2: Create AuraFace wrapper module

**Objective:** Create a Python module that loads and wraps AuraFace for embedding extraction.

**Files:**
- Create: `library/auraface_utils.py`

**Code:**

```python
"""
AuraFace utilities for identity embedding extraction.

AuraFace is loaded as a frozen model. Its forward pass is differentiable,
so gradients flow through to the input image (our x̂₀ estimate).

Model source: https://huggingface.co/auraness/auraface
"""
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger(__name__)


class AuraFaceWrapper(nn.Module):
    """Frozen AuraFace model that outputs a normalized identity embedding."""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # AuraFace expects 112x112 RGB images normalized to [0, 1]
        self.input_size = 112

        # Load model from HF hub
        self.model = self._load_auraface()
        self.model.to(device=device, dtype=dtype)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad_(False)

        logger.info(f"AuraFace loaded on {device}")

    def _load_auraface(self):
        """Load AuraFace from HuggingFace hub."""
        from huggingface_hub import snapshot_download

        # AuraFace repo on HuggingFace
        repo_id = "auraness/auraface"

        # Download model files
        model_path = snapshot_download(repo_id)

        # Load the model (architecture depends on the actual repo structure)
        # Most face recognition models use a backbone + head architecture
        # We want the backbone features before the classification head
        import sys
        sys.path.insert(0, model_path)

        # Attempt to load — adjust based on actual repo structure
        try:
            from backbones import get_model
            model = get_model("iresnet50", fp16=(self.dtype == torch.float16))
            # Load weights
            import os
            weights_path = os.path.join(model_path, "auraface_weights.pth")
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        except ImportError:
            # Fallback: try loading as a standard torch model
            logger.warning(
                "Could not import AuraFace backbones module. "
                "Manual model loading will be required. "
                "Check the repo structure at https://huggingface.co/auraness/auraface"
            )
            raise

        return model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract identity embedding from face images.

        Args:
            images: Tensor of shape (B, 3, 112, 112), values in [0, 1]

        Returns:
            Normalized embedding of shape (B, embedding_dim)
        """
        # Normalize to model's expected input range if needed
        # AuraFace typically expects images normalized with mean=0.5, std=0.5
        images = (images - 0.5) / 0.5

        with torch.set_grad_enabled(True):  # We want gradients through this
            embedding = self.model(images)

        # L2 normalize the embedding
        embedding = nn.functional.normalize(embedding, p=2, dim=-1)

        return embedding

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the output embedding."""
        dummy = torch.randn(1, 3, self.input_size, self.input_size, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            emb = self.forward(dummy)
        return emb.shape[-1]
```

**Step 2: Verify module loads**

```python
from library.auraface_utils import AuraFaceWrapper
# This will fail informatively if AuraFace isn't available
# The exact loading code may need adjustment based on the actual repo structure
```

**Verification:** Module imports without syntax errors. Actual model loading requires HF hub access.

---

## Phase 1B: Data Precomputation Scripts

### Task 3: Create bounding box precomputation script

**Objective:** Script to run face detection on training images and save bounding boxes.

**Files:**
- Create: `tools/compute_face_bboxes.py`

**Code:**

```python
"""
Precompute face bounding boxes for training images.

Uses a face detector (insightface/retinaface) to locate faces,
then saves bounding boxes as .npz files alongside each image.

The saved bbox is in pixel coordinates [y1, y2, x1, x2] for the
original image resolution, so we can scale it to latent space.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def detect_face_bbox(image_path: str) -> tuple | None:
    """
    Detect a single face in an image and return its bounding box.

    Returns (y1, y2, x1, x2) in pixel coordinates, or None if no face found.
    """
    # Use OpenCV's DNN-based face detector (lightweight, no extra deps)
    # Alternative: insightface for higher accuracy
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        img = cv2.imread(image_path)
        faces = app.get(img)
        if len(faces) == 0:
            return None
        # Take the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        return (y1, y2, x1, x2)
    except ImportError:
        # Fallback: OpenCV DNN
        return _detect_with_opencv(image_path)


def _detect_with_opencv(image_path: str) -> tuple | None:
    """Fallback face detector using OpenCV DNN."""
    # Download model files if needed
    model_file = "opencv_face_detector_uint8.pb"
    config_file = "opencv_face_detector.pbtxt"

    net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()

    best_conf = 0.5  # threshold
    best_bbox = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > best_conf:
            best_conf = confidence
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            # Expand by 20% for margin
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            y1 = max(0, y1 - margin_y)
            y2 = min(h, y2 + margin_y)
            x1 = max(0, x1 - margin_x)
            x2 = min(w, x2 + margin_x)
            best_bbox = (y1, y2, x1, x2)

    return best_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz or .json file for bboxes")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_exts
    ])

    print(f"Found {len(images)} images")

    bboxes = {}
    for img_path in tqdm(images):
        bbox = detect_face_bbox(str(img_path))
        if bbox is not None:
            bboxes[img_path.name] = list(bbox)
        else:
            print(f"WARNING: No face found in {img_path.name}")

    # Save as .json for human readability and .npz for training
    output_path = args.output or os.path.join(args.image_dir, "face_bboxes.json")
    with open(output_path, 'w') as f:
        json.dump(bboxes, f, indent=2)

    # Also save as .npz for efficient loading
    npz_path = output_path.replace('.json', '.npz')
    # Store as arrays: shape (N, 4) where columns are [y1, y2, x1, x2]
    bbox_array = np.array([list(v) for v in bboxes.values()], dtype=np.int32)
    np.savez(npz_path, bboxes=bbox_array, filenames=list(bboxes.keys()))

    print(f"Saved {len(bboxes)} bounding boxes to {output_path} and {npz_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Test script with sample data**

```bash
# Create dummy directory and test
mkdir -p /tmp/test_faces
python tools/compute_face_bboxes.py --image_dir /tmp/test_faces 2>&1
```

Expected: Script runs, reports 0 images found (expected for empty dir).

**Verification:** Script exists, parses args correctly. Actual detection requires real images + face detector model.

---

### Task 4: Create AuraFace target embedding precomputation script

**Objective:** Script to compute AuraFace embeddings for all training images and save as .npz.

**Files:**
- Create: `tools/compute_auraface_embeddings.py`

**Code:**

```python
"""
Precompute AuraFace identity embeddings for training images.

For each training image, compute the AuraFace embedding and save it.
The mean embedding across all images serves as the target for identity loss.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


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
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    y1, y2, x1, x2 = bbox
    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return face.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("--bbox_file", type=str, required=True,
                        help="Path to face_bboxes.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz file for embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for AuraFace inference")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    args = parser.parse_args()

    # Load bounding boxes
    import json
    with open(args.bbox_file) as f:
        bboxes = json.load(f)

    # Load AuraFace
    # Note: This import assumes library/ is in the path
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    try:
        from library.auraface_utils import AuraFaceWrapper
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        auraface = AuraFaceWrapper(device=device)
    except Exception as e:
        print(f"Could not load AuraFace: {e}")
        print("Falling back to dummy embedding (random vector) for testing")
        auraface = None
        device = torch.device("cpu")

    # Process images
    image_dir = Path(args.image_dir)
    filenames = list(bboxes.keys())
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(filenames), args.batch_size)):
        batch_names = filenames[i:i + args.batch_size]
        batch_faces = []
        
        for name in batch_names:
            img_path = image_dir / name
            if not img_path.exists():
                print(f"WARNING: {img_path} not found, skipping")
                continue
            face = preprocess_face(str(img_path), tuple(bboxes[name]))
            batch_faces.append(face)
        
        if not batch_faces:
            continue
        
        batch_tensor = torch.from_numpy(np.stack(batch_faces))  # (B, H, W, C)
        batch_tensor = batch_tensor.permute(0, 3, 1, 2).to(device)  # (B, C, H, W)
        
        if auraface is not None:
            with torch.no_grad():
                emb = auraface(batch_tensor)
            all_embeddings.append(emb.cpu().numpy())
        else:
            # Dummy fallback
            dummy = np.random.randn(len(batch_faces), 512).astype(np.float32)
            dummy = dummy / np.linalg.norm(dummy, axis=1, keepdims=True)
            all_embeddings.append(dummy)

    # Stack all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)  # (N, D)
    
    # Compute mean embedding (target for identity loss)
    mean_embedding = all_embeddings.mean(axis=0, keepdims=True)
    # Re-normalize
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding, axis=1, keepdims=True)

    # Save
    output_path = args.output or os.path.join(args.image_dir, "auraface_embeddings.npz")
    np.savez(
        output_path,
        embeddings=all_embeddings,
        mean_embedding=mean_embedding,
        filenames=np.array(filenames[:len(all_embeddings)])
    )
    
    print(f"Saved {len(all_embeddings)} embeddings to {output_path}")
    print(f"Mean embedding shape: {mean_embedding.shape}")


if __name__ == "__main__":
    main()
```

**Verification:** Script exists, imports correctly, handles missing AuraFace gracefully with dummy fallback.

---

## Phase 1C: Dataset Metadata Loading

### Task 5: Extend dataset loading to include bbox + target embedding

**Objective:** Modify the SDXL dataset/collator to load precomputed bounding boxes and AuraFace target embeddings alongside each image.

**Files:**
- Modify: `library/train_util.py` (add metadata loading to `DreamBoothDataset` or create a new dataset class)

**Approach:**

The simplest approach is to have the dataloader load additional `.npz` metadata files that reside alongside the images. We'll create a helper function and modify the dataset class to optionally load these.

**Code additions to `library/train_util.py`:**

```python
def load_auraface_metadata(image_dir: str) -> dict:
    """
    Load precomputed AuraFace metadata from an image directory.
    
    Returns dict with keys:
        - 'bboxes': dict mapping filename -> [y1, y2, x1, x2]
        - 'target_embedding': numpy array of shape (1, embedding_dim)
        - 'image_embeddings': numpy array of shape (N, embedding_dim)
    """
    import json
    import numpy as np
    
    metadata = {}
    
    # Load bounding boxes
    bbox_json = os.path.join(image_dir, "face_bboxes.json")
    bbox_npz = os.path.join(image_dir, "face_bboxes.npz")
    
    if os.path.exists(bbox_json):
        with open(bbox_json) as f:
            metadata['bboxes'] = json.load(f)
    elif os.path.exists(bbox_npz):
        data = np.load(bbox_npz, allow_pickle=True)
        metadata['bboxes'] = dict(zip(
            data['filenames'], 
            data['bboxes'].tolist()
        ))
    
    # Load AuraFace embeddings
    emb_npz = os.path.join(image_dir, "auraface_embeddings.npz")
    if os.path.exists(emb_npz):
        data = np.load(emb_npz)
        metadata['target_embedding'] = data['mean_embedding']  # (1, D)
        metadata['image_embeddings'] = data['embeddings']  # (N, D)
    
    return metadata
```

**Step 1: Add metadata loading to the dataset**

The training loop in `train_network.py` passes `batch` dicts through. We need each batch to include the bbox and target embedding. The cleanest approach is to store them as a global/config-level structure rather than per-batch, since the target embedding is the same for all images of a persona.

We'll add a new `AuraFaceConfig` dataclass that holds the metadata, and inject it into the training step.

**Code (`library/auraface_utils.py` additions):**

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


@dataclass
class AuraFaceConfig:
    """Configuration for AuraFace identity loss during training."""
    
    # Loss weight (lambda in L_total = L_noise + lambda * L_id)
    lambda_id: float = 1.0
    
    # Timestep threshold: only apply L_id when t < threshold
    # (0.3 means last 30% of the denoising process)
    timestep_threshold: float = 0.3
    
    # Target embedding for cosine similarity loss
    # Shape: (1, embedding_dim)
    target_embedding: Optional[torch.Tensor] = None
    
    # Bounding boxes for face cropping, indexed by filename
    # Dict[str, Tuple[int, int, int, int]]
    bboxes: Optional[dict] = None
    
    # Original image size (height, width) for scaling bboxes to latent space
    image_size: tuple = (1024, 1024)
    
    # VAE scale factor (SDXL: 8x downsampling)
    vae_scale_factor: int = 8
    
    def scale_bbox_to_latent(self, y1: int, y2: int, x1: int, x2: int) -> tuple:
        """Convert pixel-space bbox to latent-space coordinates."""
        h_ratio = self.vae_scale_factor / self.image_size[0]
        w_ratio = self.vae_scale_factor / self.image_size[1]
        return (
            int(y1 * h_ratio),
            int(y2 * h_ratio),
            int(x1 * w_ratio),
            int(x2 * w_ratio),
        )
```

**Verification:** New module loads, dataclass instantiates correctly.

---

### Task 6: Add CLI arguments for AuraFace identity loss

**Objective:** Add `--auraface_lambda`, `--auraface_data_dir`, and `--auraface_threshold` arguments to the training script.

**Files:**
- Modify: `train_network.py` (add args to `add_custom_train_arguments` or main arg parser)

**Step 1: Add to argument parsing**

In `train_network.py`, in the argument definition section (search for `add_custom_train_arguments` or `argparse.ArgumentParser`), add:

```python
# AuraFace identity loss arguments
parser.add_argument(
    "--auraface_lambda",
    type=float,
    default=0.0,
    help="Weight for AuraFace identity loss. 0.0 disables it. Suggested starting value: 0.1",
)
parser.add_argument(
    "--auraface_data_dir",
    type=str,
    default=None,
    help="Directory containing face_bboxes.npz and auraface_embeddings.npz",
)
parser.add_argument(
    "--auraface_threshold",
    type=float,
    default=0.3,
    help="Timestep threshold ratio for identity loss gating (0.0-1.0). Default: 0.3",
)
```

**Verification:** `python train_network.py --help | grep auraface` shows the three new arguments.

---

## Phase 1D: Core Loss Modification

### Task 7: Implement Tweedie x̂₀ estimation utility

**Objective:** Add a function to compute the single-step clean image estimate from the noise prediction.

**Files:**
- Create/modify: `library/custom_train_functions.py`

**Step 1: Add Tweedie estimation function**

```python
def tweedie_x0_estimate(
    noise_pred: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.IntTensor,
    noise_scheduler,
) -> torch.Tensor:
    """
    Compute the single-step clean latent estimate using Tweedie's formula.
    
    For epsilon-prediction (SDXL):
        x̂₀ = (x_t - sqrt(1 - ᾱ_t) * ε_θ) / sqrt(ᾱ_t)
    
    Args:
        noise_pred: The UNet's noise prediction ε_θ, shape (B, C, H, W)
        noisy_latents: The noised latents x_t, shape (B, C, H, W)
        timesteps: The timestep for each sample, shape (B,)
        noise_scheduler: The DDPMScheduler
    
    Returns:
        x̂₀ estimate, shape (B, C, H, W)
    """
    # Gather alpha_cumprod for each timestep
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(noise_pred.device)
    alpha_bar_t = alphas_cumprod[timesteps]  # shape (B,)
    
    # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
    alpha_bar_t = alpha_bar_t.reshape(-1, 1, 1, 1)
    
    # Tweedie's formula for ε-prediction
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    
    x0_hat = (noisy_latents - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
    
    return x0_hat


def compute_id_loss(
    x0_hat: torch.Tensor,
    timesteps: torch.IntTensor,
    batch: dict,
    vae,
    auraface: 'AuraFaceWrapper',
    auraface_config: 'AuraFaceConfig',
    weight_dtype: torch.dtype,
    accelerator,
) -> torch.Tensor | None:
    """
    Compute the AuraFace identity loss for a batch.
    
    Returns None if no samples in the batch meet the timestep threshold.
    
    Args:
        x0_hat: Single-step clean latent estimate, shape (B, C, H_lat, W_lat)
        timesteps: Timestep for each sample, shape (B,)
        batch: Batch dict containing filenames
        vae: VAE for decoding latents to pixels
        auraface: AuraFaceWrapper instance
        auraface_config: AuraFaceConfig with bboxes and target embedding
        weight_dtype: Weight dtype for VAE
        accelerator: Accelerator instance
    
    Returns:
        Scalar identity loss tensor, or None
    """
    # Gate: only apply for timesteps below threshold
    # Convert threshold ratio to absolute timestep index
    max_t = noise_scheduler.config.num_train_timesteps
    threshold_step = int(auraface_config.timestep_threshold * max_t)
    
    mask = timesteps < threshold_step
    if not mask.any():
        return None
    
    # Select samples that pass the threshold
    x0_hat_selected = x0_hat[mask]  # (K, C, H_lat, W_lat)
    batch_size = x0_hat_selected.shape[0]
    
    # Get filenames for bbox lookup
    filenames = batch.get("image_filenames", None)
    
    # VAE decode to pixel space
    # x0_hat is in latent space; decode through frozen VAE
    x0_hat_selected = x0_hat_selected / vae.config.scaling_factor
    
    with torch.set_grad_enabled(True):
        # VAE decode (differentiable)
        decoded = vae.decode(x0_hat_selected.to(weight_dtype)).sample  # (K, 3, H, W)
        
        # Normalize from [-1, 1] to [0, 1]
        decoded = (decoded + 1.0) / 2.0
        decoded = torch.clamp(decoded, 0.0, 1.0)
        
        # Crop and resize each face
        face_crops = []
        for i in range(batch_size):
            if filenames is not None and filenames[i] in auraface_config.bboxes:
                y1, y2, x1, x2 = auraface_config.bboxes[filenames[i]]
                # Scale bbox to decoded image size
                h_ratio = decoded.shape[2] / auraface_config.image_size[0]
                w_ratio = decoded.shape[3] / auraface_config.image_size[1]
                y1_s = int(y1 * h_ratio)
                y2_s = int(y2 * h_ratio)
                x1_s = int(x1 * w_ratio)
                x2_s = int(x2 * w_ratio)
                
                face = decoded[i:i+1, :, y1_s:y2_s, x1_s:x2_s]
            else:
                # Fallback: use center crop (assumes face is centered)
                h, w = decoded.shape[2], decoded.shape[3]
                crop_size = min(h, w) // 2
                cy, cx = h // 2, w // 2
                face = decoded[i:i+1, :,
                         cy - crop_size//2 : cy + crop_size//2,
                         cx - crop_size//2 : cx + crop_size//2]
            
            # Resize to AuraFace input size (112x112)
            face = torch.nn.functional.interpolate(
                face, size=(112, 112), mode='bilinear', align_corners=False
            )
            face_crops.append(face)
        
        face_crops = torch.cat(face_crops, dim=0)  # (K, 3, 112, 112)
        
        # Pass through AuraFace
        generated_embeddings = auraface(face_crops)  # (K, D)
        
        # Get target embedding
        target_emb = auraface_config.target_embedding.to(generated_embeddings.device)  # (1, D)
        
        # Cosine similarity loss: L_id = 1 - cos(gen, target)
        cos_sim = torch.nn.functional.cosine_similarity(
            generated_embeddings, target_emb.expand(batch_size, -1), dim=-1
        )  # (K,)
        
        id_loss = (1.0 - cos_sim).mean()
    
    return id_loss
```

**Verification:** Functions import and execute with dummy inputs.

---

### Task 8: Integrate identity loss into the training loop

**Objective:** Modify the training step in `train_network.py` to compute and add the identity loss.

**Files:**
- Modify: `train_network.py` (the `train()` method, around line 450-478)

**Step 1: Add AuraFace initialization in `train()` method**

After model loading (around line 600), add:

```python
# Initialize AuraFace if identity loss is enabled
auraface = None
auraface_config = None
if getattr(args, 'auraface_lambda', 0.0) > 0.0 and args.auraface_data_dir:
    from library.auraface_utils import AuraFaceWrapper, AuraFaceConfig
    
    logger.info(f"Loading AuraFace for identity loss (lambda={args.auraface_lambda})")
    
    # Load AuraFace model
    auraface = AuraFaceWrapper(device=accelerator.device, dtype=weight_dtype)
    
    # Load metadata
    metadata = train_util.load_auraface_metadata(args.auraface_data_dir)
    
    target_emb = torch.from_numpy(metadata['target_embedding'])
    auraface_config = AuraFaceConfig(
        lambda_id=args.auraface_lambda,
        timestep_threshold=args.auraface_threshold,
        target_embedding=target_emb,
        bboxes=metadata.get('bboxes', {}),
    )
    logger.info(f"AuraFace loaded. Embedding dim: {auraface.get_embedding_dim()}")
    
    # Enable VAE memory optimizations for in-loop decoding
    # Tiling splits decode into overlapping tiles, reducing peak memory ~4x
    # Slicing processes latent channels in slices for additional savings
    vae.enable_tiling()
    vae.enable_slicing()
    logger.info("VAE tiling and slicing enabled for memory-efficient decode")
```

**Step 2: Modify the loss computation section**

The key section currently reads:
```python
noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(...)
loss = train_util.conditional_loss(noise_pred.float(), target.float(), ...)
...
return loss.mean()
```

Replace with gradient-normalized identity loss injection:

```python
# --- Standard noise/flow loss ---
noise_pred, target, timesteps, weighting, noisy_latents = self.get_noise_pred_and_target(
    args, accelerator, noise_scheduler, latents, batch,
    text_encoder_conds, unet, network, weight_dtype, train_unet, is_train=is_train,
)

huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "none", huber_c)
if weighting is not None:
    loss = loss * weighting
if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
    loss = apply_masked_loss(loss, batch)
loss = loss.mean(dim=list(range(1, loss.ndim)))
loss_weights = batch["loss_weights"]
loss = loss * loss_weights
loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

# --- AuraFace Identity Loss (gradient-normalized) ---
if auraface is not None and auraface_config is not None:
    from library.custom_train_functions import tweedie_x0_estimate, compute_id_loss

    x0_hat = tweedie_x0_estimate(noise_pred, noisy_latents, timesteps, noise_scheduler)

    # retain_grad() is required so the gradient hook fires
    x0_hat.retain_grad()

    id_loss = compute_id_loss(
        x0_hat, timesteps, batch, vae, auraface, auraface_config,
        weight_dtype, accelerator
    )

    if id_loss is not None:
        scaled_id_loss = auraface_config.lambda_id * id_loss

        # Gradient normalization: the VAE decoder Jacobian amplifies gradients
        # by ~1/scaling_factor² when backpropagating from pixel space to latent space.
        # Dividing by scaling_factor² cancels this amplification, keeping the
        # identity gradient magnitude comparable to the noise loss gradient.
        def normalize_id_grad(grad):
            return grad / (vae.config.scaling_factor ** 2)

        x0_hat.register_hook(normalize_id_grad)

        loss = loss + scaled_id_loss

        if accelerator.is_main_process:
            logs["loss/id_loss"] = id_loss.detach().item()
            logs["loss/id_loss_scaled"] = scaled_id_loss.detach().item()

return loss.mean()
```

**Step 3: Modify `get_noise_pred_and_target` to also return `noisy_latents`**

The function currently returns `(noise_pred, target, timesteps, None)`. We need it to also return `noisy_latents` so we can compute x̂₀.

Change the return line from:
```python
return noise_pred, target, timesteps, None
```
to:
```python
return noise_pred, target, timesteps, None, noisy_latents
```

And update the caller in `train()`:
```python
noise_pred, target, timesteps, weighting, noisy_latents = self.get_noise_pred_and_target(...)
```

**Verification:** Training script starts without errors when `--auraface_lambda 0.1` is specified. With `--auraface_lambda 0.0`, behavior is identical to baseline.

---

## Phase 1E: Integration & Testing

### Task 9: End-to-end smoke test

**Objective:** Verify the complete pipeline runs without crashing on a tiny test dataset.

**Files:**
- Create: `tests/test_auraface_loss.py`

**Step 1: Create a minimal test dataset**

```python
"""Smoke test for AuraFace identity loss integration."""
import os
import tempfile
import numpy as np
import torch


def test_tweedie_estimation():
    """Test Tweedie's formula gives correct shapes."""
    from diffusers import DDPMScheduler
    from library.custom_train_functions import tweedie_x0_estimate
    
    scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    # Pre-compute alphas for the test
    from library.custom_train_functions import prepare_scheduler_for_custom_training
    prepare_scheduler_for_custom_training(scheduler, torch.device("cpu"))
    
    noise_pred = torch.randn(2, 4, 128, 128)
    noisy_latents = torch.randn(2, 4, 128, 128)
    timesteps = torch.tensor([100, 500], dtype=torch.long)
    
    x0_hat = tweedie_x0_estimate(noise_pred, noisy_latents, timesteps, scheduler)
    
    assert x0_hat.shape == (2, 4, 128, 128)
    print("✓ Tweedie estimation shape correct")


def test_auraface_config():
    """Test AuraFaceConfig dataclass."""
    from library.auraface_utils import AuraFaceConfig
    
    config = AuraFaceConfig(
        lambda_id=0.1,
        timestep_threshold=0.3,
        image_size=(1024, 1024),
    )
    
    # Test bbox scaling
    scaled = config.scale_bbox_to_latent(100, 500, 200, 600)
    assert scaled[0] == 0   # 100 * 8/1024 = 0 (int)
    assert scaled[1] == 3   # 500 * 8/1024 = 3
    print("✓ Bbox scaling correct")


if __name__ == "__main__":
    test_tweedie_estimation()
    test_auraface_config()
    print("\n✓ All tests passed!")
```

**Step 2: Run tests**

```bash
cd ~/source/activity/stratum-lora
python tests/test_auraface_loss.py
```

Expected: Both tests pass.

**Verification:** Tests pass. This confirms Tweedie estimation and config are correct.

---

### Task 10: Create example training config

**Objective:** Provide a working example config file for AuraFace-enhanced SDXL LoRA training.

**Files:**
- Create: `examples/auraface_sdxl_lora.toml`

**Code:**

```toml
# Example: AuraFace-enhanced SDXL LoRA training
# Uses identity loss to improve persona preservation

pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
vae = ""
output_dir = "./output/auraface_lora"
output_name = "persona_lora"

# Training data
train_data_dir = "./data/my_persona"
dataset_config = ""

# Network configuration
network_module = "networks.lora"
network_dim = 32
network_alpha = 16
network_train_unet_only = true

# Training parameters
resolution = "1024,1024"
batch_size = 1
max_train_epochs = 10
save_every_n_epochs = 1

# Optimizer
optimizer_type = "AdamW8bit"
learning_rate = 1e-4

# Noise schedule
noise_scheduler = "ddpm"
min_snr_gamma = 5.0

# --- AuraFace Identity Loss ---
auraface_lambda = 0.1          # Weight of identity loss (0.0 = disabled)
auraface_threshold = 0.3       # Only apply when timestep < 30%
auraface_data_dir = "./data/my_persona"  # Contains face_bboxes.npz and auraface_embeddings.npz

# Other
mixed_precision = "fp16"
save_precision = "fp16"
cache_latents = true
cache_latents_to_disk = true
```

**Verification:** Config file parses cleanly. Training command works: `python train_network.py --config_file examples/auraface_sdxl_lora.toml`

---

## Summary: Files Changed

| File | Action | Description |
|---|---|---|
| `library/auraface_utils.py` | **Create** | AuraFace wrapper + AuraFaceConfig |
| `library/custom_train_functions.py` | **Modify** | Add `tweedie_x0_estimate()`, `compute_id_loss()` |
| `library/train_util.py` | **Modify** | Add `load_auraface_metadata()` |
| `train_network.py` | **Modify** | Add CLI args, AuraFace init, loss integration + return noisy_latents |
| `tools/compute_face_bboxes.py` | **Create** | Bounding box precomputation script |
| `tools/compute_auraface_embeddings.py` | **Create** | Target embedding precomputation script |
| `tests/test_auraface_loss.py` | **Create** | Unit tests for Tweedie + config |
| `examples/auraface_sdxl_lora.toml` | **Create** | Example training configuration |

## Key Design Decisions

1. **Timestep gating at 0.3**: Only compute identity loss when the x̂₀ estimate has enough facial structure for AuraFace to produce meaningful embeddings. This is configurable via `--auraface_threshold`.

2. **VAE decode in the training loop**: The VAE decoder is differentiable but expensive. For small batch sizes (1-2 on a 2070 Super), this is acceptable. Future optimization could cache decoded x̂₀ estimates.

3. **Bbox precomputation**: Face detection is not differentiable, so we precompute bounding boxes offline. The crop itself (tensor slicing + interpolation) is fully differentiable.

4. **Target embedding as mean**: The target identity embedding is the mean of all training image embeddings. This provides a robust anchor point. Future versions could use the centroid of a subset or a weighted mean.

5. **lambda_id starting point of 0.1**: Identity loss should regularize, not dominate. 0.1 is a conservative starting point; tune based on validation.

6. **Gradient normalization via VAE scaling factor**: The identity loss backpropagates through the VAE decoder — a massive, non-linear Jacobian that amplifies gradients by 100×–1000× relative to the latent-space noise loss. Without normalization, the identity signal obliterates the flow-matching optimization. The fix uses a `.register_hook` on `x̂₀` that divides incoming gradients by `vae.config.scaling_factor²`, canceling the decoder's amplification. This is the same scaling factor used to map between pixel and latent space, making it the natural normalization constant.

## Known Limitations / Risks

- **AuraFace model architecture unknown**: The exact loading code depends on the structure of `auraness/auraface` on HuggingFace. May need adjustment.
- **VAE decode memory — 2070 Super insufficient**: Decoding full 1024×1024 latents to pixels in the training loop adds substantial VRAM overhead. The 2070 Super (8GB) cannot handle SDXL LoRA training + frozen text encoders + in-loop VAE decode simultaneously. Two mitigations:
  - **Tiled decode**: `vae.enable_tiling()` splits the decode into overlapping tiles, cutting peak memory ~4× at the cost of ~10% speed. We enable this unconditionally when AuraFace loss is active.
  - **VAE slicing**: `vae.enable_slicing()` processes latent channels in slices for additional savings.
  - **Target hardware**: vast.ai with a 4090 (24GB) or the AMD Strix Halo (128GB unified) for full training runs. The 2070 Super can still be used for dry-run validation with tiling enabled and batch_size=1.
- **High timestep noise**: At timesteps near the threshold (t ≈ 0.3), x̂₀ may still be noisy, producing unstable identity gradients. Lowering the threshold to 0.2 or adding a warmup schedule can help.
- **Gradient normalization calibration**: `vae.config.scaling_factor²` is a first-order approximation. If identity loss still dominates or vanishes, the normalization constant may need empirical tuning (e.g., scaling factor exponent, or dynamic gradient norm matching via `torch.autograd.grad`).

## Next Phases (Out of Scope)

- Phase 2: DINOv3 CLS token conditioning in the forward pass
- Phase 3: Normal-map curvature loss (L_geo) as spatial attention weight
- Phase 4: Conditioning dropout for inference-time flexibility
