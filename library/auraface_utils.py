"""
AuraFace / ArcFace utilities for identity embedding extraction.

Supports two backends:
  - ONNX (via insightface) for offline precomputation (non-differentiable)
  - PyTorch ArcFace (via IResNet-100) for training-time identity loss (differentiable)

ONNX Model source: https://huggingface.co/fal/AuraFace-v1
PyTorch ArcFace weights: insightface recognition/arcface_torch model zoo
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class AuraFaceWrapper(nn.Module):
    """Frozen face recognition model for identity embedding extraction.
    
    Two loading modes:
    1. ONNX (default): Loads glintr100.onnx from fal/AuraFace-v1 via insightface.
       Non-differentiable — use for offline target embedding precomputation.
    2. PyTorch ArcFace: Loads IResNet-100 weights from a .pth file.
       Fully differentiable — use for training-time identity loss.
    
    Input: (B, 3, 112, 112) tensor, values in [0, 1], RGB.
    Output: L2-normalized embedding of shape (B, 512).
    """

    def __init__(self, device: torch.device = None, dtype: torch.dtype = torch.float32,
                 arcface_weights: Optional[str] = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.input_size = 112
        self._embedding_dim = None
        self._use_pytorch = False

        if arcface_weights and os.path.exists(arcface_weights):
            self.model = self._load_pytorch_arcface(arcface_weights)
            self._use_pytorch = True
            logger.info(f"ArcFace (PyTorch) loaded on {self.device}")
        else:
            self.model = self._load_onnx_auraface()
            logger.info(f"AuraFace (ONNX) loaded on {self.device}")

    def _load_pytorch_arcface(self, weights_path: str):
        """Load ArcFace IResNet from PyTorch checkpoint (differentiable)."""
        from .iresnet_se import IResNetSE50

        model = IResNetSE50()
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
        model.to(device=self.device, dtype=self.dtype)
        model.eval()

        for param in model.parameters():
            param.requires_grad_(False)

        logger.info(f"Loaded ArcFace PyTorch weights from {weights_path}")
        return model

    def _load_onnx_auraface(self):
        """Load AuraFace ONNX model via insightface (non-differentiable)."""
        from huggingface_hub import snapshot_download
        import insightface

        repo_id = "fal/AuraFace-v1"

        try:
            model_path = snapshot_download(repo_id)
        except Exception as e:
            logger.error(f"Could not download AuraFace: {e}")
            return None

        onnx_path = os.path.join(model_path, "glintr100.onnx")
        if not os.path.exists(onnx_path):
            for fname in os.listdir(model_path):
                if fname.endswith('.onnx') and 'det' not in fname.lower():
                    onnx_path = os.path.join(model_path, fname)
                    break

        if not os.path.exists(onnx_path):
            logger.error(f"No ONNX model found in {model_path}")
            return None

        try:
            model = insightface.model_zoo.get_model(onnx_path)
            ctx_id = 0 if torch.cuda.is_available() else -1
            model.prepare(ctx_id=ctx_id)
            logger.info(f"Loaded AuraFace ONNX from {onnx_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load AuraFace ONNX: {e}")
            return None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract identity embedding from face images.

        Args:
            images: Tensor of shape (B, 3, 112, 112), values in [0, 1], RGB

        Returns:
            Normalized embedding of shape (B, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("Face recognition model not loaded")

        if self._use_pytorch:
            # PyTorch path: fully differentiable
            images = (images - 0.5) / 0.5  # Normalize to [-1, 1]
            with torch.set_grad_enabled(True):
                embedding = self.model(images)
        else:
            # ONNX path: non-differentiable (numpy bridge)
            images_np = images.detach().cpu().numpy()
            images_np = (images_np * 255.0).astype(np.uint8)

            embeddings = []
            for i in range(images_np.shape[0]):
                img = images_np[i].transpose(1, 2, 0)
                img = img[:, :, ::-1]  # RGB -> BGR
                feat = self.model.get_feat(img)
                embeddings.append(feat)

            embedding = torch.from_numpy(np.stack(embeddings)).to(
                device=self.device, dtype=self.dtype
            )
            # Squeeze extra dimension if present (get_feat may return (1, 512))
            if embedding.dim() == 3:
                embedding = embedding.squeeze(1)

        # L2 normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the output embedding."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        if self.model is None:
            raise RuntimeError("Face recognition model not loaded")
        dummy = torch.rand(1, 3, self.input_size, self.input_size,
                          device=self.device, dtype=self.dtype)
        with torch.no_grad():
            emb = self.forward(dummy)
        self._embedding_dim = emb.shape[-1]
        return self._embedding_dim

    @property
    def is_differentiable(self) -> bool:
        """Whether gradients flow through this model."""
        return self._use_pytorch


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

    def get_bbox_for_image(self, filename: str) -> Optional[tuple]:
        """Get the bounding box for a specific image."""
        if self.bboxes is None:
            return None
        # Try exact match first, then basename
        import os
        if filename in self.bboxes:
            return tuple(self.bboxes[filename])
        basename = os.path.basename(filename)
        if basename in self.bboxes:
            return tuple(self.bboxes[basename])
        return None


def load_auraface_metadata(image_dir: str) -> dict:
    """
    Load precomputed AuraFace metadata from an image directory.

    Returns dict with keys:
        - 'bboxes': dict mapping filename -> [y1, y2, x1, x2]
        - 'target_embedding': numpy array of shape (1, embedding_dim)
        - 'image_embeddings': numpy array of shape (N, embedding_dim)
    """
    import json
    import os

    metadata = {}

    # Load bounding boxes
    bbox_json = os.path.join(image_dir, "face_bboxes.json")
    bbox_npz = os.path.join(image_dir, "face_bboxes.npz")

    if os.path.exists(bbox_json):
        with open(bbox_json) as f:
            metadata['bboxes'] = json.load(f)
    elif os.path.exists(bbox_npz):
        data = np.load(bbox_npz, allow_pickle=True)
        filenames = data['filenames']
        bbox_array = data['bboxes']
        metadata['bboxes'] = {
            str(fn): bbox_array[i].tolist()
            for i, fn in enumerate(filenames)
        }

    # Load AuraFace embeddings
    emb_npz = os.path.join(image_dir, "auraface_embeddings.npz")
    if os.path.exists(emb_npz):
        data = np.load(emb_npz)
        metadata['target_embedding'] = data['mean_embedding']  # (1, D)
        metadata['image_embeddings'] = data['embeddings']  # (N, D)

    return metadata
