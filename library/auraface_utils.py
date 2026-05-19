"""
AuraFace utilities for identity embedding extraction.

AuraFace is loaded as a frozen model. Its forward pass is differentiable,
so gradients flow through to the input image (our x̂₀ estimate).

Model source: https://huggingface.co/auraness/auraface
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
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

        self._embedding_dim = None
        logger.info(f"AuraFace loaded on {device}")

    def _load_auraface(self):
        """Load AuraFace from HuggingFace hub with multi-tier fallback."""
        from huggingface_hub import snapshot_download
        import os
        import sys

        repo_id = "auraness/auraface"

        try:
            model_path = snapshot_download(repo_id)
        except Exception as e:
            logger.warning(f"Could not download AuraFace from HF: {e}")
            logger.warning("AuraFace will be unavailable. Identity loss disabled.")
            return None

        sys.path.insert(0, model_path)

        # Strategy 1: Try backbones.get_model (common face rec layout)
        try:
            from backbones import get_model
            model = get_model("iresnet50", fp16=(self.dtype == torch.float16))
            # Try to load weights
            for fname in os.listdir(model_path):
                if fname.endswith(('.pth', '.pt', '.bin')):
                    weights_path = os.path.join(model_path, fname)
                    state = torch.load(weights_path, map_location="cpu")
                    model.load_state_dict(state, strict=False)
                    logger.info(f"Loaded AuraFace weights from {fname}")
                    break
            return model
        except ImportError:
            logger.debug("backbones.get_model not available, trying fallback")
        except Exception as e:
            logger.debug(f"backbones approach failed: {e}")

        # Strategy 2: Try loading any .pth/.pt file as a full model
        for fname in os.listdir(model_path):
            if fname.endswith(('.pth', '.pt')):
                try:
                    model = torch.load(
                        os.path.join(model_path, fname),
                        map_location="cpu",
                        weights_only=False,
                    )
                    if isinstance(model, nn.Module):
                        logger.info(f"Loaded AuraFace from {fname}")
                        return model
                except Exception:
                    continue

        logger.warning(
            "Could not load AuraFace model from downloaded files. "
            "Check the repo structure at https://huggingface.co/auraness/auraface. "
            "Manual setup may be required."
        )
        return None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract identity embedding from face images.

        Args:
            images: Tensor of shape (B, 3, 112, 112), values in [0, 1]

        Returns:
            Normalized embedding of shape (B, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("AuraFace model not loaded")

        # Normalize to model's expected input range (mean=0.5, std=0.5)
        images = (images - 0.5) / 0.5

        with torch.set_grad_enabled(True):  # Gradients flow through to input
            embedding = self.model(images)

        # L2 normalize the embedding
        embedding = nn.functional.normalize(embedding, p=2, dim=-1)

        return embedding

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the output embedding."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        if self.model is None:
            raise RuntimeError("AuraFace model not loaded — cannot determine embedding dim")
        dummy = torch.randn(1, 3, self.input_size, self.input_size,
                           device=self.device, dtype=self.dtype)
        with torch.no_grad():
            emb = self.forward(dummy)
        self._embedding_dim = emb.shape[-1]
        return self._embedding_dim


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
