"""
DINOv3 CLS token extraction and projection for SDXL identity conditioning.

Loads facebook/dinov3-vitl16-pretrain-lvd1689m (ViT-L, 1024-dim CLS).
Frozen model for offline precomputation + lightweight projection MLP
for training-time UNet conditioning injection.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class DINOv3Wrapper(nn.Module):
    """Frozen DINOv3 ViT-L for CLS token extraction."""

    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device("cpu")

        self.model = self._load_dinov3()
        self.model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

        self.cls_dim = 1024  # ViT-L hidden dim
        logger.info(f"DINOv3 ViT-L loaded on {self.device}, CLS dim={self.cls_dim}")

    def _load_dinov3(self):
        """Load DINOv3 from HuggingFace transformers."""
        from transformers import AutoModel

        # Try local cache first, fall back to HF hub
        try:
            model = AutoModel.from_pretrained(
                "facebook/dinov3-vitl16-pretrain-lvd1689m",
                cache_dir="/mnt/nas-ai-models/huggingface-cache",
                local_files_only=True,
            )
            logger.info("Loaded DINOv3 from local cache")
        except Exception:
            logger.info("DINOv3 not in local cache, downloading from HF hub...")
            model = AutoModel.from_pretrained(
                "facebook/dinov3-vitl16-pretrain-lvd1689m",
            )

        return model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token from images.

        Args:
            images: (B, 3, 518, 518) tensor, ImageNet-normalized

        Returns:
            CLS token of shape (B, 1024)
        """
        with torch.no_grad():
            outputs = self.model(images)
            cls_token = outputs.last_hidden_state[:, 0, :]  # (B, 1024)
        return cls_token


class DINOProjection(nn.Module):
    """
    Projects DINOv3 CLS token (1024-dim) to SDXL conditioning space (2048-dim).

    Architecture: Linear → LayerNorm → GELU → Linear → LayerNorm
    ~4.2M trainable parameters (1024×2048 + 2048×2048).
    """

    def __init__(self, dinov3_dim: int = 1024, sdxl_dim: int = 2048):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(dinov3_dim, sdxl_dim),
            nn.LayerNorm(sdxl_dim),
            nn.GELU(),
            nn.Linear(sdxl_dim, sdxl_dim),
            nn.LayerNorm(sdxl_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Project CLS token to SDXL cross-attention dimension.

        Args:
            cls_token: (B, 1024) from DINOv3

        Returns:
            Projected token: (B, 1, 2048) ready for UNet concatenation
        """
        projected = self.projection(cls_token)  # (B, 2048)
        return projected.unsqueeze(1)  # (B, 1, 2048)
