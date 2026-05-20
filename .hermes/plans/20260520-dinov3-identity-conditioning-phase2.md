# Phase 2: DINOv3 CLS Token Conditioning — SDXL UNet

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Inject a precomputed DINOv3 CLS token into the SDXL UNet's cross-attention conditioning, teaching the model to associate a high-dimensional semantic identity vector with facial geometry. Combined with the existing ArcFace identity loss (Phase 1), this gives the model a strong identity anchor in the forward pass.

**Architecture:** Before training, extract the DINOv3 ViT-L CLS token (1024-dim) from a reference image. At each training step, project this token to 2048-dim via a lightweight MLP, concatenate it with the SDXL text embeddings along the sequence dimension, and feed to the UNet. During inference, pass the same DINO token instead of a trigger word. Condition with dropout (20-30%) so the model doesn't become reliant on the token.

**Tech Stack:** PyTorch, DINOv3 ViT-L (frozen), SDXL UNet cross-attention, existing kohya-ss fork

**Assumptions:**
- DINOv3 ViT-L cached at `/mnt/nas-ai-models/huggingface-cache/` (facebook/dinov3-vitl16-pretrain-lvd1689m)
- Phase 1 identity loss code is in place and working
- Training runs on Strix Halo (ROCm, fp32)

---

## Design: Where DINO Token Injects

SDXL UNet cross-attention accepts conditioning as two tensors:
- `text_embedding`: shape (B, 77, 2048) — token sequence from combined CLIP-L + CLIP-bigG
- `vector_embedding`: shape (B, 2816) — pooled CLIP output + size embeddings

The DINO CLS token replaces the need for a trigger word. We inject it by expanding it into a sequence and concatenating:

```
text_embedding = torch.cat([
    encoder_hidden_states1,  # (B, 77, 768)  from CLIP-L
    encoder_hidden_states2,  # (B, 77, 1280) from CLIP-bigG
], dim=2)  # → (B, 77, 2048)

# DINO injection: project + expand
dino_seq = dino_projection(dino_cls_token)  # (1, 1024) → (B, 1, 2048)
text_embedding = torch.cat([text_embedding, dino_seq], dim=1)  # → (B, 78, 2048)
```

The injection point is `sdxl_train_network.py` → `SdxlNetworkTrainer.call_unet()` at line 205 where `text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2)`.

---

## Phase 2A: DINOv3 Wrapper + Projection

### Task 1: Create DINOv3 wrapper module

**Objective:** Load frozen DINOv3 ViT-L and extract CLS tokens.

**Files:**
- Create: `library/dinov3_utils.py`

**Code:**

```python
"""
DINOv3 CLS token extraction for identity conditioning.

Loads facebook/dinov3-vitl16-pretrain-lvd1689m (ViT-L, 1024-dim CLS).
Frozen — used for offline precomputation and training-time injection.
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
        """Load DINOv3 from transformers."""
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            cache_dir="/mnt/nas-ai-models/huggingface-cache",
        )
        return model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token from images.

        Args:
            images: (B, 3, 518, 518) tensor, normalized to ImageNet stats

        Returns:
            CLS token of shape (B, 1024)
        """
        with torch.no_grad():
            outputs = self.model(images)
            cls_token = outputs.last_hidden_state[:, 0, :]  # (B, 1024)
        return cls_token


class DINOProjection(nn.Module):
    """Projects DINOv3 CLS token (1024-dim) to SDXL conditioning space (2048-dim)."""

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
        Project CLS token to SDXL conditioning dim.

        Args:
            cls_token: (B, 1024) from DINOv3

        Returns:
            Projected token: (B, 1, 2048) ready for UNet concat
        """
        projected = self.projection(cls_token)  # (B, 2048)
        return projected.unsqueeze(1)  # (B, 1, 2048)
```

**Verification:** Module imports, loads DINOv3 from cache, produces correct shapes.

---

### Task 2: Create DINO CLS token precomputation script

**Objective:** Script to compute DINOv3 CLS token from reference image(s) and save as .npy.

**Files:**
- Create: `tools/compute_dinov3_token.py`

**Code:**

```python
#!/usr/bin/env python3
"""Compute DINOv3 CLS token from a reference image and save to .npy."""

import argparse
import sys
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Reference image path")
    parser.add_argument("--output", required=True, help="Output .npy file")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from library.dinov3_utils import DINOv3Wrapper

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dinov3 = DINOv3Wrapper(device=device)

    # DINOv3 expects 518x518 with ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    cls_token = dinov3(img_tensor)  # (1, 1024)
    np.save(args.output, cls_token.cpu().numpy())
    print(f"Saved CLS token ({cls_token.shape}) to {args.output}")


if __name__ == "__main__":
    main()
```

**Verification:** Run on a test image, verify output shape is (1, 1024).

---

## Phase 2B: Training-Time Injection

### Task 3: Add CLI args for DINOv3 conditioning

**Objective:** Add `--dinov3_token`, `--dinov3_dropout` arguments.

**Files:**
- Modify: `train_network.py` → `setup_parser()`

**Add after existing AuraFace args:**

```python
parser.add_argument(
    "--dinov3_token",
    type=str,
    default=None,
    help="Path to .npy file with precomputed DINOv3 CLS token",
)
parser.add_argument(
    "--dinov3_dropout",
    type=float,
    default=0.2,
    help="Probability of dropping DINO token during training (0.0-1.0)",
)
```

---

### Task 4: Initialize DINOv3 projection in training loop

**Objective:** Load DINO projection model and CLS token in `train()`.

**Files:**
- Modify: `train_network.py` → `train()` method (after AuraFace init)

**Code (after VAE tiling section, ~line 995):**

```python
# Initialize DINOv3 conditioning if enabled
self.dinov3_token = None
self.dino_projection = None
if getattr(args, 'dinov3_token', None) and os.path.exists(args.dinov3_token):
    from library.dinov3_utils import DINOProjection
    import numpy as np

    logger.info(f"Loading DINOv3 CLS token from {args.dinov3_token}")
    token = np.load(args.dinov3_token)
    self.dinov3_token = torch.from_numpy(token).to(
        accelerator.device, dtype=weight_dtype
    )  # (1, 1024)

    self.dino_projection = DINOProjection().to(
        accelerator.device, dtype=weight_dtype
    )
    self.dino_dropout = getattr(args, 'dinov3_dropout', 0.2)
    logger.info(
        f"DINOv3 projection initialized. Dropout rate: {self.dino_dropout}"
    )
```

**Step 2: Register DINO projection parameters with optimizer**

The projection MLP must be added to the optimizer's parameter groups, or gradients will flow but the weights won't update. Find the optimizer setup section in `train()` (around line 790):

```python
trainable_params = network.prepare_optimizer_params(
    args.text_encoder_lr, args.unet_lr, args.learning_rate
)

# Register DINO projection parameters
if self.dino_projection is not None:
    trainable_params.append({
        "params": self.dino_projection.parameters(),
        "lr": args.learning_rate,
    })
    logger.info("DINOv3 projection added to optimizer")

optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(
    args, trainable_params
)
```

**Verification:** Check that `optimizer.param_groups` includes the projection MLP parameters after init.

**Note:** Currently `network_train_unet_only = true`, so `text_encoder_lr` and `unet_lr` are unused. The projection uses the base `learning_rate`. A separate `--dinov3_lr` arg could be added later for fine-tuning.

---

### Task 5: Inject DINO token into UNet conditioning

**Objective:** Modify `SdxlNetworkTrainer.call_unet()` to concatenate DINO token.

**Files:**
- Modify: `sdxl_train_network.py` → `call_unet()` (line ~205)

**Change from:**

```python
text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)
```

**To:**

```python
text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

# Inject DINOv3 CLS token if available
if self.dinov3_token is not None and self.dino_projection is not None:
    batch_size = noisy_latents.shape[0]

    # Expand CLS token to batch
    dino_token = self.dinov3_token.expand(batch_size, -1)  # (B, 1024)

    # Conditioning dropout: replace with zeros p% of the time
    if self.training and self.dino_dropout > 0:
        mask = torch.rand(batch_size, 1, device=dino_token.device) > self.dino_dropout
        dino_token = dino_token * mask.float()

    # Project to 2048-dim and add as extra sequence token
    dino_seq = self.dino_projection(dino_token)  # (B, 1, 2048)
    text_embedding = torch.cat([text_embedding, dino_seq], dim=1)  # (B, 78, 2048)
```

**Important:** The variable `self.training` is not reliable since `process_batch()` may disable gradients. Check mode via `is_train` parameter or `torch.is_grad_enabled()`.

**Verification:** Training runs without errors. DINO token shown in debug logging.

---

## Phase 2C: Integration & Testing

### Task 6: Add DINO projection logging

**Objective:** Log whether DINO token is active per step (for debugging dropout).

**Files:**
- Modify: `train_network.py` → step logging (~line 1608)

**Add:**

```python
if self.dinov3_token is not None:
    logs["dino/active"] = 1.0 if torch.is_grad_enabled() else 0.0
```

---

### Task 7: Update example config

**Objective:** Add DINOv3 options to example configs.

**Files:**
- Modify: `examples/auraface_sdxl_lora.toml`
- Create: `examples/auraface_dinov3_sdxl_lora.toml`

**New config entries:**

```toml
# DINOv3 identity conditioning (forward pass)
dinov3_token = "./data/my_person/dinov3_cls.npy"
dinov3_dropout = 0.2
```

---

### Task 8: Smoke test — forward pass with DINO token

**Objective:** Verify DINO injection doesn't break UNet forward pass.

**Files:**
- Create: `tests/test_dinov3_injection.py`

**Code:**

```python
"""Smoke test: DINOv3 CLS token injection into SDXL UNet."""
import torch


def test_dino_projection_shape():
    """Test DINO Projection gives correct output shape."""
    from library.dinov3_utils import DINOProjection

    proj = DINOProjection(dinov3_dim=1024, sdxl_dim=2048)
    x = torch.randn(2, 1024)
    out = proj(x)
    assert out.shape == (2, 1, 2048), f"Expected (2,1,2048), got {out.shape}"
    print("  ✓ DINO projection shape correct")


def test_dino_concat():
    """Test DINO token concatenation matches expected shape."""
    text_emb = torch.randn(2, 77, 2048)  # standard SDXL text embedding
    dino_seq = torch.randn(2, 1, 2048)   # projected DINO token

    combined = torch.cat([text_emb, dino_seq], dim=1)
    assert combined.shape == (2, 78, 2048), f"Expected (2,78,2048), got {combined.shape}"
    print("  ✓ DINO concat shape correct")


def test_dino_dropout():
    """Test conditioning dropout logic."""
    token = torch.ones(4, 1024)
    dropout_rate = 0.5

    # Simulate dropout: create a mask
    torch.manual_seed(42)
    mask = torch.rand(4, 1) > dropout_rate
    dropped = token * mask.float()

    # ~50% should be zeroed
    zero_rows = (dropped.abs().sum(dim=1) == 0).sum().item()
    print(f"  ✓ Dropout zeroed {zero_rows}/4 rows (expected ~2)")
    assert 1 <= zero_rows <= 3  # statistically ~2


if __name__ == "__main__":
    test_dino_projection_shape()
    test_dino_concat()
    test_dino_dropout()
    print("\n✓ All DINOv3 tests passed!")
```

---

## Summary: Files Changed

| File | Action | Description |
|---|---|---|
| `library/dinov3_utils.py` | **Create** | DINOv3Wrapper + DINOProjection |
| `tools/compute_dinov3_token.py` | **Create** | CLS token precomputation script |
| `train_network.py` | **Modify** | CLI args, DINO init in train(), logging |
| `sdxl_train_network.py` | **Modify** | DINO token injection in call_unet() |
| `examples/auraface_dinov3_sdxl_lora.toml` | **Create** | Example config with DINO |
| `tests/test_dinov3_injection.py` | **Create** | Smoke tests |

## Key Design Decisions

1. **Token as extra sequence element**: Append DINO token to the text embedding sequence (77 → 78 tokens) rather than summing into it. This preserves all existing text conditioning while adding identity-specific attention.

2. **Lightweight projection MLP**: 2-layer MLP with LayerNorm + GELU (1024 → 2048 → 2048). ~4.2M trainable parameters. Trained jointly with the LoRA (no separate optimizer needed — gradients flow through projection to the UNet).

3. **Conditioning dropout at 20%**: During training, 20% of steps drop the DINO token (replace with zeros). This prevents the model from becoming catastrophically reliant on it, ensuring inference works without the token available.

4. **Single reference image**: One DINO CLS token precomputed from a representative image. Multiple images could be averaged, but a single well-chosen reference provides sufficient identity signal.

5. **Frozen DINOv3, trainable projection**: DINOv3 itself is frozen (no gradients). Only the projection MLP learns to map the semantic identity vector into the UNet's conditioning space. The LoRA weights + projection are the only trainable parameters.

## Known Limitations / Risks

- **Cross-attention dimension**: Appending an extra token changes the sequence length from 77 to 78. SDXL's cross-attention should handle variable-length sequences, but some implementations may have hardcoded position assumptions.
- **DINOv3 image size**: DINOv3 ViT-L expects 518×518 images with ImageNet normalization. The precomputation script handles this; the training loop doesn't call DINOv3 directly.
- **Projection overfitting**: With only one reference image, the projection MLP may overfit to that specific CLS token. Using multiple reference images and averaging their CLS tokens is a safer initial approach.
- **Memory**: The projection MLP adds ~4.2M params × fp32 = ~17MB — negligible relative to the SDXL UNet.
