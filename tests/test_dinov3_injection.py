"""Smoke tests: DINOv3 CLS token injection into SDXL UNet conditioning."""


def test_dino_projection_shape():
    """Test DINOProjection gives correct output shape."""
    import torch
    from library.dinov3_utils import DINOProjection

    proj = DINOProjection(dinov3_dim=1024, sdxl_dim=2048)
    x = torch.randn(2, 1024)
    out = proj(x)
    assert out.shape == (2, 1, 2048), f"Expected (2,1,2048), got {out.shape}"
    print("  ✓ DINO projection shape correct")


def test_dino_concat():
    """Test DINO token concatenation with padding to 80 tokens."""
    import torch

    text_emb = torch.randn(2, 77, 2048)  # standard SDXL text embedding
    dino_seq = torch.randn(2, 1, 2048)   # projected DINO token
    padding = torch.zeros(2, 2, 2048)     # padding for FlashAttention

    combined = torch.cat([text_emb, dino_seq, padding], dim=1)
    assert combined.shape == (2, 80, 2048), f"Expected (2,80,2048), got {combined.shape}"
    print("  ✓ DINO concat + pad shape correct (80 tokens)")


def test_dino_dropout():
    """Test conditioning dropout logic."""
    import torch

    token = torch.ones(4, 1024)
    dropout_rate = 0.5

    torch.manual_seed(42)
    mask = torch.rand(4, 1) > dropout_rate
    dropped = token * mask.float()

    zero_rows = (dropped.abs().sum(dim=1) == 0).sum().item()
    print(f"  ✓ Dropout zeroed {zero_rows}/4 rows (expected ~2)")
    assert 1 <= zero_rows <= 3


def test_dino_projection_trainable():
    """Test DINOProjection parameters are trainable."""
    import torch
    from library.dinov3_utils import DINOProjection

    proj = DINOProjection()
    x = torch.randn(1, 1024, requires_grad=False)
    out = proj(x)
    loss = out.sum()
    loss.backward()

    # Check that gradients flowed to projection parameters
    has_grad = False
    for p in proj.parameters():
        if p.grad is not None:
            has_grad = True
            break
    assert has_grad, "Projection parameters should receive gradients"
    print("  ✓ DINO projection is trainable (gradients flow)")

    # Check param count
    n_params = sum(p.numel() for p in proj.parameters())
    print(f"  ✓ Projection params: {n_params:,} (~4.2M expected)")


if __name__ == "__main__":
    print("DINOv3 Phase 2 smoke tests\n")
    test_dino_projection_shape()
    test_dino_concat()
    test_dino_dropout()
    test_dino_projection_trainable()
    print("\n✓ All DINOv3 tests passed!")
