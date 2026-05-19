"""Smoke tests for AuraFace identity loss integration."""


def test_tweedie_estimation():
    """Test Tweedie's formula gives correct shapes and values."""
    import torch
    from diffusers import DDPMScheduler
    from library.auraface_loss import tweedie_x0_estimate
    from library.custom_train_functions import prepare_scheduler_for_custom_training

    scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    prepare_scheduler_for_custom_training(scheduler, torch.device("cpu"))

    noise_pred = torch.randn(2, 4, 128, 128)
    noisy_latents = torch.randn(2, 4, 128, 128)
    timesteps = torch.tensor([100, 500], dtype=torch.long)

    x0_hat = tweedie_x0_estimate(noise_pred, noisy_latents, timesteps, scheduler)

    assert x0_hat.shape == (2, 4, 128, 128), f"Unexpected shape: {x0_hat.shape}"
    # x0_hat should be finite
    assert torch.isfinite(x0_hat).all(), "x0_hat contains NaN or inf"
    print("  ✓ Tweedie estimation shape correct and finite")


def test_auraface_config():
    """Test AuraFaceConfig dataclass and bbox scaling."""
    from library.auraface_utils import AuraFaceConfig

    config = AuraFaceConfig(
        lambda_id=0.1,
        timestep_threshold=0.3,
        image_size=(1024, 1024),
        vae_scale_factor=8,
    )

    # Test bbox scaling to latent space
    scaled = config.scale_bbox_to_latent(0, 1024, 0, 1024)
    assert scaled == (0, 8, 0, 8), f"Unexpected scaled bbox: {scaled}"

    # Test with non-zero origin
    scaled2 = config.scale_bbox_to_latent(100, 500, 200, 600)
    assert scaled2[0] == 0, f"Expected y1=0, got {scaled2[0]}"  # 100*8/1024 = 0
    assert scaled2[1] == 3, f"Expected y2=3, got {scaled2[1]}"  # 500*8/1024 = 3

    print("  ✓ Bbox scaling correct")


def test_load_metadata():
    """Test load_auraface_metadata with non-existent directory."""
    import tempfile
    import os
    from library.auraface_utils import load_auraface_metadata

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = load_auraface_metadata(tmpdir)
        assert isinstance(metadata, dict), "Should return dict even for empty dir"
        assert len(metadata) == 0, "Empty dir should return empty dict"
        print("  ✓ Empty directory returns empty metadata dict")


def test_auraface_config_bbox_lookup():
    """Test get_bbox_for_image with various filename formats."""
    from library.auraface_utils import AuraFaceConfig

    config = AuraFaceConfig(
        bboxes={"image_01.jpg": [10, 100, 20, 200]}
    )

    # Exact match
    bbox = config.get_bbox_for_image("image_01.jpg")
    assert bbox == (10, 100, 20, 200), f"Exact match failed: {bbox}"

    # Basename match
    bbox2 = config.get_bbox_for_image("/long/path/image_01.jpg")
    assert bbox2 == (10, 100, 20, 200), f"Basename match failed: {bbox2}"

    # Missing
    bbox3 = config.get_bbox_for_image("nonexistent.jpg")
    assert bbox3 is None, f"Missing should return None: {bbox3}"

    # No bboxes at all
    config2 = AuraFaceConfig(bboxes=None)
    assert config2.get_bbox_for_image("anything.jpg") is None

    print("  ✓ Bbox lookup handles exact, basename, and missing")


if __name__ == "__main__":
    print("Running AuraFace smoke tests...\n")
    test_tweedie_estimation()
    test_auraface_config()
    test_load_metadata()
    test_auraface_config_bbox_lookup()
    print("\n✓ All tests passed!")
