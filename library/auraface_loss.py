"""
AuraFace identity loss functions for diffusion model training.

These are used during the training loop to add an identity-preserving
regularization term based on AuraFace embeddings.
"""

import torch
import torch.nn.functional as F


def tweedie_x0_estimate(
    noise_pred: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.IntTensor,
    noise_scheduler,
) -> torch.Tensor:
    """
    Compute the single-step clean latent estimate using Tweedie's formula.

    For epsilon-prediction (SDXL):
        x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)

    Args:
        noise_pred: The UNet's noise prediction, shape (B, C, H, W)
        noisy_latents: The noised latents x_t, shape (B, C, H, W)
        timesteps: The timestep for each sample, shape (B,)
        noise_scheduler: DDPMScheduler with pre-computed alphas_cumprod

    Returns:
        x0_hat estimate, shape (B, C, H, W)
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(noise_pred.device)
    alpha_bar_t = alphas_cumprod[timesteps]  # shape (B,)
    alpha_bar_t = alpha_bar_t.reshape(-1, 1, 1, 1)

    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)

    x0_hat = (noisy_latents - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
    return x0_hat


def compute_id_loss(
    x0_hat: torch.Tensor,
    timesteps: torch.IntTensor,
    batch: dict,
    vae,
    auraface,
    auraface_config,
    noise_scheduler,
    weight_dtype: torch.dtype,
) -> torch.Tensor | None:
    """
    Compute the AuraFace identity loss for a batch of training samples.

    Workflow:
        1. Gate: only apply for timesteps below threshold (t < 0.3 * max_t)
        2. VAE-decode x0_hat from latent space to pixel space
        3. Crop face region using precomputed bounding boxes
        4. Resize to 112x112 for AuraFace
        5. Extract identity embedding from generated face
        6. Compute cosine similarity loss against target embedding

    The VAE decode is differentiable, so gradients flow back through the
    entire chain to the UNet. The gradient hook on x0_hat (applied at the
    call site) normalizes magnitudes to prevent VAE decoder amplification.

    Args:
        x0_hat: Clean latent estimate, shape (B, C, H_lat, W_lat)
        timesteps: Timestep for each sample, shape (B,)
        batch: Batch dict containing 'image_filenames'
        vae: Frozen VAE decoder
        auraface: Frozen AuraFaceWrapper
        auraface_config: AuraFaceConfig with bboxes and target_embedding
        noise_scheduler: For timestep threshold calculation
        weight_dtype: Weight dtype for VAE

    Returns:
        Scalar identity loss, or None if no samples pass the threshold gate
    """
    # Gate: only apply for low timesteps where face structure exists
    max_t = noise_scheduler.config.num_train_timesteps
    threshold_step = int(auraface_config.timestep_threshold * max_t)

    mask = timesteps < threshold_step
    if not mask.any():
        return None

    # Select samples that pass the threshold
    x0_hat_selected = x0_hat[mask]
    batch_size = x0_hat_selected.shape[0]
    filenames = batch.get("image_filenames", None)

    # Scale latents before VAE decode
    x0_hat_selected = x0_hat_selected / vae.config.scaling_factor

    with torch.set_grad_enabled(True):
        # VAE decode: latent -> pixel (differentiable)
        decoded = vae.decode(x0_hat_selected.to(weight_dtype)).sample

        # Normalize from [-1, 1] to [0, 1]
        decoded = (decoded + 1.0) / 2.0
        decoded = torch.clamp(decoded, 0.0, 1.0)

        # Crop and resize each face
        face_crops = []
        for i in range(batch_size):
            bbox = None
            if (filenames is not None
                    and auraface_config.bboxes is not None
                    and i < len(filenames)):
                bbox = auraface_config.get_bbox_for_image(filenames[i])

            if bbox is not None:
                y1, y2, x1, x2 = bbox
                h_ratio = decoded.shape[2] / auraface_config.image_size[0]
                w_ratio = decoded.shape[3] / auraface_config.image_size[1]
                y1_s = max(0, int(y1 * h_ratio))
                y2_s = min(decoded.shape[2], int(y2 * h_ratio))
                x1_s = max(0, int(x1 * w_ratio))
                x2_s = min(decoded.shape[3], int(x2 * w_ratio))
                face = decoded[i:i+1, :, y1_s:y2_s, x1_s:x2_s]
            else:
                # Fallback: center crop (assumes face is centered)
                h, w = decoded.shape[2], decoded.shape[3]
                crop_size = min(h, w) // 2
                cy, cx = h // 2, w // 2
                y_start = max(0, cy - crop_size // 2)
                y_end = min(h, cy + crop_size // 2)
                x_start = max(0, cx - crop_size // 2)
                x_end = min(w, cx + crop_size // 2)
                face = decoded[i:i+1, :, y_start:y_end, x_start:x_end]

            # Resize to AuraFace input size
            face = F.interpolate(
                face, size=(112, 112), mode='bilinear', align_corners=False
            )
            face_crops.append(face)

        face_crops = torch.cat(face_crops, dim=0)  # (K, 3, 112, 112)

        # Extract identity embeddings
        generated_embeddings = auraface(face_crops)  # (K, D)

        # Cosine similarity loss against target
        target_emb = auraface_config.target_embedding.to(
            generated_embeddings.device
        )
        cos_sim = F.cosine_similarity(
            generated_embeddings,
            target_emb.expand(batch_size, -1),
            dim=-1,
        )
        id_loss = (1.0 - cos_sim).mean()

    return id_loss
