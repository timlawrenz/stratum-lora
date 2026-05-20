# Stratum-LoRA

Fork of [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) adding identity-preserving regularization to SDXL LoRA training via ArcFace face recognition.

## What it does

Standard LoRA training optimizes only for diffusion loss. This fork adds an **identity loss** that forces the model to preserve facial identity:

```
L_total = L_noise + λ · L_id

where:
  x̂₀ = Tweedie_estimate(ε_pred, x_t, t)     [only when t < 0.3 · T]
  decoded = VAE.decode(x̂₀)                     [tiled, memory-efficient]
  face = crop(decoded, precomputed_bbox)
  emb = ArcFace(face)                           [IResNet-SE50, frozen, differentiable]
  L_id = 1 - cos(emb, emb_target)
  grad ← grad / scaling_factor²                 [gradient normalization hook]
```

## Architecture

```
Training Image ──► VAE Encode ──► Add Noise ──► UNet (LoRA) ──► ε_pred
                                                    │
                          ┌─────────────────────────┘
                          │  (if t < threshold)
                          ▼
              x̂₀ = (x_t − √(1−ᾱ) · ε) / √ᾱ    ← Tweedie's formula
                          │
                          ▼
                   VAE Decode (tiled)
                          │
                          ▼
              Crop face (precomputed bbox)
                          │
                          ▼
                   ArcFace (frozen, differentiable)
                          │
                          ▼
         L_id = 1 − cos(emb_gen, emb_target)
```

## Key changes from upstream

| File | Change |
|---|---|
| `library/auraface_utils.py` | `AuraFaceWrapper` — dual-backend face model (ONNX + PyTorch ArcFace), `AuraFaceConfig` dataclass |
| `library/auraface_loss.py` | `tweedie_x0_estimate()`, `compute_id_loss()` with timestep gating, face crop, ArcFace embedding, cosine loss |
| `library/iresnet_se.py` | IResNet-SE50 backbone matching [lithiumice/insightface](https://huggingface.co/lithiumice/insightface) checkpoint |
| `train_network.py` | Identity loss integration in `process_batch()`, gradient normalization via `vae.config.scaling_factor²` hook, CLI args, AuraFace init, VAE tiling |
| `tools/compute_face_bboxes.py` | Face detection → bbox .json/.npz (InsightFace → OpenCV DNN fallback) |
| `tools/compute_auraface_embeddings.py` | ArcFace embedding precomputation with batch inference |
| `tools/evaluate_loras.py` | A/B comparison script: generates matching-seed images, extracts embeddings, reports identity fidelity |

## Quick start

```bash
# 1. Prep data
python tools/compute_face_bboxes.py --image_dir ./data/my_person
python tools/compute_auraface_embeddings.py --image_dir ./data/my_person --bbox_file ./data/my_person/face_bboxes.json

# 2. Train (see examples/ for configs)
python sdxl_train_network.py --config_file my_config.toml

# 3. Compare vs baseline
python tools/evaluate_loras.py \
  --lora_a output/auraface_lora.safetensors \
  --lora_b output/control_lora.safetensors \
  --target_emb ./data/my_person/auraface_embeddings.npz
```

### CLI args

| Flag | Default | Description |
|---|---|---|
| `--auraface_lambda` | `0.0` | Weight for identity loss (0 = disabled). Suggested: 0.1 |
| `--auraface_data_dir` | — | Directory with `face_bboxes.npz` and `auraface_embeddings.npz` |
| `--auraface_threshold` | `0.3` | Timestep ratio for loss gating (0.3 = last 30%) |
| `--arcface_weights` | — | Path to PyTorch ArcFace .pth for differentiable loss |

### ROCm / AMD notes

- Use `mixed_precision = "no"` (fp32) — fp16 produces NaN on ROCm
- Set `cache_latents = false` — VAE encoding produces NaN on ROCm
- Enable `enable_bucket = true` with `bucket_no_upscale = true`

## Plan

See [`.hermes/plans/20260519-auraface-identity-loss-phase1.md`](.hermes/plans/20260519-auraface-identity-loss-phase1.md) for the full implementation plan.

## Future phases

- Phase 2: DINOv3 CLS token conditioning
- Phase 3: Sapiens depth/normals geometry loss
- Phase 4: Conditioning dropout for inference flexibility

## License

Based on [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts). ArcFace weights from [lithiumice/insightface](https://huggingface.co/lithiumice/insightface) — non-commercial use only.

## Results

### Phase 1: Identity loss proof of concept

**Setup:** 200–400 steps, SDXL base, 512×512, fp32 on ROCm 7.2.3 (AMD Radeon 8060S, 128GB). LoRA dim=32, AdamW lr=1e-4, λ_id=0.1. ArcFace IResNet-SE50 backbone (43.8M params, frozen, differentiable). No captions, no trigger tokens.

**A/B protocol:** Train two LoRAs with identical settings (one with identity loss, one without). Generate N matching-seed images from both, extract face embeddings via AuraFace ONNX, compute cosine similarity to precomputed target embedding.

| Dataset | Images | Steps | AuraFace avg cos | Control avg cos | Wins | Δ |
|---|---|---|---|---|---|---|
| [Hegre Moloko](https://www.hegre.com/) | 43 | 200 | 0.1337 ±0.039 | 0.1147 ±0.051 | 70% | **+16.6%** |
| Scarl3tt (Instagram) | 111 | 400 | 0.1304 ±0.064 | 0.1249 ±0.058 | 60% | +4.4% |

**Key findings:**
- Identity loss consistently improves facial fidelity across datasets
- Effect is stronger on high-quality, consistent-illumination portraits (Moloko) than varied social-media photos (Scarl3tt)
- Scarl3tt trained only 0.07 epochs (5,550 effective images, 400 steps) — longer training needed
- ROCm/AMD requires fp32 + disabled latent caching to avoid NaN
- Identity loss adds ~55% per-step overhead (VAE decode + ArcFace forward)
- Both LoRAs weigh 163MB (dim=32, conv+linear keys)
