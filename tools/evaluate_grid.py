#!/usr/bin/env python3
"""Grid search evaluation: generate images for all checkpoint LoRAs and compute
identity fidelity (cosine similarity vs target embedding).

Discovers all checkpoint files under the grid_search output directory,
generates N images per LoRA with fixed seeds, extracts face embeddings,
and outputs a results CSV + heatmap.

Usage (on Strix Halo):
    python tools/evaluate_grid.py \
        --model stabilityai/stable-diffusion-xl-base-1.0 \
        --grid_dir /mnt/nas-ai-models/loras/sdxl/grid_search \
        --target_emb /mnt/nas-ai-models/training-data/loras/scarl3tt/auraface_embeddings.npz \
        --prompt "a photo of scarl3tt person" \
        --num_images 20 \
        --output_dir /mnt/nas-ai-models/loras/sdxl/grid_search/eval
"""

import argparse
import csv
import glob
import os
import re
import sys
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from tqdm import tqdm
import cv2
import json


def discover_checkpoints(grid_dir):
    """Discover all checkpoint files organized by (method, dim, step)."""
    checkpoints = {}  # key: (method, dim, step) -> path
    pattern = os.path.join(grid_dir, "*/scarl3tt*.safetensors")
    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)
        # Parse: scarl3tt_{method}_dim{dim}[-step{step}].safetensors
        base_match = re.match(r"scarl3tt_(\w+)_dim(\d+)", fname)
        if not base_match:
            continue
        method = base_match.group(1)
        dim = int(base_match.group(2))
        step_match = re.search(r"step(\d+)", fname)
        step = int(step_match.group(1)) if step_match else 3000  # final
        key = (method, dim, step)
        if key not in checkpoints:
            checkpoints[key] = fpath
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Grid search A/B evaluation")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--grid_dir", required=True, help="Root of grid search output")
    parser.add_argument("--target_emb", required=True, help="Path to auraface_embeddings.npz")
    parser.add_argument("--prompt", default="a photo of scarl3tt person")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--output_dir", default="./eval_output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load target embedding
    data = np.load(args.target_emb)
    target_emb = data["mean_embedding"]
    while target_emb.ndim > 2:
        target_emb = target_emb.squeeze(0)
    target_emb = torch.from_numpy(target_emb).to(device, dtype=torch.float32)
    print(f"Target embedding: {target_emb.shape}")

    # Load AuraFace for evaluation
    print("Loading AuraFace ONNX...")
    from library.auraface_utils import AuraFaceWrapper
    auraface = AuraFaceWrapper(device=device)

    # Load SDXL once
    print("Loading SDXL...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)

    # Discover checkpoints
    checkpoints = discover_checkpoints(args.grid_dir)
    print(f"Discovered {len(checkpoints)} checkpoints: "
          f"methods={sorted(set(k[0] for k in checkpoints))}, "
          f"dims={sorted(set(k[1] for k in checkpoints))}, "
          f"steps={sorted(set(k[2] for k in checkpoints))}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Results
    results = []  # list of dicts
    generators = [torch.Generator(device=device).manual_seed(args.seed_start + i)
                  for i in range(args.num_images)]

    for (method, dim, step), lora_path in tqdm(sorted(checkpoints.items()),
                                                 desc="Evaluating"):
        label = f"{method}_dim{dim}_step{step}"
        print(f"\n── {label} ──")
        pipe.delete_adapters("default")
        pipe.load_lora_weights(lora_path, adapter_name="default")

        cosines = []
        for i, gen in enumerate(generators):
            img = pipe(args.prompt, num_inference_steps=args.steps,
                       guidance_scale=args.cfg, generator=gen).images[0]
            # Save a few samples
            if i < 2:
                img.save(os.path.join(args.output_dir, f"{label}_seed{args.seed_start+i:04d}.png"))

            emb = extract_face_embedding(img, auraface, device)
            if emb is not None:
                sim = torch.nn.functional.cosine_similarity(
                    emb, target_emb, dim=-1).item()
                cosines.append(sim)

        if cosines:
            avg = np.mean(cosines)
            std = np.std(cosines)
            print(f"  {len(cosines)}/{args.num_images} faces, "
                  f"avg_cos={avg:.4f} ±{std:.4f}")
            results.append({
                "method": method,
                "dim": dim,
                "step": step,
                "avg_cosine": round(avg, 6),
                "std_cosine": round(std, 6),
                "faces_detected": len(cosines),
                "num_images": args.num_images,
            })

    # Save CSV
    csv_path = os.path.join(args.output_dir, "grid_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "dim", "step", "avg_cosine", "std_cosine",
            "faces_detected", "num_images"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")
    print(f"Total: {len(results)} checkpoints evaluated")

    # Quick summary
    print("\n═══ BEST BY METHOD ═══")
    methods = sorted(set(r["method"] for r in results))
    for m in methods:
        subset = [r for r in results if r["method"] == m]
        best = max(subset, key=lambda x: x["avg_cosine"])
        print(f"  {m:12s} best: dim={best['dim']:3d} step={best['step']:5d} "
              f"cosine={best['avg_cosine']:.4f}")

    print("\nDone!")


def extract_face_embedding(img: Image.Image, auraface, device):
    """Extract face embedding from a generated image using center crop."""
    import numpy as np
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    # Center crop: assume face is in center half
    face = img_np[h//4:3*h//4, w//4:3*w//4]
    face = cv2.resize(face, (112, 112))
    face_tensor = torch.from_numpy(face.astype(np.float32) / 255.0)
    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = auraface(face_tensor)
    return emb


if __name__ == "__main__":
    main()
