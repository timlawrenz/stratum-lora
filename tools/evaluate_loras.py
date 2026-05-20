"""A/B evaluation: compare AuraFace-enhanced LoRA vs baseline.

Generates images from both LoRAs with identical seeds, extracts face
embeddings using AuraFace, and reports which produces higher identity fidelity.

Usage:
  python tools/evaluate_loras.py \
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --lora_a /path/to/auraface_lora.safetensors \
    --lora_b /path/to/control_lora.safetensors \
    --target_emb /mnt/nas-ai-models/training-data/loras/hegre-moloko/auraface_embeddings.npz \
    --prompt "a photo of a person" \
    --num_images 10 \
    --output_dir ./eval_output
"""

import argparse
import os
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from tqdm import tqdm
import cv2
import json


def main():
    parser = argparse.ArgumentParser(description="A/B test AuraFace vs baseline LoRA")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora_a", required=True, help="Path to AuraFace-enhanced LoRA")
    parser.add_argument("--lora_b", required=True, help="Path to control LoRA (no identity loss)")
    parser.add_argument("--target_emb", required=True, help="Path to auraface_embeddings.npz")
    parser.add_argument("--prompt", default="a photo of a person")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--output_dir", default="./eval_output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bbox_file", default=None,
                        help="Path to face_bboxes.json for face cropping")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Load target embedding
    data = np.load(args.target_emb)
    target_emb = data['mean_embedding']
    while target_emb.ndim > 2:
        target_emb = target_emb.squeeze(0)
    target_emb = torch.from_numpy(target_emb).to(device, dtype=torch.float32)
    print(f"Target embedding: {target_emb.shape}")

    # Load AuraFace
    print("Loading AuraFace ONNX for evaluation...")
    from library.auraface_utils import AuraFaceWrapper, load_auraface_metadata
    auraface = AuraFaceWrapper(device=device)  # ONNX mode, non-diff is fine for eval

    # Load bboxes if available
    bboxes = None
    if args.bbox_file and os.path.exists(args.bbox_file):
        with open(args.bbox_file) as f:
            bboxes = json.load(f)
        print(f"Loaded {len(bboxes)} bounding boxes")

    # Load base model
    print("Loading SDXL...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    results_a = []
    results_b = []

    for i in tqdm(range(args.num_images), desc="Generating"):
        seed = args.seed_start + i

        # Generate with LoRA A (AuraFace)
        pipe.load_lora_weights(args.lora_a, adapter_name="auraface")
        pipe.set_adapters(["auraface"])
        generator = torch.Generator(device=device).manual_seed(seed)
        img_a = pipe(args.prompt, num_inference_steps=args.steps,
                     guidance_scale=args.cfg, generator=generator).images[0]
        img_a.save(os.path.join(args.output_dir, f"auraface_{seed:04d}.png"))

        # Generate with LoRA B (control)
        pipe.load_lora_weights(args.lora_b, adapter_name="control")
        pipe.set_adapters(["control"])
        generator = torch.Generator(device=device).manual_seed(seed)
        img_b = pipe(args.prompt, num_inference_steps=args.steps,
                     guidance_scale=args.cfg, generator=generator).images[0]
        img_b.save(os.path.join(args.output_dir, f"control_{seed:04d}.png"))

        # Extract face embeddings
        emb_a = extract_face_embedding(img_a, auraface, bboxes, device)
        emb_b = extract_face_embedding(img_b, auraface, bboxes, device)

        if emb_a is not None and emb_b is not None:
            sim_a = torch.nn.functional.cosine_similarity(emb_a, target_emb, dim=-1).item()
            sim_b = torch.nn.functional.cosine_similarity(emb_b, target_emb, dim=-1).item()
            results_a.append(sim_a)
            results_b.append(sim_b)
            winner = "AuraFace" if sim_a > sim_b else "Control"
            print(f"  seed={seed}: AuraFace={sim_a:.4f} Control={sim_b:.4f} → {winner}")

    # Report
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    if results_a:
        avg_a = np.mean(results_a)
        avg_b = np.mean(results_b)
        wins_a = sum(1 for a, b in zip(results_a, results_b) if a > b)
        wins_b = len(results_a) - wins_a
        print(f"AuraFace LoRA:  avg cosine={avg_a:.4f} (±{np.std(results_a):.4f})")
        print(f"Control LoRA:   avg cosine={avg_b:.4f} (±{np.std(results_b):.4f})")
        print(f"AuraFace wins:  {wins_a}/{len(results_a)} ({wins_a/len(results_a)*100:.0f}%)")
        print(f"Control wins:   {wins_b}/{len(results_a)} ({wins_b/len(results_a)*100:.0f}%)")
        print(f"Improvement:    {(avg_a - avg_b) / max(avg_b, 1e-8) * 100:+.1f}%")

    print(f"\nImages saved to: {args.output_dir}")


def extract_face_embedding(img: Image.Image, auraface, bboxes, device):
    """Extract face embedding from a generated image."""
    img_np = np.array(img)  # RGB, uint8
    h, w = img_np.shape[:2]

    # If bboxes available, use them (assume first bbox if no filename match)
    if bboxes:
        bbox = list(bboxes.values())[0]
        y1, y2, x1, x2 = bbox
        # Scale bbox to generated image size (assumes original was ~2048px)
        scale_y = h / 2048
        scale_x = w / 2048
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        face = img_np[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    else:
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
