#!/bin/bash
# Run AuraFace SDXL LoRA training on Strix Halo (192.168.86.137)
# Shared NAS at /mnt/nas-ai-models/ must be mounted

set -e

REMOTE="192.168.86.137"
PROJECT="stratum-lora"
REMOTE_DIR="~/source/activity/$PROJECT"

echo "=== Syncing code to Strix Halo ==="
rsync -avz --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
    ~/source/activity/$PROJECT/ \
    $REMOTE:$REMOTE_DIR/

echo ""
echo "=== Installing deps on Strix Halo ==="
ssh $REMOTE "cd $REMOTE_DIR && python3 -m venv .venv && source .venv/bin/activate && pip install -q torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2 && pip install -q diffusers accelerate transformers safetensors einops toml voluptuous rich imagesize opencv-python huggingface_hub insightface onnxruntime"

echo ""
echo "=== Running training ==="
ssh $REMOTE "cd $REMOTE_DIR && source .venv/bin/activate && HF_HUB_CACHE=/home/tim/.cache/huggingface HF_HUB_DISABLE_SYMLINKS_WARNING=1 python3 sdxl_train_network.py --config_file test_dryrun.toml"

echo ""
echo "=== Done ==="
