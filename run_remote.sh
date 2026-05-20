#!/bin/bash
# Run AuraFace SDXL LoRA training on Strix Halo (192.168.86.137)
# Shared NAS at /mnt/nas-ai-models/ must be mounted
# Path on Strix Halo: ~/activity/stratum-lora (no 'source' prefix)

set -e

REMOTE="192.168.86.137"
PROJECT="stratum-lora"
REMOTE_DIR="~/activity/$PROJECT"

echo "=== Syncing code to Strix Halo ==="
rsync -avz --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
    ~/source/activity/$PROJECT/ \
    $REMOTE:$REMOTE_DIR/

echo ""
echo "=== Running training on Strix Halo ==="
ssh $REMOTE "cd $REMOTE_DIR && source .venv/bin/activate && HF_HUB_DISABLE_SYMLINKS_WARNING=1 python3 sdxl_train_network.py --config_file test_dryrun.toml"

echo ""
echo "=== Done ==="
