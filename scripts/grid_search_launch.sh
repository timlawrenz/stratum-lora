#!/usr/bin/env bash
# Grid search launch script — runs on Strix Halo
# 9 configs: 3 methods (control, auraface, dinov3) × 3 dims (32, 64, 128)
# Each: 3000 steps, checkpoints at 1k/2k/3k
set -euo pipefail

cd ~/activity/stratum-lora
source .venv/bin/activate

CONFIGS=(
  "control_dim32" "control_dim64" "control_dim128"
  "auraface_dim32" "auraface_dim64" "auraface_dim128"
  "dinov3_dim32" "dinov3_dim64" "dinov3_dim128"
)

export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export MIOPEN_USER_DB_PATH=~/.config/miopen
export MIOPEN_FIND_MODE=7
export PYTORCH_TUNABLEOP_TUNING=1
export PYTORCH_TUNABLEOP_TUNING_ENABLED=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export PYTHONUNBUFFERED=1

ulimit -n 65535

TOTAL=${#CONFIGS[@]}
CURRENT=0

for config in "${CONFIGS[@]}"; do
  CURRENT=$((CURRENT + 1))
  echo "══════════════════════════════════════════════════════"
  echo "[${CURRENT}/${TOTAL}] Training ${config}"
  echo "══════════════════════════════════════════════════════"
  python sdxl_train_network.py --config_file "configs/grid_search/${config}.toml" 2>&1
  echo "✅ ${config} done"
done

echo "══════════════════════════════════════════════════════"
echo "All ${TOTAL} runs complete!"
echo "Checkpoints in /mnt/nas-ai-models/loras/sdxl/grid_search/"
echo "══════════════════════════════════════════════════════"
