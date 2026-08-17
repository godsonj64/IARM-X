#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_100m.sh
#   NPROC=2 bash scripts/train_100m.sh
# Set LOCAL_DATA=1 to use parquet downloaded by scripts/download_datasets.py.

NPROC="${NPROC:-1}"
LOCAL_DATA="${LOCAL_DATA:-0}"

if [[ "$LOCAL_DATA" == "1" ]]; then
  PRETRAIN_CONFIG="configs/iarmx_100m_pretrain_local.yaml"
  SFT_CONFIG="configs/iarmx_100m_sft_local.yaml"
else
  PRETRAIN_CONFIG="configs/iarmx_100m_pretrain.yaml"
  SFT_CONFIG="configs/iarmx_100m_sft.yaml"
fi

run_train () {
  local cfg="$1"
  if [[ "$NPROC" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NPROC" -m iarmx.training.train --config "$cfg"
  else
    python -m iarmx.training.train --config "$cfg"
  fi
}

echo "[1/2] FineWeb-Edu sample-10BT pretraining"
run_train "$PRETRAIN_CONFIG"

echo "[2/2] UltraChat-200k supervised fine-tuning"
run_train "$SFT_CONFIG"
