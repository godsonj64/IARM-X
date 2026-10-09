#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_100m.sh
#   NPROC=2 bash scripts/train_100m.sh
# Set LOCAL_DATA=1 to use parquet downloaded by scripts/download_datasets.py.
# Set MEMMAP=1 to use token shards written by scripts/pretokenize.py (exact resume).

NPROC="${NPROC:-1}"
LOCAL_DATA="${LOCAL_DATA:-0}"
MEMMAP="${MEMMAP:-0}"

if [[ "$LOCAL_DATA" == "1" ]]; then
  PRETRAIN_CONFIG="configs/iarmx_100m_pretrain_local.yaml"
  SFT_CONFIG="configs/iarmx_100m_sft_local.yaml"
else
  PRETRAIN_CONFIG="configs/iarmx_100m_pretrain.yaml"
  SFT_CONFIG="configs/iarmx_100m_sft.yaml"
fi
if [[ "$MEMMAP" == "1" ]]; then
  PRETRAIN_CONFIG="configs/iarmx_100m_pretrain_memmap.yaml"
fi

# --resume auto continues from the newest checkpoint in the stage's output_dir
# (exact data position), so rerunning this script after a preemption picks up
# where it stopped; a finished stage exits immediately. Delete output_dir to
# start a stage over.
run_train () {
  local cfg="$1"
  if [[ "$NPROC" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NPROC" -m iarmx.training.train --config "$cfg" --resume auto
  else
    python -m iarmx.training.train --config "$cfg" --resume auto
  fi
}

echo "[1/2] FineWeb-Edu sample-10BT pretraining"
run_train "$PRETRAIN_CONFIG"

echo "[2/2] UltraChat-200k supervised fine-tuning"
run_train "$SFT_CONFIG"
