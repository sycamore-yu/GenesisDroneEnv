#!/usr/bin/env bash
# Fully differentiable tracking: APG / SHAC / PPO, 100 updates, early stop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/tong/miniconda3/envs/genesis/bin/python}"
UPDATES="${UPDATES:-100}"
PATIENCE="${EARLY_STOP_PATIENCE:-1}"
SAVE="${SAVE_INTERVAL:-50}"

PYTHONUNBUFFERED=1 "$PYTHON" -u scripts/train/track_diff_train.py \
  --algo apg --fully-differentiable --horizon 32 --updates "$UPDATES" \
  --save-interval "$SAVE" --early-stop-patience "$PATIENCE"
PYTHONUNBUFFERED=1 "$PYTHON" -u scripts/train/track_diff_train.py \
  --algo shac --fully-differentiable --horizon 32 --updates "$UPDATES" \
  --save-interval "$SAVE" --early-stop-patience "$PATIENCE"
PYTHONUNBUFFERED=1 "$PYTHON" -u scripts/train/track_train.py \
  --fully-differentiable --max-iterations "$UPDATES" --num-envs 1024
echo "fully_diff tracking done"
