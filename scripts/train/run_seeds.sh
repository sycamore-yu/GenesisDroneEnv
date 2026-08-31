#!/usr/bin/env bash
# Phase 8: 3 seeds of one config. Set EXTRA flags, e.g. EXTRA='--horizon 96 --closing-velocity-weight 1.0'
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/tong/miniconda3/envs/genesis/bin/python}"
ALGO="${ALGO:-apg}"
UPDATES="${UPDATES:-200}"
PATIENCE="${EARLY_STOP_PATIENCE:-1}"
EXTRA="${EXTRA:-}"

for seed in 1 2 3; do
  echo "=== seed ${seed} ==="
  PYTHONUNBUFFERED=1 "$PYTHON" -u scripts/train/track_diff_train.py \
    --algo "$ALGO" \
    --updates "$UPDATES" \
    --early-stop-patience "$PATIENCE" \
    --seed "$seed" \
    $EXTRA
done
