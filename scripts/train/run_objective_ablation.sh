#!/usr/bin/env bash
# Phase 5: one-factor objective ablation. APG H=32, 200 updates, stop after 1 worse eval.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/tong/miniconda3/envs/genesis/bin/python}"
UPDATES="${UPDATES:-200}"
PATIENCE="${EARLY_STOP_PATIENCE:-1}"

run() {
  echo "=== $* ==="
  PYTHONUNBUFFERED=1 "$PYTHON" -u scripts/train/track_diff_train.py \
    --algo apg \
    --horizon 32 \
    --updates "$UPDATES" \
    --early-stop-patience "$PATIENCE" \
    "$@"
}

run --progress-norm l2
run --closing-velocity-weight 1.0
run --arrival-surrogate gaussian

echo "Done. Compare logs/track_diff/apg_H32_* vs apg_H32_2026-08-30_14-55-38"
