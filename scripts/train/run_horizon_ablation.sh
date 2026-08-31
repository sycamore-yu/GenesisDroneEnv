#!/usr/bin/env bash
# Horizon ablation for Track DiffRL. Keeps dt=0.01, reward, network, lr unchanged.
# Runs: APG H32/64/96, SHAC-old H32, SHAC-terminal H32/64/96.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/tong/miniconda3/envs/genesis/bin/python}"
UPDATES="${UPDATES:-400}"
NUM_ENVS="${NUM_ENVS:-1024}"
VAL="${VAL:-256}"
# ponytail: sequential patience, not ASHA. ASHA if we ever run many trials in parallel.
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"

run() {
  local algo="$1"
  local horizon="$2"
  shift 2
  echo "=== ${algo} H=${horizon} $* ==="
  PYTHONUNBUFFERED=1 "$PYTHON" -u scripts/train/track_diff_train.py \
    --algo "$algo" \
    --horizon "$horizon" \
    --updates "$UPDATES" \
    --num-envs "$NUM_ENVS" \
    --validation-scenarios "$VAL" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    "$@"
}

run apg 32
run apg 64
run apg 96
run shac 32 --no-terminal-value
run shac 32
run shac 64
run shac 96

echo "Done. Inspect logs/track_diff/*_H*/training_summary.json"
