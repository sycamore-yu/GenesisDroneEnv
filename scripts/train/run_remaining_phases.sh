#!/usr/bin/env bash
# Phase 5 ablation, then 6–8: plot, promote winner to H=96, 3 seeds, fair eval.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/tong/miniconda3/envs/genesis/bin/python}"
LOG="$ROOT/logs/track_diff/remaining_phases.log"

{
  echo "=== phase 5 objective ablation ==="
  bash scripts/train/run_objective_ablation.sh

  echo "=== pick winner ==="
  "$PYTHON" scripts/eval/pick_objective_winner.py
  SKIP=$("$PYTHON" -c "import json; print(int(json.load(open('logs/track_diff/objective_winner.json'))['skip_h96_retrain']))")
  PROMOTE=$("$PYTHON" -c "import json; print(json.load(open('logs/track_diff/objective_winner.json'))['promote_extra'])")
  SEED_EXTRA=$("$PYTHON" -c "import json; print(json.load(open('logs/track_diff/objective_winner.json'))['seed_extra'])")
  APG=$("$PYTHON" -c "import json; print(json.load(open('logs/track_diff/objective_winner.json'))['apg_eval_if_skip'])")
  SHAC=$("$PYTHON" -c "import json; print(json.load(open('logs/track_diff/objective_winner.json'))['shac_eval'])")

  if [ "$SKIP" = "0" ]; then
    echo "=== promote to H96: ${PROMOTE} ==="
    PYTHONUNBUFFERED=1 "$PYTHON" -u scripts/train/track_diff_train.py \
      --algo apg --updates "${UPDATES:-200}" --early-stop-patience "${EARLY_STOP_PATIENCE:-1}" \
      $PROMOTE
    APG=$("$PYTHON" -c "
from pathlib import Path
root = Path('logs/track_diff')
dirs = sorted(
    [p for p in root.glob('apg_H96_*') if (p / 'best.pt').exists()],
    key=lambda p: p.stat().st_mtime,
)
print(dirs[-1] / 'best.pt')
")
  else
    echo "=== skip H96 retrain; using existing APG H96 ==="
  fi

  echo "=== phase 8 seeds: ${SEED_EXTRA} ==="
  EXTRA="$SEED_EXTRA" ALGO=apg UPDATES="${UPDATES:-200}" EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-1}" \
    bash scripts/train/run_seeds.sh

  echo "=== phase 6-7 plots ==="
  "$PYTHON" scripts/eval/plot_track_diff_pareto.py

  echo "=== fair eval native+common ==="
  mkdir -p logs/track_diff/evaluation_phase8
  PYTHONUNBUFFERED=1 "$PYTHON" -u scripts/eval/track_diff_eval.py \
    --apg-checkpoint "$APG" \
    --shac-checkpoint "$SHAC" \
    --num-scenarios "${NUM_SCENARIOS:-1024}" \
    --output-dir logs/track_diff/evaluation_phase8

  echo "=== phase 6-7 plots after eval ==="
  "$PYTHON" scripts/eval/plot_track_diff_pareto.py \
    --ppo-eval logs/track_diff/evaluation_phase8/evaluation.json
  echo "ALL DONE"
} 2>&1 | tee -a "$LOG"
