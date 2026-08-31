#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/race
STATES=config/race/eval_states.pt

train_eval() {
  local algo=$1 seed=$2 extra=${3:-}
  local dir="logs/race/${algo}_s${seed}${extra:+_h$extra}"
  if [[ -n "$extra" ]]; then
    python scripts/train/race_train.py --algo "$algo" --seed "$seed" --horizon "$extra" --log-dir "$dir"
  else
    python scripts/train/race_train.py --algo "$algo" --seed "$seed" --log-dir "$dir"
  fi
  local ckpt
  ckpt="$(ls -1t "$dir"/model_*.pt 2>/dev/null | head -1 || true)"
  if [[ -z "$ckpt" ]]; then
    echo "missing checkpoint in $dir" >&2
    exit 1
  fi
  python scripts/eval/race_eval.py --algo "$algo" --checkpoint "$ckpt" --states "$STATES" --output "$dir/eval.json"
}

for seed in 1 2 3; do
  train_eval ppo "$seed"
done
for horizon in 32 64 96; do
  for seed in 1 2 3; do
    train_eval apg "$seed" "$horizon"
  done
done
for seed in 1 2 3; do
  train_eval shac "$seed"
done
python scripts/eval/race_accept.py --output logs/race/acceptance.json
echo "phase1 matrix finished"
