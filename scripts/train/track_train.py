"""Legacy tracking PPO train → shared builder."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml

import genesis as gs

from genesis_drones.experiment.builder import build_training_stack, run_spec_from_cfg
from genesis_drones.experiment.train_loop import train_ppo_stack


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fully-differentiable", action="store_true")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--dynamics", choices=("native_quad", "full_quad"), default="native_quad")
    return parser.parse_args()


def main():
    args = parse_args()
    with (PROJECT_ROOT / "config/track_rl/genesis_env.yaml").open() as file:
        genesis_yaml = yaml.safe_load(file)
    data = {
        "task": "tracking",
        "dynamics": args.dynamics,
        "algorithm": "ppo",
        "sensor": "state",
        "seed": 0,
        "updates": args.max_iterations or 100,
        "save_interval": 50,
        "log_root": "logs/track_rl",
        "network": {"hidden_sizes": [128, 128, 128]},
        "environment": {
            "dt": genesis_yaml["dt"],
            "horizon": 1,
            "fully_differentiable": bool(args.fully_differentiable),
        },
        "num_envs": {"ppo": args.num_envs or genesis_yaml.get("num_envs", 64)},
    }
    run_spec = run_spec_from_cfg(data)
    gs.init(logging_level="warning")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_dir = Path(args.log_dir) if args.log_dir else PROJECT_ROOT / f"logs/track_rl/track_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    stack = build_training_stack(run_spec, data, int(data["num_envs"]["ppo"]), log_dir=log_dir)
    train_ppo_stack(stack, run_spec, int(data["updates"]), log_dir)


if __name__ == "__main__":
    main()
