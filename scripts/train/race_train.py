"""Legacy racing train entry → shared experiment builder / train_loop."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml

import genesis as gs

from genesis_drones.algorithms.diff_rl import diff_algorithm_names
from genesis_drones.dynamics import QUAD_DYNAMICS
from genesis_drones.experiment.builder import build_training_stack, run_spec_from_cfg
from genesis_drones.experiment.train_loop import train_diff_stack, train_ppo_stack


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("ppo", *diff_algorithm_names()), required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "race" / "train.yaml")
    parser.add_argument("--dynamics", choices=QUAD_DYNAMICS)
    parser.add_argument("--horizon", type=int, choices=(32, 64, 96))
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--ppo-config", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as file:
        data = yaml.safe_load(file)
    if args.seed is not None:
        data["seed"] = args.seed
    if args.dynamics is not None:
        data["environment"]["dynamics"] = args.dynamics
    if args.horizon is not None:
        data["environment"]["horizon"] = args.horizon
        if args.algo in data:
            data[args.algo]["horizon"] = args.horizon
    data["task"] = "racing"
    data["dynamics"] = data["environment"]["dynamics"]
    data["algorithm"] = args.algo
    data["sensor"] = "relative_position"
    if args.updates is not None:
        data["updates"] = args.updates
    if args.save_interval is not None:
        data["save_interval"] = args.save_interval
    if args.ppo_config is not None:
        data["_ppo_config"] = str(args.ppo_config)
    if args.learning_rate is not None:
        data["_learning_rate"] = args.learning_rate

    run_spec = run_spec_from_cfg(data)
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, seed=run_spec.seed or 0, logging_level="warning")
    num_envs = args.num_envs or int(data["num_envs"][args.algo])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = args.log_dir or (
        PROJECT_ROOT / data["log_root"] / f"{args.algo}_{run_spec.dynamics}_H{data['environment']['horizon']}_{timestamp}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    stack = build_training_stack(run_spec, data, num_envs, log_dir=log_dir)
    if args.learning_rate is not None and stack["kind"] == "ppo":
        stack["train_config"]["algorithm"]["learning_rate"] = args.learning_rate
        stack["runner"].alg.learning_rate = args.learning_rate
    if args.resume is not None and stack["kind"] == "ppo":
        stack["runner"].load(str(args.resume))
    updates = int(data.get("updates", 1000))
    if stack["kind"] == "ppo":
        train_ppo_stack(stack, run_spec, updates, log_dir)
    else:
        train_diff_stack(stack, run_spec, data, updates, int(data.get("save_interval", 100)), log_dir)


if __name__ == "__main__":
    main()
