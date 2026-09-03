"""Legacy tracking DiffRL train → shared builder / train_loop."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

import genesis as gs

from genesis_drones.algorithms.diff_rl import diff_algorithm_names
from genesis_drones.experiment.builder import build_training_stack, run_spec_from_cfg
from genesis_drones.experiment.train_loop import train_diff_stack
from genesis_drones.utils.track_diff_config import load_track_diff_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=diff_algorithm_names(), required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "track_diff" / "train.yaml")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--fully-differentiable", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_track_diff_settings(args.config)
    data = dict(settings.raw)
    if args.horizon is not None:
        data["environment"] = {**data["environment"], "horizon": args.horizon}
        data[args.algo] = {**data[args.algo], "horizon": args.horizon}
    if args.fully_differentiable:
        data["environment"] = {**data["environment"], "fully_differentiable": True}
    if args.seed is not None:
        data["seed"] = args.seed
    if args.updates is not None:
        data["updates"] = args.updates
    if args.save_interval is not None:
        data["save_interval"] = args.save_interval
    data["task"] = "tracking"
    data["dynamics"] = "native_quad"
    data["algorithm"] = args.algo
    data["sensor"] = "state"

    run_spec = run_spec_from_cfg(data)
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, seed=run_spec.seed or 0, logging_level="warning")
    num_envs = args.num_envs or int(data["num_envs"][args.algo])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = args.log_dir or PROJECT_ROOT / data["log_root"] / f"{args.algo}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    stack = build_training_stack(run_spec, data, num_envs, log_dir=log_dir)
    train_diff_stack(stack, run_spec, data, int(data.get("updates", 400)), int(data.get("save_interval", 100)), log_dir)


if __name__ == "__main__":
    main()
