#!/usr/bin/env python3
"""Unified training entry: config → ExperimentBuilder → train loop."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import genesis as gs

from genesis_drones.experiment.builder import build_training_stack, run_spec_from_cfg
from genesis_drones.experiment.train_loop import train_diff_stack, train_ppo_stack


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    data = _to_dict(cfg)
    run_spec = run_spec_from_cfg(data)
    gs.init(
        backend=gs.gpu if torch.cuda.is_available() else gs.cpu,
        seed=run_spec.seed or 0,
        logging_level="warning",
    )
    algorithm = run_spec.algorithm
    num_envs = int(data.get("num_envs", {}).get(algorithm, data.get("num_envs", 64)))
    if isinstance(data.get("num_envs"), int):
        num_envs = int(data["num_envs"])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = data.get("log_root", "logs/experiments")
    log_dir = PROJECT_ROOT / log_root / f"{run_spec.task}_{run_spec.dynamics}_{algorithm}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))

    stack = build_training_stack(run_spec, data, num_envs, log_dir=log_dir)
    updates = int(data.get("updates", 100))
    save_interval = int(data.get("save_interval", 50))
    if stack["kind"] == "ppo":
        train_ppo_stack(stack, run_spec, updates, log_dir)
    else:
        train_diff_stack(stack, run_spec, data, updates, save_interval, log_dir)
    print(f"done log_dir={log_dir}")


if __name__ == "__main__":
    main()
