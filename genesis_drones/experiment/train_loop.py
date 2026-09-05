"""Shared training loops. Scripts call these; no duplicated update logic in scripts."""

from __future__ import annotations

from pathlib import Path

import yaml
from torch.utils.tensorboard import SummaryWriter

from genesis_drones.experiment.checkpoint import save_checkpoint
from genesis_drones.experiment.spec import RunSpec


def train_ppo_stack(stack: dict, run_spec: RunSpec, updates: int, log_dir: Path) -> None:
    runner = stack["runner"]
    runner.learn(num_learning_iterations=updates, init_at_random_ep_len=True)
    train_config = stack.get("train_config") or {}
    if train_config:
        (log_dir / "train_config.yaml").write_text(yaml.safe_dump(train_config, sort_keys=False))
    network = {
        "hidden_sizes": list(
            train_config.get("actor", {}).get(
                "hidden_dims",
                run_spec.environment.get("hidden_sizes", [256, 128]),
            )
        )
    }
    payload = runner.alg.save()
    payload["iter"] = runner.current_learning_iteration
    payload["config"] = {"network": network, "environment": run_spec.environment}
    save_checkpoint(log_dir / "model.pt", payload, run_spec)


def train_diff_stack(stack: dict, run_spec: RunSpec, cfg: dict, updates: int, save_interval: int, log_dir: Path) -> None:
    adapter = stack["adapter"]
    agent = stack["agent"]
    normalizer = stack["normalizer"]
    observation = adapter.reset_diff(seed=run_spec.seed or 0)
    writer = SummaryWriter(log_dir)
    for update in range(1, updates + 1):
        observation, stats = agent.update(adapter, observation, normalizer)
        writer.add_scalar("loss/actor", stats.actor_loss, update)
        writer.add_scalar("reward/mean", stats.mean_reward, update)
        if update % max(save_interval, 1) == 0:
            save_checkpoint(
                log_dir / f"model_{update}.pt",
                {
                    "legacy": False,
                    "agent": agent.state_dict(),
                    "policy_normalizer": normalizer.state_dict(),
                    "config": cfg,
                    "update": update,
                },
                run_spec,
            )
    writer.close()
    save_checkpoint(
        log_dir / "model.pt",
        {
            "legacy": False,
            "agent": agent.state_dict(),
            "policy_normalizer": normalizer.state_dict(),
            "config": cfg,
            "update": updates,
        },
        run_spec,
    )
