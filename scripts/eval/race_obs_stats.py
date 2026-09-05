"""Measure racing policy observation scale. Not a training run."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

import genesis as gs

from genesis_drones.experiment.builder import PROJECT_ROOT, build_environment, run_spec_from_cfg


def main() -> None:
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, seed=0, logging_level="warning")
    with (PROJECT_ROOT / "config/race/train.yaml").open() as file:
        data = yaml.safe_load(file)
    data["task"] = "racing"
    data["dynamics"] = "full_quad"
    data["algorithm"] = "ppo"
    data["environment"]["dynamics"] = "full_quad"
    data["seed"] = 0
    environment = build_environment(run_spec_from_cfg(data), 256, requires_grad=False)
    names = [
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "roll",
        "pitch",
        "yaw",
        "next_pos_x",
        "next_pos_y",
        "next_pos_z",
        "next_yaw",
    ]
    chunks = []
    observation, _ = environment.reset(seed=0)
    chunks.append(observation.detach())
    for _ in range(48):
        action = torch.zeros(environment.num_envs, environment.action_dim, device=environment.device)
        action[:, 2] = 0.3
        step = environment.step(action)
        chunks.append(step.observation.policy.detach())
    stacked = torch.cat(chunks, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    abs_mean = stacked.abs().mean(dim=0)
    report = {
        "n_rows": int(stacked.shape[0]),
        "dim": int(stacked.shape[1]),
        "per_dim": [
            {
                "name": names[i],
                "mean": float(mean[i]),
                "std": float(std[i]),
                "abs_mean": float(abs_mean[i]),
                "min": float(stacked[:, i].min()),
                "max": float(stacked[:, i].max()),
                "frac_abs_gt_10": float((stacked[:, i].abs() > 10).float().mean()),
                "frac_abs_gt_100": float((stacked[:, i].abs() > 100).float().mean()),
            }
            for i in range(stacked.shape[1])
        ],
        "max_abs": float(stacked.abs().max()),
        "frac_abs_gt_10": float((stacked.abs() > 10).float().mean()),
        "frac_abs_gt_100": float((stacked.abs() > 100).float().mean()),
    }
    dest = PROJECT_ROOT / "logs/race/rslrl55_migration/obs_stats.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps({"wrote": str(dest), "max_abs": report["max_abs"], "frac_abs_gt_10": report["frac_abs_gt_10"]}))


if __name__ == "__main__":
    main()
