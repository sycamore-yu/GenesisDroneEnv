#!/usr/bin/env python3
"""Unified evaluation entry: checkpoint → RunSpec → policy.act → EvaluationRunner / task metrics."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import genesis as gs

from genesis_drones.adapters.policy import CallablePolicyAdapter
from genesis_drones.experiment.builder import build_environment, load_policy, run_spec_from_cfg
from genesis_drones.experiment.checkpoint import extract_run_spec, load_checkpoint
from genesis_drones.evaluation.race import (
    evaluate_policy,
    evaluate_rolling,
    make_evaluation_states,
    save_summary,
    summarize_race_results,
)
from genesis_drones.evaluation.runner import EvaluationRunner
from genesis_drones.evaluation.tracking import TrackScenarios, evaluate_diff_policy, summarize_metrics
from genesis_drones.tasks.racing_core import load_evaluation_initial_states


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(cfg: DictConfig) -> None:
    data = _to_dict(cfg)
    checkpoint = data.get("checkpoint")
    if not checkpoint:
        raise SystemExit("checkpoint=... is required")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()

    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    payload = load_checkpoint(checkpoint_path, map_location=gs.device)
    # Prefer checkpoint metadata. Only apply CLI axes when checkpoint lacks them,
    # or when user forces a value that must match (extract_run_spec checks mismatches).
    cli_axes = {}
    for key in ("task", "dynamics", "algorithm", "sensor"):
        value = data.get(key)
        if isinstance(value, dict):
            value = value.get("name")
        if value is None:
            continue
        # Skip hydra default fill when checkpoint already records the axis.
        if key in payload or (isinstance(payload.get("run_spec"), dict) and key in payload["run_spec"]):
            # Still validate if user set an explicit non-default via command line is hard;
            # only inject when checkpoint is missing the axis.
            continue
        recorded_sidecar = False
        for name in ("experiment.json", "contract.json"):
            if (checkpoint_path.parent / name).exists():
                recorded_sidecar = True
                break
        if recorded_sidecar:
            continue
        cli_axes[key] = value
    run_spec = extract_run_spec(checkpoint_path, payload, overrides=cli_axes or None)
    if not run_spec.environment:
        base = run_spec_from_cfg(data)
        run_spec.environment = base.environment
        if not run_spec.algorithm_config:
            run_spec.algorithm_config = base.algorithm_config

    eval_cfg = data.get("eval") or {}
    num_envs = int(eval_cfg.get("num_envs", 1))
    environment = build_environment(run_spec, num_envs, requires_grad=False)
    policy, run_spec, payload = load_policy(
        checkpoint_path,
        environment,
        overrides={"task": run_spec.task, "dynamics": run_spec.dynamics, "algorithm": run_spec.algorithm},
        cfg=payload.get("config") or data,
    )

    action_fn = lambda obs: policy.act(obs, deterministic=True)
    steps = eval_cfg.get("steps")
    if run_spec.task == "racing":
        if steps is not None:
            # Rolling uses RaceEnv-style step tuple API via facade if present.
            from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
            from genesis_drones.experiment.builder import racing_env_config_from

            facade = RaceEnv(
                racing_env_config_from({"environment": run_spec.environment}, run_spec.dynamics),
                num_envs=num_envs,
                requires_grad=False,
            )
            policy2, _, _ = load_policy(checkpoint_path, facade.env, cfg=payload.get("config") or data)
            metrics = evaluate_rolling(facade, lambda o: policy2.act(o, deterministic=True), int(steps))
            checksum = None
        else:
            states_path = Path(eval_cfg.get("states") or "config/race/eval_states.pt")
            if not states_path.is_absolute():
                states_path = PROJECT_ROOT / states_path
            checksum = None
            if states_path.exists():
                try:
                    states, checksum = load_evaluation_initial_states(states_path)
                except Exception:
                    states = make_evaluation_states(environment, count=min(8, 100), seed=int(data.get("seed") or 0))
                    checksum = None
            else:
                states = make_evaluation_states(environment, count=min(8, 100), seed=int(data.get("seed") or 0))
            metrics = summarize_race_results(evaluate_policy(environment, action_fn, states))
        summary = {
            **metrics,
            "checksum": checksum,
            **run_spec.to_dict(),
            "protocol": "rolling" if steps is not None else "fixed_states",
        }
    else:
        from genesis_drones.envs.track_diff_env import TrackDiffEnv
        from genesis_drones.experiment.builder import tracking_env_config_from

        track_cfg = tracking_env_config_from(data, run_spec.dynamics)
        facade = TrackDiffEnv(track_cfg, num_envs=num_envs, requires_grad=False)
        policy_t, _, _ = load_policy(checkpoint_path, facade.env, cfg=payload.get("config") or data)
        scenarios = TrackScenarios.generate(
            num_envs,
            track_cfg.max_episode_steps,
            track_cfg,
            seed=int(data.get("seed") or 0),
        )
        raw = evaluate_diff_policy(facade, policy_t, scenarios)
        summary = {
            **summarize_metrics(raw, track_cfg.max_episode_steps * track_cfg.dt).to_dict(),
            **run_spec.to_dict(),
            "protocol": "scenarios",
        }
        # Also exercise generic runner once for smoke parity.
        runner = EvaluationRunner(facade.env, policy_t, max_steps=2)
        runner.run(reset_kwargs={"seed": None})

    output = eval_cfg.get("output")
    output_path = Path(output) if output else PROJECT_ROOT / "logs" / "eval" / f"{run_spec.task}_{run_spec.algorithm}_eval.json"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    save_summary(output_path, summary)
    print(json.dumps({k: summary[k] for k in summary if k != "environment"}, indent=2, default=str))


if __name__ == "__main__":
    main()
