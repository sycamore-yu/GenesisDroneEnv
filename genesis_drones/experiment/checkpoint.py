"""Checkpoint load/save with additive RunSpec metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from genesis_drones.experiment.spec import RunSpec


AXES = ("task", "dynamics", "algorithm", "network", "sensor", "seed")


def _read_sidecar(checkpoint_path: Path) -> dict[str, Any]:
    recorded: dict[str, Any] = {}
    for name in ("experiment.json", "contract.json"):
        candidate = checkpoint_path.parent / name
        if candidate.exists():
            recorded.update(json.loads(candidate.read_text()))
            break
    return recorded


def extract_run_spec(
    checkpoint_path: Path | None,
    payload: dict[str, Any] | None,
    *,
    overrides: dict[str, Any] | None = None,
) -> RunSpec:
    """Resolve RunSpec from checkpoint / sidecars / CLI overrides. Never invent axes."""
    recorded: dict[str, Any] = {}
    if checkpoint_path is not None:
        recorded.update(_read_sidecar(checkpoint_path))
    if payload:
        if "run_spec" in payload and isinstance(payload["run_spec"], dict):
            recorded.update(payload["run_spec"])
        for key in AXES:
            if key in payload:
                recorded[key] = payload[key]
        if "config" in payload and isinstance(payload["config"], dict):
            cfg = payload["config"]
            env = cfg.get("environment")
            if isinstance(env, dict):
                recorded.setdefault("environment", env)
                recorded.setdefault("dynamics", env.get("dynamics"))
            for algo in ("ppo", "apg", "shac"):
                if algo in cfg and isinstance(cfg[algo], dict):
                    recorded.setdefault("algorithm_config", cfg[algo])
            if "network" in cfg and isinstance(cfg["network"], dict):
                recorded.setdefault("network", "mlp")
    overrides = overrides or {}
    for key, value in overrides.items():
        if value is None:
            continue
        if key in recorded and recorded[key] is not None and str(recorded[key]) != str(value):
            raise ValueError(f"checkpoint {key} {recorded[key]!r} does not match override {value!r}")
        recorded[key] = value

    missing = [key for key in ("task", "dynamics", "algorithm") if not recorded.get(key)]
    if missing:
        raise ValueError(
            "cannot resolve RunSpec; missing "
            + ", ".join(missing)
            + ". Provide CLI overrides or a checkpoint with experiment metadata."
        )
    return RunSpec(
        task=str(recorded["task"]),
        dynamics=str(recorded["dynamics"]),
        algorithm=str(recorded["algorithm"]),
        network=str(recorded.get("network", "mlp")),
        sensor=str(recorded.get("sensor", "state")),
        seed=recorded.get("seed"),
        environment=dict(recorded.get("environment") or {}),
        algorithm_config=dict(recorded.get("algorithm_config") or {}),
        schema_version=int(recorded.get("schema_version", 1)),
        git_commit=recorded.get("git_commit"),
        created_at=recorded.get("created_at"),
    )


def attach_run_spec(payload: dict[str, Any], run_spec: RunSpec) -> dict[str, Any]:
    """Additive: keep legacy keys, add run_spec block and axis mirrors."""
    out = dict(payload)
    spec_dict = run_spec.to_dict()
    out["run_spec"] = spec_dict
    for key in AXES:
        out[key] = getattr(run_spec, key)
    out["schema_version"] = run_spec.schema_version
    return out


def save_checkpoint(path: Path, payload: dict[str, Any], run_spec: RunSpec | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = attach_run_spec(payload, run_spec) if run_spec is not None else payload
    torch.save(body, path)
    if run_spec is not None:
        (path.parent / "experiment.json").write_text(json.dumps(run_spec.to_dict(), indent=2))


def load_checkpoint(path: Path, map_location=None) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
