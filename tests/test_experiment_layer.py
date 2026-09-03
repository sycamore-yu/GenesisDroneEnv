"""Phase-2 experiment / policy / eval smoke (lightweight)."""

from pathlib import Path

import torch

from genesis_drones.adapters.policy import CallablePolicyAdapter, DiffRLPolicyAdapter
from genesis_drones.experiment.checkpoint import attach_run_spec, extract_run_spec
from genesis_drones.experiment.registry import ALGORITHMS, DYNAMICS, TASKS
from genesis_drones.experiment.spec import RunSpec
from genesis_drones.evaluation.runner import EvaluationRunner


def test_registries_cover_phase1_axes():
    assert set(TASKS) >= {"racing", "tracking"}
    assert set(DYNAMICS) >= {"native_quad", "full_quad"}
    assert set(ALGORITHMS) >= {"ppo", "apg", "shac"}


def test_run_spec_round_trip_and_extract():
    spec = RunSpec.stamp(task="racing", dynamics="native_quad", algorithm="shac", seed=1)
    payload = attach_run_spec({"agent": {}}, spec)
    resolved = extract_run_spec(None, payload)
    assert resolved.task == "racing"
    assert resolved.algorithm == "shac"
    assert resolved.dynamics == "native_quad"


def test_extract_requires_axes_without_guessing():
    try:
        extract_run_spec(None, {"foo": 1})
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_evaluation_runner_with_callable_policy():
    import genesis as gs
    from genesis_drones.envs.race_env import RaceEnvConfig, make_racing_env

    if not gs._initialized:
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    env = make_racing_env(RaceEnvConfig(horizon=1, max_episode_steps=3), num_envs=1, requires_grad=False)
    policy = CallablePolicyAdapter(lambda obs: env.hover_command(obs.shape[0]))
    result = EvaluationRunner(env, policy, max_steps=2).run(reset_kwargs={"seed": 0})
    assert result["steps"] == 2
    assert result["last_step"] is not None
