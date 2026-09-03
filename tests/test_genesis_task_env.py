import torch

import genesis as gs

from genesis_drones.adapters import DiffRLAdapter, RslRlAdapter
from genesis_drones.dynamics import make_dynamics_backend
from genesis_drones.envs.contracts import EnvStep
from genesis_drones.envs.genesis_task_env import GenesisTaskEnv
from genesis_drones.envs.race_env import RaceEnvConfig, make_racing_env
from genesis_drones.tasks.racing_core_task import RacingCore, RacingCoreConfig
from genesis_drones.tasks.tracking_core import TrackingCore, TrackingCoreConfig


def _ensure_genesis():
    if not gs._initialized:
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")


def test_genesis_task_env_racing_step_shapes():
    _ensure_genesis()
    env = make_racing_env(RaceEnvConfig(horizon=1, dynamics="native_quad"), num_envs=2, requires_grad=False)
    policy, state = env.reset(seed=0)
    assert policy.shape == (2, 13)
    assert state.shape == (2, 34)
    step = env.step(env.hover_command())
    assert isinstance(step, EnvStep)
    assert step.observation.policy.shape == (2, 13)
    assert step.reward.shape == (2,)
    assert step.physics_loss.shape == (2,)
    assert step.terminated.shape == step.truncated.shape == (2,)


def test_rsl_rl_and_diff_adapters_on_racing():
    _ensure_genesis()
    env = make_racing_env(RaceEnvConfig(horizon=2, dynamics="full_quad"), num_envs=2, requires_grad=True)
    adapter = DiffRLAdapter(env, horizon=2)
    obs = adapter.reset_diff(seed=1)
    assert obs.policy.shape[-1] == 13
    transition = adapter.step_diff(env.hover_command())
    assert transition.physics_loss.shape == (2,)
    assert transition.bootstrap_critic.shape[-1] == 13

    env2 = make_racing_env(RaceEnvConfig(horizon=1, dynamics="native_quad"), num_envs=2, requires_grad=False)
    task = RslRlAdapter(env2, {"algorithm": {"gamma": 0.99}})
    out = task.step(env2.hover_command())
    assert out[0]["policy"].shape == (2, 13)


def test_tracking_core_constructs_with_native_and_full_backends():
    _ensure_genesis()
    for name in ("native_quad", "full_quad"):
        from genesis_drones.envs.track_diff_env import TrackDiffEnvConfig, make_tracking_env

        env = make_tracking_env(
            TrackDiffEnvConfig(horizon=1, dynamics=name),
            num_envs=1,
            requires_grad=False,
        )
        policy, _ = env.reset(seed=None)
        assert policy.shape[-1] == 23
        step = env.step(env.hover_command())
        assert step.observation.policy.shape[-1] == 23
        assert torch.isfinite(step.reward).all()
