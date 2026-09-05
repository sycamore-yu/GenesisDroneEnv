"""Factories: config → TaskCore / Dynamics / GenesisTaskEnv / training stack / policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

import genesis as gs

from genesis_drones.adapters.diff_rl_adapter import DiffRLAdapter
from genesis_drones.adapters.policy import DiffRLPolicyAdapter, RslRlPolicyAdapter
from genesis_drones.adapters.rsl_rl_adapter import RslRlAdapter
from genesis_drones.algorithms.diff_rl import (
    NetworkConfig,
    RunningNormalizer,
    build_diff_algorithm_config,
    make_diff_agent,
)
from genesis_drones.controllers.ctbr_controller import CtbrControllerConfig
from genesis_drones.controllers.native_config import NativeQuadConfig
from genesis_drones.dynamics import make_dynamics_backend
from genesis_drones.envs.genesis_task_env import GenesisTaskEnv
from genesis_drones.envs.race_env import RaceEnvConfig, make_racing_env
from genesis_drones.envs.track_diff_env import TrackDiffEnvConfig, make_tracking_env
from genesis_drones.experiment.checkpoint import extract_run_spec, load_checkpoint
from genesis_drones.experiment.registry import require_algorithm, require_dynamics, require_task
from genesis_drones.experiment.spec import RunSpec
from genesis_drones.tasks.racing_core_task import RacingCore, RacingCoreConfig
from genesis_drones.tasks.racing_tracks import RACING_TRACK
from genesis_drones.tasks.tracking_core import RewardScales, TrackingCore, TrackingCoreConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dict(obj) -> dict:
    if obj is None:
        return {}
    if hasattr(obj, "items"):
        return dict(obj)
    return dict(obj)


def racing_env_config_from(cfg: dict[str, Any], dynamics: str) -> RaceEnvConfig:
    env = _dict(cfg.get("environment"))
    return RaceEnvConfig(
        dt=float(env.get("dt", 0.0333)),
        horizon=int(env.get("horizon", 32)),
        max_episode_steps=int(env.get("max_episode_steps", int(40.0 / 0.0333))),
        min_target_velocity=float(env.get("min_target_velocity", 5.0)),
        max_target_velocity=float(env.get("max_target_velocity", 10.0)),
        gamma=float(env.get("gamma", 0.99)),
        td_lambda=float(env.get("td_lambda", 0.95)),
        dynamics=dynamics,
        controller=CtbrControllerConfig(randomize=False),
    )


def tracking_env_config_from(cfg: dict[str, Any], dynamics: str) -> TrackDiffEnvConfig:
    """Merge Hydra/task yaml with track_rl flight defaults for NativeQuad numbers."""
    with (PROJECT_ROOT / "config/track_rl/genesis_env.yaml").open() as file:
        genesis_config = yaml.safe_load(file)
    with (PROJECT_ROOT / "config/track_rl/rl_env.yaml").open() as file:
        task_config = yaml.safe_load(file)["task"]
    with (PROJECT_ROOT / "config/track_rl/flight.yaml").open() as file:
        flight_config = yaml.safe_load(file)
    env = _dict(cfg.get("environment"))
    angle = flight_config["ang"]
    reward_data = _dict(cfg.get("reward_scales")) or task_config["reward_scales"]
    return TrackDiffEnvConfig(
        dt=float(env.get("dt", genesis_config["dt"])),
        horizon=int(env.get("horizon", 32)),
        max_episode_steps=int(env.get("max_episode_steps", task_config["max_episode_length"])),
        target_threshold=float(env.get("target_threshold", task_config["target_thr"])),
        max_collective_thrust=float(flight_config["max_t"]),
        motor_arm=float(env.get("motor_arm", 0.1)),
        thrust_coefficient=float(flight_config["kf"]),
        moment_coefficient=float(env.get("moment_coefficient", 7.94e-12)),
        thrust_to_weight_ratio=float(flight_config["TWR"]),
        base_rpm=float(flight_config["base_rpm"]),
        angle_kp=(angle["kp_r"], angle["kp_p"], angle["kp_y"]),
        angle_ki=(angle["ki_r"], angle["ki_p"], angle["ki_y"]),
        angle_kd=(angle["kd_r"], angle["kd_p"], angle["kd_y"]),
        body_collision_radius=float(env.get("body_collision_radius", 0.06)),
        body_collision_half_height=float(env.get("body_collision_half_height", 0.0125)),
        ground_termination_height=float(task_config["termination_if_close_to_ground"]),
        ground_warning_distance=float(env.get("ground_warning_distance", 0.2)),
        horizontal_termination_error=float(task_config["termination_if_x_greater_than"]),
        horizontal_warning_distance=float(env.get("horizontal_warning_distance", 0.5)),
        vertical_termination_error=float(task_config["termination_if_z_greater_than"]),
        vertical_warning_distance=float(env.get("vertical_warning_distance", 0.12)),
        safety_temperature_ratio=float(env.get("safety_temperature_ratio", 0.1)),
        roll_termination=float(task_config["termination_if_roll_greater_than"]),
        pitch_termination=float(task_config["termination_if_pitch_greater_than"]),
        yaw_lambda=float(task_config["yaw_lambda"]),
        max_horizon_vel=float(task_config["max_horizon_vel"]),
        max_vertical_vel=float(task_config["max_vertical_vel"]),
        action_delta_weight=float(cfg.get("action_delta_weight", 0.0)),
        obs_scale_position_error=float(task_config["obs_scales"]["cur_pos_error"]),
        obs_scale_linear_velocity=float(task_config["obs_scales"]["lin_vel"]),
        obs_scale_angular_velocity=float(task_config["obs_scales"]["ang_vel"]),
        enable_collision=bool(env.get("enable_collision", True)),
        initial_x_range=tuple(genesis_config["init_x_range"]),
        initial_y_range=tuple(genesis_config["init_y_range"]),
        initial_z_range=tuple(genesis_config["init_z_range"]),
        target_x_range=tuple(task_config["command_cfg"]["pos_x_range"]),
        target_y_range=tuple(task_config["command_cfg"]["pos_y_range"]),
        target_z_range=tuple(task_config["command_cfg"]["pos_z_range"]),
        reward_scales=RewardScales(
            target=float(reward_data["target"]),
            smooth=float(reward_data["smooth"]),
            yaw=float(reward_data["yaw"]),
            angular=float(reward_data["angular"]),
            crash=float(reward_data["crash"]),
            velocity=float(reward_data.get("velocity", 0.0)),
        ),
        fully_differentiable=bool(env.get("fully_differentiable", False)),
        dynamics=dynamics,
    )


def build_task(run_spec: RunSpec, num_envs: int, device=None, dtype=None):
    require_task(run_spec.task)
    device = device or gs.device
    dtype = dtype or gs.tc_float
    if run_spec.task == "racing":
        env_cfg = racing_env_config_from({"environment": run_spec.environment}, run_spec.dynamics)
        return RacingCore(
            RacingCoreConfig(
                dt=env_cfg.dt,
                max_episode_steps=env_cfg.max_episode_steps,
                min_target_velocity=env_cfg.min_target_velocity,
                max_target_velocity=env_cfg.max_target_velocity,
                gamma=env_cfg.gamma,
                td_lambda=env_cfg.td_lambda,
            ),
            num_envs=num_envs,
            device=device,
            dtype=dtype,
            track=RACING_TRACK,
        )
    track_cfg = tracking_env_config_from(
        {"environment": run_spec.environment, "reward_scales": run_spec.environment.get("reward_scales", {})},
        run_spec.dynamics,
    )
    return TrackingCore(TrackingCoreConfig.from_compat(track_cfg), num_envs, device, dtype)


def build_dynamics(run_spec: RunSpec, num_envs: int, dt: float, device=None, dtype=None):
    require_dynamics(run_spec.dynamics)
    device = device or gs.device
    dtype = dtype or gs.tc_float
    native = None
    if run_spec.task == "tracking":
        track_cfg = tracking_env_config_from({"environment": run_spec.environment}, run_spec.dynamics)
        native = track_cfg.native_quad_config()
    return make_dynamics_backend(
        run_spec.dynamics,
        num_envs,
        device,
        dtype,
        dt,
        controller_config=CtbrControllerConfig(randomize=False),
        native_config=native or NativeQuadConfig(),
    )


def build_environment(
    run_spec: RunSpec,
    num_envs: int,
    *,
    requires_grad: bool,
    show_viewer: bool = False,
) -> GenesisTaskEnv:
    require_task(run_spec.task)
    require_dynamics(run_spec.dynamics)
    if run_spec.task == "racing":
        return make_racing_env(
            racing_env_config_from({"environment": run_spec.environment}, run_spec.dynamics),
            num_envs,
            requires_grad=requires_grad,
            show_viewer=show_viewer,
        )
    return make_tracking_env(
        tracking_env_config_from({"environment": run_spec.environment}, run_spec.dynamics),
        num_envs,
        requires_grad=requires_grad,
        show_viewer=show_viewer,
    )


def build_training_stack(
    run_spec: RunSpec,
    cfg: dict[str, Any],
    num_envs: int,
    log_dir: Path | None = None,
):
    """Return (env_or_adapter, runner_or_agent_bundle) ready to train."""
    algorithm = require_algorithm(run_spec.algorithm)
    if algorithm == "ppo":
        environment = build_environment(run_spec, num_envs, requires_grad=False)
        train_config = _load_ppo_train_config(run_spec, cfg)
        adapter = RslRlAdapter(environment, train_config)
        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(adapter, train_config, str(log_dir or "."), device=str(gs.device))
        return {"kind": "ppo", "environment": environment, "adapter": adapter, "runner": runner, "train_config": train_config}

    environment = build_environment(run_spec, num_envs, requires_grad=True)
    horizon = int(run_spec.environment.get("horizon", cfg.get("environment", {}).get("horizon", 32)))
    critic_mode = "physics_plus_policy" if run_spec.task == "tracking" else "neg_reward"
    adapter = DiffRLAdapter(
        environment,
        horizon=horizon,
        action_delta_weight=float(cfg.get("action_delta_weight", 0.0)),
        critic_cost_mode=critic_mode,
    )
    network = NetworkConfig(hidden_sizes=tuple(cfg.get("network", {}).get("hidden_sizes", (256, 128))))
    algo_cfg = build_diff_algorithm_config(algorithm, cfg.get(algorithm) or run_spec.algorithm_config)
    agent = make_diff_agent(algorithm, adapter.spec, network, algo_cfg, gs.device)
    normalizer = RunningNormalizer(adapter.spec.policy_observation_dim).to(gs.device)
    return {
        "kind": "diff",
        "environment": environment,
        "adapter": adapter,
        "agent": agent,
        "normalizer": normalizer,
        "network": network,
        "algorithm_config": algo_cfg,
    }


def _load_ppo_train_config(run_spec: RunSpec, cfg: dict[str, Any]) -> dict:
    if run_spec.task == "racing":
        override = cfg.get("_ppo_config")
        if override:
            path = Path(override)
        else:
            name = "ppo_full_quad.yaml" if run_spec.dynamics == "full_quad" else "ppo.yaml"
            path = PROJECT_ROOT / "config" / "race" / name
    else:
        path = PROJECT_ROOT / "config" / "track_rl" / "rl_env.yaml"
        with path.open() as file:
            train_config = yaml.safe_load(file)["train"]
        sizes = list(cfg.get("network", {}).get("hidden_sizes", [128, 128, 128]))
        train_config["actor"]["hidden_dims"] = sizes
        train_config["critic"]["hidden_dims"] = sizes
        train_config["obs_groups"] = {"actor": ["policy"], "critic": ["policy"]}
        train_config["seed"] = run_spec.seed
        if cfg.get("_learning_rate") is not None:
            train_config["algorithm"]["learning_rate"] = cfg["_learning_rate"]
        return train_config
    with path.open() as file:
        train_config = yaml.safe_load(file)
    sizes = list(cfg.get("network", {}).get("hidden_sizes", train_config["actor"].get("hidden_dims", [256, 128])))
    train_config["actor"]["hidden_dims"] = sizes
    train_config["critic"]["hidden_dims"] = sizes
    train_config["seed"] = run_spec.seed
    if cfg.get("_learning_rate") is not None:
        train_config["algorithm"]["learning_rate"] = cfg["_learning_rate"]
    return train_config


def load_policy(
    checkpoint: Path,
    environment: GenesisTaskEnv,
    *,
    overrides: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
    device=None,
):
    """Load any supported checkpoint into a Policy with act()."""
    device = device or gs.device
    payload = load_checkpoint(checkpoint, map_location=device)
    run_spec = extract_run_spec(checkpoint, payload, overrides=overrides)
    algorithm = require_algorithm(run_spec.algorithm)
    cfg = cfg or payload.get("config") or {}
    if algorithm == "ppo":
        return _load_ppo_policy(payload, run_spec, environment, cfg, device), run_spec, payload
    return _load_diff_policy(payload, run_spec, environment, cfg, device), run_spec, payload


def _load_diff_policy(payload, run_spec, environment, cfg, device):
    from genesis_drones.envs.differentiable import DiffEnvSpec

    horizon = int(run_spec.environment.get("horizon", cfg.get("environment", {}).get("horizon", 32)))
    network_sizes = cfg.get("network", {}).get("hidden_sizes") or (256, 128)
    if run_spec.algorithm in cfg:
        algo_data = cfg[run_spec.algorithm]
    else:
        algo_data = run_spec.algorithm_config or {"horizon": horizon}
    network = NetworkConfig(hidden_sizes=tuple(network_sizes))
    spec = DiffEnvSpec(
        policy_observation_dim=environment.task.policy_observation_dim,
        critic_observation_dim=environment.task.critic_observation_dim,
        action_dim=environment.action_dim,
        horizon=horizon,
        nominal_action=environment.plant.nominal_action,
    )
    agent = make_diff_agent(
        run_spec.algorithm,
        spec,
        network,
        build_diff_algorithm_config(run_spec.algorithm, algo_data),
        device,
    )
    agent.load_state_dict(payload["agent"])
    normalizer = RunningNormalizer(spec.policy_observation_dim).to(device)
    key = "policy_normalizer" if "policy_normalizer" in payload else "normalizer"
    if key in payload:
        normalizer.load_state_dict(payload[key])
    return DiffRLPolicyAdapter(agent, normalizer)


def build_ppo_actor(train_config: dict[str, Any], obs_dim: int, action_dim: int, device):
    from tensordict import TensorDict
    from rsl_rl.utils import resolve_class

    zeros = torch.zeros(1, obs_dim, device=device)
    dummy = TensorDict({"policy": zeros, "critic": zeros.clone()}, batch_size=[1])
    obs_groups = train_config.get("obs_groups", {"actor": ["policy"], "critic": ["policy"]})
    for group_keys in obs_groups.values():
        for key in group_keys:
            if key not in dummy.keys():
                dummy[key] = zeros.clone()
    actor_class, actor_cfg = resolve_class(train_config["actor"])
    actor = actor_class(dummy, obs_groups, "actor", action_dim, **actor_cfg)
    return actor


def _load_ppo_policy(payload, run_spec, environment, cfg, device):
    train_config = _load_ppo_train_config(run_spec, cfg)
    actor = build_ppo_actor(
        train_config,
        environment.task.policy_observation_dim,
        environment.action_dim,
        device,
    )
    state = payload.get("actor_state_dict")
    if state is None:
        raise ValueError("PPO checkpoint missing actor_state_dict")
    actor.load_state_dict(state)
    actor.to(device)
    actor.eval()
    return RslRlPolicyAdapter(actor)


def run_spec_from_cfg(cfg: dict[str, Any]) -> RunSpec:
    task = str(cfg.get("task", cfg.get("task_name", "racing")))
    if isinstance(cfg.get("task"), dict):
        task = str(cfg["task"].get("name", "racing"))
    dynamics = str(cfg.get("dynamics", "native_quad"))
    if isinstance(cfg.get("dynamics"), dict):
        dynamics = str(cfg["dynamics"].get("name", "native_quad"))
    algorithm = str(cfg.get("algorithm", "ppo"))
    if isinstance(cfg.get("algorithm"), dict):
        algorithm = str(cfg["algorithm"].get("name", "ppo"))
    sensor = str(cfg.get("sensor", "state"))
    if isinstance(cfg.get("sensor"), dict):
        sensor = str(cfg["sensor"].get("name", "state"))
    env = _dict(cfg.get("environment"))
    if isinstance(cfg.get("task"), dict):
        env = {**_dict(cfg["task"].get("environment")), **env}
    if isinstance(cfg.get("dynamics"), dict):
        env = {**env, **{k: v for k, v in cfg["dynamics"].items() if k != "name"}}
    algo_cfg = {}
    if isinstance(cfg.get("algorithm"), dict):
        algo_cfg = {k: v for k, v in cfg["algorithm"].items() if k != "name"}
    elif algorithm in cfg:
        algo_cfg = _dict(cfg[algorithm])
    return RunSpec.stamp(
        task=task,
        dynamics=dynamics,
        algorithm=algorithm,
        network=str(cfg.get("network", {}).get("name", "mlp") if isinstance(cfg.get("network"), dict) else cfg.get("network", "mlp")),
        sensor=sensor,
        seed=cfg.get("seed"),
        environment=env,
        algorithm_config=algo_cfg,
    )
