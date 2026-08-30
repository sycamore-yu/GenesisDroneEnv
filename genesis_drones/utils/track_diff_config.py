from dataclasses import dataclass
from pathlib import Path

import yaml

from genesis_drones.algorithms.diff_rl import (
    ApgAgent,
    ApgConfig,
    NetworkConfig,
    RunningNormalizer,
    ShacAgent,
    ShacConfig,
)
from genesis_drones.envs.track_diff_env import RewardScales, TrackDiffEnv, TrackDiffEnvConfig


@dataclass(frozen=True)
class TrackDiffSettings:
    raw: dict
    environment: TrackDiffEnvConfig
    network: NetworkConfig
    apg: ApgConfig
    shac: ShacConfig
    seed: int
    updates: int
    save_interval: int
    log_root: Path
    validation_scenarios: int
    validation_seed: int
    test_scenarios: int
    test_seed: int
    apg_num_envs: int
    shac_num_envs: int


def build_track_diff_settings(data: dict, project_root: Path) -> TrackDiffSettings:
    with (project_root / "config" / "track_rl" / "genesis_env.yaml").open() as file:
        genesis_config = yaml.safe_load(file)
    with (project_root / "config" / "track_rl" / "rl_env.yaml").open() as file:
        task_config = yaml.safe_load(file)["task"]
    with (project_root / "config" / "track_rl" / "flight.yaml").open() as file:
        flight_config = yaml.safe_load(file)

    environment_data = data["environment"]
    angle = flight_config["ang"]
    reward_data = data.get("reward_scales", task_config["reward_scales"])
    environment = TrackDiffEnvConfig(
        dt=genesis_config["dt"],
        horizon=environment_data["horizon"],
        max_episode_steps=task_config["max_episode_length"],
        target_threshold=task_config["target_thr"],
        max_collective_thrust=flight_config["max_t"],
        motor_arm=environment_data["motor_arm"],
        thrust_coefficient=flight_config["kf"],
        moment_coefficient=environment_data["moment_coefficient"],
        thrust_to_weight_ratio=flight_config["TWR"],
        base_rpm=flight_config["base_rpm"],
        angle_kp=(angle["kp_r"], angle["kp_p"], angle["kp_y"]),
        angle_ki=(angle["ki_r"], angle["ki_p"], angle["ki_y"]),
        angle_kd=(angle["kd_r"], angle["kd_p"], angle["kd_y"]),
        body_collision_radius=environment_data["body_collision_radius"],
        body_collision_half_height=environment_data["body_collision_half_height"],
        ground_termination_height=task_config["termination_if_close_to_ground"],
        ground_warning_distance=environment_data["ground_warning_distance"],
        horizontal_termination_error=task_config["termination_if_x_greater_than"],
        horizontal_warning_distance=environment_data["horizontal_warning_distance"],
        vertical_termination_error=task_config["termination_if_z_greater_than"],
        vertical_warning_distance=environment_data["vertical_warning_distance"],
        safety_temperature_ratio=environment_data["safety_temperature_ratio"],
        roll_termination=task_config["termination_if_roll_greater_than"],
        pitch_termination=task_config["termination_if_pitch_greater_than"],
        yaw_lambda=task_config["yaw_lambda"],
        max_horizon_vel=task_config["max_horizon_vel"],
        max_vertical_vel=task_config["max_vertical_vel"],
        action_delta_weight=data.get("action_delta_weight", 0.0),
        obs_scale_position_error=task_config["obs_scales"]["cur_pos_error"],
        obs_scale_linear_velocity=task_config["obs_scales"]["lin_vel"],
        obs_scale_angular_velocity=task_config["obs_scales"]["ang_vel"],
        enable_collision=environment_data.get("enable_collision", True),
        initial_x_range=tuple(genesis_config["init_x_range"]),
        initial_y_range=tuple(genesis_config["init_y_range"]),
        initial_z_range=tuple(genesis_config["init_z_range"]),
        target_x_range=tuple(task_config["command_cfg"]["pos_x_range"]),
        target_y_range=tuple(task_config["command_cfg"]["pos_y_range"]),
        target_z_range=tuple(task_config["command_cfg"]["pos_z_range"]),
        reward_scales=RewardScales(
            target=reward_data["target"],
            smooth=reward_data["smooth"],
            yaw=reward_data["yaw"],
            angular=reward_data["angular"],
            crash=reward_data["crash"],
            velocity=reward_data.get("velocity", 0.0),
        ),
    )
    return TrackDiffSettings(
        raw=data,
        environment=environment,
        network=NetworkConfig(hidden_sizes=tuple(data["network"]["hidden_sizes"])),
        apg=ApgConfig(**data["apg"]),
        shac=ShacConfig(**data["shac"]),
        seed=data["seed"],
        updates=data["updates"],
        save_interval=data["save_interval"],
        log_root=project_root / data["log_root"],
        validation_scenarios=data["validation_scenarios"],
        validation_seed=data["validation_seed"],
        test_scenarios=data["test_scenarios"],
        test_seed=data["test_seed"],
        apg_num_envs=data["num_envs"]["apg"],
        shac_num_envs=data["num_envs"]["shac"],
    )


def load_track_diff_settings(path: Path) -> TrackDiffSettings:
    with path.open() as file:
        data = yaml.safe_load(file)
    return build_track_diff_settings(data, path.resolve().parents[2])


def make_track_diff_agent(algorithm: str, settings: TrackDiffSettings, device):
    observation_size = TrackDiffEnv.observation_dim
    ratio = settings.environment.thrust_to_weight_ratio
    if algorithm == "apg":
        return ApgAgent(observation_size, TrackDiffEnv.action_dim, ratio, settings.network, settings.apg, device)
    return ShacAgent(observation_size, TrackDiffEnv.action_dim, ratio, settings.network, settings.shac, device)


def make_track_diff_normalizer(device) -> RunningNormalizer:
    return RunningNormalizer(TrackDiffEnv.observation_dim).to(device)
