from dataclasses import dataclass
from pathlib import Path

import yaml

from genesis_drones.algorithms.diff_rl import ApgConfig, NetworkConfig, ShacConfig
from genesis_drones.envs.track_diff_env import LossWeights, TrackDiffEnvConfig


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
    environment = TrackDiffEnvConfig(
        dt=genesis_config["dt"],
        horizon=environment_data["horizon"],
        max_episode_steps=task_config["max_episode_length"],
        target_threshold=task_config["target_thr"],
        max_collective_thrust=flight_config["max_t"],
        motor_arm=environment_data["motor_arm"],
        thrust_coefficient=flight_config["kf"],
        moment_coefficient=environment_data["moment_coefficient"],
        body_collision_radius=environment_data["body_collision_radius"],
        body_collision_half_height=environment_data["body_collision_half_height"],
        ground_termination_height=task_config["termination_if_close_to_ground"],
        ground_warning_distance=environment_data["ground_warning_distance"],
        horizontal_termination_error=task_config["termination_if_x_greater_than"],
        horizontal_warning_distance=environment_data["horizontal_warning_distance"],
        vertical_termination_error=task_config["termination_if_z_greater_than"],
        vertical_warning_distance=environment_data["vertical_warning_distance"],
        safety_temperature_ratio=environment_data["safety_temperature_ratio"],
        initial_x_range=tuple(genesis_config["init_x_range"]),
        initial_y_range=tuple(genesis_config["init_y_range"]),
        initial_z_range=tuple(genesis_config["init_z_range"]),
        target_x_range=tuple(task_config["command_cfg"]["pos_x_range"]),
        target_y_range=tuple(task_config["command_cfg"]["pos_y_range"]),
        target_z_range=tuple(task_config["command_cfg"]["pos_z_range"]),
        loss_weights=LossWeights(**data["loss_weights"]),
        reward_weights=LossWeights(**data["reward_weights"]),
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
