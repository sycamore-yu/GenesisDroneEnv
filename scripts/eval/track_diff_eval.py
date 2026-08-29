import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from genesis_drones.algorithms.diff_rl import ApgAgent, RunningNormalizer, ShacAgent
from genesis_drones.envs.genesis_env import Genesis_env
from genesis_drones.envs.track_diff_env import TrackDiffEnv
from genesis_drones.evaluation.track_diff import (
    TrackScenarios,
    evaluate_diff_policy,
    paired_differences,
    success_against_ppo,
    summarize_metrics,
)
from genesis_drones.tasks.track_task import Track_task
from genesis_drones.utils.track_diff_config import load_track_diff_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FixedScenarioTrackTask(Track_task):
    def __init__(self, *args, scenarios: TrackScenarios, **kwargs):
        self.scenarios = scenarios
        self.waypoint_index = torch.zeros(scenarios.initial_position.shape[0], device=gs.device, dtype=torch.int64)
        self.arrived = torch.zeros(scenarios.initial_position.shape[0], device=gs.device, dtype=torch.bool)
        self.is_resetting = False
        super().__init__(*args, **kwargs)
        self.waypoint_sequences = scenarios.waypoint_sequences.to(gs.device)

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return
        if self.is_resetting:
            self.waypoint_index[envs_idx] = 0
        else:
            self.arrived[envs_idx] = True
            self.waypoint_index[envs_idx] = torch.clamp(
                self.waypoint_index[envs_idx] + 1,
                max=self.waypoint_sequences.shape[1] - 1,
            )
        self.command_buf[envs_idx] = self.waypoint_sequences[envs_idx, self.waypoint_index[envs_idx]]

    def reset(self, env_idx=None):
        self.is_resetting = True
        observation = super().reset(env_idx)
        self.is_resetting = False
        return observation

    def step(self, action):
        self.arrived.zero_()
        return super().step(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apg-checkpoint", type=Path, required=True)
    parser.add_argument("--shac-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ppo-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "logs" / "track_rl" / "policy_demo" / "model_500.pt",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "track_diff" / "train.yaml")
    parser.add_argument("--scenarios", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-scenarios", type=int)
    return parser.parse_args()


def load_diff_policy(
    checkpoint_path: Path,
    algorithm: str,
    settings,
) -> tuple[ApgAgent | ShacAgent, RunningNormalizer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=gs.device, weights_only=False)
    if checkpoint["algorithm"] != algorithm:
        raise ValueError(f"{checkpoint_path} is not an {algorithm} checkpoint")
    if algorithm == "apg":
        agent = ApgAgent(17, 4, 3.3, settings.network, settings.apg, gs.device)
    else:
        agent = ShacAgent(17, 4, 3.3, settings.network, settings.shac, gs.device)
    agent.actor.load_state_dict(checkpoint["agent"]["actor"])
    normalizer = RunningNormalizer(17).to(gs.device)
    normalizer.load_state_dict(checkpoint["normalizer"])
    return agent, normalizer, checkpoint


def evaluate_ppo(
    scenarios: TrackScenarios,
    checkpoint_path: Path,
    environment_config: dict,
    flight_config: dict,
    task_config: dict,
    train_config: dict,
) -> dict[str, torch.Tensor]:
    num_scenarios = scenarios.initial_position.shape[0]
    environment_config = environment_config.copy()
    environment_config.update(
        num_envs=num_scenarios,
        show_viewer=False,
        render_cam=False,
        vis_waypoints=False,
    )
    genesis_environment = Genesis_env(environment_config, flight_config, num_envs=num_scenarios)
    task = FixedScenarioTrackTask(
        genesis_env=genesis_environment,
        env_config=environment_config,
        task_config=task_config,
        train_config=train_config,
        num_envs=num_scenarios,
        scenarios=scenarios,
    )
    runner = OnPolicyRunner(task, train_config, "", device="cuda:0")
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device="cuda:0")
    observation = task.reset()

    envs_idx = torch.arange(num_scenarios, device=gs.device)
    initial_position = scenarios.initial_position.to(gs.device)
    initial_quaternion = scenarios.initial_quaternion.to(gs.device)
    genesis_environment.drone.set_pos(initial_position, zero_velocity=True)
    genesis_environment.drone.set_quat(initial_quaternion, zero_velocity=True)
    genesis_environment.drone.odom.reset(initial_quaternion, envs_idx)
    genesis_environment.drone.odom.odom_update()
    genesis_environment.drone.controller.reset(envs_idx)
    task.command_buf[:] = scenarios.waypoint_sequences[:, 0].to(gs.device)
    task.cur_pos_error[:] = task.command_buf - genesis_environment.drone.odom.world_pos
    task.last_pos_error[:] = task.cur_pos_error
    task._update_obs()
    observation = task.get_observations()

    is_alive = torch.ones(num_scenarios, device=gs.device, dtype=torch.bool)
    waypoint_count = torch.zeros(num_scenarios, device=gs.device, dtype=torch.int64)
    first_arrival_step = torch.full(
        (num_scenarios,), task_config["max_episode_length"], device=gs.device, dtype=torch.int64
    )
    crash_step = torch.full(
        (num_scenarios,), task_config["max_episode_length"] + 1, device=gs.device, dtype=torch.int64
    )
    position_error_sum = torch.zeros(num_scenarios, device=gs.device)
    position_error_steps = torch.zeros(num_scenarios, device=gs.device)

    with torch.no_grad():
        for step in range(1, task_config["max_episode_length"] + 1):
            is_alive_before = is_alive.clone()
            action = policy(observation)
            action = torch.where(is_alive_before[:, None], action, torch.zeros_like(action))
            observation, _, _, _ = task.step(action)
            is_crashed = is_alive_before & (task.crash_condition_buf | genesis_environment.drone.odom.has_nan)
            is_arrived = is_alive_before & ~is_crashed & task.arrived
            distance = torch.linalg.vector_norm(task.cur_pos_error, dim=-1)
            waypoint_count += is_arrived
            first_arrival_step = torch.where(
                is_arrived & (first_arrival_step == task_config["max_episode_length"]),
                torch.full_like(first_arrival_step, step),
                first_arrival_step,
            )
            crash_step = torch.where(is_crashed, torch.full_like(crash_step, step), crash_step)
            position_error_sum += distance * is_alive_before
            position_error_steps += is_alive_before
            is_alive = is_alive_before & ~is_crashed

    max_episode_steps = task_config["max_episode_length"]
    survival_steps = torch.where(
        crash_step <= max_episode_steps,
        crash_step,
        torch.full_like(crash_step, max_episode_steps),
    )
    return {
        "first_arrived": waypoint_count > 0,
        "waypoint_count": waypoint_count,
        "first_arrival_time": first_arrival_step * environment_config["dt"],
        "crashed": crash_step <= max_episode_steps,
        "survival_time": survival_steps * environment_config["dt"],
        "mean_position_error": position_error_sum / position_error_steps.clamp_min(1.0),
    }


def training_metadata(checkpoint: dict, checkpoint_path: Path) -> dict:
    summary_path = checkpoint_path.parent / "training_summary.json"
    run_summary = None
    if summary_path.exists():
        with summary_path.open() as file:
            run_summary = json.load(file)
    return {
        "updates": checkpoint["update"] if run_summary is None else run_summary["updates"],
        "environment_steps": (
            checkpoint["environment_steps"] if run_summary is None else run_summary["environment_steps"]
        ),
        "elapsed_seconds": (
            checkpoint["elapsed_seconds"] if run_summary is None else run_summary["elapsed_seconds"]
        ),
        "peak_memory_bytes": None if run_summary is None else run_summary["peak_memory_bytes"],
        "selected_checkpoint_update": checkpoint["update"],
        "selected_checkpoint_environment_steps": checkpoint["environment_steps"],
    }


def main() -> None:
    args = parse_args()
    settings = load_track_diff_settings(args.config)
    num_scenarios = settings.test_scenarios if args.num_scenarios is None else args.num_scenarios
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gs.init(backend=gs.gpu, seed=settings.seed, logging_level="warning")
    apg_agent, apg_normalizer, apg_checkpoint = load_diff_policy(args.apg_checkpoint, "apg", settings)
    shac_agent, shac_normalizer, shac_checkpoint = load_diff_policy(args.shac_checkpoint, "shac", settings)
    apg_initial_agent, apg_initial_normalizer, _ = load_diff_policy(
        args.apg_checkpoint.parent / "checkpoint_0000.pt", "apg", settings
    )
    shac_initial_agent, shac_initial_normalizer, _ = load_diff_policy(
        args.shac_checkpoint.parent / "checkpoint_0000.pt", "shac", settings
    )

    if args.scenarios is None:
        scenarios = TrackScenarios.generate(
            num_scenarios,
            settings.environment.max_episode_steps,
            settings.environment,
            settings.test_seed,
        )
        scenarios.save(args.output_dir / "test_scenarios.pt")
    else:
        scenarios = TrackScenarios.load(args.scenarios)
        if scenarios.initial_position.shape[0] != num_scenarios:
            raise ValueError("scenario count does not match --num-scenarios")

    diff_environment = TrackDiffEnv(settings.environment, num_scenarios, requires_grad=False)
    apg_initial_metrics = evaluate_diff_policy(
        diff_environment, apg_initial_agent, apg_initial_normalizer, scenarios
    )
    shac_initial_metrics = evaluate_diff_policy(
        diff_environment, shac_initial_agent, shac_initial_normalizer, scenarios
    )
    apg_metrics = evaluate_diff_policy(diff_environment, apg_agent, apg_normalizer, scenarios)
    shac_metrics = evaluate_diff_policy(diff_environment, shac_agent, shac_normalizer, scenarios)

    with (PROJECT_ROOT / "config" / "track_rl" / "genesis_env.yaml").open() as file:
        environment_config = yaml.safe_load(file)
    with (PROJECT_ROOT / "config" / "track_rl" / "flight.yaml").open() as file:
        flight_config = yaml.safe_load(file)
    with (PROJECT_ROOT / "config" / "track_rl" / "rl_env.yaml").open() as file:
        rl_config = yaml.safe_load(file)
    ppo_metrics = evaluate_ppo(
        scenarios,
        args.ppo_checkpoint,
        environment_config,
        flight_config,
        rl_config["task"],
        rl_config["train"],
    )

    episode_seconds = settings.environment.max_episode_steps * settings.environment.dt
    summaries = {
        "apg_untrained": summarize_metrics(apg_initial_metrics, episode_seconds),
        "shac_untrained": summarize_metrics(shac_initial_metrics, episode_seconds),
        "apg": summarize_metrics(apg_metrics, episode_seconds),
        "shac": summarize_metrics(shac_metrics, episode_seconds),
        "ppo": summarize_metrics(ppo_metrics, episode_seconds),
    }
    apg_success = success_against_ppo(summaries["apg"], summaries["ppo"])
    shac_success = success_against_ppo(summaries["shac"], summaries["ppo"])
    result = {
        "summaries": {name: summary.to_dict() for name, summary in summaries.items()},
        "paired_difference_from_ppo": {
            "apg": {
                name: asdict(value)
                for name, value in paired_differences(apg_metrics, ppo_metrics, episode_seconds).items()
            },
            "shac": {
                name: asdict(value)
                for name, value in paired_differences(shac_metrics, ppo_metrics, episode_seconds).items()
            },
        },
        "paired_difference_from_untrained": {
            "apg": {
                name: asdict(value)
                for name, value in paired_differences(
                    apg_metrics, apg_initial_metrics, episode_seconds
                ).items()
            },
            "shac": {
                name: asdict(value)
                for name, value in paired_differences(
                    shac_metrics, shac_initial_metrics, episode_seconds
                ).items()
            },
        },
        "success": {
            "apg": apg_success,
            "shac": shac_success,
            "overall": apg_success["success"] and shac_success["success"],
        },
        "training": {
            "apg": training_metadata(apg_checkpoint, args.apg_checkpoint),
            "shac": training_metadata(shac_checkpoint, args.shac_checkpoint),
            "ppo": {
                "updates": 500,
                "environment_steps": 12000 * 80 * 500,
                "elapsed_seconds": None,
                "peak_memory_bytes": None,
            },
        },
        "comparison_scope": "system-level: observations, control interfaces, and collision training differ",
    }
    torch.save(
        {
            "apg_untrained": apg_initial_metrics,
            "shac_untrained": shac_initial_metrics,
            "apg": apg_metrics,
            "shac": shac_metrics,
            "ppo": ppo_metrics,
        },
        args.output_dir / "per_scenario_metrics.pt",
    )
    with (args.output_dir / "evaluation.json").open("w") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result["success"], indent=2))


if __name__ == "__main__":
    main()
