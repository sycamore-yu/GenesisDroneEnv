import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from genesis_drones.adapters.policy import DiffRLPolicyAdapter
from genesis_drones.algorithms.diff_rl import ApgAgent, RunningNormalizer, ShacAgent
from genesis_drones.envs.genesis_env import Genesis_env
from genesis_drones.envs.track_diff_env import TrackDiffEnv
from genesis_drones.evaluation.track_diff import (
    TrackScenarios,
    classify_eval_step,
    evaluate_diff_policy,
    make_diff_observation,
    paired_differences,
    success_against_ppo,
    summarize_metrics,
)
from genesis_drones.tasks.track_task import Track_task
from genesis_drones.utils.track_diff_config import (
    load_track_diff_settings,
    make_track_diff_agent,
    make_track_diff_normalizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ScoringTrackTask(Track_task):
    """PPO env for one-life eval: never teleport. Arrival advances the fixed waypoint list."""

    def __init__(self, *args, scenarios: TrackScenarios, **kwargs):
        self.scenarios = scenarios
        self.waypoint_index = torch.zeros(scenarios.initial_position.shape[0], device=gs.device, dtype=torch.int64)
        self.lock_resets = False
        super().__init__(*args, **kwargs)
        self.waypoint_sequences = scenarios.waypoint_sequences.to(gs.device)

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.command_buf[envs_idx] = self.waypoint_sequences[envs_idx, self.waypoint_index[envs_idx]]

    def reset(self, env_idx=None):
        if self.lock_resets and env_idx is not None:
            return self.get_observations()
        if env_idx is None:
            self.waypoint_index.zero_()
        else:
            self.waypoint_index[env_idx] = 0
        return super().reset(env_idx)


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
    parser.add_argument(
        "--skip-common-plant",
        action="store_true",
        help="skip APG/SHAC eval on the PPO rotor plant",
    )
    return parser.parse_args()


def load_diff_policy(
    checkpoint_path: Path,
    algorithm: str,
    settings,
) -> tuple[ApgAgent | ShacAgent, RunningNormalizer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=gs.device, weights_only=False)
    if checkpoint["algorithm"] != algorithm:
        raise ValueError(f"{checkpoint_path} is not an {algorithm} checkpoint")
    agent = make_track_diff_agent(algorithm, settings, gs.device)
    agent.actor.load_state_dict(checkpoint["agent"]["actor"])
    normalizer = make_track_diff_normalizer(gs.device)
    normalizer.load_state_dict(checkpoint["normalizer"])
    return agent, normalizer, checkpoint


def make_scoring_task(
    scenarios: TrackScenarios,
    environment_config: dict,
    flight_config: dict,
    task_config: dict,
    train_config: dict,
) -> tuple[Genesis_env, ScoringTrackTask]:
    num_scenarios = scenarios.initial_position.shape[0]
    environment_config = environment_config.copy()
    environment_config.update(
        num_envs=num_scenarios,
        show_viewer=False,
        render_cam=False,
        vis_waypoints=False,
    )
    genesis_environment = Genesis_env(environment_config, flight_config, num_envs=num_scenarios)
    task = ScoringTrackTask(
        genesis_env=genesis_environment,
        env_config=environment_config,
        task_config=task_config,
        train_config=train_config,
        num_envs=num_scenarios,
        scenarios=scenarios,
    )
    return genesis_environment, task


def pin_rotor_task(task: ScoringTrackTask, scenarios: TrackScenarios) -> None:
    genesis_environment = task.genesis_env
    num_scenarios = scenarios.initial_position.shape[0]
    envs_idx = torch.arange(num_scenarios, device=gs.device)
    task.lock_resets = False
    task.reset()
    initial_position = scenarios.initial_position.to(gs.device)
    initial_quaternion = scenarios.initial_quaternion.to(gs.device)
    genesis_environment.drone.set_pos(initial_position, zero_velocity=True)
    genesis_environment.drone.set_quat(initial_quaternion, zero_velocity=True)
    genesis_environment.drone.odom.reset(initial_quaternion, envs_idx)
    genesis_environment.drone.odom.odom_update()
    genesis_environment.drone.controller.reset(envs_idx)
    task.waypoint_index.zero_()
    task.command_buf[:] = scenarios.waypoint_sequences[:, 0].to(gs.device)
    task.cur_pos_error[:] = task.command_buf - genesis_environment.drone.odom.world_pos
    task.last_pos_error[:] = task.cur_pos_error
    task.last_actions.zero_()
    task._update_obs()
    task.lock_resets = True


def score_rotor_rollout(task: ScoringTrackTask, choose_action, task_config: dict, dt: float) -> dict[str, torch.Tensor]:
    """One-life scoring on Genesis_env + angle PID + set_propellers_rpm."""
    genesis_environment = task.genesis_env
    num_scenarios = task.num_envs
    envs_idx = torch.arange(num_scenarios, device=gs.device)
    max_steps = task_config["max_episode_length"]
    max_waypoint_index = task.waypoint_sequences.shape[1] - 1
    is_alive = torch.ones(num_scenarios, device=gs.device, dtype=torch.bool)
    waypoint_count = torch.zeros(num_scenarios, device=gs.device, dtype=torch.int64)
    first_arrival_step = torch.full((num_scenarios,), max_steps, device=gs.device, dtype=torch.int64)
    crash_step = torch.full((num_scenarios,), max_steps + 1, device=gs.device, dtype=torch.int64)
    position_error_sum = torch.zeros(num_scenarios, device=gs.device)
    position_error_steps = torch.zeros(num_scenarios, device=gs.device)
    hit_ground_any = torch.zeros(num_scenarios, device=gs.device, dtype=torch.bool)
    left_workspace_any = torch.zeros(num_scenarios, device=gs.device, dtype=torch.bool)
    vertical_band_any = torch.zeros(num_scenarios, device=gs.device, dtype=torch.bool)
    previous_position = genesis_environment.drone.odom.world_pos.detach().clone()
    segment_start_position = previous_position.clone()
    last_action = torch.zeros((num_scenarios, 4), device=gs.device)
    path_length = torch.zeros(num_scenarios, device=gs.device)
    segment_path_length = torch.zeros(num_scenarios, device=gs.device)
    completed_path_length = torch.zeros(num_scenarios, device=gs.device)
    straight_line_distance = torch.zeros(num_scenarios, device=gs.device)
    completed_segments = torch.zeros(num_scenarios, device=gs.device)
    inter_waypoint_time_sum = torch.zeros(num_scenarios, device=gs.device)
    segment_start_step = torch.zeros(num_scenarios, device=gs.device, dtype=torch.int64)
    action_total_variation = torch.zeros(num_scenarios, device=gs.device)
    speed_sum = torch.zeros(num_scenarios, device=gs.device)
    closing_sum = torch.zeros(num_scenarios, device=gs.device)
    alive_steps = torch.zeros(num_scenarios, device=gs.device)
    max_speed = torch.zeros(num_scenarios, device=gs.device)
    speed_history = torch.zeros((num_scenarios, max_steps), device=gs.device)
    closing_history = torch.zeros((num_scenarios, max_steps), device=gs.device)
    history_mask = torch.zeros((num_scenarios, max_steps), device=gs.device, dtype=torch.bool)

    with torch.no_grad():
        for step in range(1, max_steps + 1):
            is_alive_before = is_alive.clone()
            target_before = task.command_buf.detach().clone()
            action = choose_action(task, is_alive_before, last_action)
            action = torch.where(is_alive_before[:, None], action, torch.zeros_like(action))
            task.step(action)
            position = genesis_environment.drone.odom.world_pos
            velocity = genesis_environment.drone.odom.world_linear_vel
            is_arrived, is_crashed, distance, hit_ground, left_workspace = classify_eval_step(
                is_alive_before,
                position,
                target_before,
                genesis_environment.drone.odom.has_nan,
                task_config["target_thr"],
                task_config["termination_if_close_to_ground"],
                task_config["termination_if_x_greater_than"],
            )
            step_distance = torch.linalg.vector_norm(position - previous_position, dim=-1)
            path_length = path_length + step_distance * is_alive_before
            segment_path_length = segment_path_length + step_distance * is_alive_before
            speed = torch.linalg.vector_norm(velocity, dim=-1)
            goal_vector = target_before - previous_position
            goal_distance = torch.linalg.vector_norm(goal_vector, dim=-1).clamp_min(1e-6)
            closing = (velocity * goal_vector).sum(dim=-1) / goal_distance
            speed_sum = speed_sum + speed * is_alive_before
            closing_sum = closing_sum + closing * is_alive_before
            alive_steps = alive_steps + is_alive_before.to(dtype=alive_steps.dtype)
            max_speed = torch.maximum(max_speed, speed * is_alive_before)
            speed_history[:, step - 1] = speed
            closing_history[:, step - 1] = closing
            history_mask[:, step - 1] = is_alive_before
            action_total_variation = (
                action_total_variation
                + torch.linalg.vector_norm(action - last_action, dim=-1) * is_alive_before
            )
            if is_arrived.any():
                segment_straight = torch.linalg.vector_norm(target_before - segment_start_position, dim=-1)
                segment_time = (step - segment_start_step).to(dtype=inter_waypoint_time_sum.dtype) * dt
                straight_line_distance = straight_line_distance + segment_straight * is_arrived
                completed_path_length = completed_path_length + segment_path_length * is_arrived
                inter_waypoint_time_sum = inter_waypoint_time_sum + segment_time * is_arrived
                completed_segments = completed_segments + is_arrived.to(dtype=completed_segments.dtype)
                segment_path_length = torch.where(
                    is_arrived, torch.zeros_like(segment_path_length), segment_path_length
                )
                segment_start_position = torch.where(is_arrived[:, None], position, segment_start_position)
                segment_start_step = torch.where(
                    is_arrived, torch.full_like(segment_start_step, step), segment_start_step
                )
            z_error = (position[:, 2] - target_before[:, 2]).abs()
            vertical_band_any = vertical_band_any | (
                is_alive_before & (z_error > task_config["termination_if_z_greater_than"])
            )
            hit_ground_any = hit_ground_any | (is_alive_before & hit_ground)
            left_workspace_any = left_workspace_any | (is_alive_before & left_workspace)
            waypoint_count += is_arrived
            task.waypoint_index = torch.where(
                is_arrived,
                torch.clamp(task.waypoint_index + 1, max=max_waypoint_index),
                task.waypoint_index,
            )
            task.command_buf[:] = task.waypoint_sequences[envs_idx, task.waypoint_index]
            first_arrival_step = torch.where(
                is_arrived & (first_arrival_step == max_steps),
                torch.full_like(first_arrival_step, step),
                first_arrival_step,
            )
            crash_step = torch.where(is_crashed, torch.full_like(crash_step, step), crash_step)
            position_error_sum += distance * is_alive_before
            position_error_steps += is_alive_before
            is_alive = is_alive_before & ~is_crashed
            previous_position = position.detach().clone()
            last_action = action.detach()
            task._update_obs()

    survival_steps = torch.where(
        crash_step <= max_steps,
        crash_step,
        torch.full_like(crash_step, max_steps),
    )
    safe_alive = alive_steps.clamp_min(1.0)
    safe_segments = completed_segments.clamp_min(1.0)
    path_efficiency = straight_line_distance / (completed_path_length + 1e-6)
    excess_path_ratio = completed_path_length / (straight_line_distance + 1e-6)
    no_segment = completed_segments == 0
    path_efficiency = torch.where(no_segment, torch.ones_like(path_efficiency), path_efficiency)
    excess_path_ratio = torch.where(no_segment, torch.ones_like(excess_path_ratio), excess_path_ratio)
    p95_speed = torch.zeros(num_scenarios, device=gs.device)
    p95_closing = torch.zeros(num_scenarios, device=gs.device)
    for env_index in range(num_scenarios):
        mask = history_mask[env_index]
        if mask.any():
            p95_speed[env_index] = torch.quantile(speed_history[env_index, mask].float(), 0.95)
            p95_closing[env_index] = torch.quantile(closing_history[env_index, mask].float(), 0.95)
    return {
        "first_arrived": waypoint_count > 0,
        "waypoint_count": waypoint_count,
        "first_arrival_time": first_arrival_step * dt,
        "crashed": crash_step <= max_steps,
        "survival_time": survival_steps * dt,
        "mean_position_error": position_error_sum / position_error_steps.clamp_min(1.0),
        "hit_ground": hit_ground_any,
        "left_workspace": left_workspace_any,
        "vertical_band_violation": vertical_band_any,
        "mean_inter_waypoint_time": inter_waypoint_time_sum / safe_segments,
        "mean_speed": speed_sum / safe_alive,
        "p95_speed": p95_speed,
        "max_speed": max_speed,
        "mean_closing_velocity": closing_sum / safe_alive,
        "p95_closing_velocity": p95_closing,
        "path_length": path_length,
        "straight_line_distance": straight_line_distance,
        "path_efficiency": path_efficiency,
        "excess_path_ratio": excess_path_ratio,
        "action_total_variation": action_total_variation,
        "completed_segments": completed_segments,
    }


def evaluate_ppo(
    scenarios: TrackScenarios,
    checkpoint_path: Path,
    environment_config: dict,
    flight_config: dict,
    task_config: dict,
    train_config: dict,
    task: ScoringTrackTask | None = None,
) -> dict[str, torch.Tensor]:
    if task is None:
        _, task = make_scoring_task(scenarios, environment_config, flight_config, task_config, train_config)
    runner = OnPolicyRunner(task, train_config, "", device="cuda:0")
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device="cuda:0")
    pin_rotor_task(task, scenarios)

    def choose_action(task, is_alive, _last_action):
        action = policy(task.get_observations())
        return action

    return score_rotor_rollout(task, choose_action, task_config, environment_config["dt"])


def evaluate_common_diff_policy(
    task: ScoringTrackTask,
    agent: ApgAgent | ShacAgent,
    normalizer: RunningNormalizer,
    scenarios: TrackScenarios,
    env_config,
    task_config: dict,
    dt: float,
) -> dict[str, torch.Tensor]:
    pin_rotor_task(task, scenarios)

    def choose_action(task, is_alive, last_action):
        odom = task.genesis_env.drone.odom
        observation = make_diff_observation(
            odom.world_pos,
            task.command_buf,
            odom.body_quat,
            odom.world_linear_vel,
            odom.body_ang_vel,
            last_action,
            is_alive,
            env_config,
        )
        return agent.action(observation, normalizer, deterministic=True)

    return score_rotor_rollout(task, choose_action, task_config, dt)


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
        diff_environment, DiffRLPolicyAdapter(apg_initial_agent, apg_initial_normalizer), scenarios
    )
    shac_initial_metrics = evaluate_diff_policy(
        diff_environment, DiffRLPolicyAdapter(shac_initial_agent, shac_initial_normalizer), scenarios
    )
    apg_metrics = evaluate_diff_policy(diff_environment, DiffRLPolicyAdapter(apg_agent, apg_normalizer), scenarios)
    shac_metrics = evaluate_diff_policy(diff_environment, DiffRLPolicyAdapter(shac_agent, shac_normalizer), scenarios)

    with (PROJECT_ROOT / "config" / "track_rl" / "genesis_env.yaml").open() as file:
        environment_config = yaml.safe_load(file)
    with (PROJECT_ROOT / "config" / "track_rl" / "flight.yaml").open() as file:
        flight_config = yaml.safe_load(file)
    with (PROJECT_ROOT / "config" / "track_rl" / "rl_env.yaml").open() as file:
        rl_config = yaml.safe_load(file)
    _, rotor_task = make_scoring_task(
        scenarios,
        environment_config,
        flight_config,
        rl_config["task"],
        rl_config["train"],
    )
    ppo_metrics = evaluate_ppo(
        scenarios,
        args.ppo_checkpoint,
        environment_config,
        flight_config,
        rl_config["task"],
        rl_config["train"],
        task=rotor_task,
    )
    common_apg_metrics = None
    common_shac_metrics = None
    if not args.skip_common_plant:
        common_apg_metrics = evaluate_common_diff_policy(
            rotor_task,
            apg_agent,
            apg_normalizer,
            scenarios,
            settings.environment,
            rl_config["task"],
            settings.environment.dt,
        )
        common_shac_metrics = evaluate_common_diff_policy(
            rotor_task,
            shac_agent,
            shac_normalizer,
            scenarios,
            settings.environment,
            rl_config["task"],
            settings.environment.dt,
        )

    episode_seconds = settings.environment.max_episode_steps * settings.environment.dt
    summaries = {
        "apg_untrained": summarize_metrics(apg_initial_metrics, episode_seconds),
        "shac_untrained": summarize_metrics(shac_initial_metrics, episode_seconds),
        "apg_native": summarize_metrics(apg_metrics, episode_seconds),
        "shac_native": summarize_metrics(shac_metrics, episode_seconds),
        "apg": summarize_metrics(apg_metrics, episode_seconds),
        "shac": summarize_metrics(shac_metrics, episode_seconds),
        "ppo": summarize_metrics(ppo_metrics, episode_seconds),
        "ppo_native": summarize_metrics(ppo_metrics, episode_seconds),
        "ppo_common": summarize_metrics(ppo_metrics, episode_seconds),
    }
    if common_apg_metrics is not None:
        summaries["apg_common"] = summarize_metrics(common_apg_metrics, episode_seconds)
        summaries["shac_common"] = summarize_metrics(common_shac_metrics, episode_seconds)
    apg_success = success_against_ppo(summaries["apg_native"], summaries["ppo"])
    shac_success = success_against_ppo(summaries["shac_native"], summaries["ppo"])
    result = {
        "summaries": {name: summary.to_dict() for name, summary in summaries.items()},
        "paired_difference_from_ppo": {
            "apg_native": {
                name: asdict(value)
                for name, value in paired_differences(apg_metrics, ppo_metrics, episode_seconds).items()
            },
            "shac_native": {
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
            "note": "boolean success is diagnostic only; use the performance table",
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
        "comparison_scope": {
            "native": "APG/SHAC on wrench plant; PPO on rotor plant",
            "common_plant": "APG/SHAC/PPO on Genesis_env + angle PID + set_propellers_rpm",
            "observation": "each policy keeps its training-time observation preprocessing",
        },
        "eval_protocol": {
            "one_life": True,
            "no_teleport_on_fail": True,
            "arrival": "distance to current waypoint < 0.1 m, then switch to the next fixed waypoint",
            "crash": "height < 0.1 m, or |x/y error| > 5 m, or NaN. |z error| > 1.2 m is recorded as vertical_band_violation, not a crash",
        },
        "ppo_event_rates": {
            "hit_ground": ppo_metrics["hit_ground"].float().mean().item(),
            "left_workspace": ppo_metrics["left_workspace"].float().mean().item(),
            "vertical_band_violation": ppo_metrics["vertical_band_violation"].float().mean().item(),
        },
    }
    if common_apg_metrics is not None:
        result["paired_difference_from_ppo"]["apg_common"] = {
            name: asdict(value)
            for name, value in paired_differences(common_apg_metrics, ppo_metrics, episode_seconds).items()
        }
        result["paired_difference_from_ppo"]["shac_common"] = {
            name: asdict(value)
            for name, value in paired_differences(common_shac_metrics, ppo_metrics, episode_seconds).items()
        }
    per_scenario = {
        "apg_untrained": apg_initial_metrics,
        "shac_untrained": shac_initial_metrics,
        "apg_native": apg_metrics,
        "shac_native": shac_metrics,
        "apg": apg_metrics,
        "shac": shac_metrics,
        "ppo": ppo_metrics,
        "ppo_common": ppo_metrics,
    }
    if common_apg_metrics is not None:
        per_scenario["apg_common"] = common_apg_metrics
        per_scenario["shac_common"] = common_shac_metrics
    torch.save(per_scenario, args.output_dir / "per_scenario_metrics.pt")
    with (args.output_dir / "evaluation.json").open("w") as file:
        json.dump(result, file, indent=2)
    print(
        json.dumps(
            {
                "success": result["success"],
                "eval_protocol": result["eval_protocol"],
                "comparison_scope": result["comparison_scope"],
                "ppo_event_rates": result["ppo_event_rates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
