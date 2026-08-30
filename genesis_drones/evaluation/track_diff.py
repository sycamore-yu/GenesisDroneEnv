from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from genesis_drones.algorithms.diff_rl import ApgAgent, RunningNormalizer, ShacAgent
from genesis_drones.envs.track_diff_env import TrackDiffEnv, TrackDiffEnvConfig


@dataclass(frozen=True)
class TrackScenarios:
    initial_position: torch.Tensor
    initial_quaternion: torch.Tensor
    waypoint_sequences: torch.Tensor

    @classmethod
    def generate(
        cls,
        num_scenarios: int,
        max_episode_steps: int,
        config: TrackDiffEnvConfig,
        seed: int,
    ) -> "TrackScenarios":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial_position = torch.empty((num_scenarios, 3), device="cpu")
        initial_position[:, 0].uniform_(*config.initial_x_range, generator=generator)
        initial_position[:, 1].uniform_(*config.initial_y_range, generator=generator)
        initial_position[:, 2].uniform_(*config.initial_z_range, generator=generator)
        yaw = torch.empty(num_scenarios, device="cpu").uniform_(-torch.pi, torch.pi, generator=generator)
        initial_quaternion = torch.stack(
            (torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw)), dim=-1
        )
        waypoint_sequences = torch.empty((num_scenarios, max_episode_steps + 1, 3), device="cpu")
        waypoint_sequences[..., 0].uniform_(*config.target_x_range, generator=generator)
        waypoint_sequences[..., 1].uniform_(*config.target_y_range, generator=generator)
        waypoint_sequences[..., 2].uniform_(*config.target_z_range, generator=generator)
        return cls(initial_position, initial_quaternion, waypoint_sequences)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "initial_position": self.initial_position,
                "initial_quaternion": self.initial_quaternion,
                "waypoint_sequences": self.waypoint_sequences,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TrackScenarios":
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(state["initial_position"], state["initial_quaternion"], state["waypoint_sequences"])


@dataclass(frozen=True)
class ScalarSummary:
    mean: float
    standard_deviation: float
    confidence_interval_low: float
    confidence_interval_high: float


@dataclass(frozen=True)
class EvaluationSummary:
    first_arrival_rate: ScalarSummary
    waypoint_count: ScalarSummary
    waypoints_per_minute: ScalarSummary
    capped_first_arrival_time: ScalarSummary
    successful_first_arrival_time: ScalarSummary
    successful_arrival_count: int
    crash_rate: ScalarSummary
    survival_time: ScalarSummary
    mean_position_error: ScalarSummary
    mean_inter_waypoint_time: ScalarSummary
    median_inter_waypoint_time: ScalarSummary
    mean_speed: ScalarSummary
    p95_speed: ScalarSummary
    max_speed: ScalarSummary
    mean_closing_velocity: ScalarSummary
    p95_closing_velocity: ScalarSummary
    path_length: ScalarSummary
    straight_line_distance: ScalarSummary
    path_efficiency: ScalarSummary
    excess_path_ratio: ScalarSummary
    action_total_variation: ScalarSummary

    def to_dict(self) -> dict:
        return asdict(self)


def classify_eval_step(
    is_alive: torch.Tensor,
    position: torch.Tensor,
    target: torch.Tensor,
    has_nan: torch.Tensor,
    target_threshold: float,
    ground_height: float,
    horizontal_limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-life scoring: arrive if close; die only on ground, NaN, or XY workspace exit.

    |z error| is not a crash. PPO training resets on that; the viewer teleports, so it
    looks like the drone kept flying.
    """
    position_error = target - position
    distance = torch.linalg.vector_norm(position_error, dim=-1)
    hit_ground = position[:, 2] < ground_height
    left_workspace = (position_error[:, 0].abs() > horizontal_limit) | (
        position_error[:, 1].abs() > horizontal_limit
    )
    is_crashed = is_alive & (hit_ground | left_workspace | has_nan)
    is_arrived = is_alive & ~is_crashed & (distance < target_threshold)
    return is_arrived, is_crashed, distance, hit_ground, left_workspace


def evaluate_diff_policy(
    environment: TrackDiffEnv,
    agent: ApgAgent | ShacAgent,
    normalizer: RunningNormalizer,
    scenarios: TrackScenarios,
) -> dict[str, torch.Tensor]:
    previous_vertical_rule = getattr(environment, "end_on_vertical_error", True)
    previous_respawn = getattr(environment, "respawn_on_fail", True)
    environment.end_on_vertical_error = False
    environment.respawn_on_fail = False
    observation = environment.reset(
        scenarios.initial_position,
        scenarios.initial_quaternion,
        scenarios.waypoint_sequences,
    )
    try:
        num_envs = environment.num_envs
        device = environment.device
        dt = environment.config.dt
        max_steps = environment.config.max_episode_steps
        state = environment._read_state()
        previous_position = state.position.detach().clone()
        segment_start_position = previous_position.clone()
        previous_action = torch.zeros((num_envs, environment.action_dim), device=device)
        path_length = torch.zeros(num_envs, device=device)
        segment_path_length = torch.zeros(num_envs, device=device)
        completed_path_length = torch.zeros(num_envs, device=device)
        straight_line_distance = torch.zeros(num_envs, device=device)
        completed_segments = torch.zeros(num_envs, device=device)
        inter_waypoint_time_sum = torch.zeros(num_envs, device=device)
        segment_start_step = torch.zeros(num_envs, device=device, dtype=torch.int64)
        action_total_variation = torch.zeros(num_envs, device=device)
        speed_sum = torch.zeros(num_envs, device=device)
        closing_sum = torch.zeros(num_envs, device=device)
        alive_steps = torch.zeros(num_envs, device=device)
        max_speed = torch.zeros(num_envs, device=device)
        # ponytail: store all alive-step speeds for p95; 1024×1500×4B ≈ 6MB
        speed_history = torch.zeros((num_envs, max_steps), device=device)
        closing_history = torch.zeros((num_envs, max_steps), device=device)
        history_mask = torch.zeros((num_envs, max_steps), device=device, dtype=torch.bool)

        with torch.no_grad():
            for step in range(max_steps):
                action = agent.action(observation, normalizer, deterministic=True)
                target_before = environment.target_position.detach().clone()
                is_alive_before = environment.is_alive.clone()
                observation, _, _, extras = environment.step(action)
                state = environment._read_state()
                position = state.position.detach()
                velocity = state.linear_velocity.detach()
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
                speed_history[:, step] = speed
                closing_history[:, step] = closing
                history_mask[:, step] = is_alive_before
                action_total_variation = (
                    action_total_variation
                    + torch.linalg.vector_norm(action - previous_action, dim=-1) * is_alive_before
                )
                arrived = extras["arrived"]
                if arrived.any():
                    segment_straight = torch.linalg.vector_norm(target_before - segment_start_position, dim=-1)
                    segment_time = (step + 1 - segment_start_step).to(dtype=inter_waypoint_time_sum.dtype) * dt
                    straight_line_distance = straight_line_distance + segment_straight * arrived
                    completed_path_length = completed_path_length + segment_path_length * arrived
                    inter_waypoint_time_sum = inter_waypoint_time_sum + segment_time * arrived
                    completed_segments = completed_segments + arrived.to(dtype=completed_segments.dtype)
                    segment_path_length = torch.where(arrived, torch.zeros_like(segment_path_length), segment_path_length)
                    segment_start_position = torch.where(arrived[:, None], position, segment_start_position)
                    segment_start_step = torch.where(
                        arrived, torch.full_like(segment_start_step, step + 1), segment_start_step
                    )
                previous_position = position
                previous_action = action.detach()

        base = {name: value.detach().clone() for name, value in environment.episode_metrics().items()}
        safe_alive = alive_steps.clamp_min(1.0)
        safe_segments = completed_segments.clamp_min(1.0)
        path_efficiency = straight_line_distance / (completed_path_length + 1e-6)
        excess_path_ratio = completed_path_length / (straight_line_distance + 1e-6)
        no_segment = completed_segments == 0
        path_efficiency = torch.where(no_segment, torch.ones_like(path_efficiency), path_efficiency)
        excess_path_ratio = torch.where(no_segment, torch.ones_like(excess_path_ratio), excess_path_ratio)

        p95_speed = torch.zeros(num_envs, device=device)
        p95_closing = torch.zeros(num_envs, device=device)
        for env_index in range(num_envs):
            mask = history_mask[env_index]
            if mask.any():
                p95_speed[env_index] = torch.quantile(speed_history[env_index, mask].float(), 0.95)
                p95_closing[env_index] = torch.quantile(closing_history[env_index, mask].float(), 0.95)

        base.update(
            {
                "mean_inter_waypoint_time": (inter_waypoint_time_sum / safe_segments).detach(),
                "mean_speed": (speed_sum / safe_alive).detach(),
                "p95_speed": p95_speed.detach(),
                "max_speed": max_speed.detach(),
                "mean_closing_velocity": (closing_sum / safe_alive).detach(),
                "p95_closing_velocity": p95_closing.detach(),
                "path_length": path_length.detach(),
                "straight_line_distance": straight_line_distance.detach(),
                "path_efficiency": path_efficiency.detach(),
                "excess_path_ratio": excess_path_ratio.detach(),
                "action_total_variation": action_total_variation.detach(),
                "completed_segments": completed_segments.detach(),
            }
        )
        return base
    finally:
        environment.end_on_vertical_error = previous_vertical_rule
        environment.respawn_on_fail = previous_respawn
        # RigidSolver.get_state() caches every queried state, including in non-differentiable scenes.
        # Reset the scene after evaluation so the 1500-step validation cache does not stay resident between runs.
        environment.scene.reset()


def _summarize_values(
    values: torch.Tensor,
    bootstrap_indices: torch.Tensor,
) -> ScalarSummary:
    values = values.detach().to(device="cpu", dtype=torch.float64)
    bootstrap_means = values[bootstrap_indices].mean(dim=1)
    confidence_interval = torch.quantile(
        bootstrap_means, torch.tensor([0.025, 0.975], device="cpu", dtype=torch.float64)
    )
    return ScalarSummary(
        mean=values.mean().item(),
        standard_deviation=values.std(unbiased=False).item(),
        confidence_interval_low=confidence_interval[0].item(),
        confidence_interval_high=confidence_interval[1].item(),
    )


def _median_summary(values: torch.Tensor) -> ScalarSummary:
    values = values.detach().to(device="cpu", dtype=torch.float64)
    median = values.median().item()
    return ScalarSummary(median, values.std(unbiased=False).item(), median, median)


def summarize_metrics(
    metrics: dict[str, torch.Tensor],
    episode_seconds: float,
    bootstrap_seed: int = 20250830,
    bootstrap_samples: int = 10000,
) -> EvaluationSummary:
    num_scenarios = metrics["waypoint_count"].shape[0]
    generator = torch.Generator(device="cpu").manual_seed(bootstrap_seed)
    bootstrap_indices = torch.randint(
        num_scenarios,
        (bootstrap_samples, num_scenarios),
        generator=generator,
        device="cpu",
    )
    first_arrived = metrics["first_arrived"].to(dtype=torch.float32)
    waypoint_count = metrics["waypoint_count"].to(dtype=torch.float32)
    crashed = metrics["crashed"].to(dtype=torch.float32)
    successful_arrival_times = metrics["first_arrival_time"][metrics["first_arrived"]]
    if successful_arrival_times.shape[0] == 0:
        successful_arrival_summary = ScalarSummary(float("nan"), float("nan"), float("nan"), float("nan"))
    else:
        successful_indices = torch.randint(
            successful_arrival_times.shape[0],
            (bootstrap_samples, successful_arrival_times.shape[0]),
            generator=generator,
            device="cpu",
        )
        successful_arrival_summary = _summarize_values(successful_arrival_times, successful_indices)
    return EvaluationSummary(
        first_arrival_rate=_summarize_values(first_arrived, bootstrap_indices),
        waypoint_count=_summarize_values(waypoint_count, bootstrap_indices),
        waypoints_per_minute=_summarize_values(waypoint_count * (60.0 / episode_seconds), bootstrap_indices),
        capped_first_arrival_time=_summarize_values(metrics["first_arrival_time"], bootstrap_indices),
        successful_first_arrival_time=successful_arrival_summary,
        successful_arrival_count=successful_arrival_times.shape[0],
        crash_rate=_summarize_values(crashed, bootstrap_indices),
        survival_time=_summarize_values(metrics["survival_time"], bootstrap_indices),
        mean_position_error=_summarize_values(metrics["mean_position_error"], bootstrap_indices),
        mean_inter_waypoint_time=_summarize_values(metrics["mean_inter_waypoint_time"], bootstrap_indices),
        median_inter_waypoint_time=_median_summary(metrics["mean_inter_waypoint_time"]),
        mean_speed=_summarize_values(metrics["mean_speed"], bootstrap_indices),
        p95_speed=_summarize_values(metrics["p95_speed"], bootstrap_indices),
        max_speed=_summarize_values(metrics["max_speed"], bootstrap_indices),
        mean_closing_velocity=_summarize_values(metrics["mean_closing_velocity"], bootstrap_indices),
        p95_closing_velocity=_summarize_values(metrics["p95_closing_velocity"], bootstrap_indices),
        path_length=_summarize_values(metrics["path_length"], bootstrap_indices),
        straight_line_distance=_summarize_values(metrics["straight_line_distance"], bootstrap_indices),
        path_efficiency=_summarize_values(metrics["path_efficiency"], bootstrap_indices),
        excess_path_ratio=_summarize_values(metrics["excess_path_ratio"], bootstrap_indices),
        action_total_variation=_summarize_values(metrics["action_total_variation"], bootstrap_indices),
    )


def paired_differences(
    metrics: dict[str, torch.Tensor],
    baseline_metrics: dict[str, torch.Tensor],
    episode_seconds: float,
    bootstrap_seed: int = 20250831,
    bootstrap_samples: int = 10000,
) -> dict[str, ScalarSummary]:
    num_scenarios = metrics["waypoint_count"].shape[0]
    generator = torch.Generator(device="cpu").manual_seed(bootstrap_seed)
    bootstrap_indices = torch.randint(
        num_scenarios,
        (bootstrap_samples, num_scenarios),
        generator=generator,
        device="cpu",
    )
    differences = {
        "first_arrival_rate": metrics["first_arrived"].float() - baseline_metrics["first_arrived"].float(),
        "waypoint_count": metrics["waypoint_count"].float() - baseline_metrics["waypoint_count"].float(),
        "waypoints_per_minute": (metrics["waypoint_count"].float() - baseline_metrics["waypoint_count"].float())
        * (60.0 / episode_seconds),
        "capped_first_arrival_time": metrics["first_arrival_time"] - baseline_metrics["first_arrival_time"],
        "crash_rate": metrics["crashed"].float() - baseline_metrics["crashed"].float(),
        "survival_time": metrics["survival_time"] - baseline_metrics["survival_time"],
        "mean_position_error": metrics["mean_position_error"] - baseline_metrics["mean_position_error"],
        "mean_speed": metrics["mean_speed"] - baseline_metrics["mean_speed"],
        "mean_closing_velocity": metrics["mean_closing_velocity"] - baseline_metrics["mean_closing_velocity"],
        "path_efficiency": metrics["path_efficiency"] - baseline_metrics["path_efficiency"],
        "action_total_variation": metrics["action_total_variation"] - baseline_metrics["action_total_variation"],
    }
    return {name: _summarize_values(values, bootstrap_indices) for name, values in differences.items()}


def success_against_ppo(summary: EvaluationSummary, ppo_summary: EvaluationSummary) -> dict[str, bool]:
    checks = {
        "waypoint_count": summary.waypoint_count.mean >= 0.9 * ppo_summary.waypoint_count.mean,
        "first_arrival_rate": summary.first_arrival_rate.mean >= ppo_summary.first_arrival_rate.mean - 0.05,
        "crash_rate": summary.crash_rate.mean <= ppo_summary.crash_rate.mean + 0.05,
    }
    checks["success"] = all(checks.values())
    return checks
