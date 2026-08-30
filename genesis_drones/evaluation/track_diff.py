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
    environment.end_on_vertical_error = False
    observation = environment.reset(
        scenarios.initial_position,
        scenarios.initial_quaternion,
        scenarios.waypoint_sequences,
    )
    try:
        with torch.no_grad():
            for _ in range(environment.config.max_episode_steps):
                action = agent.action(observation, normalizer, deterministic=True)
                observation, _, _, _ = environment.step(action)
        return {name: value.detach().clone() for name, value in environment.episode_metrics().items()}
    finally:
        environment.end_on_vertical_error = previous_vertical_rule
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
    }
    return {name: _summarize_values(values, bootstrap_indices) for name, values in differences.items()}


def success_against_ppo(summary: EvaluationSummary, ppo_summary: EvaluationSummary) -> dict[str, bool]:
    checks = {
        "waypoint_count": summary.waypoint_count.mean >= 0.9 * ppo_summary.waypoint_count.mean,
        "first_arrival_rate": summary.first_arrival_rate.mean >= ppo_summary.first_arrival_rate.mean - 0.05,
        "crash_rate": summary.crash_rate.mean <= ppo_summary.crash_rate.mean + 0.05,
        "mean_position_error": summary.mean_position_error.mean <= 1.1 * ppo_summary.mean_position_error.mean,
    }
    checks["success"] = all(checks.values())
    return checks
