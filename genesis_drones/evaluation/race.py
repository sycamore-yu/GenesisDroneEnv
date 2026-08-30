import json
from dataclasses import dataclass
from pathlib import Path

import torch

from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
from genesis_drones.tasks.racing_core import (
    EvaluationInitialStates,
    generate_initial_states,
)


RACING_CONTRACT = {
    "track": "fixed_seven",
    "action": "ctbr",
    "policy_observation_size": 40,
    "critic_observation_size": 41,
    "dt": 0.01,
    "max_episode_steps": 3000,
    "controller": "ctbr",
    "gamma": 0.999,
    "td_lambda": 0.95,
}


def assert_shared_racing_contract(config: RaceEnvConfig) -> None:
    assert config.dt == RACING_CONTRACT["dt"]
    assert config.max_episode_steps == RACING_CONTRACT["max_episode_steps"]
    assert config.gamma == RACING_CONTRACT["gamma"]
    assert config.td_lambda == RACING_CONTRACT["td_lambda"]


@dataclass
class RaceEpisodeResult:
    completed: bool
    analytic_collision: bool
    physics_collision: bool
    failed: bool
    completion_time: float
    gate_times: list[float]
    mean_speed: float
    path_efficiency: float


def make_evaluation_states(environment: RaceEnv, count: int = 100, seed: int = 20250830) -> EvaluationInitialStates:
    return generate_initial_states(environment.track, count, seed=seed)


def evaluate_policy(
    environment: RaceEnv,
    action_fn,
    states: EvaluationInitialStates,
) -> list[RaceEpisodeResult]:
    environment.respawn_on_fail = False
    results = []
    for index in range(states.position.shape[0]):
        one = EvaluationInitialStates(
            states.position[index : index + 1],
            states.quaternion[index : index + 1],
            states.linear_velocity[index : index + 1],
        )
        policy_observation, _ = environment.reset(initial_states=one)
        analytic_collision = False
        physics_collision = False
        completed = False
        for _ in range(environment.config.max_episode_steps):
            action = action_fn(policy_observation)
            policy_observation, _, done, extras = environment.step(action)
            analytic_collision = analytic_collision or bool(extras["analytic_collision"][0])
            physics_collision = physics_collision or bool(extras["physics_collision"][0])
            completed = completed or bool(extras["completed"][0])
            if bool(done[0]):
                break
        steps = int(environment.episode_length_buf[0].item())
        duration = max(steps, 1) * environment.config.dt
        path_length = float(environment.path_length[0].item())
        gate_times = environment.gate_pass_time[0].detach().cpu().tolist()
        results.append(
            RaceEpisodeResult(
                completed=completed,
                analytic_collision=analytic_collision,
                physics_collision=physics_collision,
                failed=analytic_collision or physics_collision or not completed,
                completion_time=duration,
                gate_times=gate_times,
                mean_speed=path_length / duration,
                path_efficiency=0.0 if path_length == 0.0 else 1.0 / path_length,
            )
        )
    return results


def summarize_race_results(results: list[RaceEpisodeResult]) -> dict:
    count = max(len(results), 1)
    success = sum(result.completed and not result.failed for result in results) / count
    analytic = sum(result.analytic_collision for result in results) / count
    physics = sum(result.physics_collision for result in results) / count
    collision = sum(result.analytic_collision or result.physics_collision for result in results) / count
    completed = [result for result in results if result.completed]
    gate_count = len(results[0].gate_times) if results else 0
    mean_gate_times = [
        sum(result.gate_times[index] for result in completed) / max(len(completed), 1) for index in range(gate_count)
    ]
    return {
        "success_rate": success,
        "collision_rate": collision,
        "analytic_collision_rate": analytic,
        "physics_collision_rate": physics,
        "mean_completion_time": sum(result.completion_time for result in completed) / max(len(completed), 1)
        if completed
        else 0.0,
        "mean_gate_times": mean_gate_times,
        "mean_speed": sum(result.mean_speed for result in results) / count,
        "mean_path_efficiency": sum(result.path_efficiency for result in results) / count,
        "count": count,
    }


def save_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
