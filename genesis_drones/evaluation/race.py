import json
from dataclasses import dataclass
from pathlib import Path

import torch

from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
from genesis_drones.tasks.racing_core import (
    CRITIC_OBSERVATION_SIZE,
    POLICY_OBSERVATION_SIZE,
    EvaluationInitialStates,
    generate_initial_states,
)


RACING_CONTRACT = {
    "track": "diffaero_racing",
    "action": "ctbr",
    "policy_observation_size": POLICY_OBSERVATION_SIZE,
    "critic_observation_size": CRITIC_OBSERVATION_SIZE,
    "dt": 0.0333,
    "max_episode_steps": int(40.0 / 0.0333),
    "controller": "ctbr",
    "gamma": 0.99,
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
    failed: bool
    completion_time: float
    gates_passed: int
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
            states.target_gate[index : index + 1],
        )
        policy_observation, _ = environment.reset(initial_states=one)
        analytic_collision = False
        extras = None
        for _ in range(environment.config.max_episode_steps + 1):
            action = action_fn(policy_observation)
            policy_observation, _, done, extras = environment.step(action)
            analytic_collision = analytic_collision or bool(extras["analytic_collision"][0])
            if bool(done[0]):
                break
        steps = int(environment.episode_length_buf[0].item())
        duration = max(steps, 1) * environment.config.dt
        path_length = float(environment.path_length[0].item())
        gates_passed = int(environment.n_passed_gates[0].item())
        completed = bool(extras["success"][0])
        failed = analytic_collision
        results.append(
            RaceEpisodeResult(
                completed=completed,
                analytic_collision=analytic_collision,
                failed=failed,
                completion_time=duration,
                gates_passed=gates_passed,
                mean_speed=path_length / duration,
                path_efficiency=0.0 if path_length == 0.0 else gates_passed / path_length,
            )
        )
    return results


def summarize_race_results(results: list[RaceEpisodeResult]) -> dict:
    count = max(len(results), 1)
    success = sum(result.completed for result in results) / count
    analytic = sum(result.analytic_collision for result in results) / count
    collision = analytic
    completed = [result for result in results if result.completed]
    return {
        "success_rate": success,
        "collision_rate": collision,
        "analytic_collision_rate": analytic,
        "mean_completion_time": sum(result.completion_time for result in completed) / max(len(completed), 1)
        if completed
        else 0.0,
        "mean_gates_passed": sum(result.gates_passed for result in results) / count,
        "mean_speed": sum(result.mean_speed for result in results) / count,
        "mean_path_efficiency": sum(result.path_efficiency for result in results) / count,
        "count": count,
    }


def save_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
