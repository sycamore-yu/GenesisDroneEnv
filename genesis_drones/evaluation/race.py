import json
from dataclasses import dataclass
from pathlib import Path

import torch

from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
from genesis_drones.tasks.racing_core import (
    POLICY_OBSERVATION_SIZE,
    STATE_OBSERVATION_SIZE,
    EvaluationInitialStates,
    generate_initial_states,
)


EXPERIMENT_AXES = ("task", "dynamics", "algorithm", "network", "sensor", "seed")
RACING_CONTRACT = {
    "track": "diffaero_racing",
    "action": "normalized_4d",
    "policy_observation_size": POLICY_OBSERVATION_SIZE,
    "critic_observation_size": POLICY_OBSERVATION_SIZE,
    "state_observation_size": STATE_OBSERVATION_SIZE,
    "dt": 0.0333,
    "max_episode_steps": int(40.0 / 0.0333),
    "gamma": 0.99,
    "td_lambda": 0.95,
}


def assert_shared_racing_contract(config: RaceEnvConfig) -> None:
    assert config.dt == RACING_CONTRACT["dt"]
    assert config.max_episode_steps == RACING_CONTRACT["max_episode_steps"]
    assert config.gamma == RACING_CONTRACT["gamma"]
    assert config.td_lambda == RACING_CONTRACT["td_lambda"]


def racing_experiment(algorithm: str, dynamics: str, seed: int) -> dict:
    return {
        "task": "racing",
        "dynamics": dynamics,
        "algorithm": algorithm,
        "network": "mlp",
        "sensor": "relative_position",
        "seed": seed,
        **RACING_CONTRACT,
    }


def write_experiment(path: Path, experiment: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(experiment, indent=2))


def recorded_experiment(checkpoint_path: Path | None, payload: dict | None = None) -> dict:
    recorded = {}
    if checkpoint_path is not None:
        for name in ("experiment.json", "contract.json"):
            candidate = checkpoint_path.parent / name
            if candidate.exists():
                recorded.update(json.loads(candidate.read_text()))
                break
    if payload:
        for key in EXPERIMENT_AXES:
            if key in payload:
                recorded[key] = payload[key]
    return recorded


def resolve_experiment(
    algorithm: str,
    dynamics: str | None,
    recorded: dict,
    fallback_dynamics: str,
) -> dict:
    recorded_algorithm = recorded.get("algorithm")
    recorded_dynamics = recorded.get("dynamics")
    if recorded_algorithm is not None and recorded_algorithm != algorithm:
        raise ValueError(f"checkpoint algorithm {recorded_algorithm!r} does not match --algo {algorithm!r}")
    if dynamics is not None and recorded_dynamics is not None and dynamics != recorded_dynamics:
        raise ValueError(f"checkpoint dynamics {recorded_dynamics!r} does not match --dynamics {dynamics!r}")
    return {
        "task": recorded.get("task", "racing"),
        "dynamics": dynamics or recorded_dynamics or fallback_dynamics,
        "algorithm": algorithm,
        "network": recorded.get("network", "mlp"),
        "sensor": recorded.get("sensor", "relative_position"),
        "seed": recorded.get("seed"),
    }


def apply_network_hidden_sizes(train_config: dict, hidden_sizes) -> None:
    sizes = list(hidden_sizes)
    train_config["policy"]["actor_hidden_dims"] = sizes
    train_config["policy"]["critic_hidden_dims"] = sizes


@dataclass
class RaceEpisodeResult:
    completed: bool
    analytic_collision: bool
    failed: bool
    completion_time: float
    gates_passed: int
    mean_speed: float
    path_efficiency: float


def make_evaluation_states(environment, count: int = 100, seed: int = 20250830) -> EvaluationInitialStates:
    track = getattr(environment, "track", None)
    if track is None:
        track = environment.task.track
    return generate_initial_states(track, count, seed=seed)


def evaluate_policy(
    environment,
    action_fn,
    states: EvaluationInitialStates,
) -> list[RaceEpisodeResult]:
    environment.respawn_on_fail = False
    core = getattr(environment, "env", environment)
    config = getattr(environment, "config", None)
    max_steps = int(getattr(config, "max_episode_steps", core.task.config.max_episode_steps))
    dt = float(getattr(config, "dt", core.dt))
    results = []
    for index in range(states.position.shape[0]):
        one = EvaluationInitialStates(
            states.position[index : index + 1],
            states.quaternion[index : index + 1],
            states.linear_velocity[index : index + 1],
            states.target_gate[index : index + 1],
        )
        reset_out = environment.reset(initial_states=one)
        policy_observation = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        analytic_collision = False
        extras = None
        done_flag = False
        for _ in range(max_steps + 1):
            action = action_fn(policy_observation)
            step_out = environment.step(action)
            if hasattr(step_out, "observation"):
                policy_observation = step_out.observation.policy
                done_flag = bool(step_out.done[0])
                extras = step_out.extras
                analytic_collision = analytic_collision or bool(extras.get("analytic_collision", extras.get("terminated"))[0])
            else:
                policy_observation, _, done, extras = step_out
                done_flag = bool(done[0])
                analytic_collision = analytic_collision or bool(extras["analytic_collision"][0])
            if done_flag:
                break
        task = getattr(environment, "task", None) or core.task
        steps = int(task.episode_length_buf[0].item())
        duration = max(steps, 1) * dt
        path_length = float(task.path_length[0].item())
        gates_passed = int(task.n_passed_gates[0].item())
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


def evaluate_rolling(environment: RaceEnv, action_fn, n_steps: int) -> dict:
    environment.respawn_on_fail = True
    policy_observation, _ = environment.reset(seed=0)
    n_resets = 0
    sums = {key: 0.0 for key in ("passed", "success", "survive", "collision", "duration", "returns")}
    step_reward_sum = 0.0
    episode_return = torch.zeros(environment.num_envs, device="cpu")
    dt = environment.config.dt
    with torch.inference_mode():
        for _ in range(n_steps):
            action = action_fn(policy_observation)
            policy_observation, (_, _, reward), done, extras = environment.step(action)
            reward = reward.detach().float().cpu()
            step_reward_sum += float(reward.mean().item())
            episode_return = episode_return + reward
            if not done.any():
                continue
            reset = done.detach().cpu().bool()
            n = int(reset.sum().item())
            n_resets += n
            sums["passed"] += float(extras["n_passed_gates"].detach().cpu()[reset].float().sum().item())
            sums["success"] += float(extras["success"].detach().cpu()[reset].float().sum().item())
            sums["survive"] += float(extras["truncated"].detach().cpu()[reset].float().sum().item())
            sums["collision"] += float(extras["terminated"].detach().cpu()[reset].float().sum().item())
            sums["duration"] += float(((extras["episode_length"].detach().cpu()[reset] - 1).float() * dt).sum().item())
            sums["returns"] += float(episode_return[reset].sum().item())
            episode_return[reset] = 0.0

    scale = max(n_resets, 1)
    return {
        "success_rate": sums["success"] / scale,
        "survive_rate": sums["survive"] / scale,
        "collision_rate": sums["collision"] / scale,
        "mean_passed_gates": sums["passed"] / scale,
        "mean_episode_seconds": sums["duration"] / scale,
        "mean_episode_return": sums["returns"] / scale,
        "mean_step_reward": step_reward_sum / max(n_steps, 1),
        "n_steps": n_steps,
        "n_envs": environment.num_envs,
        "n_resets": n_resets,
    }


def save_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
