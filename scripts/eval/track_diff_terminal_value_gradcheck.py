"""Check that SHAC terminal value V(s_H) contributes nonzero gradients to state and early actions."""

import json
from dataclasses import replace
from pathlib import Path

import torch

import genesis as gs

from genesis_drones.algorithms.diff_rl import (
    Critic,
    NetworkConfig,
    RunningNormalizer,
    as_simulation_action,
    collect_simulation_action_gradients,
)
from genesis_drones.envs.track_diff_env import TrackDiffEnv
from genesis_drones.evaluation.track_diff import TrackScenarios
from genesis_drones.utils.track_diff_config import load_track_diff_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POSITION_SLICE = slice(0, 3)
LINEAR_VELOCITY_SLICE = slice(13, 16)
QUATERNION_SLICE = slice(9, 13)


def fixed_scenarios(horizon: int) -> TrackScenarios:
    return TrackScenarios(
        initial_position=torch.tensor([[0.0, 0.0, 0.6]], device="cpu"),
        initial_quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cpu"),
        waypoint_sequences=torch.tensor([[[0.6, 0.2, 0.8]]], device="cpu").expand(1, horizon + 1, 3).clone(),
    )


def assert_close(analytical: float, numerical: float, relative_tolerance: float, absolute_tolerance: float) -> None:
    error = abs(analytical - numerical)
    tolerance = max(absolute_tolerance, relative_tolerance * abs(numerical))
    if error > tolerance:
        raise AssertionError(
            f"gradient mismatch: analytical={analytical:.8f} numerical={numerical:.8f} "
            f"error={error:.8f} tolerance={tolerance:.8f}"
        )


def check_terminal_value_state_gradient(critic: Critic, observation_size: int, device: torch.device) -> dict:
    observation = torch.randn(1, observation_size, device=device, requires_grad=True)
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    value = critic(observation)
    value.sum().backward()
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    gradient = observation.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    position_norm = torch.linalg.vector_norm(gradient[0, POSITION_SLICE]).item()
    velocity_norm = torch.linalg.vector_norm(gradient[0, LINEAR_VELOCITY_SLICE]).item()
    attitude_norm = torch.linalg.vector_norm(gradient[0, QUATERNION_SLICE]).item()
    if position_norm == 0.0 and velocity_norm == 0.0 and attitude_norm == 0.0:
        raise AssertionError("∂V/∂s_H is all zeros on position/velocity/attitude")
    return {
        "position_grad_norm": position_norm,
        "linear_velocity_grad_norm": velocity_norm,
        "quaternion_grad_norm": attitude_norm,
        "full_grad_norm": torch.linalg.vector_norm(gradient).item(),
    }


def rollout_terminal_value(
    environment: TrackDiffEnv,
    critic: Critic,
    normalizer: RunningNormalizer,
    actions: list[torch.Tensor],
    scenarios: TrackScenarios,
    gamma: float,
) -> torch.Tensor:
    observation = environment.reset(
        scenarios.initial_position,
        scenarios.initial_quaternion,
        scenarios.waypoint_sequences,
    )
    discount = observation.new_tensor(1.0)
    for action in actions:
        observation, _, _, _ = environment.step(action)
        discount = discount * gamma
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    terminal_value = critic(normalizer(observation)) * environment.is_alive.to(dtype=observation.dtype)
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    return (terminal_value * discount).mean()


def check_terminal_value_action_gradient(
    analytical_environment: TrackDiffEnv,
    fd_environment: TrackDiffEnv,
    critic: Critic,
    normalizer: RunningNormalizer,
    scenarios: TrackScenarios,
    horizon: int,
    gamma: float,
) -> dict:
    hover = 2.0 / analytical_environment.config.thrust_to_weight_ratio - 1.0
    base_actions = torch.zeros((horizon, 1, 4), device=analytical_environment.device)
    base_actions[:, :, 0] = 0.02
    base_actions[:, :, 1] = -0.015
    base_actions[:, :, 3] = hover

    sim_actions = [as_simulation_action(action.clone()) for action in base_actions]
    terminal_loss = rollout_terminal_value(
        analytical_environment, critic, normalizer, sim_actions, scenarios, gamma
    )
    analytical_environment.scene.backward(terminal_loss)
    action_gradients = collect_simulation_action_gradients(sim_actions)
    for action in sim_actions:
        action.grad = None
    analytical_environment.release_simulation_graphs(terminal_loss)

    early_analytical = action_gradients[0, 0, 0].item()
    late_analytical = action_gradients[horizon - 1, 0, 0].item()
    if abs(early_analytical) == 0.0 and abs(late_analytical) == 0.0:
        raise AssertionError("∂V(s_H)/∂a_t is zero for both early and late actions")

    epsilon = 1e-2
    step = 0
    dimension = 0
    positive = base_actions.clone()
    negative = base_actions.clone()
    positive[step, 0, dimension] += epsilon
    negative[step, 0, dimension] -= epsilon
    positive_loss = rollout_terminal_value(
        fd_environment,
        critic,
        normalizer,
        [action for action in positive],
        scenarios,
        gamma,
    ).item()
    negative_loss = rollout_terminal_value(
        fd_environment,
        critic,
        normalizer,
        [action for action in negative],
        scenarios,
        gamma,
    ).item()
    numerical = (positive_loss - negative_loss) / (2.0 * epsilon)
    assert_close(early_analytical, numerical, relative_tolerance=0.2, absolute_tolerance=1e-2)
    return {
        "early_action_analytical": early_analytical,
        "early_action_numerical": numerical,
        "late_action_analytical": late_analytical,
        "early_action_nonzero": abs(early_analytical) > 0.0,
        "late_action_nonzero": abs(late_analytical) > 0.0,
    }


def main() -> None:
    settings = load_track_diff_settings(PROJECT_ROOT / "config" / "track_diff" / "train.yaml")
    gs.init(backend=gs.gpu, seed=settings.seed, logging_level="warning")
    device = gs.device

    # Genesis differentiable runtime expects torch optimizers to exist before the scene.
    critic = Critic(TrackDiffEnv.observation_dim, NetworkConfig()).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    with torch.no_grad():
        critic.network.output.bias.fill_(0.5)
        critic.network.output.weight.normal_(0.0, 0.1)
    normalizer = RunningNormalizer(TrackDiffEnv.observation_dim).to(device)

    state_result = check_terminal_value_state_gradient(critic, TrackDiffEnv.observation_dim, device)

    horizon = 8
    gamma = settings.shac.gamma
    env_config = replace(settings.environment, horizon=horizon)
    analytical_environment = TrackDiffEnv(env_config, 1, requires_grad=True)
    fd_environment = TrackDiffEnv(env_config, 1, requires_grad=False)
    scenarios = fixed_scenarios(horizon)
    action_result = check_terminal_value_action_gradient(
        analytical_environment,
        fd_environment,
        critic,
        normalizer,
        scenarios,
        horizon,
        gamma,
    )
    del optimizer

    result = {
        "terminal_value_state_gradient": state_result,
        "terminal_value_action_gradient": action_result,
        "passed": True,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
