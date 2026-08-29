import json
from dataclasses import replace
from pathlib import Path

import torch

import genesis as gs

from genesis_drones.envs.track_diff_env import TrackDiffEnv
from genesis_drones.evaluation.track_diff import TrackScenarios
from genesis_drones.utils.track_diff_config import load_track_diff_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def fixed_scenarios(horizon: int) -> TrackScenarios:
    return TrackScenarios(
        initial_position=torch.tensor([[0.0, 0.0, 0.6]], device="cpu"),
        initial_quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cpu"),
        waypoint_sequences=torch.tensor([[[0.6, 0.2, 0.8]]], device="cpu").expand(1, horizon + 1, 3).clone(),
    )


def rollout_loss(environment: TrackDiffEnv, actions, scenarios: TrackScenarios) -> torch.Tensor:
    environment.reset(
        scenarios.initial_position,
        scenarios.initial_quaternion,
        scenarios.waypoint_sequences,
    )
    total_loss = 0.0
    for action in actions:
        _, (loss, _), _, _ = environment.step(action)
        total_loss = total_loss + loss.mean()
    return total_loss / len(actions)


def finite_difference(
    environment: TrackDiffEnv,
    actions: torch.Tensor,
    scenarios: TrackScenarios,
    step: int,
    dimension: int,
    epsilon: float,
) -> float:
    positive = actions.clone()
    negative = actions.clone()
    positive[step, 0, dimension] += epsilon
    negative[step, 0, dimension] -= epsilon
    positive_loss = rollout_loss(environment, positive, scenarios).item()
    negative_loss = rollout_loss(environment, negative, scenarios).item()
    return (positive_loss - negative_loss) / (2.0 * epsilon)


def assert_gradient(analytical: float, numerical: float, relative_tolerance: float, absolute_tolerance: float) -> None:
    error = abs(analytical - numerical)
    tolerance = max(absolute_tolerance, relative_tolerance * abs(numerical))
    if error > tolerance:
        raise AssertionError(
            f"gradient mismatch: analytical={analytical:.8f} numerical={numerical:.8f} "
            f"error={error:.8f} tolerance={tolerance:.8f}"
        )


def main() -> None:
    settings = load_track_diff_settings(PROJECT_ROOT / "config" / "track_diff" / "train.yaml")
    gs.init(backend=gs.gpu, seed=settings.seed, logging_level="warning")

    hover_action = 2.0 / 3.3 - 1.0
    action_parameters = torch.nn.Parameter(torch.zeros((32, 1, 4), device=gs.device))
    with torch.no_grad():
        action_parameters[:, :, 0] = torch.atanh(action_parameters.new_tensor(hover_action))
    action_optimizer = torch.optim.Adam((action_parameters,), lr=0.03)

    single_config = replace(settings.environment, horizon=1)
    single_analytical_environment = TrackDiffEnv(single_config, 1, requires_grad=True)
    single_fd_environment = TrackDiffEnv(single_config, 1, requires_grad=False)
    single_scenarios = fixed_scenarios(1)

    mixer_action = torch.tensor([[hover_action, 0.0, 0.0, 0.0]], device=gs.device)
    motor_thrust, actual_wrench = single_analytical_environment.mix_action(mixer_action)
    hover_thrust = settings.environment.max_collective_thrust / 3.3
    torch.testing.assert_close(motor_thrust, torch.full_like(motor_thrust, hover_thrust / 4.0))
    torch.testing.assert_close(
        actual_wrench,
        torch.tensor([[hover_thrust, 0.0, 0.0, 0.0]], device=gs.device),
        atol=1e-6,
        rtol=1e-6,
    )
    saturated_motor_thrust, saturated_wrench = single_analytical_environment.mix_action(
        torch.tensor([[1.0, 1.0, -1.0, 1.0]], device=gs.device)
    )
    assert (saturated_motor_thrust >= 0.0).all() and (
        saturated_motor_thrust <= settings.environment.max_collective_thrust / 4.0
    ).all()
    torch.testing.assert_close(
        saturated_wrench,
        saturated_motor_thrust @ single_analytical_environment.allocation.T,
    )

    single_actions = torch.tensor([[[hover_action, 0.03, -0.02, 0.01]]], device=gs.device)
    single_inputs = [gs.from_torch(single_actions[0], requires_grad=True)]
    single_loss = rollout_loss(single_analytical_environment, single_inputs, single_scenarios)
    single_analytical_environment.scene.backward(single_loss)
    single_results = []
    for dimension in range(4):
        analytical = single_inputs[0].grad[0, dimension].item()
        numerical = finite_difference(
            single_fd_environment,
            single_actions,
            single_scenarios,
            step=0,
            dimension=dimension,
            epsilon=1e-2,
        )
        assert_gradient(analytical, numerical, relative_tolerance=0.01, absolute_tolerance=1e-3)
        single_results.append({"dimension": dimension, "analytical": analytical, "numerical": numerical})

    multi_config = replace(settings.environment, horizon=10)
    multi_analytical_environment = TrackDiffEnv(multi_config, 1, requires_grad=True)
    multi_fd_environment = TrackDiffEnv(multi_config, 1, requires_grad=False)
    multi_scenarios = fixed_scenarios(10)
    multi_actions = torch.zeros((10, 1, 4), device=gs.device)
    multi_actions[:, :, 0] = hover_action
    multi_actions[:, :, 1] = 0.02
    multi_actions[:, :, 2] = -0.015
    multi_actions[:, :, 3] = 0.01
    multi_inputs = [gs.from_torch(action, requires_grad=True) for action in multi_actions]
    multi_loss = rollout_loss(multi_analytical_environment, multi_inputs, multi_scenarios)
    multi_analytical_environment.scene.backward(multi_loss)
    multi_results = []
    for step, dimension in ((0, 0), (3, 1), (6, 2), (9, 3)):
        analytical = multi_inputs[step].grad[0, dimension].item()
        numerical = finite_difference(
            multi_fd_environment,
            multi_actions,
            multi_scenarios,
            step=step,
            dimension=dimension,
            epsilon=1e-2,
        )
        assert_gradient(analytical, numerical, relative_tolerance=0.05, absolute_tolerance=1e-3)
        multi_results.append(
            {"step": step, "dimension": dimension, "analytical": analytical, "numerical": numerical}
        )

    optimization_environment = TrackDiffEnv(settings.environment, 1, requires_grad=True)
    optimization_scenarios = fixed_scenarios(32)
    optimization_losses = []
    for _ in range(25):
        action_optimizer.zero_grad()
        loss = rollout_loss(optimization_environment, torch.tanh(action_parameters), optimization_scenarios)
        optimization_environment.scene.backward(loss)
        action_optimizer.step()
        optimization_losses.append(loss.detach().item())
    if optimization_losses[-1] >= optimization_losses[0]:
        raise AssertionError(
            f"32-step action optimization failed: initial={optimization_losses[0]} final={optimization_losses[-1]}"
        )

    result = {
        "mixer": "passed",
        "single_step": single_results,
        "ten_step": multi_results,
        "optimization": {
            "initial_loss": optimization_losses[0],
            "final_loss": optimization_losses[-1],
            "iterations": len(optimization_losses),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
