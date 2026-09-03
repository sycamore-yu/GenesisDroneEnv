"""Compare wrench plant vs rotor plant on the same s0, a0.

Wrench: TrackDiffEnv mix_action + control_dofs_force, then scene.step.
Rotor: Genesis_env scene.step, then angle PID + set_propellers_rpm (1-step RPM lag).
"""

import json
from pathlib import Path

import torch
import yaml

import genesis as gs

from genesis_drones.envs.genesis_env import Genesis_env
from genesis_drones.envs.track_diff_env import TrackDiffEnv
from genesis_drones.utils.geometry import quaternion_to_roll_pitch_yaw
from genesis_drones.evaluation.track_diff import TrackScenarios
from genesis_drones.utils.track_diff_config import load_track_diff_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECK_STEPS = (1, 10, 32, 100)


def hover_action(thrust_to_weight_ratio: float, device) -> torch.Tensor:
    hover = 2.0 / thrust_to_weight_ratio - 1.0
    action = torch.zeros((1, 4), device=device)
    action[0, 3] = hover
    return action


def rpm_to_wrench(rpm: torch.Tensor, environment: TrackDiffEnv) -> torch.Tensor:
    motor_thrust = environment.config.thrust_coefficient * rpm * rpm
    return motor_thrust @ environment.allocation.T


def snapshot(position, quaternion, linear_velocity, angular_velocity) -> dict[str, list]:
    euler = quaternion_to_roll_pitch_yaw(quaternion)
    return {
        "position": position[0].detach().cpu().tolist(),
        "quaternion": quaternion[0].detach().cpu().tolist(),
        "linear_velocity": linear_velocity[0].detach().cpu().tolist(),
        "angular_velocity": angular_velocity[0].detach().cpu().tolist(),
        "roll_pitch_yaw": euler[0].detach().cpu().tolist(),
    }


def delta(left: dict, right: dict) -> dict[str, float]:
    result = {}
    for name in ("position", "quaternion", "linear_velocity", "angular_velocity", "roll_pitch_yaw"):
        difference = torch.tensor(left[name]) - torch.tensor(right[name])
        result[name] = torch.linalg.vector_norm(difference).item()
    return result


def rollout_wrench(environment: TrackDiffEnv, scenarios: TrackScenarios, actions: torch.Tensor) -> dict:
    environment.reset(scenarios.initial_position, scenarios.initial_quaternion, scenarios.waypoint_sequences)
    history = {}
    wrenches = []
    for step in range(actions.shape[0]):
        _, _, _, extras = environment.step(actions[step])
        wrenches.append(extras["actual_wrench"][0].detach().cpu())
        if step + 1 in CHECK_STEPS:
            state = environment._read_state()
            history[step + 1] = snapshot(
                state.position, state.quaternion, state.linear_velocity, state.angular_velocity_body
            )
    return {"history": history, "mean_thrust": torch.stack(wrenches)[:, 0].mean().item(), "mean_torque_norm": torch.linalg.vector_norm(torch.stack(wrenches)[:, 1:], dim=-1).mean().item()}


def rollout_rotor(
    genesis_environment: Genesis_env,
    scenarios: TrackScenarios,
    actions: torch.Tensor,
    wrench_environment: TrackDiffEnv,
) -> dict:
    drone = genesis_environment.drone
    envs_idx = torch.arange(1, device=gs.device)
    genesis_environment.reset(envs_idx)
    drone.set_pos(scenarios.initial_position.to(gs.device), zero_velocity=True)
    drone.set_quat(scenarios.initial_quaternion.to(gs.device), zero_velocity=True)
    drone.odom.reset(scenarios.initial_quaternion.to(gs.device), envs_idx)
    drone.odom.odom_update()
    drone.controller.reset(envs_idx)
    history = {}
    wrenches = []
    for step in range(actions.shape[0]):
        genesis_environment.step(actions[step])
        rpm = drone.controller.mixer(actions[step])
        wrench = rpm_to_wrench(rpm, wrench_environment)
        wrenches.append(wrench[0].detach().cpu())
        if step + 1 in CHECK_STEPS:
            drone.odom.odom_update()
            history[step + 1] = snapshot(
                drone.odom.world_pos,
                drone.odom.body_quat,
                drone.odom.world_linear_vel,
                drone.odom.body_ang_vel,
            )
    return {"history": history, "mean_thrust": torch.stack(wrenches)[:, 0].mean().item(), "mean_torque_norm": torch.linalg.vector_norm(torch.stack(wrenches)[:, 1:], dim=-1).mean().item()}


def controller_only_match(wrench_environment: TrackDiffEnv, genesis_environment: Genesis_env, action: torch.Tensor) -> dict:
    scenarios = TrackScenarios(
        initial_position=torch.tensor([[0.0, 0.0, 0.6]]),
        initial_quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        waypoint_sequences=torch.tensor([[[0.0, 0.0, 0.6]]]),
    )
    wrench_environment.reset(scenarios.initial_position, scenarios.initial_quaternion, scenarios.waypoint_sequences)
    motor_thrust, wrench = wrench_environment.mix_action(action)
    drone = genesis_environment.drone
    envs_idx = torch.arange(1, device=gs.device)
    genesis_environment.reset(envs_idx)
    drone.set_pos(scenarios.initial_position.to(gs.device), zero_velocity=True)
    drone.set_quat(scenarios.initial_quaternion.to(gs.device), zero_velocity=True)
    drone.odom.reset(scenarios.initial_quaternion.to(gs.device), envs_idx)
    drone.odom.odom_update()
    drone.controller.reset(envs_idx)
    drone.controller.step(action)
    rpm = drone.controller.mixer(action)
    rotor_wrench = rpm_to_wrench(rpm, wrench_environment)
    return {
        "motor_thrust_l2": torch.linalg.vector_norm(motor_thrust - wrench_environment.config.thrust_coefficient * rpm * rpm).item(),
        "wrench_l2": torch.linalg.vector_norm(wrench - rotor_wrench).item(),
        "wrench_thrust": wrench[0, 0].item(),
        "rotor_thrust": rotor_wrench[0, 0].item(),
        "wrench_torque": wrench[0, 1:].detach().cpu().tolist(),
        "rotor_torque": rotor_wrench[0, 1:].detach().cpu().tolist(),
    }


def main() -> None:
    settings = load_track_diff_settings(PROJECT_ROOT / "config" / "track_diff" / "train.yaml")
    gs.init(backend=gs.gpu, seed=settings.seed, logging_level="warning")
    with (PROJECT_ROOT / "config" / "track_rl" / "genesis_env.yaml").open() as file:
        environment_config = yaml.safe_load(file)
    with (PROJECT_ROOT / "config" / "track_rl" / "flight.yaml").open() as file:
        flight_config = yaml.safe_load(file)
    environment_config.update(num_envs=1, show_viewer=False, render_cam=False, vis_waypoints=False)

    hover = hover_action(settings.environment.thrust_to_weight_ratio, gs.device)
    roll = hover.clone()
    roll[0, 0] = 0.05
    cases = {"hover": hover, "roll": roll}

    wrench_environment = TrackDiffEnv(settings.environment, 1, requires_grad=False)
    genesis_environment = Genesis_env(environment_config, flight_config, num_envs=1)
    scenarios = TrackScenarios(
        initial_position=torch.tensor([[0.0, 0.0, 0.6]]),
        initial_quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        waypoint_sequences=torch.zeros((1, 101, 3)),
    )
    scenarios.waypoint_sequences[:] = torch.tensor([0.6, 0.2, 0.8])

    result = {"controller_only": {}, "rollouts": {}}
    for name, action in cases.items():
        result["controller_only"][name] = controller_only_match(wrench_environment, genesis_environment, action)
        actions = action.expand(100, 1, 4).clone()
        wrench_run = rollout_wrench(wrench_environment, scenarios, actions)
        rotor_run = rollout_rotor(genesis_environment, scenarios, actions, wrench_environment)
        step_deltas = {
            str(step): delta(wrench_run["history"][step], rotor_run["history"][step]) for step in CHECK_STEPS
        }
        result["rollouts"][name] = {
            "step_deltas": step_deltas,
            "wrench_mean_thrust": wrench_run["mean_thrust"],
            "rotor_mean_thrust": rotor_run["mean_thrust"],
            "wrench_mean_torque_norm": wrench_run["mean_torque_norm"],
            "rotor_mean_torque_norm": rotor_run["mean_torque_norm"],
        }

    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
