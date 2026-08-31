"""Split-S feasibility helpers. Training policies must not import this module."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from genesis_drones.controllers.ctbr_controller import CtbrControllerConfig
from genesis_drones.tasks.racing_core import (
    EvaluationInitialStates,
    RaceTrackSpec,
    detect_race_events,
    gate_corners_world,
    track_to_tensors,
)
from genesis_drones.tasks.racing_tracks import FIXED_SEVEN_GATE_TRACK

GATE4_INDEX = 3
GATE5_INDEX = 4
STANDOFF = 2.0


@dataclass
class SplitSPlan:
    waypoints: np.ndarray
    times: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    collective: np.ndarray
    body_rate: np.ndarray
    rotation: np.ndarray
    action: np.ndarray
    dt: float


def split_s_waypoints(track: RaceTrackSpec = FIXED_SEVEN_GATE_TRACK, standoff: float = STANDOFF) -> np.ndarray:
    tensors = track_to_tensors(track, torch.device("cpu"), torch.float32)
    g4 = tensors.positions[GATE4_INDEX].numpy()
    n4 = tensors.rotations[GATE4_INDEX, :, 0].numpy()
    g5 = tensors.positions[GATE5_INDEX].numpy()
    n5 = tensors.rotations[GATE5_INDEX, :, 0].numpy()
    return np.stack((g4 - standoff * n4, g4, g4 + standoff * n4, g5 - standoff * n5, g5, g5 + standoff * n5))


def _flatness(velocity: np.ndarray, acceleration: np.ndarray, dt: float, gravity: float):
    acc_g = acceleration + np.array((0.0, 0.0, gravity))
    collective = np.linalg.norm(acc_g, axis=1, keepdims=True).clip(min=1e-6)
    z_body = acc_g / collective
    heading = np.empty(len(velocity))
    last = np.arctan2(velocity[0, 1], velocity[0, 0])
    for index, xy_velocity in enumerate(velocity[:, :2]):
        if np.linalg.norm(xy_velocity) > 0.2:
            last = float(np.arctan2(xy_velocity[1], xy_velocity[0]))
        heading[index] = last
    x_c = np.stack((np.cos(heading), np.sin(heading), np.zeros_like(heading)), axis=1)
    y_body = np.cross(z_body, x_c)
    y_norm = np.linalg.norm(y_body, axis=1, keepdims=True).clip(min=1e-6)
    y_body = y_body / y_norm
    x_body = np.cross(y_body, z_body)
    rotation = np.stack((x_body, y_body, z_body), axis=-1)
    rotation_dot = np.gradient(rotation, dt, axis=0)
    omega_hat = np.matmul(np.transpose(rotation, (0, 2, 1)), rotation_dot)
    body_rate = np.stack((omega_hat[:, 2, 1], omega_hat[:, 0, 2], omega_hat[:, 1, 0]), axis=1)
    return collective[:, 0], body_rate, rotation


def _normalized_action(collective: np.ndarray, body_rate: np.ndarray, config: CtbrControllerConfig) -> np.ndarray:
    max_collective = config.thrust_to_weight_ratio * config.gravity
    thrust_action = 2.0 * collective / max_collective - 1.0
    rate_action = body_rate / np.asarray(config.max_body_rates)
    return np.concatenate((thrust_action[:, None], rate_action), axis=1)


def _plan_once(waypoints: np.ndarray, dt: float, cruise_speed: float, config: CtbrControllerConfig) -> SplitSPlan:
    segment = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    knots = np.concatenate(([0.0], np.cumsum(np.maximum(segment / cruise_speed, 0.5))))
    start_velocity = (waypoints[1] - waypoints[0]) / max(np.linalg.norm(waypoints[1] - waypoints[0]), 1e-6) * min(
        cruise_speed, 1.0
    )
    end_velocity = (waypoints[-1] - waypoints[-2]) / max(np.linalg.norm(waypoints[-1] - waypoints[-2]), 1e-6) * 0.3
    spline = CubicSpline(knots, waypoints, bc_type=((1, start_velocity), (1, end_velocity)))
    times = np.arange(0.0, knots[-1] + dt, dt)
    position = spline(times)
    velocity = spline(times, 1)
    acceleration = spline(times, 2)
    jerk = spline(times, 3)
    collective, body_rate, rotation = _flatness(velocity, acceleration, dt, config.gravity)
    action = _normalized_action(collective, body_rate, config)
    return SplitSPlan(waypoints, times, position, velocity, acceleration, jerk, collective, body_rate, rotation, action, dt)


def plan_split_s_trajectory(
    track: RaceTrackSpec = FIXED_SEVEN_GATE_TRACK,
    dt: float = 0.01,
    cruise_speed: float = 1.6,
    standoff: float = STANDOFF,
    config: CtbrControllerConfig | None = None,
) -> SplitSPlan:
    config = config or CtbrControllerConfig()
    waypoints = split_s_waypoints(track, standoff)
    scale = 1.0
    plan = _plan_once(waypoints, dt, cruise_speed, config)
    for _ in range(12):
        if float(np.max(np.abs(plan.action))) <= 0.95:
            return plan
        scale *= 1.25
        plan = _plan_once(waypoints, dt, cruise_speed / scale, config)
    raise RuntimeError("planned Split-S exceeds CTBR limits after time scaling")


def assert_ctbr_within_limits(plan: SplitSPlan, config: CtbrControllerConfig | None = None) -> None:
    config = config or CtbrControllerConfig()
    max_collective = config.thrust_to_weight_ratio * config.gravity
    assert np.all(plan.collective >= 0.05 * config.gravity)
    assert np.all(plan.collective <= 0.99 * max_collective)
    assert np.all(np.abs(plan.body_rate) <= np.asarray(config.max_body_rates) * 0.99)
    assert np.all(np.abs(plan.action) <= 0.95)


def analytic_split_s_events(plan: SplitSPlan, track: RaceTrackSpec = FIXED_SEVEN_GATE_TRACK, safety_radius: float = 0.06):
    tensors = track_to_tensors(track, torch.device("cpu"), torch.float32)
    position = torch.tensor(plan.position, dtype=torch.float32)
    gate_index = torch.tensor([GATE4_INDEX], dtype=torch.int64)
    active = torch.ones(1, dtype=torch.bool)
    passed4 = False
    passed5 = False
    for step in range(position.shape[0] - 1):
        events = detect_race_events(
            position[step : step + 1],
            position[step + 1 : step + 2],
            tensors,
            gate_index,
            safety_radius,
            active,
        )
        if bool(events.analytic_collision[0]) or bool(events.wrong_way[0]):
            return {"passed4": passed4, "passed5": passed5, "collision": True, "wrong_way": bool(events.wrong_way[0])}
        if bool(events.passed[0]):
            if int(gate_index[0]) == GATE4_INDEX:
                passed4 = True
            elif int(gate_index[0]) == GATE5_INDEX:
                passed5 = True
        gate_index = events.next_gate_index
    return {"passed4": passed4, "passed5": passed5, "collision": False, "wrong_way": False}


def start_quaternion(plan: SplitSPlan) -> np.ndarray:
    xyzw = Rotation.from_matrix(plan.rotation[0]).as_quat()
    return np.array((xyzw[3], xyzw[0], xyzw[1], xyzw[2]))


def tracking_action(
    plan: SplitSPlan,
    step: int,
    position: np.ndarray,
    velocity: np.ndarray,
    config: CtbrControllerConfig | None = None,
    position_gain: float = 5.0,
    velocity_gain: float = 2.5,
) -> np.ndarray:
    config = config or CtbrControllerConfig()
    index = min(step, len(plan.position) - 1)
    acceleration = (
        plan.acceleration[index]
        + position_gain * (plan.position[index] - position)
        + velocity_gain * (plan.velocity[index] - velocity)
    )
    acc_g = acceleration + np.array((0.0, 0.0, config.gravity))
    collective = float(np.linalg.norm(acc_g).clip(min=1e-6))
    z_desired = acc_g / collective
    z_feedforward = plan.rotation[index, :, 2]
    rate_correction = plan.rotation[index].T @ np.cross(z_feedforward, z_desired)
    body_rate = plan.body_rate[index] + 4.0 * rate_correction
    return _normalized_action(np.array((collective,)), body_rate[None], config)[0]


def gate_corners(track: RaceTrackSpec, gate_index: int) -> np.ndarray:
    tensors = track_to_tensors(track, torch.device("cpu"), torch.float32)
    return gate_corners_world(tensors, torch.tensor([gate_index]))[0].numpy()


def replay_split_s(environment, plan: SplitSPlan) -> dict:
    dtype = environment.last_action.dtype
    device = environment.last_action.device
    start = EvaluationInitialStates(
        position=torch.tensor(plan.position[:1], device=device, dtype=dtype),
        quaternion=torch.tensor(start_quaternion(plan)[None], device=device, dtype=dtype),
        linear_velocity=torch.tensor(plan.velocity[:1], device=device, dtype=dtype),
    )
    environment.reset(initial_states=start)
    environment.respawn_on_fail = False
    environment.gate_index.fill_(GATE4_INDEX)
    passed4 = False
    passed5 = False
    for step in range(len(plan.times)):
        state = environment._read_state()
        action = tracking_action(
            plan,
            step,
            state.position[0].detach().cpu().numpy(),
            state.linear_velocity[0].detach().cpu().numpy(),
            environment.controller.config,
        )
        _, _, _, extras = environment.step(torch.tensor(action[None], device=device, dtype=dtype))
        if bool(extras["analytic_collision"][0]) or bool(extras["physics_collision"][0]) or bool(extras["wrong_way"][0]):
            return {
                "passed4": passed4,
                "passed5": passed5,
                "collision": True,
                "physics_collision": bool(extras["physics_collision"][0]),
                "gate_index": int(extras["gate_index"][0]),
            }
        if bool(extras["passed"][0]) and int(extras["gate_index"][0]) == GATE5_INDEX:
            passed4 = True
        if bool(extras["passed"][0]) and int(extras["gate_index"][0]) >= 5:
            passed5 = True
            break
    return {
        "passed4": passed4,
        "passed5": passed5,
        "collision": False,
        "physics_collision": False,
        "gate_index": int(environment.gate_index[0]),
    }
