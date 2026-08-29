from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from torch.nn import functional as F

import genesis as gs


ASSETS_PATH = Path(__file__).resolve().parents[1] / "robots" / "assets"


@dataclass(frozen=True)
class LossWeights:
    position: float = 3.0
    velocity: float = 1.0
    tilt: float = 0.1
    yaw: float = 0.1
    angular_velocity: float = 0.001
    action_delta: float = 0.001
    safety: float = 5.0
    terminal: float = 10.0


@dataclass(frozen=True)
class TrackDiffEnvConfig:
    dt: float = 0.01
    horizon: int = 32
    max_episode_steps: int = 1500
    target_threshold: float = 0.1
    max_collective_thrust: float = 16.1865
    motor_arm: float = 0.1
    thrust_coefficient: float = 3.16e-10
    moment_coefficient: float = 7.94e-12
    body_collision_radius: float = 0.06
    body_collision_half_height: float = 0.0125
    ground_termination_height: float = 0.1
    ground_warning_distance: float = 0.2
    horizontal_termination_error: float = 5.0
    horizontal_warning_distance: float = 0.5
    vertical_termination_error: float = 1.2
    vertical_warning_distance: float = 0.12
    safety_temperature_ratio: float = 0.1
    initial_x_range: tuple[float, float] = (-0.05, 0.05)
    initial_y_range: tuple[float, float] = (-0.05, 0.05)
    initial_z_range: tuple[float, float] = (0.6, 0.61)
    target_x_range: tuple[float, float] = (-1.2, 1.2)
    target_y_range: tuple[float, float] = (-1.2, 1.2)
    target_z_range: tuple[float, float] = (0.6, 1.0)
    loss_weights: LossWeights = LossWeights()
    reward_weights: LossWeights = LossWeights(terminal=0.0)


class DroneState(NamedTuple):
    position: torch.Tensor
    quaternion: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity_body: torch.Tensor


def smooth_safety_penalty(distance: torch.Tensor, warning_distance: float, temperature_ratio: float) -> torch.Tensor:
    temperature = warning_distance * temperature_ratio
    normalizer = F.softplus(distance.new_tensor(warning_distance / temperature))
    return torch.square(F.softplus((warning_distance - distance) / temperature) / normalizer)


def detached_torch_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    return value.as_subclass(torch.Tensor) if isinstance(value, gs.Tensor) else value


class TrackDiffEnv:
    action_dim = 4
    observation_dim = 17

    def __init__(self, config: TrackDiffEnvConfig, num_envs: int, requires_grad: bool = True):
        self.config = config
        self.num_envs = num_envs
        self.requires_grad = requires_grad
        self.device = gs.device

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=config.dt,
                substeps=1,
                substeps_local=config.horizon if requires_grad else 1,
                requires_grad=requires_grad,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=False,
                enable_joint_limit=True,
            ),
            show_viewer=False,
        )
        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(
                file=str(ASSETS_PATH / "drone_urdf" / "drone.urdf"),
                pos=(0.0, 0.0, 0.6),
                euler=(0.0, 0.0, 0.0),
                default_armature=2.6e-7,
            ),
        )
        self.scene.build(n_envs=num_envs)
        self.drone.set_dofs_damping([0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4])

        motor_moment_ratio = config.moment_coefficient / config.thrust_coefficient
        self.allocation = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [-config.motor_arm, -config.motor_arm, config.motor_arm, config.motor_arm],
                [-config.motor_arm, config.motor_arm, config.motor_arm, -config.motor_arm],
                [-motor_moment_ratio, motor_moment_ratio, -motor_moment_ratio, motor_moment_ratio],
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.allocation_inverse = torch.linalg.inv(self.allocation)
        self.motor_max_thrust = config.max_collective_thrust / 4.0
        self.torque_scale = torch.tensor(
            [
                config.motor_arm * config.max_collective_thrust / 2.0,
                config.motor_arm * config.max_collective_thrust / 2.0,
                motor_moment_ratio * config.max_collective_thrust / 2.0,
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.envs_idx = torch.arange(num_envs, device=self.device)
        self.target_lower = torch.tensor(
            [config.target_x_range[0], config.target_y_range[0], config.target_z_range[0]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.target_upper = torch.tensor(
            [config.target_x_range[1], config.target_y_range[1], config.target_z_range[1]],
            device=self.device,
            dtype=gs.tc_float,
        )

        self.target_position = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_action = torch.zeros((num_envs, self.action_dim), device=self.device, dtype=gs.tc_float)
        self.is_alive = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        self.waypoint_sequences = None
        self.waypoint_index = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self.episode_step = 0
        self.waypoint_count = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self.first_arrival_step = torch.full(
            (num_envs,), config.max_episode_steps, device=self.device, dtype=torch.int64
        )
        self.crash_step = torch.full(
            (num_envs,), config.max_episode_steps + 1, device=self.device, dtype=torch.int64
        )
        self.position_error_sum = torch.zeros(num_envs, device=self.device, dtype=gs.tc_float)
        self.position_error_steps = torch.zeros(num_envs, device=self.device, dtype=gs.tc_float)

    def _read_state(self) -> DroneState:
        solver_state = self.scene.rigid_solver.get_state()
        link_idx = self.drone.base_link_idx
        dof_start = self.drone.dof_start
        quaternion = solver_state.links_quat[:, link_idx]
        quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
        return DroneState(
            solver_state.links_pos[:, link_idx],
            quaternion,
            solver_state.dofs_vel[:, dof_start : dof_start + 3],
            solver_state.dofs_vel[:, dof_start + 3 : dof_start + 6],
        )

    def _make_observation(self, state: DroneState) -> torch.Tensor:
        quaternion = torch.where(state.quaternion[:, :1] < 0.0, -state.quaternion, state.quaternion)
        w = quaternion[:, 0]
        x = quaternion[:, 1]
        y = quaternion[:, 2]
        z = quaternion[:, 3]
        cosine_yaw = 1.0 - 2.0 * (y * y + z * z)
        sine_yaw = 2.0 * (w * z + x * y)
        position_error = self.target_position - state.position
        position_local = torch.stack(
            (
                cosine_yaw * position_error[:, 0] + sine_yaw * position_error[:, 1],
                -sine_yaw * position_error[:, 0] + cosine_yaw * position_error[:, 1],
                position_error[:, 2],
            ),
            dim=-1,
        )
        velocity_local = torch.stack(
            (
                cosine_yaw * state.linear_velocity[:, 0] + sine_yaw * state.linear_velocity[:, 1],
                -sine_yaw * state.linear_velocity[:, 0] + cosine_yaw * state.linear_velocity[:, 1],
                state.linear_velocity[:, 2],
            ),
            dim=-1,
        )
        observation = torch.cat(
            (position_local, quaternion, velocity_local, state.angular_velocity_body, self.last_action), dim=-1
        )
        return torch.where(self.is_alive[:, None], observation, torch.zeros_like(observation))

    def reset(
        self,
        initial_position: torch.Tensor | None = None,
        initial_quaternion: torch.Tensor | None = None,
        waypoint_sequences: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.scene.reset()
        if initial_position is None:
            initial_position = torch.empty((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
            initial_position[:, 0].uniform_(*self.config.initial_x_range)
            initial_position[:, 1].uniform_(*self.config.initial_y_range)
            initial_position[:, 2].uniform_(*self.config.initial_z_range)
        else:
            initial_position = initial_position.to(device=self.device, dtype=gs.tc_float)
        if initial_quaternion is None:
            yaw = torch.empty(self.num_envs, device=self.device, dtype=gs.tc_float).uniform_(-torch.pi, torch.pi)
            initial_quaternion = torch.stack(
                (torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw)), dim=-1
            )
        else:
            initial_quaternion = initial_quaternion.to(device=self.device, dtype=gs.tc_float)

        self.drone.set_pos(initial_position, zero_velocity=True)
        self.drone.set_quat(initial_quaternion, zero_velocity=True)
        self.last_action = torch.zeros_like(self.last_action)
        self.is_alive.fill_(True)
        self.episode_step = 0
        self.waypoint_index.zero_()
        self.waypoint_count.zero_()
        self.first_arrival_step.fill_(self.config.max_episode_steps)
        self.crash_step.fill_(self.config.max_episode_steps + 1)
        self.position_error_sum.zero_()
        self.position_error_steps.zero_()

        if waypoint_sequences is None:
            self.waypoint_sequences = None
            self.target_position = self.target_lower + (self.target_upper - self.target_lower) * torch.rand(
                (self.num_envs, 3), device=self.device, dtype=gs.tc_float
            )
        else:
            if waypoint_sequences.shape[0] != self.num_envs:
                raise ValueError("waypoint_sequences must have one sequence per environment")
            self.waypoint_sequences = waypoint_sequences.to(device=self.device, dtype=gs.tc_float)
            self.target_position = self.waypoint_sequences[:, 0]

        return self._make_observation(self._read_state()).detach()

    def mix_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action = action.clamp(-1.0, 1.0)
        collective_thrust = 0.5 * (action[:, :1] + 1.0) * self.config.max_collective_thrust
        desired_wrench = torch.cat((collective_thrust, action[:, 1:] * self.torque_scale), dim=-1)
        motor_thrust = torch.clamp(desired_wrench @ self.allocation_inverse.T, 0.0, self.motor_max_thrust)
        return motor_thrust, motor_thrust @ self.allocation.T


    def step(self, action: torch.Tensor):
        action = action.clamp(-1.0, 1.0)
        is_alive_before = self.is_alive.clone()
        state_before = self._read_state()
        motor_thrust, actual_wrench = self.mix_action(action)
        actual_wrench = actual_wrench * is_alive_before[:, None]
        w_before = state_before.quaternion[:, 0]
        x_before = state_before.quaternion[:, 1]
        y_before = state_before.quaternion[:, 2]
        z_before = state_before.quaternion[:, 3]
        thrust_axis_world = torch.stack(
            (
                2.0 * (x_before * z_before + w_before * y_before),
                2.0 * (y_before * z_before - w_before * x_before),
                1.0 - 2.0 * (x_before * x_before + y_before * y_before),
            ),
            dim=-1,
        )
        force_world = thrust_axis_world * actual_wrench[:, :1]
        generalized_force = torch.cat((force_world, actual_wrench[:, 1:]), dim=-1)
        if generalized_force.requires_grad and not isinstance(generalized_force, gs.Tensor):
            generalized_force = gs.from_torch(generalized_force, detach=False)
        self.drone.control_dofs_force(generalized_force)
        self.scene.step()

        state = self._read_state()
        quaternion = state.quaternion
        w = quaternion[:, 0]
        x = quaternion[:, 1]
        y = quaternion[:, 2]
        z = quaternion[:, 3]
        position_error = self.target_position - state.position
        distance = torch.linalg.vector_norm(position_error, dim=-1)
        closeness = torch.exp(-distance)
        position_loss = 1.0 - closeness
        velocity_loss = F.smooth_l1_loss(
            torch.linalg.vector_norm(state.linear_velocity, dim=-1),
            torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float),
            reduction="none",
        )
        rotation_zz = 1.0 - 2.0 * (x * x + y * y)
        tilt_loss = closeness * (1.0 - rotation_zz)
        rotation_xx = 1.0 - 2.0 * (y * y + z * z)
        rotation_yx = 2.0 * (x * y + w * z)
        yaw_cosine = rotation_xx / torch.sqrt(rotation_xx * rotation_xx + rotation_yx * rotation_yx + 1e-8)
        yaw_loss = closeness * (1.0 - yaw_cosine)
        angular_velocity_loss = torch.linalg.vector_norm(state.angular_velocity_body, dim=-1)
        action_delta_loss = torch.sum(torch.square(action - self.last_action), dim=-1)

        rotation_zx = 2.0 * (x * z - w * y)
        rotation_zy = 2.0 * (y * z + w * x)
        lowest_height = (
            state.position[:, 2]
            - self.config.body_collision_radius
            * torch.sqrt(rotation_zx * rotation_zx + rotation_zy * rotation_zy + 1e-8)
            - self.config.body_collision_half_height * rotation_zz.abs()
        )
        ground_safety_loss = smooth_safety_penalty(
            lowest_height, self.config.ground_warning_distance, self.config.safety_temperature_ratio
        )
        horizontal_distance_x = self.config.horizontal_termination_error - position_error[:, 0].abs()
        horizontal_distance_y = self.config.horizontal_termination_error - position_error[:, 1].abs()
        vertical_distance = self.config.vertical_termination_error - position_error[:, 2].abs()
        safety_loss = (
            ground_safety_loss
            + smooth_safety_penalty(
                horizontal_distance_x,
                self.config.horizontal_warning_distance,
                self.config.safety_temperature_ratio,
            )
            + smooth_safety_penalty(
                horizontal_distance_y,
                self.config.horizontal_warning_distance,
                self.config.safety_temperature_ratio,
            )
            + smooth_safety_penalty(
                vertical_distance,
                self.config.vertical_warning_distance,
                self.config.safety_temperature_ratio,
            )
        )

        has_finite_state = (
            torch.isfinite(state.position).all(dim=-1)
            & torch.isfinite(quaternion).all(dim=-1)
            & torch.isfinite(state.linear_velocity).all(dim=-1)
            & torch.isfinite(state.angular_velocity_body).all(dim=-1)
        )
        is_crashed = (
            (state.position[:, 2] < self.config.ground_termination_height)
            | (horizontal_distance_x < 0.0)
            | (horizontal_distance_y < 0.0)
            | (vertical_distance < 0.0)
            | ~has_finite_state
        )
        is_newly_dead = is_alive_before & is_crashed
        is_arrived = is_alive_before & ~is_newly_dead & (distance < self.config.target_threshold)

        loss_weights = self.config.loss_weights
        continuous_loss = (
            loss_weights.position * position_loss
            + loss_weights.velocity * velocity_loss
            + loss_weights.tilt * tilt_loss
            + loss_weights.yaw * yaw_loss
            + loss_weights.angular_velocity * angular_velocity_loss
            + loss_weights.action_delta * action_delta_loss
            + loss_weights.safety * safety_loss
        )
        loss = continuous_loss * is_alive_before + loss_weights.terminal * is_newly_dead

        reward_weights = self.config.reward_weights
        reward_penalty = (
            reward_weights.position * position_loss
            + reward_weights.velocity * velocity_loss
            + reward_weights.tilt * tilt_loss
            + reward_weights.yaw * yaw_loss
            + reward_weights.angular_velocity * angular_velocity_loss
            + reward_weights.action_delta * action_delta_loss
            + reward_weights.safety * safety_loss
        )
        reward = ((1.0 - reward_penalty) * is_alive_before).detach()

        self.episode_step += 1
        self.waypoint_count += is_arrived
        self.first_arrival_step = torch.where(
            is_arrived & (self.first_arrival_step == self.config.max_episode_steps),
            torch.full_like(self.first_arrival_step, self.episode_step),
            self.first_arrival_step,
        )
        self.crash_step = torch.where(
            is_newly_dead, torch.full_like(self.crash_step, self.episode_step), self.crash_step
        )
        self.position_error_sum += distance.detach() * is_alive_before
        self.position_error_steps += is_alive_before
        self.is_alive = is_alive_before & ~is_newly_dead

        if self.waypoint_sequences is None:
            next_target = self.target_lower + (self.target_upper - self.target_lower) * torch.rand(
                (self.num_envs, 3), device=self.device, dtype=gs.tc_float
            )
            self.target_position = torch.where(is_arrived[:, None], next_target, self.target_position)
        else:
            self.waypoint_index = torch.clamp(
                self.waypoint_index + is_arrived,
                max=self.waypoint_sequences.shape[1] - 1,
            )
            self.target_position = self.waypoint_sequences[self.envs_idx, self.waypoint_index]

        self.last_action = torch.where(self.is_alive[:, None], action, torch.zeros_like(action))
        is_time_limit = self.episode_step == self.config.max_episode_steps
        is_truncated = self.is_alive & is_time_limit
        done = is_newly_dead | is_truncated
        observation = self._make_observation(state)
        self.is_alive = self.is_alive & ~is_truncated
        self.last_action = torch.where(self.is_alive[:, None], self.last_action, torch.zeros_like(self.last_action))
        extras = {
            "terminated": is_newly_dead.detach(),
            "truncated": is_truncated.detach(),
            "alive": self.is_alive.detach(),
            "arrived": is_arrived.detach(),
            "actual_wrench": actual_wrench.detach(),
            "motor_thrust": motor_thrust.detach(),
            "loss_components": {
                "position": (position_loss * is_alive_before).detach(),
                "velocity": (velocity_loss * is_alive_before).detach(),
                "tilt": (tilt_loss * is_alive_before).detach(),
                "yaw": (yaw_loss * is_alive_before).detach(),
                "angular_velocity": (angular_velocity_loss * is_alive_before).detach(),
                "action_delta": (action_delta_loss * is_alive_before).detach(),
                "safety": (safety_loss * is_alive_before).detach(),
                "terminal": is_newly_dead.detach(),
            },
            "metrics": {
                "position_error": (distance * is_alive_before).detach(),
            },
        }
        return observation, (loss, reward), done.detach(), extras

    def state_dict(self) -> dict:
        drone_state = self._read_state()
        dofs_velocity = torch.cat((drone_state.linear_velocity, drone_state.angular_velocity_body), dim=-1)
        return {
            "position": detached_torch_tensor(drone_state.position),
            "quaternion": detached_torch_tensor(drone_state.quaternion),
            "dofs_velocity": detached_torch_tensor(dofs_velocity),
            "target_position": detached_torch_tensor(self.target_position),
            "last_action": detached_torch_tensor(self.last_action),
            "is_alive": detached_torch_tensor(self.is_alive),
            "waypoint_sequences": (
                None if self.waypoint_sequences is None else detached_torch_tensor(self.waypoint_sequences)
            ),
            "waypoint_index": detached_torch_tensor(self.waypoint_index),
            "episode_step": self.episode_step,
            "waypoint_count": detached_torch_tensor(self.waypoint_count),
            "first_arrival_step": detached_torch_tensor(self.first_arrival_step),
            "crash_step": detached_torch_tensor(self.crash_step),
            "position_error_sum": detached_torch_tensor(self.position_error_sum),
            "position_error_steps": detached_torch_tensor(self.position_error_steps),
        }

    def load_state_dict(self, state: dict) -> torch.Tensor:
        self.scene.reset()
        self.drone.set_pos(state["position"].to(self.device), zero_velocity=True)
        self.drone.set_quat(state["quaternion"].to(self.device), zero_velocity=True)
        self.drone.set_dofs_velocity(state["dofs_velocity"].to(self.device))
        self.target_position = state["target_position"].to(self.device)
        self.last_action = state["last_action"].to(self.device)
        self.is_alive = state["is_alive"].to(self.device)
        self.waypoint_sequences = (
            None if state["waypoint_sequences"] is None else state["waypoint_sequences"].to(self.device)
        )
        self.waypoint_index = state["waypoint_index"].to(self.device)
        self.episode_step = state["episode_step"]
        self.waypoint_count = state["waypoint_count"].to(self.device)
        self.first_arrival_step = state["first_arrival_step"].to(self.device)
        self.crash_step = state["crash_step"].to(self.device)
        self.position_error_sum = state["position_error_sum"].to(self.device)
        self.position_error_steps = state["position_error_steps"].to(self.device)
        return self.detach_window()

    def detach_window(self) -> torch.Tensor:
        # Scene.backward() runs torch.autograd.backward with retain_graph=True so the simulator unroll can re-enter
        # the graph, and it never releases it. Without this reset the retained graph and the simulator's queried
        # states accumulate every window (~113 MiB per update at 1024 envs) until the process runs out of memory.
        self.scene.sim.reset_grad()
        self.target_position = detached_torch_tensor(self.target_position)
        self.last_action = detached_torch_tensor(self.last_action)
        self.is_alive = detached_torch_tensor(self.is_alive)
        self.waypoint_index = detached_torch_tensor(self.waypoint_index)
        self.waypoint_count = detached_torch_tensor(self.waypoint_count)
        self.first_arrival_step = detached_torch_tensor(self.first_arrival_step)
        self.crash_step = detached_torch_tensor(self.crash_step)
        self.position_error_sum = detached_torch_tensor(self.position_error_sum)
        self.position_error_steps = detached_torch_tensor(self.position_error_steps)
        return self._make_observation(self._read_state()).detach()

    def episode_metrics(self) -> dict[str, torch.Tensor]:
        capped_steps = min(self.episode_step, self.config.max_episode_steps)
        survival_steps = torch.where(
            self.crash_step <= self.config.max_episode_steps,
            self.crash_step,
            torch.full_like(self.crash_step, capped_steps),
        )
        return {
            "first_arrived": (self.waypoint_count > 0).detach(),
            "waypoint_count": self.waypoint_count.detach(),
            "first_arrival_time": (self.first_arrival_step * self.config.dt).detach(),
            "crashed": (self.crash_step <= self.config.max_episode_steps).detach(),
            "survival_time": (survival_steps * self.config.dt).detach(),
            "mean_position_error": (
                self.position_error_sum / self.position_error_steps.clamp_min(1.0)
            ).detach(),
        }
