from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from torch.nn import functional as F

import genesis as gs


ASSETS_PATH = Path(__file__).resolve().parents[1] / "robots" / "assets"


@dataclass(frozen=True)
class RewardScales:
    target: float = 10.0
    smooth: float = -1.0e-4
    yaw: float = 0.01
    angular: float = -2.0e-4
    crash: float = -10.0
    velocity: float = 0.0


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
    thrust_to_weight_ratio: float = 3.3
    base_rpm: float = 62293.9641914
    angle_kp: tuple[float, float, float] = (0.04, 0.04, 0.04)
    angle_ki: tuple[float, float, float] = (0.004, 0.004, 0.004)
    angle_kd: tuple[float, float, float] = (0.0001, 0.0001, 0.0001)
    body_collision_radius: float = 0.06
    body_collision_half_height: float = 0.0125
    ground_termination_height: float = 0.1
    ground_warning_distance: float = 0.2
    horizontal_termination_error: float = 5.0
    horizontal_warning_distance: float = 0.5
    vertical_termination_error: float = 1.2
    vertical_warning_distance: float = 0.12
    safety_temperature_ratio: float = 0.1
    roll_termination: float = 180.0
    pitch_termination: float = 180.0
    yaw_lambda: float = -10.0
    max_horizon_vel: float = 1.5
    max_vertical_vel: float = 0.5
    action_delta_weight: float = 0.0
    obs_scale_position_error: float = 1.0 / 3.0
    obs_scale_linear_velocity: float = 1.0 / 3.0
    obs_scale_angular_velocity: float = 1.0 / 3.14159
    enable_collision: bool = True
    initial_x_range: tuple[float, float] = (-0.05, 0.05)
    initial_y_range: tuple[float, float] = (-0.05, 0.05)
    initial_z_range: tuple[float, float] = (0.6, 0.61)
    target_x_range: tuple[float, float] = (-1.2, 1.2)
    target_y_range: tuple[float, float] = (-1.2, 1.2)
    target_z_range: tuple[float, float] = (0.6, 1.0)
    reward_scales: RewardScales = RewardScales()


class DroneState(NamedTuple):
    position: torch.Tensor
    quaternion: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity_body: torch.Tensor


def smooth_safety_penalty(distance: torch.Tensor, warning_distance: float, temperature_ratio: float) -> torch.Tensor:
    temperature = warning_distance * temperature_ratio
    normalizer = F.softplus(distance.new_tensor(warning_distance / temperature))
    return torch.square(F.softplus((warning_distance - distance) / temperature) / normalizer)


def quaternion_to_roll_pitch_yaw(quaternion: torch.Tensor) -> torch.Tensor:
    w = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]
    sine_pitch = w * y - x * z
    sine_roll_cosine_pitch = w * x + y * z
    sine_yaw_cosine_pitch = w * z + x * y
    cosine_roll_cosine_pitch = 0.5 * (w * w - x * x - y * y + z * z)
    cosine_yaw_cosine_pitch = 0.5 * (w * w + x * x - y * y - z * z)
    cosine_pitch = torch.sqrt(
        cosine_yaw_cosine_pitch * cosine_yaw_cosine_pitch + sine_yaw_cosine_pitch * sine_yaw_cosine_pitch
    )
    return torch.stack(
        (
            torch.atan2(sine_roll_cosine_pitch, cosine_roll_cosine_pitch),
            torch.atan2(sine_pitch, cosine_pitch),
            torch.atan2(sine_yaw_cosine_pitch, cosine_yaw_cosine_pitch),
        ),
        dim=-1,
    )


def detached_torch_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    return value.as_subclass(torch.Tensor) if isinstance(value, gs.Tensor) else value


def _detach_cached_torch_tensor(value) -> None:
    candidates = [value]
    unwrap = getattr(value, "_unwrap", None)
    if callable(unwrap):
        candidates.append(unwrap())
    for candidate in candidates:
        if candidate is None:
            continue
        for name in ("_tc", "_T_tc"):
            tensor = getattr(candidate, name, None)
            if isinstance(tensor, torch.Tensor) and tensor.grad_fn is not None:
                setattr(candidate, name, tensor.detach())


class TrackDiffEnv:
    action_dim = 4
    observation_dim = 23

    def __init__(
        self,
        config: TrackDiffEnvConfig,
        num_envs: int,
        requires_grad: bool = True,
        show_viewer: bool = False,
        visualize: bool | None = None,
    ):
        self.config = config
        self.num_envs = num_envs
        self.requires_grad = requires_grad
        self.device = gs.device
        self.visualize = show_viewer if visualize is None else visualize

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=config.dt,
                substeps=1,
                substeps_local=config.horizon if requires_grad else 1,
                requires_grad=requires_grad,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=config.enable_collision,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        self.plane = self.scene.add_entity(gs.morphs.Plane()) if (self.visualize or config.enable_collision) else None
        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(
                file=str(ASSETS_PATH / "drone_urdf" / "drone.urdf"),
                pos=(0.0, 0.0, 0.6),
                euler=(0.0, 0.0, 0.0),
                default_armature=2.6e-7,
            ),
        )
        if self.visualize:
            self.target_visual = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=str(ASSETS_PATH / "primitives" / "sphere.obj"),
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5)),
                ),
            )
        else:
            self.target_visual = None
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
        self.angle_kp = torch.tensor(config.angle_kp, device=self.device, dtype=gs.tc_float)
        self.angle_ki = torch.tensor(config.angle_ki, device=self.device, dtype=gs.tc_float)
        self.angle_kd = torch.tensor(config.angle_kd, device=self.device, dtype=gs.tc_float)
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
        self.pid_integral = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_angular_velocity = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
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
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self.body_ang_acc = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self._simulation_forces: list[torch.Tensor] = []
        self.end_on_vertical_error = True
        self.respawn_on_fail = True

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

    def backward(self, physics_loss: torch.Tensor, policy_loss: torch.Tensor | None = None) -> None:
        self.scene.backward(physics_loss)
        if policy_loss is not None:
            torch.autograd.backward(policy_loss)
        self.release_simulation_graphs()

    def release_simulation_graphs(self, physics_loss: torch.Tensor | None = None) -> None:
        # Scene.backward keeps the torch graph (retain_graph=True). A second torch.autograd.backward
        # with retain_graph=False frees it. Call torch.autograd.backward, not gs.Tensor.backward:
        # the latter would re-enter Scene._backward.
        if physics_loss is not None and physics_loss.grad_fn is not None:
            torch.autograd.backward(physics_loss)
        self._simulation_forces.clear()
        _detach_cached_torch_tensor(self.scene.rigid_solver.dyn_state.dofs.ctrl_force)
        self.last_action = detached_torch_tensor(self.last_action)
        self.pid_integral = detached_torch_tensor(self.pid_integral)
        self.last_angular_velocity = detached_torch_tensor(self.last_angular_velocity)

    def _make_observation(self, state: DroneState) -> torch.Tensor:
        position_error = (self.target_position - state.position) * self.config.obs_scale_position_error
        observation = torch.cat(
            (
                state.position,
                self.target_position,
                position_error,
                state.quaternion,
                state.linear_velocity * self.config.obs_scale_linear_velocity,
                state.angular_velocity_body * self.config.obs_scale_angular_velocity,
                self.last_action,
            ),
            dim=-1,
        )
        return torch.where(self.is_alive[:, None], observation, torch.zeros_like(observation))

    def _sync_target_visual(self) -> None:
        if self.target_visual is None:
            return
        self.target_visual.set_pos(self.target_position, zero_velocity=True)

    def reset(
        self,
        initial_position: torch.Tensor | None = None,
        initial_quaternion: torch.Tensor | None = None,
        waypoint_sequences: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.release_simulation_graphs()
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
        self.pid_integral = torch.zeros_like(self.pid_integral)
        self.last_angular_velocity = torch.zeros_like(self.last_angular_velocity)
        self.body_ang_acc = torch.zeros_like(self.body_ang_acc)
        self.is_alive.fill_(True)
        self.episode_step = 0
        self.episode_length_buf.zero_()
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

        self._sync_target_visual()
        return self._make_observation(self._read_state()).detach()

    def mix_action(self, action: torch.Tensor, state: DroneState | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        action = action.clamp(-1.0, 1.0)
        if state is None:
            state = self._read_state()
        body_euler = quaternion_to_roll_pitch_yaw(state.quaternion)
        body_setpoint = -body_euler + action[:, :3]
        angle_error = body_setpoint * 15.0 - state.angular_velocity_body
        proportional = angle_error * self.angle_kp
        integral = torch.clamp(self.pid_integral + angle_error * self.angle_ki * self.config.dt, -0.5, 0.5)
        derivative = torch.clamp(
            (self.last_angular_velocity - state.angular_velocity_body) * self.angle_kd / self.config.dt,
            -0.5,
            0.5,
        )
        pid_output = proportional + integral + derivative
        self.pid_integral = torch.where(self.is_alive[:, None], integral, torch.zeros_like(integral))
        throttle = (action[:, 3] + 1.0) * 0.5
        motor_command = torch.stack(
            (
                throttle - pid_output[:, 0] - pid_output[:, 1] - pid_output[:, 2],
                throttle - pid_output[:, 0] + pid_output[:, 1] + pid_output[:, 2],
                throttle + pid_output[:, 0] + pid_output[:, 1] - pid_output[:, 2],
                throttle + pid_output[:, 0] - pid_output[:, 1] + pid_output[:, 2],
            ),
            dim=-1,
        ).clamp(0.0, 1.0)
        motor_rpm = torch.sqrt((motor_command * self.config.thrust_to_weight_ratio).clamp_min(1e-12)) * self.config.base_rpm
        motor_thrust = self.config.thrust_coefficient * motor_rpm * motor_rpm
        return motor_thrust, motor_thrust @ self.allocation.T


    def step(self, action: torch.Tensor):
        action = action.clamp(-1.0, 1.0)
        is_alive_before = self.is_alive.clone()
        state_before = self._read_state()
        motor_thrust, actual_wrench = self.mix_action(action, state_before)
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
        if generalized_force.requires_grad:
            if isinstance(generalized_force, gs.Tensor):
                generalized_force.scene = None
            else:
                generalized_force = gs.from_torch(generalized_force, detach=False, requires_grad=True)
            self._simulation_forces.append(generalized_force)
        self.drone.control_dofs_force(generalized_force)
        self.scene.step()

        state = self._read_state()
        quaternion = state.quaternion
        position_error = self.target_position - state.position
        last_position_error = self.target_position - state_before.position
        distance = torch.linalg.vector_norm(position_error, dim=-1)
        body_euler = quaternion_to_roll_pitch_yaw(quaternion)
        # Match Odom body_ang_acc: change of body rate over this control step.
        body_ang_acc = (state.angular_velocity_body - state_before.angular_velocity_body) / self.config.dt
        self.body_ang_acc = body_ang_acc
        self.last_angular_velocity = torch.where(
            is_alive_before[:, None], state.angular_velocity_body, torch.zeros_like(state.angular_velocity_body)
        )
        target_reward = -torch.sum(torch.square(position_error), dim=-1) * 0.1
        target_reward = target_reward + torch.sum(last_position_error.abs() - position_error.abs(), dim=-1)
        smooth_reward = torch.linalg.vector_norm(action[:, :3] - self.last_action[:, :3], dim=-1)
        smooth_reward = smooth_reward + (action[:, 3] - self.last_action[:, 3]).abs() * 5.0
        yaw_reward = torch.exp(self.config.yaw_lambda * body_euler[:, 2].abs()) - 1.0
        angular_reward = torch.sum(body_ang_acc.abs(), dim=-1)

        horizontal_distance_x = self.config.horizontal_termination_error - position_error[:, 0].abs()
        horizontal_distance_y = self.config.horizontal_termination_error - position_error[:, 1].abs()
        vertical_distance = self.config.vertical_termination_error - position_error[:, 2].abs()

        has_finite_state = (
            torch.isfinite(state.position).all(dim=-1)
            & torch.isfinite(quaternion).all(dim=-1)
            & torch.isfinite(state.linear_velocity).all(dim=-1)
            & torch.isfinite(state.angular_velocity_body).all(dim=-1)
        )
        is_crashed = (
            (state.position[:, 2] < self.config.ground_termination_height)
            | (body_euler[:, 0].abs() > self.config.roll_termination)
            | (body_euler[:, 1].abs() > self.config.pitch_termination)
            | (horizontal_distance_x < 0.0)
            | (horizontal_distance_y < 0.0)
            | ~has_finite_state
        )
        if self.end_on_vertical_error:
            is_crashed = is_crashed | (vertical_distance < 0.0)
        is_newly_dead = is_alive_before & is_crashed
        is_arrived = is_alive_before & ~is_newly_dead & (distance < self.config.target_threshold)
        target_reward = target_reward + 20.0 * is_arrived.to(dtype=target_reward.dtype)
        crash_reward = is_newly_dead.to(dtype=target_reward.dtype)

        scales = self.config.reward_scales
        step_scale = self.config.dt
        reward = (
            scales.target * target_reward
            + scales.smooth * smooth_reward
            + scales.yaw * yaw_reward
            + scales.angular * angular_reward
            + scales.crash * crash_reward
        ) * step_scale
        reward = torch.where(is_alive_before, reward, torch.zeros_like(reward))
        physics_loss = -reward
        policy_loss = (self.config.action_delta_weight * torch.sum(torch.square(action - self.last_action), dim=-1)) * (
            is_alive_before
        )

        self.episode_step += 1
        self.episode_length_buf += is_alive_before.to(dtype=self.episode_length_buf.dtype)
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

        self._sync_target_visual()
        self.last_action = torch.where(self.is_alive[:, None], action, torch.zeros_like(action))
        is_time_limit = self.episode_length_buf >= self.config.max_episode_steps
        is_truncated = self.is_alive & is_time_limit
        done = is_newly_dead | is_truncated
        # Mid-step respawn mutates buffers used by the autograd graph. Only respawn outside
        # differentiable windows (see detach_window); keep dead masks during BPTT.
        if self.respawn_on_fail and not self.requires_grad:
            reset_idx = (is_newly_dead | is_truncated).nonzero(as_tuple=False).flatten()
            if reset_idx.numel() > 0:
                self._reset_envs(reset_idx)
            observation = self._make_observation(self._read_state())
        else:
            self.is_alive = self.is_alive & ~is_truncated
            self.last_action = torch.where(self.is_alive[:, None], self.last_action, torch.zeros_like(self.last_action))
            observation = self._make_observation(state)
        extras = {
            "terminated": is_newly_dead.detach(),
            "truncated": is_truncated.detach(),
            "alive": self.is_alive.detach(),
            "arrived": is_arrived.detach(),
            "actual_wrench": actual_wrench.detach(),
            "motor_thrust": motor_thrust.detach(),
            "loss_components": {
                "target": (target_reward * is_alive_before).detach(),
                "smooth": (smooth_reward * is_alive_before).detach(),
                "yaw": (yaw_reward * is_alive_before).detach(),
                "angular": (angular_reward * is_alive_before).detach(),
                "crash": crash_reward.detach(),
            },
            "metrics": {
                "position_error": (distance * is_alive_before).detach(),
            },
        }
        return observation, (physics_loss, policy_loss, reward), done.detach(), extras

    def _reset_envs(self, env_idx: torch.Tensor) -> None:
        count = int(env_idx.numel())
        if count == 0:
            return
        initial_position = torch.empty((count, 3), device=self.device, dtype=gs.tc_float)
        initial_position[:, 0].uniform_(*self.config.initial_x_range)
        initial_position[:, 1].uniform_(*self.config.initial_y_range)
        initial_position[:, 2].uniform_(*self.config.initial_z_range)
        yaw = torch.empty(count, device=self.device, dtype=gs.tc_float).uniform_(-torch.pi, torch.pi)
        initial_quaternion = torch.stack(
            (torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw)), dim=-1
        )
        self.drone.set_pos(initial_position, envs_idx=env_idx, zero_velocity=True)
        self.drone.set_quat(initial_quaternion, envs_idx=env_idx, zero_velocity=True)
        self.last_action[env_idx] = 0.0
        self.pid_integral[env_idx] = 0.0
        self.last_angular_velocity[env_idx] = 0.0
        self.body_ang_acc[env_idx] = 0.0
        self.episode_length_buf[env_idx] = 0
        self.is_alive[env_idx] = True
        if self.waypoint_sequences is None:
            self.target_position[env_idx] = self.target_lower + (self.target_upper - self.target_lower) * torch.rand(
                (count, 3), device=self.device, dtype=gs.tc_float
            )
        else:
            self.waypoint_index[env_idx] = 0
            self.target_position[env_idx] = self.waypoint_sequences[env_idx, 0]
        self.waypoint_count[env_idx] = 0
        self.first_arrival_step[env_idx] = self.config.max_episode_steps
        self.crash_step[env_idx] = self.config.max_episode_steps + 1
        self.position_error_sum[env_idx] = 0.0
        self.position_error_steps[env_idx] = 0.0
        self._sync_target_visual()

    def state_dict(self) -> dict:
        drone_state = self._read_state()
        dofs_velocity = torch.cat((drone_state.linear_velocity, drone_state.angular_velocity_body), dim=-1)
        return {
            "position": detached_torch_tensor(drone_state.position),
            "quaternion": detached_torch_tensor(drone_state.quaternion),
            "dofs_velocity": detached_torch_tensor(dofs_velocity),
            "target_position": detached_torch_tensor(self.target_position),
            "last_action": detached_torch_tensor(self.last_action),
            "pid_integral": detached_torch_tensor(self.pid_integral),
            "last_angular_velocity": detached_torch_tensor(self.last_angular_velocity),
            "is_alive": detached_torch_tensor(self.is_alive),
            "waypoint_sequences": (
                None if self.waypoint_sequences is None else detached_torch_tensor(self.waypoint_sequences)
            ),
            "waypoint_index": detached_torch_tensor(self.waypoint_index),
            "episode_step": self.episode_step,
            "episode_length_buf": detached_torch_tensor(self.episode_length_buf),
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
        self.pid_integral = state.get("pid_integral", torch.zeros_like(self.pid_integral)).to(self.device)
        self.last_angular_velocity = state.get(
            "last_angular_velocity", torch.zeros_like(self.last_angular_velocity)
        ).to(self.device)
        self.is_alive = state["is_alive"].to(self.device)
        self.waypoint_sequences = (
            None if state["waypoint_sequences"] is None else state["waypoint_sequences"].to(self.device)
        )
        self.waypoint_index = state["waypoint_index"].to(self.device)
        self.episode_step = state["episode_step"]
        self.episode_length_buf = state.get(
            "episode_length_buf", torch.zeros_like(self.episode_length_buf)
        ).to(self.device)
        self.waypoint_count = state["waypoint_count"].to(self.device)
        self.first_arrival_step = state["first_arrival_step"].to(self.device)
        self.crash_step = state["crash_step"].to(self.device)
        self.position_error_sum = state["position_error_sum"].to(self.device)
        self.position_error_steps = state["position_error_steps"].to(self.device)
        return self.detach_window()

    def detach_window(self) -> torch.Tensor:
        self.release_simulation_graphs()
        self.target_position = detached_torch_tensor(self.target_position)
        self.last_action = detached_torch_tensor(self.last_action)
        self.pid_integral = detached_torch_tensor(self.pid_integral)
        self.last_angular_velocity = detached_torch_tensor(self.last_angular_velocity)
        self.body_ang_acc = detached_torch_tensor(self.body_ang_acc)
        self.is_alive = detached_torch_tensor(self.is_alive)
        self.episode_length_buf = detached_torch_tensor(self.episode_length_buf)
        if self.waypoint_sequences is not None:
            self.waypoint_sequences = detached_torch_tensor(self.waypoint_sequences)
        self.waypoint_index = detached_torch_tensor(self.waypoint_index)
        self.waypoint_count = detached_torch_tensor(self.waypoint_count)
        self.first_arrival_step = detached_torch_tensor(self.first_arrival_step)
        self.crash_step = detached_torch_tensor(self.crash_step)
        self.position_error_sum = detached_torch_tensor(self.position_error_sum)
        self.position_error_steps = detached_torch_tensor(self.position_error_steps)
        if self.respawn_on_fail:
            reset_idx = (~self.is_alive).nonzero(as_tuple=False).flatten()
            if reset_idx.numel() > 0:
                self._reset_envs(reset_idx)
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
