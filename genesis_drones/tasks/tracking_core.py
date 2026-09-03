from dataclasses import dataclass, fields
from typing import NamedTuple

import torch
from torch.nn import functional as F

from genesis_drones.utils.geometry import quaternion_to_roll_pitch_yaw

OBSERVATION_SIZE = 23


@dataclass(frozen=True)
class RewardScales:
    target: float = 10.0
    smooth: float = -1.0e-4
    yaw: float = 0.01
    angular: float = -2.0e-4
    crash: float = -10.0
    velocity: float = 0.0


@dataclass(frozen=True)
class TrackingCoreConfig:
    dt: float = 0.01
    max_episode_steps: int = 1500
    target_threshold: float = 0.1
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
    initial_x_range: tuple[float, float] = (-0.05, 0.05)
    initial_y_range: tuple[float, float] = (-0.05, 0.05)
    initial_z_range: tuple[float, float] = (0.6, 0.61)
    target_x_range: tuple[float, float] = (-1.2, 1.2)
    target_y_range: tuple[float, float] = (-1.2, 1.2)
    target_z_range: tuple[float, float] = (0.6, 1.0)
    reward_scales: RewardScales = RewardScales()
    progress_norm: str = "l1"
    closing_velocity_weight: float = 0.0
    arrival_surrogate: str = "none"
    arrival_surrogate_sigma: float = 0.15
    fully_differentiable: bool = False
    tracking_position_weight: float = 1.0
    tracking_attitude_weight: float = 0.2
    tracking_velocity_weight: float = 0.05
    tracking_angular_rate_weight: float = 0.02
    tracking_action_smooth_weight: float = 1.0e-4
    tracking_safety_weight: float = 1.0

    @classmethod
    def from_compat(cls, config) -> "TrackingCoreConfig":
        return cls(**{field.name: getattr(config, field.name) for field in fields(cls)})


class TrackingEvents(NamedTuple):
    arrived: torch.Tensor
    crashed: torch.Tensor


class InitialPose(NamedTuple):
    position: torch.Tensor
    quaternion: torch.Tensor
    linear_velocity: torch.Tensor


class TrackingEvalResult(NamedTuple):
    reward: torch.Tensor
    physics_loss: torch.Tensor
    policy_loss: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    events: TrackingEvents
    body_ang_acc: torch.Tensor
    distance: torch.Tensor
    loss_components: dict
    metrics: dict


def smooth_safety_penalty(distance: torch.Tensor, warning_distance: float, temperature_ratio: float) -> torch.Tensor:
    temperature = warning_distance * temperature_ratio
    normalizer = F.softplus(distance.new_tensor(warning_distance / temperature))
    return torch.square(F.softplus((warning_distance - distance) / temperature) / normalizer)


def tracking_progress(last_position_error: torch.Tensor, position_error: torch.Tensor, norm: str) -> torch.Tensor:
    if norm == "l2":
        return torch.linalg.vector_norm(last_position_error, dim=-1) - torch.linalg.vector_norm(position_error, dim=-1)
    return torch.sum(last_position_error.abs() - position_error.abs(), dim=-1)


def closing_velocity(linear_velocity: torch.Tensor, position_error: torch.Tensor) -> torch.Tensor:
    distance = torch.linalg.vector_norm(position_error, dim=-1).clamp_min(1e-6)
    return (linear_velocity * position_error).sum(dim=-1) / distance


def arrival_surrogate_bonus(distance: torch.Tensor, kind: str, sigma: float) -> torch.Tensor:
    if kind == "gaussian":
        return torch.exp(-torch.square(distance) / (2.0 * sigma * sigma))
    if kind == "sigmoid":
        return torch.sigmoid((0.1 - distance) / max(sigma, 1e-6))
    return torch.zeros_like(distance)


def attitude_error(quaternion: torch.Tensor, target_quaternion: torch.Tensor) -> torch.Tensor:
    alignment = (quaternion * target_quaternion).sum(dim=-1)
    return 1.0 - alignment * alignment


def tracking_safety_penalty(
    position: torch.Tensor,
    position_error: torch.Tensor,
    config: TrackingCoreConfig,
) -> torch.Tensor:
    ratio = config.safety_temperature_ratio
    return (
        smooth_safety_penalty(position[:, 2], config.ground_warning_distance, ratio)
        + smooth_safety_penalty(
            config.horizontal_termination_error - position_error[:, 0].abs(),
            config.horizontal_warning_distance,
            ratio,
        )
        + smooth_safety_penalty(
            config.horizontal_termination_error - position_error[:, 1].abs(),
            config.horizontal_warning_distance,
            ratio,
        )
        + smooth_safety_penalty(
            config.vertical_termination_error - position_error[:, 2].abs(),
            config.vertical_warning_distance,
            ratio,
        )
    )


def differentiable_tracking_reward(
    position: torch.Tensor,
    target_position: torch.Tensor,
    linear_velocity: torch.Tensor,
    quaternion: torch.Tensor,
    target_quaternion: torch.Tensor,
    angular_velocity: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    safety_penalty: torch.Tensor,
    config: TrackingCoreConfig,
) -> torch.Tensor:
    # ponytail: static hover refs v*=0, w*=0, q*=identity. Time-indexed trajectory when Racing needs it.
    position_term = torch.sum(torch.square(position - target_position), dim=-1)
    velocity_term = torch.sum(torch.square(linear_velocity), dim=-1)
    rate_term = torch.sum(torch.square(angular_velocity), dim=-1)
    smooth_term = torch.sum(torch.square(action - last_action), dim=-1)
    return -(
        config.tracking_position_weight * position_term
        + config.tracking_velocity_weight * velocity_term
        + config.tracking_attitude_weight * attitude_error(quaternion, target_quaternion)
        + config.tracking_angular_rate_weight * rate_term
        + config.tracking_action_smooth_weight * smooth_term
        + config.tracking_safety_weight * safety_penalty
    )


def _angular_velocity(state) -> torch.Tensor:
    angular = getattr(state, "angular_velocity_body", None)
    if angular is None:
        angular = state.angular_velocity
    return angular


class TrackingCore:
    observation_dim = OBSERVATION_SIZE
    policy_observation_dim = OBSERVATION_SIZE
    critic_observation_dim = OBSERVATION_SIZE
    action_dim = 4

    def __init__(
        self,
        config: TrackingCoreConfig,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.envs_idx = torch.arange(num_envs, device=self.device)
        self.target_lower = torch.tensor(
            [config.target_x_range[0], config.target_y_range[0], config.target_z_range[0]],
            device=self.device,
            dtype=dtype,
        )
        self.target_upper = torch.tensor(
            [config.target_x_range[1], config.target_y_range[1], config.target_z_range[1]],
            device=self.device,
            dtype=dtype,
        )
        self.target_position = torch.zeros((num_envs, 3), device=self.device, dtype=dtype)
        self.last_action = torch.zeros((num_envs, self.action_dim), device=self.device, dtype=dtype)
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
        self.position_error_sum = torch.zeros(num_envs, device=self.device, dtype=dtype)
        self.position_error_steps = torch.zeros(num_envs, device=self.device, dtype=dtype)
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self.body_ang_acc = torch.zeros((num_envs, 3), device=self.device, dtype=dtype)
        self.end_on_vertical_error = True

    def sample_target(self, count: int | None = None) -> torch.Tensor:
        n = self.num_envs if count is None else count
        return self.target_lower + (self.target_upper - self.target_lower) * torch.rand(
            (n, 3), device=self.device, dtype=self.dtype
        )

    def sample_initial_pose(
        self,
        count: int,
        seed: int | None = None,
        initial_states: InitialPose | None = None,
    ) -> InitialPose:
        if initial_states is not None:
            return InitialPose(
                initial_states.position.to(device=self.device, dtype=self.dtype),
                initial_states.quaternion.to(device=self.device, dtype=self.dtype),
                initial_states.linear_velocity.to(device=self.device, dtype=self.dtype),
            )
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)
        position = torch.empty((count, 3), device=self.device, dtype=self.dtype)
        position[:, 0].uniform_(*self.config.initial_x_range, generator=generator)
        position[:, 1].uniform_(*self.config.initial_y_range, generator=generator)
        position[:, 2].uniform_(*self.config.initial_z_range, generator=generator)
        yaw = torch.empty(count, device=self.device, dtype=self.dtype)
        yaw.uniform_(-torch.pi, torch.pi, generator=generator)
        quaternion = torch.stack(
            (torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw)),
            dim=-1,
        )
        return InitialPose(position, quaternion, torch.zeros_like(position))

    def crash_mask(
        self,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        position_error: torch.Tensor,
        body_euler: torch.Tensor | None = None,
        end_on_vertical_error: bool | None = None,
    ) -> torch.Tensor:
        if body_euler is None:
            body_euler = quaternion_to_roll_pitch_yaw(quaternion)
        config = self.config
        horizontal_distance_x = config.horizontal_termination_error - position_error[:, 0].abs()
        horizontal_distance_y = config.horizontal_termination_error - position_error[:, 1].abs()
        vertical_distance = config.vertical_termination_error - position_error[:, 2].abs()
        has_finite_state = (
            torch.isfinite(position).all(dim=-1)
            & torch.isfinite(quaternion).all(dim=-1)
            & torch.isfinite(linear_velocity).all(dim=-1)
            & torch.isfinite(angular_velocity).all(dim=-1)
        )
        is_crashed = (
            (position[:, 2] < config.ground_termination_height)
            | (body_euler[:, 0].abs() > config.roll_termination)
            | (body_euler[:, 1].abs() > config.pitch_termination)
            | (horizontal_distance_x < 0.0)
            | (horizontal_distance_y < 0.0)
            | ~has_finite_state
        )
        if self.end_on_vertical_error if end_on_vertical_error is None else end_on_vertical_error:
            is_crashed = is_crashed | (vertical_distance < 0.0)
        return is_crashed

    def arrival_mask(self, distance: torch.Tensor) -> torch.Tensor:
        return distance < self.config.target_threshold

    def reset_task(
        self,
        indices: torch.Tensor | None = None,
        initial_states: InitialPose | None = None,
        seed: int | None = None,
        waypoint_sequences: torch.Tensor | None = None,
        env_idx: torch.Tensor | None = None,
    ) -> InitialPose:
        # Compat: older callers used env_idx= / waypoint_sequences= without pose return.
        if env_idx is not None:
            indices = env_idx
        if indices is None:
            indices = self.envs_idx
            full_reset = True
        else:
            full_reset = int(indices.numel()) == self.num_envs and torch.equal(
                indices, self.envs_idx
            )
        count = int(indices.numel())
        pose = self.sample_initial_pose(count, seed=seed, initial_states=initial_states)
        if count == 0:
            return pose

        if full_reset:
            self.last_action.zero_()
            self.is_alive.fill_(True)
            self.episode_step = 0
            self.episode_length_buf.zero_()
            self.waypoint_index.zero_()
            self.waypoint_count.zero_()
            self.first_arrival_step.fill_(self.config.max_episode_steps)
            self.crash_step.fill_(self.config.max_episode_steps + 1)
            self.position_error_sum.zero_()
            self.position_error_steps.zero_()
            self.body_ang_acc.zero_()
            if waypoint_sequences is None and self.waypoint_sequences is None:
                self.waypoint_sequences = None
                self.target_position = self.sample_target()
            elif waypoint_sequences is not None:
                if waypoint_sequences.shape[0] != self.num_envs:
                    raise ValueError("waypoint_sequences must have one sequence per environment")
                self.waypoint_sequences = waypoint_sequences.to(device=self.device, dtype=self.dtype)
                self.target_position = self.waypoint_sequences[:, 0]
            else:
                self.waypoint_index.zero_()
                self.target_position = self.waypoint_sequences[:, 0]
            return pose

        self.last_action[indices] = 0.0
        self.episode_length_buf[indices] = 0
        self.is_alive[indices] = True
        self.body_ang_acc[indices] = 0.0
        if waypoint_sequences is not None:
            self.waypoint_sequences = waypoint_sequences.to(device=self.device, dtype=self.dtype)
        if self.waypoint_sequences is None:
            self.target_position[indices] = self.sample_target(count)
        else:
            self.waypoint_index[indices] = 0
            self.target_position[indices] = self.waypoint_sequences[indices, 0]
        self.waypoint_count[indices] = 0
        self.first_arrival_step[indices] = self.config.max_episode_steps
        self.crash_step[indices] = self.config.max_episode_steps + 1
        self.position_error_sum[indices] = 0.0
        self.position_error_steps[indices] = 0.0
        return pose

    def observe(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        position_error = (self.target_position - state.position) * self.config.obs_scale_position_error
        observation = torch.cat(
            (
                state.position,
                self.target_position,
                position_error,
                state.quaternion,
                state.linear_velocity * self.config.obs_scale_linear_velocity,
                _angular_velocity(state) * self.config.obs_scale_angular_velocity,
                self.last_action,
            ),
            dim=-1,
        )
        observation = torch.where(self.is_alive[:, None], observation, torch.zeros_like(observation))
        return observation, observation

    def evaluate(
        self,
        state_before,
        state_after,
        action: torch.Tensor,
        is_alive_before: torch.Tensor,
        end_on_vertical_error: bool | None = None,
    ) -> TrackingEvalResult:
        action = action.clamp(-1.0, 1.0)
        config = self.config
        position_error = self.target_position - state_after.position
        last_position_error = self.target_position - state_before.position
        distance = torch.linalg.vector_norm(position_error, dim=-1)
        quaternion = state_after.quaternion
        angular_velocity = _angular_velocity(state_after)
        body_euler = quaternion_to_roll_pitch_yaw(quaternion)
        body_ang_acc = (angular_velocity - _angular_velocity(state_before)) / config.dt
        self.body_ang_acc = body_ang_acc

        target_reward = -torch.sum(torch.square(position_error), dim=-1) * 0.1
        target_reward = target_reward + tracking_progress(last_position_error, position_error, config.progress_norm)
        target_reward = target_reward + config.closing_velocity_weight * closing_velocity(
            state_after.linear_velocity, position_error
        )
        target_reward = target_reward + arrival_surrogate_bonus(
            distance, config.arrival_surrogate, config.arrival_surrogate_sigma
        )
        smooth_reward = torch.linalg.vector_norm(action[:, :3] - self.last_action[:, :3], dim=-1)
        smooth_reward = smooth_reward + (action[:, 3] - self.last_action[:, 3]).abs() * 5.0
        yaw_reward = torch.exp(config.yaw_lambda * body_euler[:, 2].abs()) - 1.0
        angular_reward = torch.sum(body_ang_acc.abs(), dim=-1)

        is_crashed = self.crash_mask(
            state_after.position,
            quaternion,
            state_after.linear_velocity,
            angular_velocity,
            position_error,
            body_euler=body_euler,
            end_on_vertical_error=end_on_vertical_error,
        )
        is_newly_dead = is_alive_before & is_crashed
        is_arrived = is_alive_before & ~is_newly_dead & self.arrival_mask(distance)
        crash_reward = is_newly_dead.to(dtype=target_reward.dtype)
        scales = config.reward_scales
        step_scale = config.dt
        if config.fully_differentiable:
            hover_quaternion = quaternion.new_tensor([1.0, 0.0, 0.0, 0.0]).expand_as(quaternion)
            reward = (
                differentiable_tracking_reward(
                    state_after.position,
                    self.target_position,
                    state_after.linear_velocity,
                    quaternion,
                    hover_quaternion,
                    angular_velocity,
                    action,
                    self.last_action,
                    tracking_safety_penalty(state_after.position, position_error, config),
                    config,
                )
                * step_scale
            )
        else:
            target_reward = target_reward + 20.0 * is_arrived.to(dtype=target_reward.dtype)
            reward = (
                scales.target * target_reward
                + scales.smooth * smooth_reward
                + scales.yaw * yaw_reward
                + scales.angular * angular_reward
                + scales.crash * crash_reward
            ) * step_scale
        reward = torch.where(is_alive_before, reward, torch.zeros_like(reward))
        physics_loss = -reward
        policy_loss = (config.action_delta_weight * torch.sum(torch.square(action - self.last_action), dim=-1)) * (
            is_alive_before
        )

        self.episode_step += 1
        self.episode_length_buf += is_alive_before.to(dtype=self.episode_length_buf.dtype)
        newly_arrived = is_arrived & (self.first_arrival_step == config.max_episode_steps)
        self.waypoint_count += newly_arrived if config.fully_differentiable else is_arrived
        self.first_arrival_step = torch.where(
            is_arrived & (self.first_arrival_step == config.max_episode_steps),
            torch.full_like(self.first_arrival_step, self.episode_step),
            self.first_arrival_step,
        )
        self.crash_step = torch.where(
            is_newly_dead, torch.full_like(self.crash_step, self.episode_step), self.crash_step
        )
        self.position_error_sum += distance.detach() * is_alive_before
        self.position_error_steps += is_alive_before
        self.is_alive = is_alive_before & ~is_newly_dead

        if not config.fully_differentiable:
            if self.waypoint_sequences is None:
                next_target = self.sample_target()
                self.target_position = torch.where(is_arrived[:, None], next_target, self.target_position)
            else:
                self.waypoint_index = torch.clamp(
                    self.waypoint_index + is_arrived,
                    max=self.waypoint_sequences.shape[1] - 1,
                )
                self.target_position = self.waypoint_sequences[self.envs_idx, self.waypoint_index]

        self.last_action = torch.where(self.is_alive[:, None], action, torch.zeros_like(action))
        is_time_limit = self.episode_length_buf >= config.max_episode_steps
        is_truncated = self.is_alive & is_time_limit
        loss_components = {
            "target": (target_reward * is_alive_before).detach(),
            "smooth": (smooth_reward * is_alive_before).detach(),
            "yaw": (yaw_reward * is_alive_before).detach(),
            "angular": (angular_reward * is_alive_before).detach(),
            "crash": crash_reward.detach(),
        }
        return TrackingEvalResult(
            reward=reward,
            physics_loss=physics_loss,
            policy_loss=policy_loss,
            terminated=is_newly_dead,
            truncated=is_truncated,
            events=TrackingEvents(arrived=is_arrived, crashed=is_newly_dead),
            body_ang_acc=body_ang_acc,
            distance=distance,
            loss_components=loss_components,
            metrics={
                "position_error": (distance * is_alive_before).detach(),
                "loss_components": loss_components,
                "arrived": is_arrived.detach(),
            },
        )

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

    def detach_buffers(self, detach_fn) -> None:
        self.target_position = detach_fn(self.target_position)
        self.last_action = detach_fn(self.last_action)
        self.body_ang_acc = detach_fn(self.body_ang_acc)
        self.is_alive = detach_fn(self.is_alive)
        self.episode_length_buf = detach_fn(self.episode_length_buf)
        if self.waypoint_sequences is not None:
            self.waypoint_sequences = detach_fn(self.waypoint_sequences)
        self.waypoint_index = detach_fn(self.waypoint_index)
        self.waypoint_count = detach_fn(self.waypoint_count)
        self.first_arrival_step = detach_fn(self.first_arrival_step)
        self.crash_step = detach_fn(self.crash_step)
        self.position_error_sum = detach_fn(self.position_error_sum)
        self.position_error_steps = detach_fn(self.position_error_steps)
