from dataclasses import dataclass
from typing import NamedTuple

import torch

from genesis_drones.dynamics.base import DroneState
from genesis_drones.tasks.racing_core import (
    POLICY_OBSERVATION_SIZE,
    STATE_OBSERVATION_SIZE,
    EvaluationInitialStates,
    RaceTrackSpec,
    RaceTrackTensors,
    detect_race_events,
    generate_initial_states,
    racing_loss_reward,
    racing_observations,
    track_to_tensors,
)
from genesis_drones.tasks.racing_tracks import RACING_TRACK


@dataclass(frozen=True)
class RacingCoreConfig:
    dt: float = 0.0333
    max_episode_steps: int = int(40.0 / 0.0333)
    min_target_velocity: float = 5.0
    max_target_velocity: float = 10.0
    gamma: float = 0.99
    td_lambda: float = 0.95


class RacingEvents(NamedTuple):
    passed: torch.Tensor
    wrong_way: torch.Tensor
    analytic_collision: torch.Tensor
    completed: torch.Tensor
    next_gate_index: torch.Tensor


class RacingEvalResult(NamedTuple):
    reward: torch.Tensor
    physics_loss: torch.Tensor
    policy_loss: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    events: RacingEvents
    metrics: dict


class RacingCore:
    policy_observation_dim = POLICY_OBSERVATION_SIZE
    critic_observation_dim = POLICY_OBSERVATION_SIZE
    state_observation_dim = STATE_OBSERVATION_SIZE

    def __init__(
        self,
        config: RacingCoreConfig,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        track: RaceTrackSpec = RACING_TRACK,
    ):
        self.config = config
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        self.track_spec = track
        self.track: RaceTrackTensors = track_to_tensors(track, device, dtype)
        self.target_gates = torch.zeros(num_envs, device=device, dtype=torch.int64)
        self.n_passed_gates = torch.zeros(num_envs, device=device, dtype=torch.int64)
        self.path_length = torch.zeros(num_envs, device=device, dtype=dtype)
        self.max_velocity = torch.full(
            (num_envs,), config.min_target_velocity, device=device, dtype=dtype
        )
        self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.int64)

    def reset_task(
        self,
        indices: torch.Tensor,
        initial_states: EvaluationInitialStates | None = None,
        seed: int | None = None,
        waypoint_sequences=None,
    ) -> EvaluationInitialStates:
        count = int(indices.numel())
        if initial_states is None:
            initial_states = generate_initial_states(self.track, count, seed=0 if seed is None else seed)
        self.target_gates[indices] = initial_states.target_gate.to(device=self.device, dtype=torch.int64)
        self.n_passed_gates[indices] = 0
        self.path_length[indices] = 0.0
        self.episode_length_buf[indices] = 0
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed + 1)
        self.max_velocity[indices] = torch.rand(
            count, device=self.device, dtype=self.dtype, generator=generator
        ) * (self.config.max_target_velocity - self.config.min_target_velocity) + self.config.min_target_velocity
        return initial_states

    def observe(self, state: DroneState) -> tuple[torch.Tensor, torch.Tensor]:
        return racing_observations(
            state.position,
            state.quaternion,
            state.linear_velocity,
            self.track,
            self.target_gates,
        )

    def evaluate(
        self,
        state_before: DroneState,
        state_after: DroneState,
        action: torch.Tensor,
        is_alive_before: torch.Tensor,
        **_kwargs,
    ) -> RacingEvalResult:
        events = detect_race_events(
            state_before.position,
            state_after.position,
            self.track,
            self.target_gates,
            is_alive_before,
        )
        self.target_gates = events.next_gate_index
        self.n_passed_gates = self.n_passed_gates + events.passed.to(dtype=self.n_passed_gates.dtype)
        target_position = self.track.positions[self.track.order[self.target_gates]]
        physics_loss, reward, loss_components = racing_loss_reward(
            state_after.position,
            state_before.position,
            state_after.quaternion,
            state_after.linear_velocity,
            state_after.angular_velocity,
            target_position,
            self.max_velocity,
            events.analytic_collision,
        )
        policy_loss = action.new_zeros(action.shape[0])
        out_of_bounds = torch.any(torch.abs(state_after.position[:, :2]) > 5.0, dim=-1) | (
            state_after.position[:, 2] > 7.0
        )
        terminated = is_alive_before & events.analytic_collision
        truncated = is_alive_before & (
            out_of_bounds | (self.episode_length_buf >= self.config.max_episode_steps)
        )
        self.episode_length_buf = self.episode_length_buf + is_alive_before.to(dtype=self.episode_length_buf.dtype)
        success = truncated & (self.episode_length_buf >= self.config.max_episode_steps)
        self.path_length = self.path_length + torch.linalg.vector_norm(
            state_after.position - state_before.position, dim=-1
        ) * is_alive_before
        racing_events = RacingEvents(
            events.passed,
            events.wrong_way,
            events.analytic_collision,
            events.completed,
            events.next_gate_index,
        )
        metrics = {
            "success": success,
            "n_passed_gates": self.n_passed_gates.clone(),
            "episode_length": self.episode_length_buf.clone(),
            "target_gate": self.target_gates.detach(),
            "loss_components": {key: value.detach() for key, value in loss_components.items()},
        }
        return RacingEvalResult(
            reward,
            physics_loss,
            policy_loss,
            terminated,
            truncated,
            racing_events,
            metrics,
        )
