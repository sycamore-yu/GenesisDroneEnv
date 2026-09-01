from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch

import genesis as gs

from genesis_drones.controllers.ctbr_controller import CtbrController, CtbrControllerConfig
from genesis_drones.envs.differentiable import (
    DiffEnvSpec,
    DiffObservation,
    DiffTransition,
    finish_simulation_window,
)
from genesis_drones.tasks.racing_core import (
    CRITIC_OBSERVATION_SIZE,
    POLICY_OBSERVATION_SIZE,
    RaceTrackSpec,
    detect_race_events,
    gate_collision_loss,
    gate_progress_loss,
    gate_vector_field_loss,
    generate_initial_states,
    quaternion_to_rotation_matrix,
    racing_observations,
    track_to_tensors,
)
from genesis_drones.tasks.racing_tracks import FIXED_SEVEN_GATE_TRACK, add_track_gates


ASSETS_PATH = Path(__file__).resolve().parents[1] / "robots" / "assets"


@dataclass(frozen=True)
class RacingRewardScales:
    progress: float = 0.4
    gate_pass: float = 5.0
    complete: float = 10.0
    collision: float = -10.0
    smooth: float = -0.01


@dataclass(frozen=True)
class RacingLossScales:
    progress: float = 0.4
    acceleration: float = 0.01
    jerk: float = 0.001
    vector_field: float = 0.4
    collision: float = 3.0


@dataclass(frozen=True)
class RaceEnvConfig:
    dt: float = 0.01
    horizon: int = 32
    max_episode_steps: int = 3000
    ground_termination_height: float = 0.1
    safety_radius: float = 0.06
    smooth_min_temperature: float = 0.01
    collision_beta_1: float = 1.0
    collision_beta_2: float = 4.0
    enable_gate_contact: bool = False
    gamma: float = 0.999
    td_lambda: float = 0.95
    controller: CtbrControllerConfig = CtbrControllerConfig()
    reward_scales: RacingRewardScales = RacingRewardScales()
    loss_scales: RacingLossScales = RacingLossScales()


class DroneState(NamedTuple):
    position: torch.Tensor
    quaternion: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity_body: torch.Tensor


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


class RaceEnv:
    action_dim = 4
    policy_observation_dim = POLICY_OBSERVATION_SIZE
    critic_observation_dim = CRITIC_OBSERVATION_SIZE

    @classmethod
    def spec_from_config(cls, config: RaceEnvConfig) -> DiffEnvSpec:
        hover_action = 2.0 / config.controller.thrust_to_weight_ratio - 1.0
        return DiffEnvSpec(
            policy_observation_dim=cls.policy_observation_dim,
            critic_observation_dim=cls.critic_observation_dim,
            action_dim=cls.action_dim,
            horizon=config.horizon,
            nominal_action=(hover_action, 0.0, 0.0, 0.0),
        )

    def __init__(
        self,
        config: RaceEnvConfig,
        num_envs: int,
        track: RaceTrackSpec = FIXED_SEVEN_GATE_TRACK,
        requires_grad: bool = True,
        show_viewer: bool = False,
    ):
        self.config = config
        self.num_envs = num_envs
        self.requires_grad = requires_grad
        self.device = gs.device
        self.track_spec = track
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
                enable_collision=config.enable_gate_contact,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(
                file=str(ASSETS_PATH / "drone_urdf" / "drone.urdf"),
                pos=(0.0, 0.0, 1.0),
                euler=(0.0, 0.0, 0.0),
                default_armature=2.6e-7,
            ),
        )
        self.gate_entities = []
        if show_viewer or config.enable_gate_contact:
            self.gate_entities = add_track_gates(self.scene, track, enable_contact=config.enable_gate_contact)
        self.scene.build(n_envs=num_envs)
        self.drone.set_dofs_damping([0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4])
        self.track = track_to_tensors(track, self.device, gs.tc_float)
        self.controller = CtbrController(config.controller, num_envs, self.device, gs.tc_float)
        self.spec = self.spec_from_config(config)
        self.last_action = torch.zeros((num_envs, self.action_dim), device=self.device, dtype=gs.tc_float)
        self.gate_index = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self.is_alive = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self.last_acceleration = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.gate_pass_count = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self.gate_pass_time = torch.full(
            (num_envs, len(track.gates)), float(config.max_episode_steps), device=self.device, dtype=gs.tc_float
        )
        self.path_length = torch.zeros(num_envs, device=self.device, dtype=gs.tc_float)
        self._simulation_forces: list[torch.Tensor] = []
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

    def _observations(self, state: DroneState) -> tuple[torch.Tensor, torch.Tensor]:
        policy, critic = racing_observations(
            state.position,
            state.quaternion,
            state.linear_velocity,
            state.angular_velocity_body,
            self.last_action,
            self.track,
            self.gate_index,
        )
        alive = self.is_alive[:, None]
        return torch.where(alive, policy, torch.zeros_like(policy)), torch.where(alive, critic, torch.zeros_like(critic))

    def release_simulation_graphs(self, physics_loss: torch.Tensor | None = None) -> None:
        if physics_loss is not None and physics_loss.grad_fn is not None:
            torch.autograd.backward(physics_loss)
        self._simulation_forces.clear()
        _detach_cached_torch_tensor(self.scene.rigid_solver.dyn_state.dofs.ctrl_force)
        self.last_action = detached_torch_tensor(self.last_action)
        self.controller.detach()

    def reset(self, initial_states=None, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        self.release_simulation_graphs()
        self.scene.reset()
        if initial_states is None:
            initial_states = generate_initial_states(self.track, self.num_envs, seed=seed)
        position = initial_states.position.to(device=self.device, dtype=gs.tc_float)
        quaternion = initial_states.quaternion.to(device=self.device, dtype=gs.tc_float)
        linear_velocity = initial_states.linear_velocity.to(device=self.device, dtype=gs.tc_float)
        self.drone.set_pos(position, zero_velocity=True)
        self.drone.set_quat(quaternion, zero_velocity=True)
        self.drone.set_dofs_velocity(torch.cat((linear_velocity, torch.zeros_like(linear_velocity)), dim=-1))
        self.last_action.zero_()
        self.controller.reset()
        self.gate_index.zero_()
        self.is_alive.fill_(True)
        self.episode_length_buf.zero_()
        self.last_acceleration.zero_()
        self.gate_pass_count.zero_()
        self.gate_pass_time.fill_(self.config.max_episode_steps)
        self.path_length.zero_()
        return self._observations(self._read_state())

    def reset_diff(self, seed: int | None = None) -> DiffObservation:
        policy, critic = self.reset(seed=0 if seed is None else seed)
        return DiffObservation(policy=policy, critic=critic)

    def mix_action(self, action: torch.Tensor, state: DroneState | None = None):
        if state is None:
            state = self._read_state()
        return self.controller.step(action, state.angular_velocity_body, self.is_alive)

    def _physics_contacts(self) -> torch.Tensor:
        if not self.config.enable_gate_contact:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        contacts = self.drone.get_contacts()
        force = None if contacts is None else contacts.get("force")
        if force is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        force = force if isinstance(force, torch.Tensor) else torch.as_tensor(force, device=self.device)
        return force.reshape(force.shape[0], -1).abs().sum(dim=-1) > 1e-6

    def step(self, action: torch.Tensor):
        action = action.clamp(-1.0, 1.0)
        is_alive_before = self.is_alive.clone()
        state_before = self._read_state()
        output = self.mix_action(action, state_before)
        actual_wrench = output.wrench * is_alive_before[:, None]
        rotation = quaternion_to_rotation_matrix(state_before.quaternion)
        force_world = rotation[:, :, 2] * actual_wrench[:, :1]
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
        events = detect_race_events(
            state_before.position,
            state.position,
            self.track,
            self.gate_index,
            self.config.safety_radius,
            is_alive_before,
        )
        acceleration = (state.linear_velocity - state_before.linear_velocity) / self.config.dt
        jerk = (acceleration - self.last_acceleration) / self.config.dt
        collision_loss, _, _ = gate_collision_loss(
            state.position,
            state.linear_velocity,
            self.track,
            self.gate_index,
            self.config.safety_radius,
            self.config.smooth_min_temperature,
            self.config.collision_beta_1,
            self.config.collision_beta_2,
        )
        progress_loss = gate_progress_loss(state.position, state.linear_velocity, self.track, self.gate_index)
        vector_field_loss = gate_vector_field_loss(
            state.position, state.linear_velocity, self.track, self.gate_index
        )
        scales = self.config.loss_scales
        physics_loss = (
            scales.progress * progress_loss
            + scales.acceleration * torch.sum(torch.square(acceleration), dim=-1)
            + scales.jerk * torch.sum(torch.square(jerk), dim=-1)
            + scales.vector_field * vector_field_loss
            + scales.collision * collision_loss
        ) * is_alive_before.to(dtype=action.dtype)
        policy_loss = action.new_zeros(action.shape[0])
        reward_scales = self.config.reward_scales
        reward = (
            reward_scales.progress * (-progress_loss)
            + reward_scales.gate_pass * events.passed.to(dtype=action.dtype)
            + reward_scales.complete * events.completed.to(dtype=action.dtype)
            + reward_scales.collision * events.analytic_collision.to(dtype=action.dtype)
            + reward_scales.smooth * torch.sum(torch.square(action - self.last_action), dim=-1)
        ) * is_alive_before.to(dtype=action.dtype)

        has_finite_state = (
            torch.isfinite(state.position).all(dim=-1)
            & torch.isfinite(state.quaternion).all(dim=-1)
            & torch.isfinite(state.linear_velocity).all(dim=-1)
        )
        physics_collision = self._physics_contacts()
        crashed = (
            (state.position[:, 2] < self.config.ground_termination_height)
            | events.analytic_collision
            | physics_collision
            | ~has_finite_state
        )
        is_newly_dead = is_alive_before & (crashed | events.completed)
        self.gate_index = events.next_gate_index
        self.gate_pass_count = self.gate_pass_count + events.passed.to(dtype=self.gate_pass_count.dtype)
        env_ids = torch.arange(self.num_envs, device=self.device)
        passed_ids = env_ids[events.passed]
        if passed_ids.numel() > 0:
            finished_gate = (self.gate_index[passed_ids] - 1).clamp(min=0)
            self.gate_pass_time[passed_ids, finished_gate] = (
                self.episode_length_buf[passed_ids].to(dtype=self.gate_pass_time.dtype) * self.config.dt
            )
        self.path_length = self.path_length + torch.linalg.vector_norm(state.position - state_before.position, dim=-1)
        self.episode_length_buf += is_alive_before.to(dtype=self.episode_length_buf.dtype)
        self.is_alive = is_alive_before & ~is_newly_dead
        is_truncated = self.is_alive & (self.episode_length_buf >= self.config.max_episode_steps)
        done = is_newly_dead | is_truncated
        self.last_action = torch.where(self.is_alive[:, None], action, torch.zeros_like(action))
        self.last_acceleration = acceleration
        critic_observation_before_reset = self._observations(state)[1]
        if self.respawn_on_fail and not self.requires_grad:
            reset_idx = done.nonzero(as_tuple=False).flatten()
            if reset_idx.numel() > 0:
                self._reset_envs(reset_idx)
            policy_observation, critic_observation = self._observations(self._read_state())
        else:
            self.is_alive = self.is_alive & ~is_truncated
            policy_observation, critic_observation = self._observations(state)
        extras = {
            "terminated": is_newly_dead.detach(),
            "truncated": is_truncated.detach(),
            "alive": self.is_alive.detach(),
            "passed": events.passed.detach(),
            "wrong_way": events.wrong_way.detach(),
            "completed": events.completed.detach(),
            "analytic_collision": events.analytic_collision.detach(),
            "physics_collision": physics_collision.detach(),
            "gate_index": self.gate_index.detach(),
            "critic_observation": critic_observation_before_reset,
            "actual_wrench": actual_wrench,
            "motor_thrust": output.motor_thrust,
            "command": output.command,
            "critic_observation_live": critic_observation,
        }
        return policy_observation, (physics_loss, policy_loss, reward.detach()), done.detach(), extras

    def step_diff(self, action: torch.Tensor) -> DiffTransition:
        policy, losses, done, extras = self.step(action)
        physics_loss, policy_loss, reward = losses
        return DiffTransition(
            observation=DiffObservation(policy=policy, critic=extras["critic_observation_live"]),
            bootstrap_critic=extras["critic_observation"],
            physics_loss=physics_loss,
            policy_loss=policy_loss,
            critic_cost=(-reward).detach(),
            reward=reward,
            done=done,
            terminated=extras["terminated"],
        )

    def _reset_envs(self, env_idx: torch.Tensor) -> None:
        count = int(env_idx.numel())
        if count == 0:
            return
        initial = generate_initial_states(self.track, count, seed=int(env_idx[0].item()) + 1)
        self.drone.set_pos(initial.position.to(self.device), envs_idx=env_idx, zero_velocity=True)
        self.drone.set_quat(initial.quaternion.to(self.device), envs_idx=env_idx, zero_velocity=True)
        linear_velocity = initial.linear_velocity.to(self.device)
        self.drone.set_dofs_velocity(
            torch.cat((linear_velocity, torch.zeros_like(linear_velocity)), dim=-1), envs_idx=env_idx
        )
        self.controller.reset(env_idx)
        self.last_action[env_idx] = 0.0
        self.gate_index[env_idx] = 0
        self.is_alive[env_idx] = True
        self.episode_length_buf[env_idx] = 0
        self.gate_pass_count[env_idx] = 0
        self.path_length[env_idx] = 0.0
        self.last_acceleration[env_idx] = 0.0

    def detach_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.release_simulation_graphs()
        self.last_action = detached_torch_tensor(self.last_action)
        self.gate_index = detached_torch_tensor(self.gate_index)
        self.is_alive = detached_torch_tensor(self.is_alive)
        self.episode_length_buf = detached_torch_tensor(self.episode_length_buf)
        if self.respawn_on_fail:
            reset_idx = (~self.is_alive).nonzero(as_tuple=False).flatten()
            if reset_idx.numel() > 0:
                self._reset_envs(reset_idx)
        observations = self._observations(self._read_state())
        return detached_torch_tensor(observations[0]), detached_torch_tensor(observations[1])

    def finish_window(
        self,
        physics_loss: torch.Tensor,
        simulation_actions: list[torch.Tensor],
    ) -> tuple[DiffObservation, torch.Tensor]:
        action_gradients = finish_simulation_window(self, physics_loss, simulation_actions)
        policy, critic = self.detach_window()
        return DiffObservation(policy=policy, critic=critic), action_gradients
