from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch

import genesis as gs

from genesis_drones.controllers.ctbr_controller import CtbrController, CtbrControllerConfig
from genesis_drones.envs.differentiable import DiffEnvSpec, DiffObservation, DiffTransition, finish_simulation_window
from genesis_drones.tasks.racing_core import (
    CRITIC_OBSERVATION_SIZE,
    POLICY_OBSERVATION_SIZE,
    EvaluationInitialStates,
    RaceTrackSpec,
    detect_race_events,
    generate_initial_states,
    quaternion_to_rotation_matrix,
    racing_loss_reward,
    racing_observations,
    track_to_tensors,
)
from genesis_drones.tasks.racing_tracks import RACING_TRACK, add_track_gates


ASSETS_PATH = Path(__file__).resolve().parents[1] / "robots" / "assets"


@dataclass(frozen=True)
class RaceEnvConfig:
    dt: float = 0.0333
    horizon: int = 32
    max_episode_steps: int = int(40.0 / 0.0333)
    min_target_velocity: float = 5.0
    max_target_velocity: float = 10.0
    gamma: float = 0.99
    td_lambda: float = 0.95
    controller: CtbrControllerConfig = CtbrControllerConfig()


class DroneState(NamedTuple):
    position: torch.Tensor
    quaternion: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity_world: torch.Tensor


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
        return DiffEnvSpec(
            policy_observation_dim=cls.policy_observation_dim,
            critic_observation_dim=cls.critic_observation_dim,
            action_dim=cls.action_dim,
            horizon=config.horizon,
            nominal_action=(2.0 / config.controller.max_normalized_thrust - 1.0, 0.0, 0.0, 0.0),
        )

    def __init__(
        self,
        config: RaceEnvConfig,
        num_envs: int,
        track: RaceTrackSpec = RACING_TRACK,
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
                enable_collision=False,
                enable_joint_limit=True,
                batch_links_info=True,
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
        self.gate_entities = add_track_gates(self.scene, track) if show_viewer else []
        self.scene.build(n_envs=num_envs)
        self.drone.set_dofs_damping([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.track = track_to_tensors(track, self.device, gs.tc_float)
        self.controller = CtbrController(config.controller, num_envs, self.device, gs.tc_float)
        self._inertia_scale = torch.ones(num_envs, device=self.device, dtype=gs.tc_float)
        self.spec = self.spec_from_config(config)
        self.last_action = torch.zeros((num_envs, self.action_dim), device=self.device, dtype=gs.tc_float)
        self.target_gates = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self.is_alive = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self.n_passed_gates = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self.path_length = torch.zeros(num_envs, device=self.device, dtype=gs.tc_float)
        self.max_velocity = torch.full(
            (num_envs,), config.min_target_velocity, device=self.device, dtype=gs.tc_float
        )
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

    def _apply_body_params(self, indices: torch.Tensor, environment_indices: torch.Tensor | None) -> None:
        # ponytail: Genesis set_links_inertia is a scalar ratio, so only Jxy is applied to the solver.
        # RateController still uses the full (Jxy, Jxy, Jz) tensor. Upgrade if yaw tracking fails.
        urdf_inertia = 1.4e-3
        desired_scale = self.controller.inertia[indices, 0] / urdf_inertia
        ratio = desired_scale / self._inertia_scale[indices].clamp_min(1e-8)
        self.scene.rigid_solver.set_links_inertia(
            ratio[:, None], links_idx=[self.drone.base_link_idx], envs_idx=environment_indices
        )
        self._inertia_scale[indices] = desired_scale
        self.drone.set_links_inertial_mass(
            self.controller.mass[indices, None], links_idx_local=[0], envs_idx=environment_indices
        )

    def _observations(self, state: DroneState) -> tuple[torch.Tensor, torch.Tensor]:
        return racing_observations(
            state.position,
            state.quaternion,
            state.linear_velocity,
            self.track,
            self.target_gates,
        )

    def release_simulation_graphs(self, physics_loss: torch.Tensor | None = None) -> None:
        if physics_loss is not None and physics_loss.grad_fn is not None:
            torch.autograd.backward(physics_loss)
        self._simulation_forces.clear()
        _detach_cached_torch_tensor(self.scene.rigid_solver.dyn_state.dofs.ctrl_force)
        self.last_action = detached_torch_tensor(self.last_action)
        self.controller.detach()

    def _set_initial_states(
        self,
        initial_states: EvaluationInitialStates,
        environment_indices: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> None:
        position = initial_states.position.to(device=self.device, dtype=gs.tc_float)
        quaternion = initial_states.quaternion.to(device=self.device, dtype=gs.tc_float)
        linear_velocity = initial_states.linear_velocity.to(device=self.device, dtype=gs.tc_float)
        target_gate = initial_states.target_gate.to(device=self.device, dtype=torch.int64)
        self.drone.set_pos(position, envs_idx=environment_indices, zero_velocity=True)
        self.drone.set_quat(quaternion, envs_idx=environment_indices, zero_velocity=True)
        self.drone.set_dofs_velocity(
            torch.cat((linear_velocity, torch.zeros_like(linear_velocity)), dim=-1), envs_idx=environment_indices
        )
        indices = torch.arange(self.num_envs, device=self.device) if environment_indices is None else environment_indices
        self.controller.randomize(indices, seed=seed)
        self._apply_body_params(indices, environment_indices)
        self.target_gates[indices] = target_gate
        self.last_action[indices] = 0.0
        self.is_alive[indices] = True
        self.episode_length_buf[indices] = 0
        self.n_passed_gates[indices] = 0
        self.path_length[indices] = 0.0
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed + 1)
        self.max_velocity[indices] = torch.rand(
            indices.numel(), device=self.device, dtype=gs.tc_float, generator=generator
        ) * (self.config.max_target_velocity - self.config.min_target_velocity) + self.config.min_target_velocity

    def reset(self, initial_states: EvaluationInitialStates | None = None, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        self.release_simulation_graphs()
        self.scene.reset()
        if initial_states is None:
            initial_states = generate_initial_states(self.track, self.num_envs, seed=seed)
        self._set_initial_states(initial_states, seed=seed)
        return self._observations(self._read_state())

    def reset_diff(self, seed: int | None = None) -> DiffObservation:
        policy, critic = self.reset(seed=0 if seed is None else seed)
        return DiffObservation(policy=policy, critic=critic)

    def mix_action(self, action: torch.Tensor, state: DroneState | None = None):
        if state is None:
            state = self._read_state()
        rotation = quaternion_to_rotation_matrix(state.quaternion)
        omega_body = torch.einsum("nji,nj->ni", rotation, state.angular_velocity_world)
        return self.controller.step(action, omega_body, self.is_alive)

    def step(self, action: torch.Tensor):
        action = action.clamp(-1.0, 1.0)
        is_alive_before = self.is_alive.clone()
        state_before = self._read_state()
        rotation = quaternion_to_rotation_matrix(state_before.quaternion)
        omega_body = torch.einsum("nji,nj->ni", rotation, state_before.angular_velocity_world)
        output = self.controller.step(action, omega_body, self.is_alive)
        body_velocity = torch.einsum("nji,nj->ni", rotation, state_before.linear_velocity)
        drag_world = torch.einsum("nij,nj->ni", rotation, -self.controller.drag * body_velocity)
        force_world = rotation[:, :, 2] * output.wrench[:, :1] + drag_world
        torque_world = torch.einsum("nij,nj->ni", rotation, output.wrench[:, 1:])
        generalized_force = torch.cat((force_world, torque_world), dim=-1)
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
            self.target_gates,
            is_alive_before,
        )
        self.target_gates = events.next_gate_index
        self.n_passed_gates = self.n_passed_gates + events.passed.to(dtype=self.n_passed_gates.dtype)
        target_position = self.track.positions[self.track.order[self.target_gates]]
        physics_loss, reward, loss_components = racing_loss_reward(
            state.position,
            state_before.position,
            state.quaternion,
            state.linear_velocity,
            state.angular_velocity_world,
            target_position,
            self.max_velocity,
            events.analytic_collision,
        )
        policy_loss = action.new_zeros(action.shape[0])
        out_of_bounds = torch.any(torch.abs(state.position[:, :2]) > 5.0, dim=-1) | (state.position[:, 2] > 7.0)
        terminated = is_alive_before & events.analytic_collision
        truncated = is_alive_before & (out_of_bounds | (self.episode_length_buf >= self.config.max_episode_steps))
        self.episode_length_buf += is_alive_before.to(dtype=self.episode_length_buf.dtype)
        success = truncated & (self.episode_length_buf >= self.config.max_episode_steps)
        done = terminated | truncated
        self.is_alive = is_alive_before & ~done
        self.last_action = torch.where(self.is_alive[:, None], action, torch.zeros_like(action))
        self.path_length = self.path_length + torch.linalg.vector_norm(
            state.position - state_before.position, dim=-1
        ) * is_alive_before

        critic_observation_before_reset = self._observations(state)[1]
        episode_length = self.episode_length_buf.clone()
        passed_gates = self.n_passed_gates.clone()
        if self.respawn_on_fail and not self.requires_grad:
            reset_idx = done.nonzero(as_tuple=False).flatten()
            if reset_idx.numel() > 0:
                self._reset_envs(reset_idx)
            policy_observation, critic_observation = self._observations(self._read_state())
        else:
            policy_observation, critic_observation = self._observations(state)
        extras = {
            "terminated": terminated.detach(),
            "truncated": truncated.detach(),
            "reset": done.detach(),
            "success": success.detach(),
            "passed": events.passed.detach(),
            "wrong_way": events.wrong_way.detach(),
            "analytic_collision": events.analytic_collision.detach(),
            "target_gate": self.target_gates.detach(),
            "n_passed_gates": passed_gates.detach(),
            "episode_length": episode_length.detach(),
            "critic_observation": critic_observation_before_reset,
            "critic_observation_live": critic_observation,
            "actual_wrench": output.wrench,
            "motor_thrust": output.motor_thrust,
            "command": output.command,
            "loss_components": {key: value.detach() for key, value in loss_components.items()},
        }
        return policy_observation, (physics_loss, policy_loss, reward), done.detach(), extras

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

    def _reset_envs(self, environment_indices: torch.Tensor) -> None:
        count = int(environment_indices.numel())
        if count == 0:
            return
        seed = int(torch.randint(0, 2**31 - 1, (), device=self.device).item())
        initial_states = generate_initial_states(self.track, count, seed=seed)
        self._set_initial_states(initial_states, environment_indices, seed=seed)

    def detach_window(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.release_simulation_graphs()
        self.last_action = detached_torch_tensor(self.last_action)
        self.target_gates = detached_torch_tensor(self.target_gates)
        self.is_alive = detached_torch_tensor(self.is_alive)
        self.episode_length_buf = detached_torch_tensor(self.episode_length_buf)
        if self.respawn_on_fail:
            reset_idx = (~self.is_alive).nonzero(as_tuple=False).flatten()
            if reset_idx.numel() > 0:
                self._reset_envs(reset_idx)
        return tuple(map(detached_torch_tensor, self._observations(self._read_state())))

    def finish_window(
        self,
        physics_loss: torch.Tensor,
        simulation_actions: list[torch.Tensor],
    ) -> tuple[DiffObservation, torch.Tensor]:
        action_gradients = finish_simulation_window(self, physics_loss, simulation_actions)
        policy, critic = self.detach_window()
        return DiffObservation(policy=policy, critic=critic), action_gradients
