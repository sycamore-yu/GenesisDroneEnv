from typing import NamedTuple

import torch

from genesis_drones.controllers.ctbr_controller import CtbrController, CtbrControllerConfig
from genesis_drones.controllers.native_mixer import NativeQuadMixer
from genesis_drones.tasks.racing_core import quaternion_to_rotation_matrix


class DroneState(NamedTuple):
    position: torch.Tensor
    quaternion: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity_world: torch.Tensor


class NativeQuadPlant:
    name = "native_quad"
    batch_links_info = False
    dof_damping = [0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4]

    def __init__(self, num_envs: int, device: torch.device, dtype: torch.dtype, dt: float, _controller_config=None):
        self.mixer = NativeQuadMixer(num_envs, device, dtype, dt)

    @staticmethod
    def nominal_action(_controller_config=None) -> tuple[float, ...]:
        hover = NativeQuadMixer.hover()
        return (0.0, 0.0, 0.0, hover)

    def hover_command(self, count: int) -> torch.Tensor:
        action = torch.zeros((count, 4), device=self.mixer.device, dtype=self.mixer.dtype)
        action[:, 3] = self.mixer.hover_action
        return action

    def attach(self, drone, _scene) -> None:
        drone.set_dofs_damping(self.dof_damping)

    def reset(self, indices: torch.Tensor, seed: int | None = None, environment_indices: torch.Tensor | None = None) -> None:
        self.mixer.reset(indices)

    def mix(self, action: torch.Tensor, state: DroneState, is_alive: torch.Tensor):
        return self.mixer.mix(action, state.quaternion, state.angular_velocity_world, is_alive)

    def control_wrench(self, action: torch.Tensor, state: DroneState, is_alive: torch.Tensor):
        motor_thrust, wrench = self.mix(action, state, is_alive)
        wrench = wrench * is_alive[:, None]
        return NativeQuadMixer.generalized_force(state.quaternion, wrench), wrench, motor_thrust

    def after_step(self, state: DroneState, is_alive_before: torch.Tensor) -> None:
        self.mixer.remember_rate(state.angular_velocity_world, is_alive_before)

    def detach(self) -> None:
        self.mixer.detach()


class FullQuadPlant:
    name = "full_quad"
    batch_links_info = True
    dof_damping = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        _dt: float,
        controller_config: CtbrControllerConfig,
    ):
        self.controller = CtbrController(controller_config, num_envs, device, dtype)
        self._inertia_scale = torch.ones(num_envs, device=device, dtype=dtype)
        self._drone = None
        self._scene = None

    @staticmethod
    def nominal_action(controller_config: CtbrControllerConfig) -> tuple[float, ...]:
        hover = 2.0 / controller_config.max_normalized_thrust - 1.0
        return (hover, 0.0, 0.0, 0.0)

    def hover_command(self, count: int) -> torch.Tensor:
        action = torch.zeros((count, 4), device=self.controller.device, dtype=self.controller.dtype)
        action[:, 0] = self.controller.hover_action
        return action

    def attach(self, drone, scene) -> None:
        self._drone = drone
        self._scene = scene
        drone.set_dofs_damping(self.dof_damping)

    def reset(self, indices: torch.Tensor, seed: int | None = None, environment_indices: torch.Tensor | None = None) -> None:
        self.controller.randomize(indices, seed=seed)
        self._apply_body_params(indices, environment_indices)

    def mix(self, action: torch.Tensor, state: DroneState, is_alive: torch.Tensor):
        rotation = quaternion_to_rotation_matrix(state.quaternion)
        omega_body = torch.einsum("nji,nj->ni", rotation, state.angular_velocity_world)
        return self.controller.step(action, omega_body, is_alive)

    def control_wrench(self, action: torch.Tensor, state: DroneState, is_alive: torch.Tensor):
        rotation = quaternion_to_rotation_matrix(state.quaternion)
        omega_body = torch.einsum("nji,nj->ni", rotation, state.angular_velocity_world)
        output = self.controller.step(action, omega_body, is_alive)
        body_velocity = torch.einsum("nji,nj->ni", rotation, state.linear_velocity)
        drag_world = torch.einsum("nij,nj->ni", rotation, -self.controller.drag * body_velocity)
        force_world = rotation[:, :, 2] * output.wrench[:, :1] + drag_world
        torque_world = torch.einsum("nij,nj->ni", rotation, output.wrench[:, 1:])
        return torch.cat((force_world, torque_world), dim=-1), output.wrench, output.motor_thrust

    def after_step(self, _state: DroneState, _is_alive_before: torch.Tensor) -> None:
        return

    def detach(self) -> None:
        self.controller.detach()

    def _apply_body_params(self, indices: torch.Tensor, environment_indices: torch.Tensor | None) -> None:
        # ponytail: Genesis set_links_inertia is a scalar ratio, so only Jxy is applied to the solver.
        # RateController still uses the full (Jxy, Jxy, Jz) tensor. Upgrade if yaw tracking fails.
        urdf_inertia = 1.4e-3
        desired_scale = self.controller.inertia[indices, 0] / urdf_inertia
        ratio = desired_scale / self._inertia_scale[indices].clamp_min(1e-8)
        self._scene.rigid_solver.set_links_inertia(
            ratio[:, None], links_idx=[self._drone.base_link_idx], envs_idx=environment_indices
        )
        self._inertia_scale[indices] = desired_scale
        self._drone.set_links_inertial_mass(
            self.controller.mass[indices, None], links_idx_local=[0], envs_idx=environment_indices
        )


PLANTS = {
    "native_quad": NativeQuadPlant,
    "full_quad": FullQuadPlant,
}
QUAD_DYNAMICS = tuple(PLANTS)


def make_quad_plant(name: str, num_envs: int, device: torch.device, dtype: torch.dtype, dt: float, controller_config):
    if name not in PLANTS:
        raise ValueError(f"unsupported dynamics: {name}")
    return PLANTS[name](num_envs, device, dtype, dt, controller_config)
