import torch

from genesis_drones.controllers.ctbr_controller import CtbrController, CtbrControllerConfig
from genesis_drones.dynamics.base import ControlOutput, DroneState
from genesis_drones.utils.geometry import quaternion_to_rotation_matrix


class FullQuadBackend:
    name = "full_quad"
    action_dim = 4
    batch_links_info = True
    dof_damping = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        dt: float = 0.0333,
        controller_config: CtbrControllerConfig | None = None,
        **_unused,
    ):
        self.controller_config = controller_config or CtbrControllerConfig()
        self.controller = CtbrController(self.controller_config, num_envs, device, dtype)
        self._inertia_scale = torch.ones(num_envs, device=device, dtype=dtype)
        self._drone = None
        self._scene = None

    @classmethod
    def default_nominal_action(cls, controller_config: CtbrControllerConfig | None = None, **_unused) -> tuple[float, ...]:
        cfg = controller_config or CtbrControllerConfig()
        hover = 2.0 / cfg.max_normalized_thrust - 1.0
        return (hover, 0.0, 0.0, 0.0)

    @property
    def nominal_action(self) -> tuple[float, ...]:
        return self.default_nominal_action(self.controller_config)

    def hover_command(self, count: int) -> torch.Tensor:
        action = torch.zeros((count, 4), device=self.controller.device, dtype=self.controller.dtype)
        action[:, 0] = self.controller.hover_action
        return action

    def attach(self, drone, scene) -> None:
        self._drone = drone
        self._scene = scene
        drone.set_dofs_damping(self.dof_damping)

    def reset(
        self,
        indices: torch.Tensor,
        seed: int | None = None,
        environment_indices: torch.Tensor | None = None,
    ) -> None:
        self.controller.randomize(indices, seed=seed)
        self._apply_body_params(indices, environment_indices)

    def mix(self, action: torch.Tensor, state: DroneState, is_alive: torch.Tensor):
        rotation = quaternion_to_rotation_matrix(state.quaternion)
        omega_body = torch.einsum("nji,nj->ni", rotation, state.angular_velocity)
        return self.controller.step(action, omega_body, is_alive)

    def control(self, action: torch.Tensor, state: DroneState, is_alive: torch.Tensor) -> ControlOutput:
        rotation = quaternion_to_rotation_matrix(state.quaternion)
        omega_body = torch.einsum("nji,nj->ni", rotation, state.angular_velocity)
        output = self.controller.step(action, omega_body, is_alive)
        body_velocity = torch.einsum("nji,nj->ni", rotation, state.linear_velocity)
        drag_world = torch.einsum("nij,nj->ni", rotation, -self.controller.drag * body_velocity)
        force_world = rotation[:, :, 2] * output.wrench[:, :1] + drag_world
        torque_world = torch.einsum("nij,nj->ni", rotation, output.wrench[:, 1:])
        force = torch.cat((force_world, torque_world), dim=-1)
        return ControlOutput(force, output.wrench, output.motor_thrust)

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
        # ponytail: set_links_inertial_mass does not refresh dofs.invweight; the integrator uses that, so hover used URDF mass 0.5. Recompute after mass/J writes. Call public setter if Genesis adds one.
        qpos = self._drone.get_qpos(envs_idx=environment_indices)
        self._scene.rigid_solver._init_invweight_and_meaninertia(
            envs_idx=environment_indices, force_update=True
        )
        self._drone.set_qpos(qpos, envs_idx=environment_indices, zero_velocity=False)
