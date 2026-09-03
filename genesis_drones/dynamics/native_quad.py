import torch

from genesis_drones.controllers.native_config import NativeQuadConfig
from genesis_drones.controllers.native_mixer import NativeQuadMixer
from genesis_drones.dynamics.base import ControlOutput, DroneState


class NativeQuadBackend:
    name = "native_quad"
    action_dim = 4
    batch_links_info = False
    dof_damping = [0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4]

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        dt: float,
        config: NativeQuadConfig | None = None,
        **_unused,
    ):
        self.config = config or NativeQuadConfig()
        self.mixer = NativeQuadMixer(num_envs, device, dtype, dt, self.config)

    @classmethod
    def default_nominal_action(cls, config: NativeQuadConfig | None = None, **_unused) -> tuple[float, ...]:
        hover = NativeQuadMixer.hover(config or NativeQuadConfig())
        return (0.0, 0.0, 0.0, hover)

    @property
    def nominal_action(self) -> tuple[float, ...]:
        return self.default_nominal_action(self.config)

    def hover_command(self, count: int) -> torch.Tensor:
        action = torch.zeros((count, 4), device=self.mixer.device, dtype=self.mixer.dtype)
        action[:, 3] = self.mixer.hover_action
        return action

    def attach(self, drone, _scene) -> None:
        drone.set_dofs_damping(self.dof_damping)

    def reset(
        self,
        indices: torch.Tensor,
        seed: int | None = None,
        environment_indices: torch.Tensor | None = None,
    ) -> None:
        self.mixer.reset(indices)

    def mix(self, action: torch.Tensor, state: DroneState, is_alive: torch.Tensor):
        return self.mixer.mix(action, state.quaternion, state.angular_velocity, is_alive)

    def control(self, action: torch.Tensor, state: DroneState, is_alive: torch.Tensor) -> ControlOutput:
        motor_thrust, wrench = self.mix(action, state, is_alive)
        wrench = wrench * is_alive[:, None]
        force = NativeQuadMixer.generalized_force(state.quaternion, wrench)
        return ControlOutput(force, wrench, motor_thrust)

    def after_step(self, state: DroneState, is_alive_before: torch.Tensor) -> None:
        self.mixer.remember_rate(state.angular_velocity, is_alive_before)

    def detach(self) -> None:
        self.mixer.detach()
