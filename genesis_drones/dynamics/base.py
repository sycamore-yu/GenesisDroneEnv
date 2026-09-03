from typing import NamedTuple, Protocol

import torch


class DroneState(NamedTuple):
    position: torch.Tensor
    quaternion: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor


class ControlOutput(NamedTuple):
    generalized_force: torch.Tensor
    wrench: torch.Tensor
    motor_thrust: torch.Tensor


class DynamicsBackend(Protocol):
    name: str
    action_dim: int
    batch_links_info: bool

    @property
    def nominal_action(self) -> tuple[float, ...]: ...

    def attach(self, drone, scene) -> None: ...

    def reset(
        self,
        indices: torch.Tensor,
        seed: int | None = None,
        environment_indices: torch.Tensor | None = None,
    ) -> None: ...

    def control(self, action: torch.Tensor, state: DroneState, is_alive: torch.Tensor) -> ControlOutput: ...

    def after_step(self, state: DroneState, is_alive_before: torch.Tensor) -> None: ...

    def detach(self) -> None: ...

    def hover_command(self, count: int) -> torch.Tensor: ...
