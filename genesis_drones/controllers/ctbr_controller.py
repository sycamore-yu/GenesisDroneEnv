from dataclasses import dataclass
from typing import NamedTuple

import torch


# DiffAero cfg/dynamics/quad.yaml defaults actually used by RateController.
_MASS_DEFAULT = 1.0
_INERTIA_XY_DEFAULT = 0.01
_INERTIA_Z_DEFAULT = 0.02
_DRAG_DEFAULT = 0.6


@dataclass(frozen=True)
class CtbrControllerConfig:
    gravity: float = 9.81
    max_normalized_thrust: float = 5.0
    max_body_rates: tuple[float, float, float] = (3.14, 3.14, 3.14)
    mass_range: tuple[float, float] = (0.8, 1.2)
    inertia_xy_range: tuple[float, float] = (0.008, 0.012)
    inertia_z_range: tuple[float, float] = (0.015, 0.025)
    drag_xy_range: tuple[float, float] = (0.5, 0.7)
    drag_z_range: tuple[float, float] = (0.5, 0.7)
    randomize: bool = True


class CtbrOutput(NamedTuple):
    command: torch.Tensor
    motor_thrust: torch.Tensor
    wrench: torch.Tensor


class CtbrController:
    def __init__(self, config: CtbrControllerConfig, num_envs: int, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        self.max_body_rates = torch.tensor(config.max_body_rates, device=device, dtype=dtype)
        self.mass = torch.full((num_envs,), _MASS_DEFAULT, device=device, dtype=dtype)
        self.inertia = torch.tensor(
            (_INERTIA_XY_DEFAULT, _INERTIA_XY_DEFAULT, _INERTIA_Z_DEFAULT), device=device, dtype=dtype
        ).expand(num_envs, -1).clone()
        self.drag = torch.full((num_envs, 3), _DRAG_DEFAULT, device=device, dtype=dtype)

    @property
    def hover_action(self) -> float:
        return 2.0 / self.config.max_normalized_thrust - 1.0

    def _fill_defaults(self, environment_indices: torch.Tensor) -> None:
        self.mass[environment_indices] = _MASS_DEFAULT
        self.inertia[environment_indices, 0] = _INERTIA_XY_DEFAULT
        self.inertia[environment_indices, 1] = _INERTIA_XY_DEFAULT
        self.inertia[environment_indices, 2] = _INERTIA_Z_DEFAULT
        self.drag[environment_indices] = _DRAG_DEFAULT

    def randomize(self, environment_indices: torch.Tensor | None = None, seed: int | None = None) -> None:
        if environment_indices is None:
            environment_indices = torch.arange(self.num_envs, device=self.device)
        if not self.config.randomize:
            self._fill_defaults(environment_indices)
            return
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)
        count = environment_indices.numel()

        def sample(bounds: tuple[float, float]) -> torch.Tensor:
            return torch.rand(count, device=self.device, dtype=self.dtype, generator=generator) * (
                bounds[1] - bounds[0]
            ) + bounds[0]

        self.mass[environment_indices] = sample(self.config.mass_range)
        inertia_xy = sample(self.config.inertia_xy_range)
        self.inertia[environment_indices, 0] = inertia_xy
        self.inertia[environment_indices, 1] = inertia_xy
        self.inertia[environment_indices, 2] = sample(self.config.inertia_z_range)
        drag_xy = sample(self.config.drag_xy_range)
        self.drag[environment_indices, 0] = drag_xy
        self.drag[environment_indices, 1] = drag_xy
        self.drag[environment_indices, 2] = sample(self.config.drag_z_range)

    def step(
        self, normalized_action: torch.Tensor, angular_velocity_body: torch.Tensor, is_active: torch.Tensor
    ) -> CtbrOutput:
        normalized_action = normalized_action.clamp(-1.0, 1.0)
        normalized_thrust = 0.5 * (normalized_action[:, 0] + 1.0) * self.config.max_normalized_thrust
        body_rate_command = normalized_action[:, 1:] * self.max_body_rates
        angular_acceleration = body_rate_command - angular_velocity_body
        angular_momentum = self.inertia * angular_velocity_body
        gyroscopic = torch.linalg.cross(angular_velocity_body, angular_momentum, dim=-1)
        gyroscopic = gyroscopic / torch.maximum(
            gyroscopic.norm(dim=-1, keepdim=True) / 100.0,
            gyroscopic.new_ones(()),
        ).detach()
        torque = self.inertia * angular_acceleration + gyroscopic
        thrust = normalized_thrust * self.config.gravity * self.mass
        command = torch.cat((normalized_thrust[:, None], body_rate_command), dim=-1)
        motor_thrust = thrust[:, None].expand(-1, 4) * 0.25
        wrench = torch.cat((thrust[:, None], torque), dim=-1)
        active = is_active[:, None].to(dtype=wrench.dtype)
        return CtbrOutput(command * active, motor_thrust * active, wrench * active)

    def reset(self, environment_indices: torch.Tensor | None = None) -> None:
        pass

    def detach(self) -> None:
        self.mass = self.mass.detach()
        self.inertia = self.inertia.detach()
        self.drag = self.drag.detach()
