from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class CtbrControllerConfig:
    dt: float = 0.01
    gravity: float = 9.81
    thrust_to_weight_ratio: float = 3.3
    base_rpm: float = 62293.9641914
    thrust_coefficient: float = 3.16e-10
    moment_coefficient: float = 7.94e-12
    motor_arm: float = 0.1
    max_body_rates: tuple[float, float, float] = (6.0, 6.0, 3.0)
    rate_kp: tuple[float, float, float] = (0.07, 0.07, 0.07)
    rate_ki: tuple[float, float, float] = (0.002, 0.002, 0.002)
    rate_kd: tuple[float, float, float] = (0.0, 0.0, 0.0)


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
        self.rate_kp = torch.tensor(config.rate_kp, device=device, dtype=dtype)
        self.rate_ki = torch.tensor(config.rate_ki, device=device, dtype=dtype)
        self.rate_kd = torch.tensor(config.rate_kd, device=device, dtype=dtype)
        moment_ratio = config.moment_coefficient / config.thrust_coefficient
        self.allocation = torch.tensor(
            (
                (1.0, 1.0, 1.0, 1.0),
                (-config.motor_arm, -config.motor_arm, config.motor_arm, config.motor_arm),
                (-config.motor_arm, config.motor_arm, config.motor_arm, -config.motor_arm),
                (-moment_ratio, moment_ratio, -moment_ratio, moment_ratio),
            ),
            device=device,
            dtype=dtype,
        )
        self.integral = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.last_angular_velocity = torch.zeros_like(self.integral)

    @property
    def hover_action(self) -> float:
        return 2.0 / self.config.thrust_to_weight_ratio - 1.0

    def step(
        self, normalized_action: torch.Tensor, angular_velocity_body: torch.Tensor, is_active: torch.Tensor
    ) -> CtbrOutput:
        normalized_action = normalized_action.clamp(-1.0, 1.0)
        collective_acceleration = (
            0.5
            * (normalized_action[:, 0] + 1.0)
            * self.config.thrust_to_weight_ratio
            * self.config.gravity
        )
        body_rate_command = normalized_action[:, 1:] * self.max_body_rates
        command = torch.cat((collective_acceleration[:, None], body_rate_command), dim=-1)

        error = body_rate_command - angular_velocity_body
        integral = torch.clamp(self.integral + error * self.rate_ki * self.config.dt, -0.5, 0.5)
        derivative = (self.last_angular_velocity - angular_velocity_body) * self.rate_kd / self.config.dt
        pid_output = error * self.rate_kp + integral + derivative
        self.integral = torch.where(is_active[:, None], integral, torch.zeros_like(integral))
        self.last_angular_velocity = torch.where(
            is_active[:, None], angular_velocity_body, torch.zeros_like(angular_velocity_body)
        )

        collective_fraction = collective_acceleration / (
            self.config.thrust_to_weight_ratio * self.config.gravity
        )
        motor_fraction = torch.stack(
            (
                collective_fraction - pid_output[:, 0] - pid_output[:, 1] - pid_output[:, 2],
                collective_fraction - pid_output[:, 0] + pid_output[:, 1] + pid_output[:, 2],
                collective_fraction + pid_output[:, 0] + pid_output[:, 1] - pid_output[:, 2],
                collective_fraction + pid_output[:, 0] - pid_output[:, 1] + pid_output[:, 2],
            ),
            dim=-1,
        ).clamp(0.0, 1.0)
        motor_rpm = torch.sqrt(
            (motor_fraction * self.config.thrust_to_weight_ratio).clamp_min(1e-12)
        ) * self.config.base_rpm
        motor_thrust = self.config.thrust_coefficient * motor_rpm * motor_rpm
        wrench = motor_thrust @ self.allocation.T
        return CtbrOutput(command, motor_thrust, wrench)

    def reset(self, environment_indices: torch.Tensor | None = None) -> None:
        if environment_indices is None:
            self.integral.zero_()
            self.last_angular_velocity.zero_()
            return
        self.integral[environment_indices] = 0.0
        self.last_angular_velocity[environment_indices] = 0.0

    def detach(self) -> None:
        self.integral = self.integral.detach()
        self.last_angular_velocity = self.last_angular_velocity.detach()
