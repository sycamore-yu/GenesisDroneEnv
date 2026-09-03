import torch

from genesis_drones.controllers.native_config import NativeQuadConfig
from genesis_drones.utils.geometry import quaternion_to_roll_pitch_yaw


class NativeQuadMixer:
    """Native quadrotor motor PID mixer. Control numbers match TrackDiffEnvConfig."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        dt: float,
        config: NativeQuadConfig | None = None,
    ):
        self.config = config or NativeQuadConfig()
        self.dt = dt
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        motor_moment_ratio = self.config.moment_coefficient / self.config.thrust_coefficient
        arm = self.config.motor_arm
        self.allocation = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [-arm, -arm, arm, arm],
                [-arm, arm, arm, -arm],
                [-motor_moment_ratio, motor_moment_ratio, -motor_moment_ratio, motor_moment_ratio],
            ],
            device=device,
            dtype=dtype,
        )
        self.angle_kp = torch.tensor(self.config.angle_kp, device=device, dtype=dtype)
        self.angle_ki = torch.tensor(self.config.angle_ki, device=device, dtype=dtype)
        self.angle_kd = torch.tensor(self.config.angle_kd, device=device, dtype=dtype)
        self.pid_integral = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.last_angular_velocity = torch.zeros((num_envs, 3), device=device, dtype=dtype)

    @staticmethod
    def hover(config: NativeQuadConfig | None = None) -> float:
        cfg = config or NativeQuadConfig()
        return 2.0 / cfg.thrust_to_weight_ratio - 1.0

    @property
    def hover_action(self) -> float:
        return self.hover(self.config)

    def reset(self, environment_indices: torch.Tensor) -> None:
        self.pid_integral[environment_indices] = 0.0
        self.last_angular_velocity[environment_indices] = 0.0

    def mix(
        self,
        action: torch.Tensor,
        quaternion: torch.Tensor,
        angular_velocity: torch.Tensor,
        is_alive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = action.clamp(-1.0, 1.0)
        body_euler = quaternion_to_roll_pitch_yaw(quaternion)
        body_setpoint = -body_euler + action[:, :3]
        angle_error = body_setpoint * 15.0 - angular_velocity
        proportional = angle_error * self.angle_kp
        integral = torch.clamp(self.pid_integral + angle_error * self.angle_ki * self.dt, -0.5, 0.5)
        derivative = torch.clamp(
            (self.last_angular_velocity - angular_velocity) * self.angle_kd / self.dt,
            -0.5,
            0.5,
        )
        pid_output = proportional + integral + derivative
        self.pid_integral = torch.where(is_alive[:, None], integral, torch.zeros_like(integral))
        throttle = (action[:, 3] + 1.0) * 0.5
        motor_command = torch.stack(
            (
                throttle - pid_output[:, 0] - pid_output[:, 1] - pid_output[:, 2],
                throttle - pid_output[:, 0] + pid_output[:, 1] + pid_output[:, 2],
                throttle + pid_output[:, 0] + pid_output[:, 1] - pid_output[:, 2],
                throttle + pid_output[:, 0] - pid_output[:, 1] + pid_output[:, 2],
            ),
            dim=-1,
        ).clamp(0.0, 1.0)
        motor_rpm = (
            torch.sqrt((motor_command * self.config.thrust_to_weight_ratio).clamp_min(1e-12)) * self.config.base_rpm
        )
        motor_thrust = self.config.thrust_coefficient * motor_rpm * motor_rpm
        return motor_thrust, motor_thrust @ self.allocation.T

    @staticmethod
    def generalized_force(quaternion: torch.Tensor, wrench: torch.Tensor) -> torch.Tensor:
        w = quaternion[:, 0]
        x = quaternion[:, 1]
        y = quaternion[:, 2]
        z = quaternion[:, 3]
        thrust_axis_world = torch.stack(
            (
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ),
            dim=-1,
        )
        force_world = thrust_axis_world * wrench[:, :1]
        return torch.cat((force_world, wrench[:, 1:]), dim=-1)

    def remember_rate(self, angular_velocity: torch.Tensor, is_alive: torch.Tensor) -> None:
        self.last_angular_velocity = torch.where(
            is_alive[:, None], angular_velocity, torch.zeros_like(angular_velocity)
        )

    def detach(self) -> None:
        self.pid_integral = self.pid_integral.detach()
        self.last_angular_velocity = self.last_angular_velocity.detach()
