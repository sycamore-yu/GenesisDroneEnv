from dataclasses import dataclass


@dataclass(frozen=True)
class NativeQuadConfig:
    # Defaults match TrackDiffEnvConfig motor/PID fields exactly.
    motor_arm: float = 0.1
    thrust_coefficient: float = 3.16e-10
    moment_coefficient: float = 7.94e-12
    thrust_to_weight_ratio: float = 3.3
    base_rpm: float = 62293.9641914
    angle_kp: tuple[float, float, float] = (0.04, 0.04, 0.04)
    angle_ki: tuple[float, float, float] = (0.004, 0.004, 0.004)
    angle_kd: tuple[float, float, float] = (0.0001, 0.0001, 0.0001)
