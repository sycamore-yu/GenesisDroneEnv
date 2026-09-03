from dataclasses import dataclass

import torch

import genesis as gs

from genesis_drones.adapters.diff_rl_adapter import DiffRLAdapter
from genesis_drones.controllers.native_config import NativeQuadConfig
from genesis_drones.dynamics import make_dynamics_backend
from genesis_drones.envs.differentiable import DiffEnvSpec, DiffObservation, DiffTransition
from genesis_drones.envs.genesis_task_env import GenesisTaskEnv, detached_torch_tensor
from genesis_drones.tasks.tracking_core import (
    OBSERVATION_SIZE,
    InitialPose,
    RewardScales,
    TrackingCore,
    TrackingCoreConfig,
    arrival_surrogate_bonus,
    attitude_error,
    closing_velocity,
    differentiable_tracking_reward,
    smooth_safety_penalty,
    tracking_progress,
    tracking_safety_penalty,
)


@dataclass(frozen=True)
class TrackDiffEnvConfig:
    dt: float = 0.01
    horizon: int = 32
    max_episode_steps: int = 1500
    target_threshold: float = 0.1
    max_collective_thrust: float = 16.1865
    motor_arm: float = 0.1
    thrust_coefficient: float = 3.16e-10
    moment_coefficient: float = 7.94e-12
    thrust_to_weight_ratio: float = 3.3
    base_rpm: float = 62293.9641914
    angle_kp: tuple[float, float, float] = (0.04, 0.04, 0.04)
    angle_ki: tuple[float, float, float] = (0.004, 0.004, 0.004)
    angle_kd: tuple[float, float, float] = (0.0001, 0.0001, 0.0001)
    body_collision_radius: float = 0.06
    body_collision_half_height: float = 0.0125
    ground_termination_height: float = 0.1
    ground_warning_distance: float = 0.2
    horizontal_termination_error: float = 5.0
    horizontal_warning_distance: float = 0.5
    vertical_termination_error: float = 1.2
    vertical_warning_distance: float = 0.12
    safety_temperature_ratio: float = 0.1
    roll_termination: float = 180.0
    pitch_termination: float = 180.0
    yaw_lambda: float = -10.0
    max_horizon_vel: float = 1.5
    max_vertical_vel: float = 0.5
    action_delta_weight: float = 0.0
    obs_scale_position_error: float = 1.0 / 3.0
    obs_scale_linear_velocity: float = 1.0 / 3.0
    obs_scale_angular_velocity: float = 1.0 / 3.14159
    enable_collision: bool = True
    initial_x_range: tuple[float, float] = (-0.05, 0.05)
    initial_y_range: tuple[float, float] = (-0.05, 0.05)
    initial_z_range: tuple[float, float] = (0.6, 0.61)
    target_x_range: tuple[float, float] = (-1.2, 1.2)
    target_y_range: tuple[float, float] = (-1.2, 1.2)
    target_z_range: tuple[float, float] = (0.6, 1.0)
    reward_scales: RewardScales = RewardScales()
    progress_norm: str = "l1"
    closing_velocity_weight: float = 0.0
    arrival_surrogate: str = "none"
    arrival_surrogate_sigma: float = 0.15
    fully_differentiable: bool = False
    tracking_position_weight: float = 1.0
    tracking_attitude_weight: float = 0.2
    tracking_velocity_weight: float = 0.05
    tracking_angular_rate_weight: float = 0.02
    tracking_action_smooth_weight: float = 1.0e-4
    tracking_safety_weight: float = 1.0
    dynamics: str = "native_quad"

    def native_quad_config(self) -> NativeQuadConfig:
        return NativeQuadConfig(
            motor_arm=self.motor_arm,
            thrust_coefficient=self.thrust_coefficient,
            moment_coefficient=self.moment_coefficient,
            thrust_to_weight_ratio=self.thrust_to_weight_ratio,
            base_rpm=self.base_rpm,
            angle_kp=self.angle_kp,
            angle_ki=self.angle_ki,
            angle_kd=self.angle_kd,
        )


def make_tracking_env(
    config: TrackDiffEnvConfig,
    num_envs: int,
    requires_grad: bool = True,
    show_viewer: bool = False,
    visualize: bool = False,
) -> GenesisTaskEnv:
    task = TrackingCore(
        TrackingCoreConfig.from_compat(config),
        num_envs,
        gs.device,
        dtype=gs.tc_float,
    )
    backend = make_dynamics_backend(
        config.dynamics,
        num_envs,
        gs.device,
        gs.tc_float,
        config.dt,
        native_config=config.native_quad_config(),
    )
    add_plane = show_viewer or visualize or config.enable_collision
    return GenesisTaskEnv(
        task_core=task,
        dynamics_backend=backend,
        num_envs=num_envs,
        requires_grad=requires_grad,
        show_viewer=show_viewer,
        dt=config.dt,
        horizon=config.horizon,
        enable_collision=config.enable_collision,
        drone_initial_pos=(0.0, 0.0, 0.6),
        add_plane=add_plane,
        add_target_visual=visualize,
    )


class TrackDiffEnv:
    """Thin tracking entry: GenesisTaskEnv + DiffRLAdapter. No duplicated scene/PID logic."""

    action_dim = 4
    observation_dim = OBSERVATION_SIZE

    @classmethod
    def spec_from_config(cls, config: TrackDiffEnvConfig) -> DiffEnvSpec:
        hover_action = 2.0 / config.thrust_to_weight_ratio - 1.0
        return DiffEnvSpec(
            policy_observation_dim=cls.observation_dim,
            critic_observation_dim=cls.observation_dim,
            action_dim=cls.action_dim,
            horizon=config.horizon,
            nominal_action=(0.0, 0.0, 0.0, hover_action),
            action_delta_weight=config.action_delta_weight,
        )

    def __init__(
        self,
        config: TrackDiffEnvConfig,
        num_envs: int,
        requires_grad: bool = True,
        show_viewer: bool = False,
        visualize: bool | None = None,
    ):
        self.config = config
        self.num_envs = num_envs
        self.requires_grad = requires_grad
        self.device = gs.device
        self.visualize = show_viewer if visualize is None else visualize
        self.spec = self.spec_from_config(config)
        self.env = make_tracking_env(
            config, num_envs, requires_grad, show_viewer=show_viewer, visualize=self.visualize
        )
        self.diff = DiffRLAdapter(
            self.env,
            horizon=config.horizon,
            action_delta_weight=config.action_delta_weight,
            critic_cost_mode="physics_plus_policy",
        )
        self.task = self.env.task
        self.scene = self.env.scene
        self.drone = self.env.drone
        self.plant = self.env.plant
        self.target_visual = self.env.target_visual
        self.respawn_on_fail = True

    @property
    def is_alive(self):
        return self.env.is_alive

    @is_alive.setter
    def is_alive(self, value):
        self.env.is_alive = value
        self.task.is_alive = value

    @property
    def target_position(self):
        return self.task.target_position

    @target_position.setter
    def target_position(self, value):
        self.task.target_position = value

    @property
    def last_action(self):
        return self.env.last_action

    @last_action.setter
    def last_action(self, value):
        self.env.last_action = value
        self.task.last_action = value

    @property
    def waypoint_sequences(self):
        return self.task.waypoint_sequences

    @waypoint_sequences.setter
    def waypoint_sequences(self, value):
        self.task.waypoint_sequences = value

    @property
    def waypoint_index(self):
        return self.task.waypoint_index

    @waypoint_index.setter
    def waypoint_index(self, value):
        self.task.waypoint_index = value

    @property
    def episode_step(self):
        return self.task.episode_step

    @episode_step.setter
    def episode_step(self, value):
        self.task.episode_step = value

    @property
    def episode_length_buf(self):
        return self.task.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.task.episode_length_buf = value

    @property
    def waypoint_count(self):
        return self.task.waypoint_count

    @waypoint_count.setter
    def waypoint_count(self, value):
        self.task.waypoint_count = value

    @property
    def first_arrival_step(self):
        return self.task.first_arrival_step

    @first_arrival_step.setter
    def first_arrival_step(self, value):
        self.task.first_arrival_step = value

    @property
    def crash_step(self):
        return self.task.crash_step

    @crash_step.setter
    def crash_step(self, value):
        self.task.crash_step = value

    @property
    def position_error_sum(self):
        return self.task.position_error_sum

    @position_error_sum.setter
    def position_error_sum(self, value):
        self.task.position_error_sum = value

    @property
    def position_error_steps(self):
        return self.task.position_error_steps

    @position_error_steps.setter
    def position_error_steps(self, value):
        self.task.position_error_steps = value

    @property
    def body_ang_acc(self):
        return self.task.body_ang_acc

    @body_ang_acc.setter
    def body_ang_acc(self, value):
        self.task.body_ang_acc = value

    @property
    def end_on_vertical_error(self):
        return self.task.end_on_vertical_error

    @end_on_vertical_error.setter
    def end_on_vertical_error(self, value):
        self.task.end_on_vertical_error = value

    def _read_state(self):
        return self.env._read_state()

    def mix_action(self, action: torch.Tensor, state=None):
        if state is None:
            state = self.env._read_state()
        return self.env.plant.mix(action, state, self.env.is_alive)

    def release_simulation_graphs(self, physics_loss: torch.Tensor | None = None) -> None:
        return self.env.release_simulation_graphs(physics_loss)

    def reset(
        self,
        initial_position: torch.Tensor | None = None,
        initial_quaternion: torch.Tensor | None = None,
        waypoint_sequences: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.env.respawn_on_fail = self.respawn_on_fail
        pose = None
        if initial_position is not None or initial_quaternion is not None:
            if initial_position is None or initial_quaternion is None:
                raise ValueError("initial_position and initial_quaternion must be provided together")
            linear = torch.zeros_like(initial_position)
            pose = InitialPose(initial_position, initial_quaternion, linear)
        # seed=None: sample poses from global RNG so reset_diff(seed) via torch.manual_seed works.
        policy, _ = self.env.reset(initial_states=pose, seed=None, waypoint_sequences=waypoint_sequences)
        return policy.detach()

    def reset_diff(self, seed: int | None = None) -> DiffObservation:
        self.env.respawn_on_fail = self.respawn_on_fail
        if seed is not None:
            torch.manual_seed(seed)
        observation = self.reset()
        return DiffObservation(policy=observation, critic=observation)

    def step(self, action: torch.Tensor):
        self.env.respawn_on_fail = self.respawn_on_fail
        env_step = self.env.step(action)
        extras = {
            "terminated": env_step.extras["terminated"],
            "truncated": env_step.extras["truncated"],
            "alive": env_step.extras["alive"],
            "arrived": env_step.extras.get("arrived", env_step.extras["terminated"].new_zeros(())),
            "actual_wrench": env_step.extras["actual_wrench"].detach()
            if hasattr(env_step.extras["actual_wrench"], "detach")
            else env_step.extras["actual_wrench"],
            "motor_thrust": env_step.extras["motor_thrust"].detach()
            if hasattr(env_step.extras["motor_thrust"], "detach")
            else env_step.extras["motor_thrust"],
            "loss_components": env_step.extras.get("loss_components", {}),
            "metrics": {
                "position_error": env_step.extras.get("position_error", env_step.reward.new_zeros(self.num_envs)),
            },
        }
        if extras["arrived"] is not None and hasattr(extras["arrived"], "detach"):
            extras["arrived"] = extras["arrived"].detach()
        return (
            env_step.observation.policy,
            (env_step.physics_loss, env_step.policy_loss, env_step.reward),
            env_step.done,
            extras,
        )

    def step_diff(self, action: torch.Tensor) -> DiffTransition:
        self.env.respawn_on_fail = self.respawn_on_fail
        return self.diff.step_diff(action)

    def state_dict(self) -> dict:
        drone_state = self.env._read_state()
        dofs_velocity = torch.cat((drone_state.linear_velocity, drone_state.angular_velocity), dim=-1)
        mixer = getattr(self.env.plant, "mixer", None)
        return {
            "position": detached_torch_tensor(drone_state.position),
            "quaternion": detached_torch_tensor(drone_state.quaternion),
            "dofs_velocity": detached_torch_tensor(dofs_velocity),
            "target_position": detached_torch_tensor(self.target_position),
            "last_action": detached_torch_tensor(self.last_action),
            "pid_integral": detached_torch_tensor(mixer.pid_integral) if mixer is not None else None,
            "last_angular_velocity": detached_torch_tensor(mixer.last_angular_velocity)
            if mixer is not None
            else None,
            "is_alive": detached_torch_tensor(self.is_alive),
            "waypoint_sequences": (
                None if self.waypoint_sequences is None else detached_torch_tensor(self.waypoint_sequences)
            ),
            "waypoint_index": detached_torch_tensor(self.waypoint_index),
            "episode_step": self.episode_step,
            "episode_length_buf": detached_torch_tensor(self.episode_length_buf),
            "waypoint_count": detached_torch_tensor(self.waypoint_count),
            "first_arrival_step": detached_torch_tensor(self.first_arrival_step),
            "crash_step": detached_torch_tensor(self.crash_step),
            "position_error_sum": detached_torch_tensor(self.position_error_sum),
            "position_error_steps": detached_torch_tensor(self.position_error_steps),
        }

    def load_state_dict(self, state: dict) -> torch.Tensor:
        self.env.release_simulation_graphs()
        self.env.scene.reset()
        self.env.drone.set_pos(state["position"].to(self.device), zero_velocity=True)
        self.env.drone.set_quat(state["quaternion"].to(self.device), zero_velocity=True)
        self.env.drone.set_dofs_velocity(state["dofs_velocity"].to(self.device))
        self.target_position = state["target_position"].to(self.device)
        self.last_action = state["last_action"].to(self.device)
        mixer = getattr(self.env.plant, "mixer", None)
        if mixer is not None:
            if state.get("pid_integral") is not None:
                mixer.pid_integral = state["pid_integral"].to(self.device)
            if state.get("last_angular_velocity") is not None:
                mixer.last_angular_velocity = state["last_angular_velocity"].to(self.device)
        self.is_alive = state["is_alive"].to(self.device)
        self.waypoint_sequences = (
            None if state["waypoint_sequences"] is None else state["waypoint_sequences"].to(self.device)
        )
        self.waypoint_index = state["waypoint_index"].to(self.device)
        self.episode_step = state["episode_step"]
        self.episode_length_buf = state.get(
            "episode_length_buf", torch.zeros_like(self.episode_length_buf)
        ).to(self.device)
        self.waypoint_count = state["waypoint_count"].to(self.device)
        self.first_arrival_step = state["first_arrival_step"].to(self.device)
        self.crash_step = state["crash_step"].to(self.device)
        self.position_error_sum = state["position_error_sum"].to(self.device)
        self.position_error_steps = state["position_error_steps"].to(self.device)
        return self.detach_window()

    def detach_window(self) -> torch.Tensor:
        self.env.respawn_on_fail = self.respawn_on_fail
        policy, _ = self.env.detach_window()
        return policy.detach()

    def finish_window(
        self,
        physics_loss: torch.Tensor,
        simulation_actions: list[torch.Tensor],
    ) -> tuple[DiffObservation, torch.Tensor]:
        self.env.respawn_on_fail = self.respawn_on_fail
        return self.diff.finish_window(physics_loss, simulation_actions)

    def episode_metrics(self) -> dict[str, torch.Tensor]:
        return self.task.episode_metrics()
