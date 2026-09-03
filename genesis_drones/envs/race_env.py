from dataclasses import dataclass
from pathlib import Path

import torch

import genesis as gs

from genesis_drones.adapters.diff_rl_adapter import DiffRLAdapter
from genesis_drones.controllers.ctbr_controller import CtbrControllerConfig
from genesis_drones.dynamics import BACKENDS, make_dynamics_backend
from genesis_drones.envs.differentiable import DiffEnvSpec, DiffObservation, DiffTransition
from genesis_drones.envs.genesis_task_env import GenesisTaskEnv, detached_torch_tensor
from genesis_drones.tasks.racing_core import POLICY_OBSERVATION_SIZE, EvaluationInitialStates, RaceTrackSpec
from genesis_drones.tasks.racing_core_task import RacingCore, RacingCoreConfig
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
    dynamics: str = "native_quad"
    controller: CtbrControllerConfig = CtbrControllerConfig()


def make_racing_env(
    config: RaceEnvConfig,
    num_envs: int,
    track: RaceTrackSpec = RACING_TRACK,
    requires_grad: bool = True,
    show_viewer: bool = False,
) -> GenesisTaskEnv:
    if config.dynamics not in BACKENDS:
        raise ValueError(f"unsupported dynamics: {config.dynamics}")
    task = RacingCore(
        RacingCoreConfig(
            dt=config.dt,
            max_episode_steps=config.max_episode_steps,
            min_target_velocity=config.min_target_velocity,
            max_target_velocity=config.max_target_velocity,
            gamma=config.gamma,
            td_lambda=config.td_lambda,
        ),
        num_envs=num_envs,
        device=gs.device,
        dtype=gs.tc_float,
        track=track,
    )
    backend = make_dynamics_backend(
        config.dynamics,
        num_envs,
        gs.device,
        gs.tc_float,
        config.dt,
        controller_config=config.controller,
    )
    return GenesisTaskEnv(
        task_core=task,
        dynamics_backend=backend,
        num_envs=num_envs,
        requires_grad=requires_grad,
        show_viewer=show_viewer,
        dt=config.dt,
        horizon=config.horizon,
        gate_entities_fn=(lambda scene: add_track_gates(scene, track)) if show_viewer else None,
    )


class RaceEnv:
    """Thin racing entry: builds GenesisTaskEnv + DiffRLAdapter. No duplicated task logic."""

    action_dim = 4
    policy_observation_dim = POLICY_OBSERVATION_SIZE
    critic_observation_dim = POLICY_OBSERVATION_SIZE

    @classmethod
    def spec_from_config(cls, config: RaceEnvConfig) -> DiffEnvSpec:
        if config.dynamics not in BACKENDS:
            raise ValueError(f"unsupported dynamics: {config.dynamics}")
        return DiffEnvSpec(
            policy_observation_dim=cls.policy_observation_dim,
            critic_observation_dim=cls.policy_observation_dim,
            action_dim=cls.action_dim,
            horizon=config.horizon,
            nominal_action=BACKENDS[config.dynamics].default_nominal_action(
                controller_config=config.controller
            ),
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
        self.env = make_racing_env(config, num_envs, track, requires_grad, show_viewer)
        self.diff = DiffRLAdapter(self.env, horizon=config.horizon)
        self.num_envs = num_envs
        self.requires_grad = requires_grad
        self.device = self.env.device
        self.dynamics = self.env.dynamics
        self.plant = self.env.plant
        self.scene = self.env.scene
        self.drone = self.env.drone
        self.track = self.env.task.track
        self.track_spec = track
        self.spec = self.diff.spec
        self.respawn_on_fail = True

    @property
    def is_alive(self):
        return self.env.is_alive

    @is_alive.setter
    def is_alive(self, value):
        self.env.is_alive = value

    @property
    def last_action(self):
        return self.env.last_action

    @last_action.setter
    def last_action(self, value):
        self.env.last_action = value

    @property
    def episode_length_buf(self):
        return self.env.task.episode_length_buf

    @property
    def n_passed_gates(self):
        return self.env.task.n_passed_gates

    @property
    def path_length(self):
        return self.env.task.path_length

    @property
    def target_gates(self):
        return self.env.task.target_gates

    def hover_command(self, count=None):
        return self.env.hover_command(count)

    def _read_state(self):
        return self.env._read_state()

    def mix_action(self, action, state=None):
        if state is None:
            state = self.env._read_state()
        return self.env.plant.mix(action, state, self.env.is_alive)

    def reset(self, initial_states: EvaluationInitialStates | None = None, seed: int = 0):
        self.env.respawn_on_fail = self.respawn_on_fail
        return self.env.reset(initial_states=initial_states, seed=seed)

    def reset_diff(self, seed: int | None = None) -> DiffObservation:
        self.env.respawn_on_fail = self.respawn_on_fail
        return self.diff.reset_diff(seed=seed)

    def step(self, action: torch.Tensor):
        self.env.respawn_on_fail = self.respawn_on_fail
        env_step = self.env.step(action)
        return (
            env_step.observation.policy,
            (env_step.physics_loss, env_step.policy_loss, env_step.reward),
            env_step.done,
            env_step.extras,
        )

    def step_diff(self, action: torch.Tensor) -> DiffTransition:
        self.env.respawn_on_fail = self.respawn_on_fail
        return self.diff.step_diff(action)

    def release_simulation_graphs(self, physics_loss=None):
        return self.env.release_simulation_graphs(physics_loss)

    def detach_window(self):
        return self.env.detach_window()

    def finish_window(self, physics_loss, simulation_actions):
        return self.diff.finish_window(physics_loss, simulation_actions)
