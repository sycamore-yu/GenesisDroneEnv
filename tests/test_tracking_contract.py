from pathlib import Path

import genesis as gs
import torch
import yaml

from genesis_drones.controllers.pid_controller import PIDcontroller
from genesis_drones.envs.genesis_env import Genesis_env
from genesis_drones.tasks.track_task import Track_task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THRUST_TO_WEIGHT_RATIO = 3.3
HOVER_ACTION = 2.0 / THRUST_TO_WEIGHT_RATIO - 1.0


class _FakeOdom:
    def __init__(self, num_envs: int, device: torch.device):
        for name in ("body_ang_acc", "body_ang_vel", "body_euler", "body_linear_vel"):
            setattr(self, name, torch.zeros((num_envs, 3), device=device, dtype=gs.tc_float))
        self.body_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_envs, 1)
        self.has_nan = torch.zeros(num_envs, device=device, dtype=torch.bool)
        for name in ("last_world_pos", "world_linear_vel", "world_pos"):
            setattr(self, name, torch.zeros((num_envs, 3), device=device, dtype=gs.tc_float))


class _FakeDrone:
    def __init__(self, num_envs: int, device: torch.device):
        self.odom = _FakeOdom(num_envs, device)


class _FakeGenesisEnv:
    """Stands in for Genesis_env: only drone odometry is read by Track_task."""

    def __init__(self, num_envs: int, device: torch.device):
        self.drone = _FakeDrone(num_envs, device)
        self.target = None

    def reset(self, envs_idx):
        self.drone.odom.world_pos[envs_idx] = torch.tensor([0.0, 0.0, 0.6], device=envs_idx.device)
        self.drone.odom.last_world_pos[envs_idx] = self.drone.odom.world_pos[envs_idx]

    def step(self, actions):
        self.drone.odom.last_world_pos[:] = self.drone.odom.world_pos


def _ensure_genesis() -> torch.device:
    if not gs._initialized:
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _task_config(**overrides) -> dict:
    with (PROJECT_ROOT / "config" / "track_rl" / "rl_env.yaml").open() as file:
        config = yaml.safe_load(file)["task"]
    config.update(overrides)
    return config


def _fake_task(device: torch.device, num_envs: int = 2, **overrides) -> Track_task:
    return Track_task(
        genesis_env=_FakeGenesisEnv(num_envs, device),
        env_config={"dt": 0.01, "num_envs": num_envs},
        train_config={"algorithm": {"gamma": 0.99}},
        task_config=_task_config(**overrides),
    )


def test_observation_layout_matches_configured_width():
    device = _ensure_genesis()
    task = _fake_task(device)

    task.reset()
    task.step(torch.zeros((2, 4), device=device))
    observation = task.get_observations()["state"]

    assert observation.shape == (2, task.task_config["num_obs"])
    # world_pos, command, position error, quaternion, linear velocity, angular velocity, last action
    assert torch.allclose(observation[:, 6:9], task.cur_pos_error)
    assert torch.allclose(observation[:, :3], task.genesis_env.drone.odom.world_pos)


def test_position_error_is_command_minus_position():
    device = _ensure_genesis()
    task = _fake_task(
        device,
        command_cfg={"pos_x_range": [0.0, 0.0], "pos_y_range": [0.0, 0.0], "pos_z_range": [1.0, 1.0]},
        termination_if_z_greater_than=5.0,
    )

    task.reset()
    assert torch.allclose(task.command_buf, torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(2, 1))

    task.step(torch.zeros((2, 4), device=device))
    assert torch.allclose(task.cur_pos_error, torch.tensor([[0.0, 0.0, 0.4]], device=device).repeat(2, 1))


def test_episode_truncates_at_max_episode_length():
    device = _ensure_genesis()
    task = _fake_task(
        device,
        max_episode_length=1,
        command_cfg={"pos_x_range": [0.0, 0.0], "pos_y_range": [0.0, 0.0], "pos_z_range": [1.0, 1.0]},
        termination_if_z_greater_than=5.0,
    )

    task.reset()
    _, _, dones, _ = task.step(torch.zeros((2, 4), device=device))
    assert not dones.any()
    assert not task.crash_condition_buf.any()

    _, _, dones, _ = task.step(torch.zeros((2, 4), device=device))
    assert dones.all()
    assert (task.episode_length_buf == 0).all()


def test_crash_conditions_follow_configured_limits():
    device = _ensure_genesis()
    task = _fake_task(device)

    task.reset()
    # env 0: below the ground termination height. env 1: vertical error beyond the configured limit.
    task.genesis_env.drone.odom.world_pos[0, 2] = 0.05
    task.genesis_env.drone.odom.world_pos[1, 2] = 3.0
    _, _, dones, _ = task.step(torch.zeros((2, 4), device=device))

    assert torch.equal(task.crash_condition_buf, torch.tensor([True, True], device=device))
    assert dones.all()


def test_fully_differentiable_ppo_keeps_command_and_skips_arrival_bonus():
    device = _ensure_genesis()
    task = _fake_task(
        device,
        fully_differentiable=True,
        command_cfg={"pos_x_range": [0.0, 0.0], "pos_y_range": [0.0, 0.0], "pos_z_range": [0.6, 0.6]},
        termination_if_z_greater_than=5.0,
    )
    task.reset()
    command = task.command_buf.clone()
    task.genesis_env.drone.odom.world_pos[:] = command
    task.genesis_env.drone.odom.last_world_pos[:] = command
    _, reward, _, _ = task.step(torch.zeros((2, 4), device=device))
    assert torch.allclose(task.command_buf, command)
    arrival_bonus = 20.0 * 10.0 * 0.01
    assert float(reward.max()) < 0.25 * arrival_bonus


def test_mixer_produces_hover_rpm_at_hover_action():
    controller = PIDcontroller.__new__(PIDcontroller)
    controller.TWR = THRUST_TO_WEIGHT_RATIO
    controller.base_rpm = 100.0
    controller.pid_output = torch.zeros((2, 3))
    controller.rc_command = torch.zeros(7)
    controller.throttle_command = torch.zeros(2)

    hover_actions = torch.zeros((2, 4))
    hover_actions[:, 3] = HOVER_ACTION
    torch.testing.assert_close(controller.mixer(hover_actions), torch.full((2, 4), 100.0))

    offset_actions = torch.zeros((2, 4))
    offset_actions[:, 3] = HOVER_ACTION + torch.tensor([-0.2, 0.2])
    thrust_ratios = torch.square(controller.mixer(offset_actions)[:, 0] / controller.base_rpm)
    # Collective thrust is linear in the throttle action, so equal offsets give equal and opposite ratios.
    torch.testing.assert_close(
        thrust_ratios - 1.0, torch.tensor([-0.33, 0.33]), atol=1e-6, rtol=0.0
    )


def test_attitude_commands_move_the_drone_horizontally():
    device = _ensure_genesis()

    with (PROJECT_ROOT / "config" / "track_rl" / "genesis_env.yaml").open() as file:
        env_config = yaml.safe_load(file)
    with (PROJECT_ROOT / "config" / "track_rl" / "flight.yaml").open() as file:
        flight_config = yaml.safe_load(file)
    env_config.update(
        num_envs=4,
        show_viewer=False,
        render_cam=False,
        vis_waypoints=False,
        fixed_init_pos=True,
        drone_init_pos=[0.0, 0.0, 1.0],
    )

    real_env = Genesis_env(env_config, flight_config)
    real_env.reset(torch.arange(4, device=device))
    actions = torch.zeros((4, 4), device=device)
    actions[0, 1] = 0.1
    actions[1, 1] = -0.1
    actions[2, 0] = 0.1
    actions[3, 0] = -0.1
    for _ in range(100):
        real_env.step(actions)

    horizontal_displacement = torch.linalg.vector_norm(real_env.drone.odom.world_pos[:, :2], dim=1)
    assert torch.all(horizontal_displacement > 0.01)
