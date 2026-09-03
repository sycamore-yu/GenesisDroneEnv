from pathlib import Path

import torch

import genesis as gs

from genesis_drones.algorithms.diff_rl import as_simulation_action
from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig


def _ensure_genesis():
    if not gs._initialized:
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    return gs.device


def _full_quad(**kwargs) -> RaceEnvConfig:
    return RaceEnvConfig(dynamics="full_quad", **kwargs)


def test_ctbr_matches_between_non_diff_and_diff_modes():
    _ensure_genesis()
    action = torch.tensor([[-0.4, 0.1, -0.2, 0.0]], device=gs.device)
    results = []
    for requires_grad in (False, True):
        environment = RaceEnv(_full_quad(horizon=2), num_envs=1, requires_grad=requires_grad)
        environment.reset(seed=1)
        state = environment._read_state()
        output = environment.mix_action(action, state)
        results.append((output.command.detach(), output.motor_thrust.detach(), output.wrench.detach()))
        environment.scene.destroy() if hasattr(environment.scene, "destroy") else None
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-5, rtol=0.0)
    torch.testing.assert_close(results[0][1], results[1][1], atol=1e-5, rtol=0.0)
    torch.testing.assert_close(results[0][2], results[1][2], atol=1e-5, rtol=0.0)


def _loss_gradients_for_window(environment: RaceEnv) -> torch.Tensor:
    environment.reset_diff(seed=2)
    hover = environment.hover_command(environment.num_envs)
    physics_loss = None
    actions = []
    for _ in range(environment.spec.horizon):
        action = hover.clone()
        action[:, 1] = 0.2
        action_sim = as_simulation_action(action)
        transition = environment.step_diff(action_sim)
        physics_loss = transition.physics_loss if physics_loss is None else physics_loss + transition.physics_loss
        actions.append(action_sim)
    _, action_gradients = environment.finish_window(physics_loss, actions)
    return action_gradients


def test_full_quad_loss_has_gradient_to_action():
    _ensure_genesis()
    environment = RaceEnv(_full_quad(horizon=2), num_envs=1, requires_grad=True)
    action_gradients = _loss_gradients_for_window(environment)
    assert torch.isfinite(action_gradients).all()
    assert action_gradients.abs().sum() > 0.0


def test_native_quad_loss_has_gradient_to_action():
    _ensure_genesis()
    environment = RaceEnv(RaceEnvConfig(horizon=2), num_envs=1, requires_grad=True)
    action_gradients = _loss_gradients_for_window(environment)
    assert torch.isfinite(action_gradients).all()
    assert action_gradients.abs().sum() > 0.0


def test_hover_and_thrust_and_roll_are_physically_reasonable():
    _ensure_genesis()
    environment = RaceEnv(_full_quad(horizon=1), num_envs=1, requires_grad=False)
    environment.reset(seed=0)
    hover = environment.hover_command(1)
    start = environment._read_state().position.clone()
    for _ in range(20):
        environment.step(hover)
    hovered = environment._read_state().position
    assert torch.isfinite(hovered).all()
    assert (hovered[0, 2] - start[0, 2]).abs() < 0.5

    environment.reset(seed=0)
    extra_thrust = hover.clone()
    extra_thrust[0, 0] = 0.5
    environment.step(extra_thrust)
    after_thrust = environment._read_state()
    assert after_thrust.linear_velocity[0, 2] > 0.0

    environment.reset(seed=0)
    roll = hover.clone()
    roll[0, 1] = 1.0
    environment.step(roll)
    after_roll = environment._read_state()
    assert after_roll.angular_velocity_world[0, 0].abs() > after_roll.angular_velocity_world[0, 1].abs()


def test_native_quad_hover_stays_near_start_height():
    _ensure_genesis()
    environment = RaceEnv(RaceEnvConfig(horizon=1), num_envs=1, requires_grad=False)
    environment.reset(seed=0)
    hover = environment.hover_command(1)
    start = environment._read_state().position.clone()
    for _ in range(20):
        environment.step(hover)
    hovered = environment._read_state().position
    assert torch.isfinite(hovered).all()
    assert (hovered[0, 2] - start[0, 2]).abs() < 0.5


def test_time_limit_is_truncation_and_collision_is_termination():
    _ensure_genesis()
    environment = RaceEnv(RaceEnvConfig(horizon=1, max_episode_steps=1), num_envs=1, requires_grad=False)
    environment.respawn_on_fail = False
    environment.reset(seed=0)
    hover = environment.hover_command(1)
    environment.step(hover)
    _, _, done, extras = environment.step(hover)
    assert bool(done[0])
    assert bool(extras["truncated"][0])
    assert not bool(extras["terminated"][0])
    source = Path(__file__).resolve().parents[1] / "genesis_drones" / "envs" / "race_env.py"
    assert "split_s" not in source.read_text()


TASK_OUTPUT_KEYS = {
    "terminated",
    "truncated",
    "reset",
    "success",
    "passed",
    "wrong_way",
    "analytic_collision",
    "target_gate",
    "n_passed_gates",
    "episode_length",
    "critic_observation",
    "critic_observation_live",
    "state",
    "state_live",
    "actual_wrench",
    "motor_thrust",
    "loss_components",
}


def test_native_and_full_quad_share_race_task_output():
    _ensure_genesis()
    extra_keys = []
    for dynamics in ("native_quad", "full_quad"):
        environment = RaceEnv(RaceEnvConfig(horizon=1, dynamics=dynamics), num_envs=1, requires_grad=False)
        policy, state = environment.reset(seed=0)
        assert policy.shape == (1, 13)
        assert state.shape == (1, 34)
        action = environment.hover_command(1)
        assert action.shape == (1, 4)
        assert action.abs().max() <= 1.0
        next_policy, losses, done, extras = environment.step(action)
        physics_loss, policy_loss, reward = losses
        assert next_policy.shape == (1, 13)
        assert physics_loss.shape == reward.shape == done.shape == (1,)
        assert extras["critic_observation"].shape[-1] == 13
        assert extras["state"].shape[-1] == 34
        extra_keys.append(set(extras))
        if hasattr(environment.scene, "destroy"):
            environment.scene.destroy()
    assert extra_keys[0] == extra_keys[1]
    assert TASK_OUTPUT_KEYS <= extra_keys[0]
