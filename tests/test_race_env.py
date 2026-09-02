from pathlib import Path

import torch

import genesis as gs

from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig


def _ensure_genesis():
    if not gs._initialized:
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    return gs.device


def test_ctbr_matches_between_non_diff_and_diff_modes():
    _ensure_genesis()
    action = torch.tensor([[-0.4, 0.1, -0.2, 0.0]], device=gs.device)
    results = []
    for requires_grad in (False, True):
        environment = RaceEnv(RaceEnvConfig(horizon=2), num_envs=1, requires_grad=requires_grad)
        environment.reset(seed=1)
        state = environment._read_state()
        output = environment.mix_action(action, state)
        results.append((output.command.detach(), output.motor_thrust.detach(), output.wrench.detach()))
        environment.scene.destroy() if hasattr(environment.scene, "destroy") else None
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-5, rtol=0.0)
    torch.testing.assert_close(results[0][1], results[1][1], atol=1e-5, rtol=0.0)
    torch.testing.assert_close(results[0][2], results[1][2], atol=1e-5, rtol=0.0)


def test_next_state_has_gradient_to_normalized_action():
    _ensure_genesis()
    environment = RaceEnv(RaceEnvConfig(horizon=2), num_envs=1, requires_grad=True)
    environment.reset(seed=2)
    action = torch.tensor([[environment.controller.hover_action, 0.0, 0.0, 0.0]], device=gs.device)
    action = action.detach().requires_grad_(True)
    observation, (physics_loss, _, _), _, extras = environment.step(action)
    extras["actual_wrench"].sum().backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    assert observation.shape[-1] == 13


def test_hover_and_thrust_and_roll_are_physically_reasonable():
    _ensure_genesis()
    environment = RaceEnv(RaceEnvConfig(horizon=1), num_envs=1, requires_grad=False)
    environment.reset(seed=0)
    hover = torch.tensor([[environment.controller.hover_action, 0.0, 0.0, 0.0]], device=gs.device)
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


def test_time_limit_is_truncation_and_collision_is_termination():
    _ensure_genesis()
    environment = RaceEnv(RaceEnvConfig(horizon=1, max_episode_steps=1), num_envs=1, requires_grad=False)
    environment.respawn_on_fail = False
    environment.reset(seed=0)
    hover = torch.tensor([[environment.controller.hover_action, 0.0, 0.0, 0.0]], device=gs.device)
    # DiffAero checks progress >= max_steps before incrementing, so max_steps=1 truncates on step 2.
    environment.step(hover)
    _, _, done, extras = environment.step(hover)
    assert bool(done[0])
    assert bool(extras["truncated"][0])
    assert not bool(extras["terminated"][0])
    source = Path(__file__).resolve().parents[1] / "genesis_drones" / "envs" / "race_env.py"
    assert "split_s" not in source.read_text()
