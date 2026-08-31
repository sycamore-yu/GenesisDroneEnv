from pathlib import Path

import torch

import genesis as gs

from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
from genesis_drones.evaluation.split_s import plan_split_s_trajectory, replay_split_s


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
    assert observation.shape[-1] == 40


def test_split_s_is_feasible_with_shared_ctbr_and_real_dynamics():
    _ensure_genesis()
    plan = plan_split_s_trajectory()
    environment = RaceEnv(
        RaceEnvConfig(horizon=1, max_episode_steps=len(plan.times) + 50, enable_gate_contact=True),
        num_envs=1,
        requires_grad=False,
    )
    result = replay_split_s(environment, plan)
    assert result["passed4"]
    assert result["passed5"]
    assert not result["collision"]
    source = Path(__file__).resolve().parents[1] / "genesis_drones" / "envs" / "race_env.py"
    assert "split_s" not in source.read_text()
