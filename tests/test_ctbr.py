import torch

from genesis_drones.controllers.ctbr_controller import CtbrController, CtbrControllerConfig


def test_action_order_is_thrust_then_body_rates():
    config = CtbrControllerConfig()
    controller = CtbrController(config, num_envs=1, device=torch.device("cpu"), dtype=torch.float32)
    hover = controller.hover_action
    zeros = torch.zeros((1, 3))
    active = torch.ones(1, dtype=torch.bool)

    hover_action = torch.tensor([[hover, 0.0, 0.0, 0.0]])
    hover_out = controller.step(hover_action, zeros, active)
    mass_gravity = 4.0 * config.thrust_coefficient * config.base_rpm * config.base_rpm
    torch.testing.assert_close(hover_out.wrench[0, 0], torch.tensor(mass_gravity), atol=1e-3, rtol=0.0)
    torch.testing.assert_close(hover_out.wrench[0, 1:], torch.zeros(3), atol=1e-4, rtol=0.0)

    roll_action = torch.tensor([[hover, 1.0, 0.0, 0.0]])
    controller.reset()
    roll_out = controller.step(roll_action, zeros, active)
    assert roll_out.command[0, 1].abs() > roll_out.command[0, 2].abs()
    assert roll_out.wrench[0, 1].abs() > roll_out.wrench[0, 2].abs()


def test_hover_action_matches_thrust_to_weight_ratio():
    config = CtbrControllerConfig(thrust_to_weight_ratio=3.3)
    assert abs(2.0 / 3.3 - 1.0 - CtbrController(config, 1, torch.device("cpu"), torch.float32).hover_action) < 1e-6


def test_zero_rate_command_does_not_add_torque_at_rest():
    controller = CtbrController(CtbrControllerConfig(), 1, torch.device("cpu"), torch.float32)
    action = torch.tensor([[controller.hover_action, 0.0, 0.0, 0.0]])
    output = controller.step(action, torch.zeros((1, 3)), torch.ones(1, dtype=torch.bool))
    torch.testing.assert_close(output.wrench[0, 1:], torch.zeros(3), atol=1e-4, rtol=0.0)


def test_next_wrench_has_gradient_to_normalized_action():
    controller = CtbrController(CtbrControllerConfig(), 1, torch.device("cpu"), torch.float32)
    action = torch.tensor([[controller.hover_action, 0.2, -0.1, 0.0]], requires_grad=True)
    output = controller.step(action, torch.zeros((1, 3)), torch.ones(1, dtype=torch.bool))
    output.wrench.sum().backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    assert action.grad.abs().sum() > 0.0
