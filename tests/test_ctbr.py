import torch

from genesis_drones.controllers.ctbr_controller import CtbrController, CtbrControllerConfig


def test_hover_action_maps_to_weight():
    config = CtbrControllerConfig(randomize=False)
    controller = CtbrController(config, num_envs=1, device=torch.device("cpu"), dtype=torch.float32)
    hover = torch.tensor([[controller.hover_action, 0.0, 0.0, 0.0]])
    output = controller.step(hover, torch.zeros((1, 3)), torch.ones(1, dtype=torch.bool))
    torch.testing.assert_close(output.wrench[0, 0], torch.tensor(config.gravity), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(output.wrench[0, 1:], torch.zeros(3), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(output.command[0, 0], torch.tensor(1.0), atol=1e-5, rtol=0.0)


def test_roll_command_makes_roll_torque():
    controller = CtbrController(CtbrControllerConfig(randomize=False), 1, torch.device("cpu"), torch.float32)
    action = torch.tensor([[controller.hover_action, 1.0, 0.0, 0.0]])
    output = controller.step(action, torch.zeros((1, 3)), torch.ones(1, dtype=torch.bool))
    assert output.command[0, 1].abs() > output.command[0, 2].abs()
    assert output.wrench[0, 1].abs() > output.wrench[0, 2].abs()


def test_zero_rate_command_does_not_add_torque_at_rest():
    controller = CtbrController(CtbrControllerConfig(randomize=False), 1, torch.device("cpu"), torch.float32)
    action = torch.tensor([[controller.hover_action, 0.0, 0.0, 0.0]])
    output = controller.step(action, torch.zeros((1, 3)), torch.ones(1, dtype=torch.bool))
    torch.testing.assert_close(output.wrench[0, 1:], torch.zeros(3), atol=1e-5, rtol=0.0)


def test_next_wrench_has_gradient_to_normalized_action():
    controller = CtbrController(CtbrControllerConfig(randomize=False), 1, torch.device("cpu"), torch.float32)
    action = torch.tensor([[controller.hover_action, 0.2, -0.1, 0.0]], requires_grad=True)
    output = controller.step(action, torch.zeros((1, 3)), torch.ones(1, dtype=torch.bool))
    output.wrench.sum().backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    assert action.grad.abs().sum() > 0.0
