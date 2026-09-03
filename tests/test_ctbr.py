import torch

from genesis_drones.controllers.ctbr_controller import CtbrController, CtbrControllerConfig


def _cpu_controller():
    return CtbrController(CtbrControllerConfig(randomize=False), 1, torch.device("cpu"), torch.float32)


def test_hover_action_maps_to_weight():
    config = CtbrControllerConfig(randomize=False)
    controller = CtbrController(config, num_envs=1, device=torch.device("cpu"), dtype=torch.float32)
    hover = torch.tensor([[controller.hover_action, 0.0, 0.0, 0.0]], device="cpu")
    omega = torch.zeros((1, 3), device="cpu")
    alive = torch.ones(1, dtype=torch.bool, device="cpu")
    output = controller.step(hover, omega, alive)
    torch.testing.assert_close(output.wrench[0, 0], torch.tensor(config.gravity, device="cpu"), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(output.wrench[0, 1:], torch.zeros(3, device="cpu"), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(output.command[0, 0], torch.tensor(1.0, device="cpu"), atol=1e-5, rtol=0.0)


def test_roll_command_makes_roll_torque():
    controller = _cpu_controller()
    action = torch.tensor([[controller.hover_action, 1.0, 0.0, 0.0]], device="cpu")
    output = controller.step(action, torch.zeros((1, 3), device="cpu"), torch.ones(1, dtype=torch.bool, device="cpu"))
    assert output.command[0, 1].abs() > output.command[0, 2].abs()
    assert output.wrench[0, 1].abs() > output.wrench[0, 2].abs()


def test_zero_rate_command_does_not_add_torque_at_rest():
    controller = _cpu_controller()
    action = torch.tensor([[controller.hover_action, 0.0, 0.0, 0.0]], device="cpu")
    output = controller.step(action, torch.zeros((1, 3), device="cpu"), torch.ones(1, dtype=torch.bool, device="cpu"))
    torch.testing.assert_close(output.wrench[0, 1:], torch.zeros(3, device="cpu"), atol=1e-5, rtol=0.0)


def test_next_wrench_has_gradient_to_normalized_action():
    controller = _cpu_controller()
    action = torch.tensor([[controller.hover_action, 0.2, -0.1, 0.0]], device="cpu", requires_grad=True)
    output = controller.step(action, torch.zeros((1, 3), device="cpu"), torch.ones(1, dtype=torch.bool, device="cpu"))
    output.wrench.sum().backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    assert action.grad.abs().sum() > 0.0
