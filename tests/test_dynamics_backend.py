import torch

from genesis_drones.controllers.ctbr_controller import CtbrController, CtbrControllerConfig
from genesis_drones.controllers.native_config import NativeQuadConfig
from genesis_drones.controllers.native_mixer import NativeQuadMixer
from genesis_drones.dynamics import DroneState, make_dynamics_backend
from genesis_drones.utils.geometry import quaternion_to_rotation_matrix


def _state(device="cpu") -> DroneState:
    return DroneState(
        position=torch.tensor([[0.1, -0.2, 1.0]], device=device),
        quaternion=torch.tensor([[0.98, 0.1, -0.05, 0.02]], device=device),
        linear_velocity=torch.tensor([[0.3, -0.1, 0.05]], device=device),
        angular_velocity=torch.tensor([[0.2, -0.15, 0.05]], device=device),
    )


def test_native_quad_backend_matches_mixer_wrench():
    device = torch.device("cpu")
    dtype = torch.float32
    dt = 0.0333
    config = NativeQuadConfig()
    mixer = NativeQuadMixer(1, device, dtype, dt, config)
    backend = make_dynamics_backend("native_quad", 1, device, dtype, dt, native_config=config)
    state = _state(device)
    action = torch.tensor([[0.1, -0.2, 0.05, mixer.hover_action]], device=device)
    alive = torch.ones(1, dtype=torch.bool, device=device)
    motor_thrust, wrench = mixer.mix(action, state.quaternion, state.angular_velocity, alive)
    expected_force = NativeQuadMixer.generalized_force(state.quaternion, wrench * alive[:, None])
    output = backend.control(action, state, alive)
    torch.testing.assert_close(output.motor_thrust, motor_thrust)
    torch.testing.assert_close(output.wrench, wrench * alive[:, None])
    torch.testing.assert_close(output.generalized_force, expected_force)
    assert backend.nominal_action == NativeQuadBackend_nominal(config)
    assert backend.name == "native_quad"
    assert backend.action_dim == 4


def NativeQuadBackend_nominal(config: NativeQuadConfig) -> tuple[float, ...]:
    hover = NativeQuadMixer.hover(config)
    return (0.0, 0.0, 0.0, hover)


def test_full_quad_backend_matches_ctbr_wrench():
    device = torch.device("cpu")
    dtype = torch.float32
    cfg = CtbrControllerConfig(randomize=False)
    controller = CtbrController(cfg, 1, device, dtype)
    backend = make_dynamics_backend("full_quad", 1, device, dtype, 0.0333, controller_config=cfg)
    state = _state(device)
    action = torch.tensor([[controller.hover_action, 0.2, -0.1, 0.05]], device=device)
    alive = torch.ones(1, dtype=torch.bool, device=device)
    rotation = quaternion_to_rotation_matrix(state.quaternion)
    omega_body = torch.einsum("nji,nj->ni", rotation, state.angular_velocity)
    reference = controller.step(action, omega_body, alive)
    body_velocity = torch.einsum("nji,nj->ni", rotation, state.linear_velocity)
    drag_world = torch.einsum("nij,nj->ni", rotation, -controller.drag * body_velocity)
    force_world = rotation[:, :, 2] * reference.wrench[:, :1] + drag_world
    torque_world = torch.einsum("nij,nj->ni", rotation, reference.wrench[:, 1:])
    expected = torch.cat((force_world, torque_world), dim=-1)
    output = backend.control(action, state, alive)
    torch.testing.assert_close(output.wrench, reference.wrench)
    torch.testing.assert_close(output.motor_thrust, reference.motor_thrust)
    torch.testing.assert_close(output.generalized_force, expected)
    hover = 2.0 / cfg.max_normalized_thrust - 1.0
    assert backend.nominal_action == (hover, 0.0, 0.0, 0.0)


def test_factory_rejects_unimplemented_names():
    try:
        make_dynamics_backend("pmc", 1, torch.device("cpu"), torch.float32, 0.01)
    except ValueError as error:
        assert "pmc" in str(error)
    else:
        raise AssertionError("expected ValueError for pmc")
