import torch

from genesis_drones.controllers.native_mixer import NativeQuadMixer


def test_native_hover_makes_positive_collective_thrust():
    mixer = NativeQuadMixer(1, torch.device("cpu"), torch.float32, dt=0.0333)
    action = torch.tensor([[0.0, 0.0, 0.0, mixer.hover_action]])
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    omega = torch.zeros(1, 3)
    alive = torch.ones(1, dtype=torch.bool)
    motor_thrust, wrench = mixer.mix(action, quaternion, omega, alive)
    assert motor_thrust.sum() > 0.0
    assert wrench[0, 0] > 0.0
