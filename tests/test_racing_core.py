import math
from pathlib import Path

import torch

from genesis_drones.tasks.racing_core import (
    CRITIC_OBSERVATION_SIZE,
    POLICY_OBSERVATION_SIZE,
    GateSpec,
    RaceTrackSpec,
    detect_race_events,
    evaluation_states_checksum,
    gate_collision_loss,
    gate_progress_loss,
    gate_vector_field_loss,
    generate_initial_states,
    load_evaluation_initial_states,
    racing_observations,
    save_evaluation_initial_states,
    track_to_tensors,
)
from genesis_drones.tasks.racing_tracks import FIXED_SEVEN_GATE_TRACK


SAFETY_RADIUS = 0.06


def _device() -> torch.device:
    return torch.device("cpu")


def _gate(
    position: tuple[float, float, float],
    yaw: float,
    opening: float = 2.0,
    outer: float = 3.0,
    depth: float = 0.2,
) -> GateSpec:
    return GateSpec(
        position=position,
        quaternion=(math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)),
        opening_width=opening,
        opening_height=opening,
        outer_width=outer,
        outer_height=outer,
        depth=depth,
    )


def _track(*gates: GateSpec) -> RaceTrackSpec:
    return RaceTrackSpec(name="test", gates=gates, order=tuple(range(len(gates))))


def _active(count: int, device: torch.device) -> torch.Tensor:
    return torch.ones(count, device=device, dtype=torch.bool)


def test_fixed_seven_coordinates_are_not_in_racing_core_source():
    source = (Path(__file__).resolve().parents[1] / "genesis_drones" / "tasks" / "racing_core.py").read_text()
    assert "10.0" not in source
    assert "3.5668" not in source
    assert "fixed_seven" not in source


def test_fourth_and_fifth_gates_keep_overlapping_opposite_normals():
    track = track_to_tensors(FIXED_SEVEN_GATE_TRACK, device=_device(), dtype=torch.float32)
    fourth = track.positions[3]
    fifth = track.positions[4]
    torch.testing.assert_close(fourth[:2], fifth[:2])
    assert fourth[2] > fifth[2]
    fourth_normal = track.rotations[3, :, 0]
    fifth_normal = track.rotations[4, :, 0]
    assert torch.dot(fourth_normal, fifth_normal) < 0.0


def test_actor_observation_is_40d_and_critic_adds_remaining_gates():
    device = _device()
    track = track_to_tensors(_track(_gate((0.0, 0.0, 1.0), 0.0), _gate((4.0, 0.0, 1.0), 0.0)), device, torch.float32)
    position = torch.tensor([[-2.0, 0.1, 1.0]], device=device)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    policy, critic = racing_observations(
        position,
        quaternion,
        torch.tensor([[1.0, 0.0, 0.0]], device=device),
        torch.zeros((1, 3), device=device),
        torch.tensor([[0.1, 0.0, 0.0, 0.0]], device=device),
        track,
        torch.zeros(1, device=device, dtype=torch.long),
    )
    assert policy.shape == (1, POLICY_OBSERVATION_SIZE)
    assert critic.shape == (1, CRITIC_OBSERVATION_SIZE)
    torch.testing.assert_close(critic[:, :40], policy)
    torch.testing.assert_close(critic[:, 40], torch.tensor([2.0]))


def test_actor_observation_ignores_world_translation():
    device = _device()
    track = track_to_tensors(_track(_gate((0.0, 0.0, 1.0), 0.0)), device, torch.float32)
    position = torch.tensor([[-2.0, 0.0, 1.0]], device=device)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    velocity = torch.tensor([[0.5, 0.0, 0.0]], device=device)
    zeros = torch.zeros((1, 3), device=device)
    last_action = torch.zeros((1, 4), device=device)
    gate_index = torch.zeros(1, device=device, dtype=torch.long)
    policy, _ = racing_observations(position, quaternion, velocity, zeros, last_action, track, gate_index)
    shifted_track = track_to_tensors(_track(_gate((3.0, 4.0, 2.0), 0.0)), device, torch.float32)
    shifted_position = position + torch.tensor([[3.0, 4.0, 1.0]], device=device)
    shifted, _ = racing_observations(
        shifted_position, quaternion, velocity, zeros, last_action, shifted_track, gate_index
    )
    torch.testing.assert_close(policy, shifted, atol=1e-5, rtol=0.0)


def test_high_speed_segment_detects_forward_pass_and_rejects_wrong_way():
    device = _device()
    track = track_to_tensors(_track(_gate((0.0, 0.0, 1.0), 0.0)), device, torch.float32)
    gate_index = torch.zeros(2, device=device, dtype=torch.long)
    events = detect_race_events(
        torch.tensor([[-10.0, 0.0, 1.0], [10.0, 0.0, 1.0]], device=device),
        torch.tensor([[10.0, 0.0, 1.0], [-10.0, 0.0, 1.0]], device=device),
        track,
        gate_index,
        SAFETY_RADIUS,
        _active(2, device),
    )
    assert events.passed.tolist() == [True, False]
    assert events.wrong_way.tolist() == [False, True]
    assert events.completed.tolist() == [True, False]


def test_frame_crossing_outside_safe_opening_is_analytic_collision():
    device = _device()
    track = track_to_tensors(_track(_gate((0.0, 0.0, 1.0), 0.0)), device, torch.float32)
    events = detect_race_events(
        torch.tensor([[-1.0, 1.4, 1.0]], device=device),
        torch.tensor([[1.0, 1.4, 1.0]], device=device),
        track,
        torch.zeros(1, device=device, dtype=torch.long),
        SAFETY_RADIUS,
        _active(1, device),
    )
    assert events.passed.tolist() == [False]
    assert events.analytic_collision.tolist() == [True]


def test_passing_last_gate_completes_and_remaining_gates_are_zero():
    device = _device()
    track = track_to_tensors(_track(_gate((0.0, 0.0, 1.0), 0.0), _gate((4.0, 0.0, 1.0), 0.0)), device, torch.float32)
    events = detect_race_events(
        torch.tensor([[3.0, 0.0, 1.0]], device=device),
        torch.tensor([[5.0, 0.0, 1.0]], device=device),
        track,
        torch.ones(1, device=device, dtype=torch.long),
        SAFETY_RADIUS,
        _active(1, device),
    )
    assert events.passed.tolist() == [True]
    assert events.completed.tolist() == [True]
    assert events.next_gate_index.tolist() == [2]
    _, critic = racing_observations(
        torch.tensor([[5.0, 0.0, 1.0]], device=device),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        torch.zeros((1, 3), device=device),
        torch.zeros((1, 3), device=device),
        torch.zeros((1, 4), device=device),
        track,
        events.next_gate_index,
    )
    torch.testing.assert_close(critic[:, 40], torch.tensor([0.0]))


def test_two_gate_track_advances_index_only_on_forward_pass():
    device = _device()
    track = track_to_tensors(_track(_gate((0.0, 0.0, 1.0), 0.0), _gate((4.0, 0.0, 1.0), 0.0)), device, torch.float32)
    events = detect_race_events(
        torch.tensor([[-1.0, 0.0, 1.0]], device=device),
        torch.tensor([[1.0, 0.0, 1.0]], device=device),
        track,
        torch.zeros(1, device=device, dtype=torch.long),
        SAFETY_RADIUS,
        _active(1, device),
    )
    assert events.next_gate_index.tolist() == [1]
    assert events.completed.tolist() == [False]


def test_initial_states_are_relative_to_first_gate_and_repeatable():
    device = _device()
    track = track_to_tensors(FIXED_SEVEN_GATE_TRACK, device, torch.float32)
    first = generate_initial_states(track, count=8, seed=7)
    second = generate_initial_states(track, count=8, seed=7)
    different = generate_initial_states(track, count=8, seed=8)
    torch.testing.assert_close(first.position, second.position)
    torch.testing.assert_close(first.quaternion, second.quaternion)
    torch.testing.assert_close(first.linear_velocity, second.linear_velocity)
    assert not torch.equal(first.position, different.position)
    gate_position = track.positions[track.order[0]]
    gate_normal = track.rotations[0, :, 0]
    offset = first.position - gate_position
    along_normal = torch.sum(offset * gate_normal, dim=-1)
    torch.testing.assert_close(along_normal, torch.full_like(along_normal, -2.0), atol=0.1, rtol=0.0)


def test_collision_loss_grows_near_frame_and_negative_gradient_points_away(tmp_path=None):
    device = _device()
    track = track_to_tensors(_track(_gate((0.0, 0.0, 1.0), 0.0)), device, torch.float32)
    far = torch.tensor([[0.0, 0.0, 1.0]], device=device, requires_grad=True)
    near = torch.tensor([[0.0, 0.95, 1.0]], device=device, requires_grad=True)
    zeros = torch.zeros((1, 3), device=device)
    gate_index = torch.zeros(1, device=device, dtype=torch.long)
    far_loss, far_clearance, _ = gate_collision_loss(far, zeros, track, gate_index, SAFETY_RADIUS, 0.01, 1.0, 4.0)
    near_loss, near_clearance, _ = gate_collision_loss(near, zeros, track, gate_index, SAFETY_RADIUS, 0.01, 1.0, 4.0)
    assert near_loss.item() > far_loss.item()
    assert near_clearance.item() < far_clearance.item()
    near_loss.backward()
    assert torch.isfinite(near.grad).all()
    toward_frame = torch.tensor([0.0, 1.0, 0.0], device=device)
    assert torch.dot(near.grad[0], toward_frame) > 0.0


def test_closing_speed_does_not_create_a_velocity_gradient_path():
    device = _device()
    track = track_to_tensors(_track(_gate((0.0, 0.0, 1.0), 0.0)), device, torch.float32)
    position = torch.tensor([[0.0, 0.95, 1.0]], device=device, requires_grad=True)
    velocity = torch.tensor([[0.0, 2.0, 0.0]], device=device, requires_grad=True)
    loss, _, _ = gate_collision_loss(
        position,
        velocity,
        track,
        torch.zeros(1, device=device, dtype=torch.long),
        SAFETY_RADIUS,
        0.01,
        1.0,
        4.0,
    )
    loss.backward()
    assert position.grad is not None
    assert velocity.grad is None or torch.equal(velocity.grad, torch.zeros_like(velocity))


def test_progress_and_vector_field_losses_are_finite_for_any_gate_count():
    device = _device()
    for count in (1, 2, 7):
        gates = tuple(_gate((float(index) * 3.0, 0.0, 1.0), 0.0) for index in range(count))
        track = track_to_tensors(_track(*gates), device, torch.float32)
        position = torch.tensor([[-1.0, 0.0, 1.0]], device=device, requires_grad=True)
        velocity = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        gate_index = torch.zeros(1, device=device, dtype=torch.long)
        progress = gate_progress_loss(position, velocity, track, gate_index)
        field = gate_vector_field_loss(position, velocity, track, gate_index)
        (progress + field).backward()
        assert torch.isfinite(progress)
        assert torch.isfinite(field)
        assert torch.isfinite(position.grad).all()


def test_evaluation_initial_states_round_trip_with_checksum(tmp_path):
    device = _device()
    track = track_to_tensors(FIXED_SEVEN_GATE_TRACK, device, torch.float32)
    states = generate_initial_states(track, count=100, seed=20250830)
    path = tmp_path / "eval_states.pt"
    checksum = save_evaluation_initial_states(states, path)
    loaded, loaded_checksum = load_evaluation_initial_states(path)
    assert checksum == loaded_checksum
    assert checksum == evaluation_states_checksum(states)
    torch.testing.assert_close(loaded.position, states.position)
    torch.testing.assert_close(loaded.quaternion, states.quaternion)
    torch.testing.assert_close(loaded.linear_velocity, states.linear_velocity)
