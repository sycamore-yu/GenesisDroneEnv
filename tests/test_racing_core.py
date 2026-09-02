import math

import torch

from genesis_drones.tasks.racing_core import (
    CRITIC_OBSERVATION_SIZE,
    POLICY_OBSERVATION_SIZE,
    GateSpec,
    RaceTrackSpec,
    detect_race_events,
    evaluation_states_checksum,
    generate_initial_states,
    load_evaluation_initial_states,
    racing_loss_reward,
    racing_observations,
    save_evaluation_initial_states,
    track_to_tensors,
)
from genesis_drones.tasks.racing_tracks import RACING_TRACK


def _device() -> torch.device:
    return torch.device("cpu")


def _gate(position: tuple[float, float, float], yaw: float) -> GateSpec:
    return GateSpec(
        position=position,
        quaternion=(math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)),
        opening_width=3.0,
        opening_height=3.0,
        outer_width=3.0,
        outer_height=3.0,
        depth=0.0,
    )


def _track(*gates: GateSpec) -> RaceTrackSpec:
    return RaceTrackSpec(name="test", gates=gates, order=tuple(range(len(gates))))


def test_racing_track_matches_diffaero_eight_gate_loop():
    expected_positions = torch.tensor(
        (
            (1.5, -1.5, 1.5),
            (0.0, 0.0, 1.5),
            (-1.5, 1.5, 1.5),
            (0.0, 3.0, 1.5),
            (1.5, 1.5, 1.5),
            (0.0, 0.0, 1.5),
            (-1.5, -1.5, 1.5),
            (0.0, -3.0, 1.5),
        )
    )
    expected_yaws = torch.tensor((math.pi / 2, math.pi, math.pi / 2, 0.0, -math.pi / 2, -math.pi, -math.pi / 2, 0.0))
    track = track_to_tensors(RACING_TRACK, _device(), torch.float32)
    torch.testing.assert_close(track.positions, expected_positions)
    normals = track.rotations[:, :, 0]
    torch.testing.assert_close(torch.atan2(normals[:, 1], normals[:, 0]), expected_yaws, atol=1e-6, rtol=0.0)


def test_racing_observation_matches_diffaero_gate_frame_layout():
    track = track_to_tensors(RACING_TRACK, _device(), torch.float32)
    target_gate = torch.zeros(1, dtype=torch.long)
    position = track.positions[:1] - torch.tensor([[0.0, 1.0, 0.0]])
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    velocity = torch.zeros(1, 3)
    policy, critic = racing_observations(position, quaternion, velocity, track, target_gate)
    assert policy.shape == (1, POLICY_OBSERVATION_SIZE)
    assert critic.shape == (1, CRITIC_OBSERVATION_SIZE)
    torch.testing.assert_close(policy[0, :3], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(critic[0, :7], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))


def test_gate_events_require_forward_crossing_and_l1_aperture():
    track = track_to_tensors(_track(_gate((0.0, 0.0, 1.5), 0.0), _gate((2.0, 0.0, 1.5), 0.0)), _device(), torch.float32)
    target_gate = torch.tensor([0, 0, 0, 1])
    previous = torch.tensor(((-1.0, 0.0, 1.5), (-1.0, 1.6, 1.5), (-1.0, 0.0, 1.5), (1.0, 0.0, 1.5)))
    current = torch.tensor(((1.0, 0.0, 1.5), (1.0, 1.6, 1.5), (-0.5, 0.0, 1.5), (3.0, 0.0, 1.5)))
    events = detect_race_events(previous, current, track, target_gate, torch.ones(4, dtype=torch.bool))
    assert events.passed.tolist() == [True, False, False, True]
    assert events.analytic_collision.tolist() == [False, True, False, False]
    assert events.next_gate_index.tolist() == [1, 0, 0, 0]
    assert not events.completed.any()


def test_backward_crossing_is_not_a_pass_or_collision():
    track = track_to_tensors(_track(_gate((0.0, 0.0, 1.5), 0.0)), _device(), torch.float32)
    events = detect_race_events(
        torch.tensor(((1.0, 0.0, 1.5),)),
        torch.tensor(((-1.0, 0.0, 1.5),)),
        track,
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, dtype=torch.bool),
    )
    assert not bool(events.passed[0])
    assert not bool(events.analytic_collision[0])
    assert int(events.next_gate_index[0]) == 0


def test_target_gate_wraps_after_last_gate():
    track = track_to_tensors(RACING_TRACK, _device(), torch.float32)
    last = track.order.shape[0] - 1
    gate_position = track.positions[last]
    gate_normal = track.rotations[last, :, 0]
    previous = (gate_position - gate_normal).unsqueeze(0)
    current = (gate_position + gate_normal).unsqueeze(0)
    events = detect_race_events(
        previous, current, track, torch.tensor([last]), torch.ones(1, dtype=torch.bool)
    )
    assert bool(events.passed[0])
    assert int(events.next_gate_index[0]) == 0


def test_reset_distribution_uses_random_gate_one_meter_behind():
    track = track_to_tensors(RACING_TRACK, _device(), torch.float32)
    first = generate_initial_states(track, count=64, seed=7)
    second = generate_initial_states(track, count=64, seed=7)
    assert torch.equal(first.target_gate, second.target_gate)
    torch.testing.assert_close(first.position, second.position)
    gate_position = track.positions[first.target_gate]
    gate_normal = track.rotations[first.target_gate, :, 0]
    torch.testing.assert_close(first.position, gate_position - gate_normal)
    torch.testing.assert_close(first.quaternion, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(64, 1))
    torch.testing.assert_close(first.linear_velocity, torch.zeros(64, 3))
    assert first.target_gate.unique().numel() > 1


def test_racing_loss_and_reward_use_diffaero_quad_weights():
    position = torch.tensor([[0.0, 0.0, 1.5]])
    previous = torch.tensor([[-0.1, 0.0, 1.5]])
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    velocity = torch.tensor([[1.0, 0.0, 0.0]])
    angular_velocity = torch.tensor([[0.0, 0.0, 0.2]])
    target = torch.tensor([[1.0, 0.0, 1.5]])
    loss, reward, components = racing_loss_reward(
        position,
        previous,
        quaternion,
        velocity,
        angular_velocity,
        target,
        torch.tensor([5.0]),
        torch.tensor([False]),
    )
    torch.testing.assert_close(components["progress_loss"], torch.tensor([-0.1]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(loss, components["vel_loss"] + 0.001 * components["jerk_loss"] + 3.0 * components["pos_loss"] + 0.1 * components["attitude_loss"])
    torch.testing.assert_close(reward, -0.1 * components["jerk_loss"] - 10.0 * components["progress_loss"])


def test_evaluation_initial_states_round_trip_with_checksum(tmp_path):
    track = track_to_tensors(RACING_TRACK, _device(), torch.float32)
    states = generate_initial_states(track, count=100, seed=20250830)
    path = tmp_path / "eval_states.pt"
    checksum = save_evaluation_initial_states(states, path)
    loaded, loaded_checksum = load_evaluation_initial_states(path)
    assert checksum == loaded_checksum == evaluation_states_checksum(states)
    torch.testing.assert_close(loaded.position, states.position)
    torch.testing.assert_close(loaded.quaternion, states.quaternion)
    torch.testing.assert_close(loaded.linear_velocity, states.linear_velocity)
    torch.testing.assert_close(loaded.target_gate, states.target_gate)
