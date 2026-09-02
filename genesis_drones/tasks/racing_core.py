import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from torch.nn import functional as F


POLICY_OBSERVATION_SIZE = 13
CRITIC_OBSERVATION_SIZE = 34
GATE_APERTURE_L1 = 1.5


@dataclass(frozen=True)
class GateSpec:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    opening_width: float
    opening_height: float
    outer_width: float
    outer_height: float
    depth: float


@dataclass(frozen=True)
class RaceTrackSpec:
    name: str
    gates: tuple[GateSpec, ...]
    order: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.gates:
            raise ValueError("A race track needs at least one gate")
        if sorted(self.order) != list(range(len(self.gates))):
            raise ValueError("Gate order must contain each gate exactly once")


class RaceTrackTensors(NamedTuple):
    positions: torch.Tensor
    rotations: torch.Tensor
    opening_sizes: torch.Tensor
    outer_sizes: torch.Tensor
    depths: torch.Tensor
    order: torch.Tensor


class RaceEvents(NamedTuple):
    passed: torch.Tensor
    wrong_way: torch.Tensor
    analytic_collision: torch.Tensor
    completed: torch.Tensor
    next_gate_index: torch.Tensor
    crossing_point: torch.Tensor


class EvaluationInitialStates(NamedTuple):
    position: torch.Tensor
    quaternion: torch.Tensor
    linear_velocity: torch.Tensor
    target_gate: torch.Tensor


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
    # gs.Tensor.unbind returns None (Genesis __torch_function__ shortcut).
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def track_to_tensors(track: RaceTrackSpec, device: torch.device, dtype: torch.dtype) -> RaceTrackTensors:
    positions = torch.tensor([gate.position for gate in track.gates], device=device, dtype=dtype)
    quaternions = torch.tensor([gate.quaternion for gate in track.gates], device=device, dtype=dtype)
    opening_sizes = torch.tensor(
        [(gate.opening_width, gate.opening_height) for gate in track.gates], device=device, dtype=dtype
    )
    outer_sizes = torch.tensor(
        [(gate.outer_width, gate.outer_height) for gate in track.gates], device=device, dtype=dtype
    )
    depths = torch.tensor([gate.depth for gate in track.gates], device=device, dtype=dtype)
    return RaceTrackTensors(
        positions,
        quaternion_to_rotation_matrix(quaternions),
        opening_sizes,
        outer_sizes,
        depths,
        torch.tensor(track.order, device=device, dtype=torch.long),
    )


def current_gate_tensors(
    track: RaceTrackTensors, gate_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_id = track.order[gate_index % track.order.shape[0]]
    return (
        track.positions[gate_id],
        track.rotations[gate_id],
        track.opening_sizes[gate_id],
        track.outer_sizes[gate_id],
        track.depths[gate_id],
    )


def _matrix_to_roll_pitch_yaw(rotation: torch.Tensor) -> torch.Tensor:
    roll = torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2])
    pitch = torch.atan2(-rotation[:, 2, 0], torch.sqrt(rotation[:, 0, 0].square() + rotation[:, 1, 0].square()))
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    return torch.stack((roll, pitch, yaw), dim=-1)


def _gate_frame_state(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    linear_velocity_world: torch.Tensor,
    track: RaceTrackTensors,
    gate_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_position, gate_rotation, _, _, _ = current_gate_tensors(track, gate_index)
    rotation_world_to_gate = gate_rotation.transpose(-1, -2)
    position_gate = torch.einsum("nij,nj->ni", rotation_world_to_gate, gate_position - position)
    velocity_gate = torch.einsum("nij,nj->ni", rotation_world_to_gate, linear_velocity_world)
    body_to_gate = rotation_world_to_gate @ quaternion_to_rotation_matrix(quaternion)
    return position_gate, velocity_gate, _matrix_to_roll_pitch_yaw(body_to_gate)


def racing_observations(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    linear_velocity_world: torch.Tensor,
    track: RaceTrackTensors,
    gate_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    position_gate, velocity_gate, rpy_gate = _gate_frame_state(
        position, quaternion, linear_velocity_world, track, gate_index
    )
    next_gate = (gate_index + 1) % track.order.shape[0]
    current_position, current_rotation, _, _, _ = current_gate_tensors(track, gate_index)
    next_position, next_rotation, _, _, _ = current_gate_tensors(track, next_gate)
    next_relative_position = torch.einsum(
        "nij,nj->ni", current_rotation.transpose(-1, -2), next_position - current_position
    )
    current_yaw = torch.atan2(current_rotation[:, 1, 0], current_rotation[:, 0, 0])
    next_yaw = torch.atan2(next_rotation[:, 1, 0], next_rotation[:, 0, 0])
    next_relative_yaw = torch.atan2(torch.sin(next_yaw - current_yaw), torch.cos(next_yaw - current_yaw))
    policy = torch.cat(
        (position_gate, velocity_gate, rpy_gate, next_relative_position, next_relative_yaw[:, None]), dim=-1
    )

    state = [linear_velocity_world, torch.roll(quaternion, shifts=-1, dims=-1)]
    for offset in range(3):
        state.extend(_gate_frame_state(position, quaternion, linear_velocity_world, track, gate_index + offset))
    return policy, torch.cat(state, dim=-1)


def detect_race_events(
    previous_position: torch.Tensor,
    position: torch.Tensor,
    track: RaceTrackTensors,
    gate_index: torch.Tensor,
    is_active: torch.Tensor,
) -> RaceEvents:
    gate_position, gate_rotation, _, _, _ = current_gate_tensors(track, gate_index)
    rotation_world_to_gate = gate_rotation.transpose(-1, -2)
    previous_local = torch.einsum("nij,nj->ni", rotation_world_to_gate, previous_position - gate_position)
    current_local = torch.einsum("nij,nj->ni", rotation_world_to_gate, position - gate_position)
    forward_crossing = (previous_local[:, 0] < 0.0) & (current_local[:, 0] > 0.0)
    backward_crossing = (previous_local[:, 0] > 0.0) & (current_local[:, 0] < 0.0)
    inside_gate = torch.linalg.vector_norm(current_local[:, 1:], ord=1, dim=-1) < GATE_APERTURE_L1
    passed = is_active & forward_crossing & inside_gate
    wrong_way = is_active & backward_crossing & inside_gate
    collision = is_active & forward_crossing & ~inside_gate
    next_gate_index = torch.where(passed, (gate_index + 1) % track.order.shape[0], gate_index)
    return RaceEvents(
        passed,
        wrong_way,
        collision,
        torch.zeros_like(passed),
        next_gate_index,
        position,
    )


def generate_initial_states(
    track: RaceTrackTensors,
    count: int,
    seed: int,
) -> EvaluationInitialStates:
    generator = torch.Generator(device=track.positions.device).manual_seed(seed)
    target_gate = torch.randint(0, track.order.shape[0], (count,), generator=generator, device=track.positions.device)
    gate_position, gate_rotation, _, _, _ = current_gate_tensors(track, target_gate)
    position = gate_position - gate_rotation[:, :, 0]
    quaternion = position.new_zeros((count, 4))
    quaternion[:, 0] = 1.0
    return EvaluationInitialStates(position, quaternion, torch.zeros_like(position), target_gate)


def racing_loss_reward(
    position: torch.Tensor,
    previous_position: torch.Tensor,
    quaternion: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    target_position: torch.Tensor,
    max_velocity: torch.Tensor,
    gate_collision: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    distance = torch.linalg.vector_norm(target_position - position, dim=-1)
    target_velocity = (target_position - position) / torch.maximum(
        distance / max_velocity, torch.ones_like(distance)
    )[:, None]
    velocity_difference = torch.linalg.vector_norm(linear_velocity - target_velocity, dim=-1)
    velocity_loss = F.smooth_l1_loss(velocity_difference, torch.zeros_like(velocity_difference), reduction="none")
    jerk_loss = torch.linalg.vector_norm(angular_velocity, dim=-1)
    rpy = _matrix_to_roll_pitch_yaw(quaternion_to_rotation_matrix(quaternion))
    attitude_loss = rpy[:, 0].square() + rpy[:, 1].square()
    position_loss = 1.0 - torch.exp(-distance)
    previous_distance = torch.linalg.vector_norm(previous_position - target_position, dim=-1)
    progress_loss = distance - previous_distance.detach()
    collision_loss = gate_collision.to(dtype=position.dtype)
    total_loss = velocity_loss + 0.001 * jerk_loss + 3.0 * position_loss + 0.1 * attitude_loss
    total_reward = (-0.1 * jerk_loss - 10.0 * progress_loss - 10.0 * collision_loss).detach()
    components = {
        "vel_loss": velocity_loss,
        "jerk_loss": jerk_loss,
        "attitude_loss": attitude_loss,
        "pos_loss": position_loss,
        "progress_loss": progress_loss,
        "collision_loss": collision_loss,
    }
    return total_loss, total_reward, components


def evaluation_states_checksum(states: EvaluationInitialStates) -> str:
    payload = torch.cat(
        (
            states.position.detach().cpu().reshape(-1),
            states.quaternion.detach().cpu().reshape(-1),
            states.linear_velocity.detach().cpu().reshape(-1),
            states.target_gate.detach().cpu().to(dtype=states.position.dtype).reshape(-1),
        )
    ).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def save_evaluation_initial_states(states: EvaluationInitialStates, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum = evaluation_states_checksum(states)
    torch.save(
        {
            "position": states.position.detach().cpu(),
            "quaternion": states.quaternion.detach().cpu(),
            "linear_velocity": states.linear_velocity.detach().cpu(),
            "target_gate": states.target_gate.detach().cpu(),
            "checksum": checksum,
        },
        path,
    )
    return checksum


def load_evaluation_initial_states(path: Path) -> tuple[EvaluationInitialStates, str]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    states = EvaluationInitialStates(
        data["position"], data["quaternion"], data["linear_velocity"], data["target_gate"]
    )
    checksum = evaluation_states_checksum(states)
    if data["checksum"] != checksum:
        raise ValueError("evaluation initial-state checksum does not match")
    return states, checksum
