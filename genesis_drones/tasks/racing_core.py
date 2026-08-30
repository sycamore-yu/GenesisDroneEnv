import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from torch.nn import functional as F

POLICY_OBSERVATION_SIZE = 40
CRITIC_OBSERVATION_SIZE = 41


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


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
    w = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]
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
    lookup = gate_index.clamp(max=track.order.shape[0] - 1)
    gate_id = track.order[lookup]
    return (
        track.positions[gate_id],
        track.rotations[gate_id],
        track.opening_sizes[gate_id],
        track.outer_sizes[gate_id],
        track.depths[gate_id],
    )


def gate_corners_world(track: RaceTrackTensors, gate_index: torch.Tensor) -> torch.Tensor:
    center, rotation, opening_size, _, _ = current_gate_tensors(track, gate_index)
    signs = center.new_tensor(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    local = center.new_zeros((center.shape[0], 4, 3))
    local[:, :, 1:] = signs[None] * opening_size[:, None] * 0.5
    return center[:, None] + torch.einsum("nij,nkj->nki", rotation, local)


def racing_observations(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    linear_velocity_world: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    last_action: torch.Tensor,
    track: RaceTrackTensors,
    gate_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    body_rotation = quaternion_to_rotation_matrix(quaternion)
    rotation_6d = body_rotation[:, :, :2].transpose(1, 2).reshape(position.shape[0], 6)
    body_velocity = torch.einsum("nji,nj->ni", body_rotation, linear_velocity_world)
    current_index = gate_index.clamp(max=track.order.shape[0] - 1)
    next_gate_index = (gate_index + 1).clamp(max=track.order.shape[0] - 1)
    current_corners = gate_corners_world(track, current_index)
    next_corners = gate_corners_world(track, next_gate_index)
    current_body = torch.einsum("nji,nkj->nki", body_rotation, current_corners - position[:, None]).reshape(-1, 12)
    next_body = torch.einsum("nji,nkj->nki", body_rotation, next_corners - position[:, None]).reshape(-1, 12)
    policy = torch.cat(
        (rotation_6d, body_velocity, angular_velocity_body, last_action, current_body, next_body), dim=-1
    )
    remaining = (track.order.shape[0] - gate_index).clamp(min=0).to(dtype=policy.dtype)[:, None]
    return policy, torch.cat((policy, remaining), dim=-1)


def _gate_frame_boxes(
    opening_size: torch.Tensor, outer_size: torch.Tensor, depth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    side_width = (outer_size[:, 0] - opening_size[:, 0]) * 0.5
    horizontal_height = (outer_size[:, 1] - opening_size[:, 1]) * 0.5
    side_center = (outer_size[:, 0] + opening_size[:, 0]) * 0.25
    horizontal_center = (outer_size[:, 1] + opening_size[:, 1]) * 0.25
    zeros = torch.zeros_like(depth)
    centers = torch.stack(
        (
            torch.stack((zeros, -side_center, zeros), dim=-1),
            torch.stack((zeros, side_center, zeros), dim=-1),
            torch.stack((zeros, zeros, -horizontal_center), dim=-1),
            torch.stack((zeros, zeros, horizontal_center), dim=-1),
        ),
        dim=1,
    )
    half_sizes = torch.stack(
        (
            torch.stack((depth * 0.5, side_width * 0.5, outer_size[:, 1] * 0.5), dim=-1),
            torch.stack((depth * 0.5, side_width * 0.5, outer_size[:, 1] * 0.5), dim=-1),
            torch.stack((depth * 0.5, outer_size[:, 0] * 0.5, horizontal_height * 0.5), dim=-1),
            torch.stack((depth * 0.5, outer_size[:, 0] * 0.5, horizontal_height * 0.5), dim=-1),
        ),
        dim=1,
    )
    return centers, half_sizes


def point_to_box_signed_distance(
    points: torch.Tensor, centers: torch.Tensor, half_sizes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    relative = points[:, None] - centers
    offset = relative.abs() - half_sizes
    distance = torch.linalg.vector_norm(torch.relu(offset), dim=-1) + torch.minimum(
        offset.amax(dim=-1), torch.zeros_like(offset[..., 0])
    )
    clamped = torch.maximum(torch.minimum(relative, half_sizes), -half_sizes)
    inside = (offset <= 0.0).all(dim=-1)
    margin = half_sizes - relative.abs()
    face_axis = F.one_hot(margin.argmin(dim=-1), num_classes=3).to(dtype=points.dtype)
    face_sign = torch.where(relative >= 0.0, torch.ones_like(relative), -torch.ones_like(relative))
    inside_surface = relative * (1.0 - face_axis) + face_sign * half_sizes * face_axis
    nearest = centers + torch.where(inside[:, :, None], inside_surface, clamped)
    return distance, nearest


def gate_frame_distance(
    position: torch.Tensor,
    track: RaceTrackTensors,
    gate_index: torch.Tensor,
    smooth_min_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    center, rotation, opening_size, outer_size, depth = current_gate_tensors(track, gate_index)
    local_position = torch.einsum("nji,nj->ni", rotation, position - center)
    box_centers, box_half_sizes = _gate_frame_boxes(opening_size, outer_size, depth)
    box_distance, nearest_local = point_to_box_signed_distance(local_position, box_centers, box_half_sizes)
    temperature = position.new_tensor(smooth_min_temperature)
    weights = torch.softmax(-box_distance / temperature, dim=-1)
    signed_distance = -temperature * torch.logsumexp(-box_distance / temperature, dim=-1)
    nearest_local = torch.sum(weights[:, :, None] * nearest_local, dim=1)
    nearest_world = center + torch.einsum("nij,nj->ni", rotation, nearest_local)
    return signed_distance, nearest_world


def gate_collision_loss(
    position: torch.Tensor,
    linear_velocity_world: torch.Tensor,
    track: RaceTrackTensors,
    gate_index: torch.Tensor,
    safety_radius: float,
    smooth_min_temperature: float,
    beta_1: float,
    beta_2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    signed_distance, nearest = gate_frame_distance(position, track, gate_index, smooth_min_temperature)
    clearance = signed_distance - safety_radius
    clearance_loss = beta_1 * F.softplus(-beta_2 * clearance)
    nearest_vector = nearest - position
    direction = nearest_vector / torch.linalg.vector_norm(nearest_vector, dim=-1, keepdim=True).clamp_min(1e-6)
    closing_speed = torch.relu(torch.sum(linear_velocity_world * direction, dim=-1)).detach()
    collision_loss = closing_speed * torch.square(torch.relu(1.0 - clearance))
    return clearance_loss + collision_loss, clearance, closing_speed


def detect_race_events(
    previous_position: torch.Tensor,
    position: torch.Tensor,
    track: RaceTrackTensors,
    gate_index: torch.Tensor,
    safety_radius: float,
    is_active: torch.Tensor,
) -> RaceEvents:
    center, rotation, opening_size, outer_size, _ = current_gate_tensors(track, gate_index)
    previous_local = torch.einsum("nji,nj->ni", rotation, previous_position - center)
    current_local = torch.einsum("nji,nj->ni", rotation, position - center)
    denominator = current_local[:, 0] - previous_local[:, 0]
    crossing_fraction = (-previous_local[:, 0] / denominator.clamp(min=1e-8)).clamp(0.0, 1.0)
    crossing_local = previous_local + crossing_fraction[:, None] * (current_local - previous_local)
    forward_crossing = (previous_local[:, 0] <= 0.0) & (current_local[:, 0] > 0.0)
    backward_crossing = (previous_local[:, 0] >= 0.0) & (current_local[:, 0] < 0.0)
    safe_half_size = opening_size * 0.5 - safety_radius
    inside_safe_opening = (crossing_local[:, 1:].abs() <= safe_half_size).all(dim=-1)
    inside_outer_frame = (crossing_local[:, 1:].abs() <= outer_size * 0.5 + safety_radius).all(dim=-1)
    is_racing = is_active & (gate_index < track.order.shape[0])
    passed = is_racing & forward_crossing & inside_safe_opening
    wrong_way = is_racing & backward_crossing & inside_safe_opening
    _, clearance, _ = gate_collision_loss(
        position,
        position.new_zeros(position.shape),
        track,
        gate_index,
        safety_radius,
        smooth_min_temperature=0.01,
        beta_1=1.0,
        beta_2=1.0,
    )
    crossed_frame = (forward_crossing | backward_crossing) & inside_outer_frame & ~inside_safe_opening
    analytic_collision = is_racing & ((clearance <= 0.0) | crossed_frame)
    last_gate = gate_index == track.order.shape[0] - 1
    completed = passed & last_gate
    next_gate_index = torch.where(passed, gate_index + 1, gate_index)
    crossing_point = center + torch.einsum("nij,nj->ni", rotation, crossing_local)
    return RaceEvents(passed, wrong_way, analytic_collision, completed, next_gate_index, crossing_point)


def generate_initial_states(
    track: RaceTrackTensors,
    count: int,
    seed: int,
    position_noise: float = 0.1,
    yaw_noise_degrees: float = 5.0,
    velocity_noise: float = 0.1,
) -> EvaluationInitialStates:
    generator = torch.Generator(device=track.positions.device).manual_seed(seed)
    gate_id = track.order[0]
    gate_position = track.positions[gate_id]
    gate_normal = track.rotations[gate_id, :, 0]
    position = gate_position[None].repeat(count, 1) - 2.0 * gate_normal[None]
    position += (2.0 * torch.rand((count, 3), generator=generator, device=position.device) - 1.0) * position_noise
    base_yaw = torch.atan2(gate_normal[1], gate_normal[0])
    yaw_noise = torch.deg2rad(position.new_tensor(yaw_noise_degrees))
    yaw = base_yaw + (2.0 * torch.rand(count, generator=generator, device=position.device) - 1.0) * yaw_noise
    quaternion = torch.stack(
        (torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw)), dim=-1
    )
    linear_velocity = (
        2.0 * torch.rand((count, 3), generator=generator, device=position.device) - 1.0
    ) * velocity_noise
    return EvaluationInitialStates(position, quaternion, linear_velocity)


def gate_progress_loss(
    position: torch.Tensor,
    linear_velocity_world: torch.Tensor,
    track: RaceTrackTensors,
    gate_index: torch.Tensor,
) -> torch.Tensor:
    center, _, _, _, _ = current_gate_tensors(track, gate_index)
    to_gate = center - position
    direction = to_gate / torch.linalg.vector_norm(to_gate, dim=-1, keepdim=True).clamp_min(1e-6)
    return -torch.sum(linear_velocity_world * direction, dim=-1)


def _finite_wire_field(point: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    start_vector = start - point
    end_vector = end - point
    start_norm = torch.linalg.vector_norm(start_vector, dim=-1, keepdim=True).clamp_min(1e-6)
    end_norm = torch.linalg.vector_norm(end_vector, dim=-1, keepdim=True).clamp_min(1e-6)
    cross = torch.linalg.cross(start_vector, end_vector, dim=-1)
    cosine = (start_vector * end_vector).sum(dim=-1, keepdim=True)
    denominator = (start_norm * end_norm + cosine).clamp_min(1e-6)
    return cross * (1.0 / start_norm + 1.0 / end_norm) / denominator


def gate_vector_field_loss(
    position: torch.Tensor,
    linear_velocity_world: torch.Tensor,
    track: RaceTrackTensors,
    gate_index: torch.Tensor,
) -> torch.Tensor:
    center, rotation, opening_size, _, _ = current_gate_tensors(track, gate_index)
    half_width = opening_size[:, 0] * 0.5
    half_height = opening_size[:, 1] * 0.5
    local_corners = position.new_zeros((position.shape[0], 4, 3))
    local_corners[:, 0, 1] = -half_width
    local_corners[:, 0, 2] = -half_height
    local_corners[:, 1, 1] = half_width
    local_corners[:, 1, 2] = -half_height
    local_corners[:, 2, 1] = half_width
    local_corners[:, 2, 2] = half_height
    local_corners[:, 3, 1] = -half_width
    local_corners[:, 3, 2] = half_height
    local_position = torch.einsum("nji,nj->ni", rotation, position - center)
    field_local = (
        _finite_wire_field(local_position, local_corners[:, 0], local_corners[:, 1])
        + _finite_wire_field(local_position, local_corners[:, 1], local_corners[:, 2])
        + _finite_wire_field(local_position, local_corners[:, 2], local_corners[:, 3])
        + _finite_wire_field(local_position, local_corners[:, 3], local_corners[:, 0])
    )
    field_world = torch.einsum("nij,nj->ni", rotation, field_local)
    direction = field_world / torch.linalg.vector_norm(field_world, dim=-1, keepdim=True).clamp_min(1e-6)
    return -torch.sum(position * direction.detach(), dim=-1)


def evaluation_states_checksum(states: EvaluationInitialStates) -> str:
    payload = torch.cat(
        (
            states.position.detach().cpu().reshape(-1),
            states.quaternion.detach().cpu().reshape(-1),
            states.linear_velocity.detach().cpu().reshape(-1),
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
            "checksum": checksum,
        },
        path,
    )
    return checksum


def load_evaluation_initial_states(path: Path) -> tuple[EvaluationInitialStates, str]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    states = EvaluationInitialStates(data["position"], data["quaternion"], data["linear_velocity"])
    checksum = evaluation_states_checksum(states)
    if data["checksum"] != checksum:
        raise ValueError("evaluation initial-state checksum does not match")
    return states, checksum

