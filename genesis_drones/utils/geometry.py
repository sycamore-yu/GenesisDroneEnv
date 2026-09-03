import torch


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


def quaternion_to_roll_pitch_yaw(quaternion: torch.Tensor) -> torch.Tensor:
    w = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]
    sine_pitch = w * y - x * z
    sine_roll_cosine_pitch = w * x + y * z
    sine_yaw_cosine_pitch = w * z + x * y
    cosine_roll_cosine_pitch = 0.5 * (w * w - x * x - y * y + z * z)
    cosine_yaw_cosine_pitch = 0.5 * (w * w + x * x - y * y - z * z)
    cosine_pitch = torch.sqrt(
        cosine_yaw_cosine_pitch * cosine_yaw_cosine_pitch + sine_yaw_cosine_pitch * sine_yaw_cosine_pitch
    )
    return torch.stack(
        (
            torch.atan2(sine_roll_cosine_pitch, cosine_roll_cosine_pitch),
            torch.atan2(sine_pitch, cosine_pitch),
            torch.atan2(sine_yaw_cosine_pitch, cosine_yaw_cosine_pitch),
        ),
        dim=-1,
    )
