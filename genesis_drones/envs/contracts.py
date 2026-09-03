from dataclasses import dataclass
from typing import Any, NamedTuple

import torch


@dataclass(frozen=True)
class ObservationBundle:
    policy: torch.Tensor
    critic: torch.Tensor
    state: torch.Tensor | None = None


class EnvStep(NamedTuple):
    observation: ObservationBundle
    reward: torch.Tensor
    physics_loss: torch.Tensor
    policy_loss: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    done: torch.Tensor
    events: Any
    metrics: dict
    extras: dict
