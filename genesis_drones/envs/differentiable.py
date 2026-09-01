from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class DiffEnvSpec:
    policy_observation_dim: int
    critic_observation_dim: int
    action_dim: int
    horizon: int
    nominal_action: tuple[float, ...]
    action_delta_weight: float = 0.0

    def __post_init__(self) -> None:
        sizes = (self.policy_observation_dim, self.critic_observation_dim, self.action_dim, self.horizon)
        if min(sizes) <= 0:
            raise ValueError("differentiable environment dimensions and horizon must be positive")
        if len(self.nominal_action) != self.action_dim:
            raise ValueError("nominal action length must match action dimension")
        if any(value <= -1.0 or value >= 1.0 for value in self.nominal_action):
            raise ValueError("nominal action values must be strictly between -1 and 1")


@dataclass(frozen=True)
class DiffObservation:
    policy: torch.Tensor
    critic: torch.Tensor


@dataclass(frozen=True)
class DiffTransition:
    observation: DiffObservation
    bootstrap_critic: torch.Tensor
    physics_loss: torch.Tensor
    policy_loss: torch.Tensor
    critic_cost: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    terminated: torch.Tensor


class DifferentiableEnvironment(Protocol):
    spec: DiffEnvSpec
    num_envs: int
    device: torch.device
    is_alive: torch.Tensor
    last_action: torch.Tensor

    def reset_diff(self, seed: int | None = None) -> DiffObservation: ...

    def step_diff(self, action: torch.Tensor) -> DiffTransition: ...

    def finish_window(
        self,
        physics_loss: torch.Tensor,
        simulation_actions: list[torch.Tensor],
    ) -> tuple[DiffObservation, torch.Tensor]: ...


def finish_simulation_window(environment, physics_loss: torch.Tensor, actions: list[torch.Tensor]) -> torch.Tensor:
    environment.scene.backward(physics_loss)
    gradients = []
    for action in actions:
        gradients.append(torch.zeros_like(action) if action.grad is None else action.grad.detach().clone())
        action.grad = None
    environment.release_simulation_graphs(physics_loss)
    return torch.stack(gradients)
