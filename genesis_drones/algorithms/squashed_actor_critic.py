from __future__ import annotations

import torch
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.modules import ActorCritic


class SquashedActorCritic(ActorCritic):
    """DiffAero PPO action: latent Gaussian -> tanh. log_prob matches executed action."""

    log_std_min = -5.0
    log_std_max = 2.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if hasattr(self, "log_std"):
            with torch.no_grad():
                self.log_std.zero_()

    def _mapped_std(self) -> torch.Tensor:
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(self.log_std) + 1.0)
        return torch.exp(log_std).clamp_min(1e-6)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        mean = self.actor(obs).clamp(-5.0, 5.0)
        std = self._mapped_std().expand(mean.shape[0], -1)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        mean = self.actor(obs).clamp(-5.0, 5.0)
        std = self._mapped_std()
        self.distribution = Normal(mean, std.expand(mean.shape[0], -1))
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype)
        return torch.tanh(mean + std * noise)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return torch.tanh(self.actor(obs).clamp(-5.0, 5.0))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        latent = torch.atanh(actions)
        log_prob = self.distribution.log_prob(latent) - torch.log(1.0 - actions.square() + 1e-8)
        return torch.clamp(log_prob.sum(dim=-1), -20.0, 20.0)


def register_squashed_actor_critic() -> None:
    from rsl_rl.runners import on_policy_runner

    on_policy_runner.SquashedActorCritic = SquashedActorCritic
