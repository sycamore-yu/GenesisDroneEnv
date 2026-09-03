"""Unified evaluation policy: act(obs) only."""

from __future__ import annotations

from typing import Protocol

import torch

from genesis_drones.envs.genesis_task_env import detached_torch_tensor


class Policy(Protocol):
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor: ...


class DiffRLPolicyAdapter:
    def __init__(self, agent, normalizer):
        self.agent = agent
        self.normalizer = normalizer

    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.agent.action(observation, self.normalizer, deterministic=deterministic)


class RslRlPolicyAdapter:
    def __init__(self, policy):
        self.policy = policy
        self.policy.eval()

    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        from tensordict import TensorDict

        obs_t = detached_torch_tensor(observation)
        batch = int(obs_t.shape[0])
        obs = TensorDict({"policy": obs_t, "critic": obs_t}, batch_size=[batch])
        if deterministic and hasattr(self.policy, "act_inference"):
            return self.policy.act_inference(obs)
        return self.policy.act(obs)


class CallablePolicyAdapter:
    def __init__(self, fn):
        self.fn = fn

    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.fn(observation)
