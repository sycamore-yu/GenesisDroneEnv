"""Generic rollout loop. No task or algorithm logic."""

from __future__ import annotations

from typing import Any, Callable

import torch

from genesis_drones.envs.contracts import EnvStep
from genesis_drones.envs.genesis_task_env import GenesisTaskEnv


class EvaluationRunner:
    def __init__(
        self,
        environment: GenesisTaskEnv,
        policy,
        *,
        max_steps: int | None = None,
        consume: Callable[[EnvStep, dict[str, Any]], None] | None = None,
    ):
        self.environment = environment
        self.policy = policy
        self.max_steps = max_steps or int(getattr(environment.task.config, "max_episode_steps", 1000))
        self.consume = consume

    def run(
        self,
        *,
        reset_kwargs: dict[str, Any] | None = None,
        steps: int | None = None,
    ) -> dict[str, Any]:
        reset_kwargs = reset_kwargs or {}
        policy_obs, _ = self.environment.reset(**reset_kwargs)
        context: dict[str, Any] = {"steps": 0, "last_step": None}
        limit = steps if steps is not None else self.max_steps
        with torch.inference_mode():
            for _ in range(limit):
                action = self.policy.act(policy_obs, deterministic=True)
                step = self.environment.step(action)
                context["steps"] += 1
                context["last_step"] = step
                if self.consume is not None:
                    self.consume(step, context)
                policy_obs = step.observation.policy
                if bool(step.done.all()):
                    break
        return context
