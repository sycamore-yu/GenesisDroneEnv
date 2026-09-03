import torch

from genesis_drones.envs.differentiable import DiffEnvSpec, DiffObservation, DiffTransition
from genesis_drones.envs.genesis_task_env import GenesisTaskEnv


class DiffRLAdapter:
    """Differentiable training surface over GenesisTaskEnv. Owns DiffEnvSpec / windows."""

    def __init__(
        self,
        environment: GenesisTaskEnv,
        horizon: int,
        action_delta_weight: float = 0.0,
        critic_cost_mode: str = "neg_reward",
    ):
        self.environment = environment
        self.critic_cost_mode = critic_cost_mode
        self.spec = DiffEnvSpec(
            policy_observation_dim=environment.task.policy_observation_dim,
            critic_observation_dim=environment.task.critic_observation_dim,
            action_dim=environment.action_dim,
            horizon=horizon,
            nominal_action=environment.plant.nominal_action,
            action_delta_weight=action_delta_weight,
        )
        self.num_envs = environment.num_envs
        self.device = environment.device

    @property
    def is_alive(self) -> torch.Tensor:
        return self.environment.is_alive

    @property
    def last_action(self) -> torch.Tensor:
        return self.environment.last_action

    @property
    def config(self):
        return self.environment.task.config

    def reset_diff(self, seed: int | None = None) -> DiffObservation:
        policy, _ = self.environment.reset(seed=0 if seed is None else seed)
        return DiffObservation(policy=policy, critic=policy)

    def step_diff(self, action: torch.Tensor) -> DiffTransition:
        env_step = self.environment.step(action)
        if self.critic_cost_mode == "physics_plus_policy":
            critic_cost = (env_step.physics_loss + env_step.policy_loss).detach()
        else:
            critic_cost = (-env_step.reward).detach()
        return DiffTransition(
            observation=DiffObservation(
                policy=env_step.observation.policy,
                critic=env_step.extras["critic_observation_live"],
            ),
            bootstrap_critic=env_step.extras["critic_observation"],
            physics_loss=env_step.physics_loss,
            policy_loss=env_step.policy_loss,
            critic_cost=critic_cost,
            reward=env_step.reward,
            done=env_step.done,
            terminated=env_step.terminated,
        )

    def finish_window(
        self,
        physics_loss: torch.Tensor,
        simulation_actions: list[torch.Tensor],
    ) -> tuple[DiffObservation, torch.Tensor]:
        policy, critic, action_gradients = self.environment.finish_window(physics_loss, simulation_actions)
        return DiffObservation(policy=policy, critic=critic), action_gradients
