from copy import deepcopy

import torch
from torch.nn import functional as F

from genesis_drones.algorithms.diff_rl import (
    ApgConfig,
    Critic,
    DeterministicActor,
    NetworkConfig,
    RunningNormalizer,
    ShacConfig,
    StochasticActor,
    UpdateStats,
    apply_actor_action_gradients,
    as_simulation_action,
    collect_simulation_action_gradients,
)
from genesis_drones.envs.race_env import RaceEnv


class RacingApgAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hover_action: float,
        network_config: NetworkConfig,
        config: ApgConfig,
        device: torch.device,
    ):
        self.actor = DeterministicActor(
            observation_size, action_size, network_config, hover_action, hover_index=0
        ).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate, betas=(0.7, 0.95))
        self.config = config

    def action(self, observation: torch.Tensor, normalizer: RunningNormalizer, deterministic: bool = True) -> torch.Tensor:
        return self.actor(normalizer(observation))

    def update(
        self, environment: RaceEnv, observation: torch.Tensor, normalizer: RunningNormalizer
    ) -> tuple[torch.Tensor, torch.Tensor, UpdateStats]:
        steps = self.config.horizon
        tracked = environment.is_alive.clone()
        physics_loss = observation.new_zeros(())
        reward_sum = observation.new_zeros(())
        actor_actions = []
        sim_actions = []
        for _ in range(steps):
            is_alive_before = environment.is_alive.clone()
            normalizer.update(observation, is_alive_before)
            action_actor = self.actor(normalizer(observation))
            action_sim = as_simulation_action(action_actor)
            observation, (step_physics_loss, _, reward), _, extras = environment.step(action_sim)
            physics_loss = physics_loss + step_physics_loss.sum()
            reward_sum = reward_sum + reward.sum()
            actor_actions.append(action_actor)
            sim_actions.append(action_sim)
        denominator = tracked.sum().clamp_min(1) * steps
        physics_loss = physics_loss / denominator
        actor_loss = physics_loss.detach()
        self.optimizer.zero_grad(set_to_none=True)
        environment.scene.backward(physics_loss)
        action_gradients = collect_simulation_action_gradients(sim_actions)
        for action in sim_actions:
            action.grad = None
        environment.release_simulation_graphs(physics_loss)
        apply_actor_action_gradients(actor_actions, action_gradients, observation.new_zeros(()))
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        observation, critic_observation = environment.detach_window()
        return observation, critic_observation, UpdateStats(
            actor_loss=actor_loss.item(),
            actor_grad_norm=actor_grad_norm.item(),
            mean_reward=(reward_sum / denominator).detach().item(),
            steps=steps,
            valid_transitions=(tracked.sum() * steps).item(),
        )

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.optimizer.load_state_dict(state["optimizer"])


class RacingShacAgent:
    def __init__(
        self,
        policy_observation_size: int,
        critic_observation_size: int,
        action_size: int,
        hover_action: float,
        network_config: NetworkConfig,
        config: ShacConfig,
        device: torch.device,
    ):
        self.actor = StochasticActor(
            policy_observation_size,
            action_size,
            network_config,
            hover_action,
            config.log_standard_deviation_min,
            config.log_standard_deviation_max,
            hover_index=0,
        ).to(device)
        self.critic = Critic(critic_observation_size, network_config).to(device)
        self.target_critic = deepcopy(self.critic)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate, betas=(0.7, 0.95)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate, betas=(0.7, 0.95)
        )
        self.config = config

    def action(self, observation: torch.Tensor, normalizer: RunningNormalizer, deterministic: bool = True) -> torch.Tensor:
        normalized = normalizer(observation)
        return self.actor.deterministic(normalized) if deterministic else self.actor.sample(normalized)[0]

    def update(
        self,
        environment: RaceEnv,
        observation: torch.Tensor,
        critic_observation: torch.Tensor,
        policy_normalizer: RunningNormalizer,
        critic_normalizer: RunningNormalizer,
    ) -> tuple[torch.Tensor, torch.Tensor, UpdateStats]:
        steps = self.config.horizon
        tracked = environment.is_alive.clone()
        physics_actor_loss = observation.new_zeros(())
        reward_sum = observation.new_zeros(())
        actor_actions = []
        sim_actions = []
        critic_observations = []
        next_critic_values = []
        costs = []
        dones = []
        terminated = []
        valid = []
        discount = observation.new_ones(observation.shape[0])
        for _ in range(steps):
            is_alive_before = environment.is_alive.clone()
            policy_normalizer.update(observation, is_alive_before)
            critic_normalizer.update(critic_observation, is_alive_before)
            action_actor = self.actor.deterministic(policy_normalizer(observation))
            action_sim = as_simulation_action(action_actor)
            next_observation, (physics_loss, _, reward), done, extras = environment.step(action_sim)
            next_critic = extras["critic_observation"]
            physics_actor_loss = physics_actor_loss + (physics_loss * discount).sum()
            reward_sum = reward_sum + reward.sum()
            critic_observations.append(critic_normalizer(critic_observation).detach())
            costs.append((-reward).detach())
            dones.append(done)
            terminated.append(extras["terminated"])
            valid.append(is_alive_before)
            with torch.no_grad():
                next_critic_values.append(
                    self.target_critic(critic_normalizer(next_critic)).clamp(-20.0, 20.0)
                )
            actor_actions.append(action_actor)
            sim_actions.append(action_sim)
            observation = next_observation
            critic_observation = next_critic
            discount = discount * self.config.gamma

        terminal_observation = critic_normalizer(critic_observation)
        terminal_cost = self.target_critic(terminal_observation)
        still_alive = environment.is_alive.to(dtype=terminal_cost.dtype)
        terminal_cost = terminal_cost * still_alive
        physics_actor_loss = physics_actor_loss + (terminal_cost * discount).sum()
        denominator = tracked.sum().clamp_min(1) * steps
        physics_actor_loss = physics_actor_loss / denominator
        actor_loss = physics_actor_loss.detach()
        self.actor_optimizer.zero_grad(set_to_none=True)
        environment.scene.backward(physics_actor_loss)
        action_gradients = collect_simulation_action_gradients(sim_actions)
        for action in sim_actions:
            action.grad = None
        environment.release_simulation_graphs(physics_actor_loss)
        apply_actor_action_gradients(actor_actions, action_gradients, observation.new_zeros(()))
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.actor_max_grad_norm)
        self.actor_optimizer.step()
        observation, live_critic = environment.detach_window()

        costs_t = torch.stack(costs)
        values = self.critic(torch.stack(critic_observations)).detach()
        stacked_terminated = torch.stack(terminated)
        stacked_done = torch.stack(dones)
        stacked_valid = torch.stack(valid)
        with torch.no_grad():
            next_values = torch.stack(next_critic_values)
            advantages = torch.zeros_like(costs_t)
            next_advantage = torch.zeros(environment.num_envs, device=environment.device, dtype=costs_t.dtype)
            for step in reversed(range(steps)):
                bootstrap_value = next_values[step] * ~stacked_terminated[step]
                delta = costs_t[step] + self.config.gamma * bootstrap_value - values[step]
                is_boundary = stacked_done[step] | (step == steps - 1)
                next_advantage = delta + self.config.gamma * self.config.td_lambda * ~is_boundary * next_advantage
                advantages[step] = next_advantage
            target_values = values + advantages

        observations = torch.stack(critic_observations)[stacked_valid]
        target_values = target_values[stacked_valid]
        critic_loss_sum = 0.0
        critic_grad_norm_sum = 0.0
        critic_updates = 0
        for _ in range(self.config.critic_iterations):
            permutation = torch.randperm(observations.shape[0], device=environment.device)
            for indices in torch.chunk(permutation, self.config.critic_minibatches):
                if indices.shape[0] == 0:
                    continue
                critic_loss = F.smooth_l1_loss(self.critic(observations[indices]), target_values[indices])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.critic_max_grad_norm
                )
                self.critic_optimizer.step()
                critic_loss_sum += critic_loss.detach().item()
                critic_grad_norm_sum += critic_grad_norm.item()
                critic_updates += 1
        with torch.no_grad():
            for parameter, target_parameter in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_parameter.lerp_(parameter, self.config.target_update_rate)
        return observation, live_critic, UpdateStats(
            actor_loss=actor_loss.item(),
            actor_grad_norm=actor_grad_norm.item(),
            mean_reward=(reward_sum / denominator).detach().item(),
            critic_loss=0.0 if critic_updates == 0 else critic_loss_sum / critic_updates,
            critic_grad_norm=0.0 if critic_updates == 0 else critic_grad_norm_sum / critic_updates,
            steps=steps,
            valid_transitions=stacked_valid.sum().item(),
        )

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
