# Portions adapted from DiffAero under the BSD 3-Clause License.
# Copyright (c) 2025, State Key Lab of Autonomous Intelligent Unmanned Systems, Beijing Institute of Technology
# Copyright (c) 2025, Zhongguancun Academy

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

import genesis as gs

from genesis_drones.envs.track_diff_env import TrackDiffEnv


@dataclass(frozen=True)
class NetworkConfig:
    hidden_sizes: tuple[int, int] = (256, 128)


@dataclass(frozen=True)
class ApgConfig:
    horizon: int = 32
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class ShacConfig:
    horizon: int = 32
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 3e-3
    gamma: float = 0.99
    td_lambda: float = 0.95
    entropy_weight: float = 0.01
    actor_max_grad_norm: float = 1.0
    critic_max_grad_norm: float = 1.0
    critic_minibatches: int = 8
    target_update_rate: float = 0.005
    log_standard_deviation_min: float = -5.0
    log_standard_deviation_max: float = 2.0


@dataclass(frozen=True)
class UpdateStats:
    actor_loss: float
    actor_grad_norm: float
    mean_reward: float = 0.0
    critic_loss: float = 0.0
    critic_grad_norm: float = 0.0
    entropy: float = 0.0
    steps: int = 0
    valid_transitions: int = 0


@dataclass
class ShacRollout:
    observations: list[torch.Tensor]
    losses: list[torch.Tensor]
    values: list[torch.Tensor]
    next_values: list[torch.Tensor]
    dones: list[torch.Tensor]
    terminated: list[torch.Tensor]
    valid: list[torch.Tensor]


def as_torch_tensor(values: torch.Tensor) -> torch.Tensor:
    return values.as_subclass(torch.Tensor) if isinstance(values, gs.Tensor) else values


class RunningNormalizer(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("variance", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        values = as_torch_tensor(values.detach())[mask]
        if values.shape[0] == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_variance = values.var(dim=0, unbiased=False)
        batch_count = values.new_tensor(values.shape[0])
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean.add_(delta * batch_count / total_count)
        combined_second_moment = (
            self.variance * self.count
            + batch_variance * batch_count
            + torch.square(delta) * self.count * batch_count / total_count
        )
        self.variance.copy_(combined_second_moment / total_count)
        self.count.copy_(total_count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = as_torch_tensor(values)
        return torch.clamp((values - self.mean) / torch.sqrt(self.variance + 1e-8), -10.0, 10.0)


class NormedLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.normalization = nn.LayerNorm(output_size)
        self.activation = nn.ELU(inplace=True)
        nn.init.zeros_(self.linear.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.normalization(self.linear(values)))


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: NetworkConfig):
        super().__init__()
        layers = []
        previous_size = input_size
        for hidden_size in config.hidden_sizes:
            layers.append(NormedLinear(previous_size, hidden_size))
            previous_size = hidden_size
        output = nn.Linear(previous_size, output_size)
        nn.init.uniform_(output.weight, -0.01, 0.01)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.layers = nn.Sequential(*layers)

    @property
    def output(self) -> nn.Linear:
        return self.layers[-1]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class DeterministicActor(nn.Module):
    def __init__(self, observation_size: int, action_size: int, config: NetworkConfig, hover_action: float):
        super().__init__()
        self.network = MultilayerPerceptron(observation_size, action_size, config)
        with torch.no_grad():
            self.network.output.bias[0] = torch.atanh(torch.tensor(hover_action))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(observation))


class StochasticActor(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: NetworkConfig,
        hover_action: float,
        log_standard_deviation_min: float,
        log_standard_deviation_max: float,
    ):
        super().__init__()
        self.mean_network = MultilayerPerceptron(observation_size, action_size, config)
        with torch.no_grad():
            self.mean_network.output.bias[0] = torch.atanh(torch.tensor(hover_action))
        self.log_standard_deviation = nn.Parameter(torch.zeros(action_size))
        self.log_standard_deviation_min = log_standard_deviation_min
        self.log_standard_deviation_max = log_standard_deviation_max

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_network(observation)
        log_standard_deviation = self.log_standard_deviation_min + 0.5 * (
            self.log_standard_deviation_max - self.log_standard_deviation_min
        ) * (torch.tanh(self.log_standard_deviation) + 1.0)
        return torch.distributions.Normal(mean, torch.exp(log_standard_deviation).expand_as(mean))

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        return torch.tanh(distribution.rsample()), distribution.entropy().sum(dim=-1)

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean_network(observation))


class Critic(nn.Module):
    def __init__(self, observation_size: int, config: NetworkConfig):
        super().__init__()
        self.network = MultilayerPerceptron(observation_size, 1, config)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


class ApgAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        thrust_to_weight_ratio: float,
        network_config: NetworkConfig,
        config: ApgConfig,
        device: torch.device,
    ):
        hover_action = 2.0 / thrust_to_weight_ratio - 1.0
        self.actor = DeterministicActor(observation_size, action_size, network_config, hover_action).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.config = config

    def action(
        self, observation: torch.Tensor, normalizer: RunningNormalizer, deterministic: bool = True
    ) -> torch.Tensor:
        return self.actor(normalizer(observation))

    def update(
        self,
        environment: TrackDiffEnv,
        observation: torch.Tensor,
        normalizer: RunningNormalizer,
    ) -> tuple[torch.Tensor, UpdateStats]:
        steps = self.config.horizon
        tracked = environment.is_alive.clone()
        actor_loss = observation.new_zeros(())
        reward_sum = observation.new_zeros(())
        for _ in range(steps):
            is_alive_before = environment.is_alive.clone()
            normalizer.update(observation, is_alive_before)
            action = self.actor(normalizer(observation))
            observation, (loss, reward), _, _ = environment.step(action)
            actor_loss = actor_loss + loss.sum()
            reward_sum = reward_sum + reward.sum()

        denominator = tracked.sum().clamp_min(1) * steps
        actor_loss = actor_loss / denominator
        self.optimizer.zero_grad()
        environment.scene.backward(actor_loss)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        observation = environment.detach_window()
        return observation, UpdateStats(
            actor_loss=actor_loss.detach().item(),
            actor_grad_norm=actor_grad_norm.item(),
            mean_reward=(reward_sum / denominator).detach().item(),
            steps=steps,
            valid_transitions=(tracked.sum() * steps).item(),
        )

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.optimizer.load_state_dict(state["optimizer"])


class ShacAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        thrust_to_weight_ratio: float,
        network_config: NetworkConfig,
        config: ShacConfig,
        device: torch.device,
    ):
        hover_action = 2.0 / thrust_to_weight_ratio - 1.0
        self.actor = StochasticActor(
            observation_size,
            action_size,
            network_config,
            hover_action,
            config.log_standard_deviation_min,
            config.log_standard_deviation_max,
        ).to(device)
        self.critic = Critic(observation_size, network_config).to(device)
        self.target_critic = deepcopy(self.critic)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_learning_rate)
        self.config = config

    def action(
        self, observation: torch.Tensor, normalizer: RunningNormalizer, deterministic: bool = True
    ) -> torch.Tensor:
        normalized = normalizer(observation)
        return self.actor.deterministic(normalized) if deterministic else self.actor.sample(normalized)[0]

    def update(
        self,
        environment: TrackDiffEnv,
        observation: torch.Tensor,
        normalizer: RunningNormalizer,
    ) -> tuple[torch.Tensor, UpdateStats]:
        steps = self.config.horizon
        tracked = environment.is_alive.clone()
        discount = tracked.to(dtype=observation.dtype)
        actor_loss = observation.new_zeros(())
        entropy_sum = observation.new_zeros(())
        reward_sum = observation.new_zeros(())
        rollout = ShacRollout([], [], [], [], [], [], [])

        for _ in range(steps):
            is_alive_before = environment.is_alive.clone()
            normalizer.update(observation, is_alive_before)
            normalized_observation = normalizer(observation)
            action, entropy = self.actor.sample(normalized_observation)
            with torch.no_grad():
                value = self.critic(normalized_observation.detach())
            next_observation, (loss, reward), done, extras = environment.step(action)
            normalized_next_observation = normalizer(next_observation)
            if environment.episode_step == environment.config.max_episode_steps:
                next_value_for_actor = self.target_critic(normalized_next_observation)
                actor_loss = actor_loss + (
                    discount * self.config.gamma * next_value_for_actor * extras["truncated"]
                ).sum()
                next_value = next_value_for_actor.detach()
            else:
                with torch.no_grad():
                    next_value = self.target_critic(normalized_next_observation.detach())

            actor_loss = actor_loss + (discount * loss).sum()
            entropy_sum = entropy_sum + (entropy * is_alive_before).sum()
            reward_sum = reward_sum + reward.sum()
            discount = discount * self.config.gamma * extras["alive"]
            rollout.observations.append(normalized_observation.detach())
            rollout.losses.append(loss.detach())
            rollout.values.append(value)
            rollout.next_values.append(next_value)
            rollout.dones.append(done)
            rollout.terminated.append(extras["terminated"])
            rollout.valid.append(is_alive_before)
            observation = next_observation

        actor_loss = actor_loss + (discount * self.target_critic(normalizer(observation))).sum()
        denominator = tracked.sum().clamp_min(1) * steps
        actor_loss = actor_loss / denominator
        entropy = entropy_sum / denominator
        total_actor_loss = actor_loss - self.config.entropy_weight * entropy
        self.actor_optimizer.zero_grad()
        environment.scene.backward(total_actor_loss)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.actor_max_grad_norm)
        self.actor_optimizer.step()
        observation = environment.detach_window()

        losses = torch.stack(rollout.losses)
        values = torch.stack(rollout.values)
        next_values = torch.stack(rollout.next_values)
        dones = torch.stack(rollout.dones)
        terminated = torch.stack(rollout.terminated)
        valid = torch.stack(rollout.valid)
        with torch.no_grad():
            advantages = torch.zeros_like(losses)
            next_advantage = torch.zeros(environment.num_envs, device=environment.device, dtype=losses.dtype)
            for step in reversed(range(steps)):
                is_boundary = dones[step] | (step == steps - 1)
                delta = losses[step] + self.config.gamma * next_values[step] * ~terminated[step] - values[step]
                next_advantage = delta + self.config.gamma * self.config.td_lambda * ~is_boundary * next_advantage
                advantages[step] = next_advantage
            target_values = values + advantages

        observations = torch.stack(rollout.observations)[valid]
        target_values = target_values[valid]
        permutation = torch.randperm(observations.shape[0], device=environment.device)
        critic_loss_sum = 0.0
        critic_grad_norm_sum = 0.0
        critic_updates = 0
        for indices in torch.chunk(permutation, self.config.critic_minibatches):
            if indices.shape[0] == 0:
                continue
            critic_loss = F.mse_loss(self.critic(observations[indices]), target_values[indices])
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

        return observation, UpdateStats(
            actor_loss=actor_loss.detach().item(),
            actor_grad_norm=actor_grad_norm.item(),
            mean_reward=(reward_sum / denominator).detach().item(),
            critic_loss=critic_loss_sum / critic_updates,
            critic_grad_norm=critic_grad_norm_sum / critic_updates,
            entropy=entropy.detach().item(),
            steps=steps,
            valid_transitions=valid.sum().item(),
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
