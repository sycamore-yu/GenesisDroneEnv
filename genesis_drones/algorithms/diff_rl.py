# Portions adapted from DiffAero under the BSD 3-Clause License.
# Copyright (c) 2025, State Key Lab of Autonomous Intelligent Unmanned Systems, Beijing Institute of Technology
# Copyright (c) 2025, Zhongguancun Academy

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

import genesis as gs

from genesis_drones.envs.differentiable import (
    DiffEnvSpec,
    DiffObservation,
    DifferentiableEnvironment,
)


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
    entropy_weight: float = 0.0
    actor_max_grad_norm: float = 1.0
    critic_max_grad_norm: float = 1.0
    critic_minibatches: int = 8
    critic_iterations: int = 1
    target_update_rate: float = 0.005
    log_standard_deviation_min: float = -5.0
    log_standard_deviation_max: float = -1.0
    use_terminal_value: bool = True


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


def as_simulation_action(action_actor: torch.Tensor) -> torch.Tensor:
    return action_actor.detach().requires_grad_(True)


def collect_simulation_action_gradients(sim_actions: list[torch.Tensor]) -> torch.Tensor:
    gradients = []
    for action in sim_actions:
        if action.grad is None:
            gradients.append(torch.zeros_like(action))
        else:
            gradients.append(action.grad.detach().clone())
    return torch.stack(gradients)


def apply_actor_action_gradients(
    actor_actions: list[torch.Tensor],
    action_gradients: torch.Tensor,
    policy_loss: torch.Tensor,
) -> None:
    outputs = [torch.stack(actor_actions)]
    gradients = [action_gradients]
    if policy_loss.requires_grad:
        outputs.append(policy_loss)
        gradients.append(torch.ones_like(policy_loss))
    torch.autograd.backward(outputs, gradients)


def actor_action_delta_loss(
    action_actor: torch.Tensor,
    previous_action: torch.Tensor,
    is_alive: torch.Tensor,
    weight: float,
    discount: torch.Tensor | None = None,
) -> torch.Tensor:
    if weight == 0.0:
        return action_actor.new_zeros(())
    per_environment = weight * torch.sum(torch.square(action_actor - previous_action), dim=-1) * is_alive
    if discount is not None:
        per_environment = per_environment * discount
    return per_environment.sum()


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
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: NetworkConfig,
        hover_action: float | tuple[float, ...],
        hover_index: int = 3,
    ):
        super().__init__()
        self.network = MultilayerPerceptron(observation_size, action_size, config)
        with torch.no_grad():
            if isinstance(hover_action, tuple):
                initial_action = self.network.output.bias.new_tensor(hover_action)
                self.network.output.bias.copy_(torch.atanh(initial_action))
            else:
                self.network.output.bias[hover_index] = torch.atanh(torch.tensor(hover_action))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(observation))


class StochasticActor(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: NetworkConfig,
        hover_action: float | tuple[float, ...],
        log_standard_deviation_min: float,
        log_standard_deviation_max: float,
        hover_index: int = 3,
    ):
        super().__init__()
        self.mean_network = MultilayerPerceptron(observation_size, action_size, config)
        with torch.no_grad():
            if isinstance(hover_action, tuple):
                initial_action = self.mean_network.output.bias.new_tensor(hover_action)
                self.mean_network.output.bias.copy_(torch.atanh(initial_action))
            else:
                self.mean_network.output.bias[hover_index] = torch.atanh(torch.tensor(hover_action))
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
        spec: DiffEnvSpec,
        network_config: NetworkConfig,
        config: ApgConfig,
        device: torch.device,
    ):
        self.actor = DeterministicActor(
            spec.policy_observation_dim,
            spec.action_dim,
            network_config,
            spec.nominal_action,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate, betas=(0.7, 0.95))
        self.config = config

    def action(
        self, observation: torch.Tensor, normalizer: RunningNormalizer, deterministic: bool = True
    ) -> torch.Tensor:
        return self.actor(normalizer(observation))

    def update(
        self,
        environment: DifferentiableEnvironment,
        observation: DiffObservation,
        normalizer: RunningNormalizer,
        _critic_normalizer: RunningNormalizer | None = None,
    ) -> tuple[DiffObservation, UpdateStats]:
        steps = self.config.horizon
        if steps != environment.spec.horizon:
            raise ValueError("algorithm horizon must match differentiable environment horizon")
        tracked = environment.is_alive.clone()
        physics_loss = observation.policy.new_zeros(())
        policy_loss = observation.policy.new_zeros(())
        reward_sum = observation.policy.new_zeros(())
        actor_actions = []
        sim_actions = []
        previous_action = environment.last_action.detach()
        for _ in range(steps):
            is_alive_before = environment.is_alive.clone()
            normalizer.update(observation.policy, is_alive_before)
            action_actor = self.actor(normalizer(observation.policy.detach()))
            action_sim = as_simulation_action(action_actor)
            transition = environment.step_diff(action_sim)
            observation = transition.observation
            physics_loss = physics_loss + transition.physics_loss.sum()
            policy_loss = policy_loss + actor_action_delta_loss(
                action_actor,
                previous_action,
                is_alive_before,
                environment.spec.action_delta_weight,
            )
            previous_action = action_actor
            reward_sum = reward_sum + transition.reward.sum()
            actor_actions.append(action_actor)
            sim_actions.append(action_sim)

        denominator = tracked.sum().clamp_min(1) * steps
        physics_loss = physics_loss / denominator
        policy_loss = policy_loss / denominator
        actor_loss = (physics_loss + policy_loss).detach()
        self.optimizer.zero_grad(set_to_none=True)
        observation, action_gradients = environment.finish_window(physics_loss, sim_actions)
        physics_loss = physics_loss.detach()
        apply_actor_action_gradients(actor_actions, action_gradients, policy_loss)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return observation, UpdateStats(
            actor_loss=actor_loss.item(),
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
        spec: DiffEnvSpec,
        network_config: NetworkConfig,
        config: ShacConfig,
        device: torch.device,
    ):
        self.actor = StochasticActor(
            spec.policy_observation_dim,
            spec.action_dim,
            network_config,
            spec.nominal_action,
            config.log_standard_deviation_min,
            config.log_standard_deviation_max,
        ).to(device)
        self.critic = Critic(spec.critic_observation_dim, network_config).to(device)
        self.target_critic = deepcopy(self.critic)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate, betas=(0.7, 0.95)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate, betas=(0.7, 0.95)
        )
        self.config = config

    def action(
        self, observation: torch.Tensor, normalizer: RunningNormalizer, deterministic: bool = True
    ) -> torch.Tensor:
        normalized = normalizer(observation)
        return self.actor.deterministic(normalized) if deterministic else self.actor.sample(normalized)[0]

    def update(
        self,
        environment: DifferentiableEnvironment,
        observation: DiffObservation,
        normalizer: RunningNormalizer,
        critic_normalizer: RunningNormalizer | None = None,
    ) -> tuple[DiffObservation, UpdateStats]:
        steps = self.config.horizon
        if steps != environment.spec.horizon:
            raise ValueError("algorithm horizon must match differentiable environment horizon")
        critic_normalizer = normalizer if critic_normalizer is None else critic_normalizer
        tracked = environment.is_alive.clone()
        physics_actor_loss = observation.policy.new_zeros(())
        policy_actor_loss = observation.policy.new_zeros(())
        entropy_sum = observation.policy.new_zeros(())
        reward_sum = observation.policy.new_zeros(())
        actor_actions = []
        sim_actions = []
        previous_action = environment.last_action.detach()
        rollout = ShacRollout([], [], [], [], [], [], [])
        discount = observation.policy.new_ones(observation.policy.shape[0])

        for _ in range(steps):
            is_alive_before = environment.is_alive.clone()
            normalizer.update(observation.policy, is_alive_before)
            if critic_normalizer is not normalizer:
                critic_normalizer.update(observation.critic, is_alive_before)
            normalized_observation = normalizer(observation.policy.detach())
            normalized_critic_observation = critic_normalizer(observation.critic.detach())
            if self.config.entropy_weight == 0.0:
                action_actor = self.actor.deterministic(normalized_observation)
                entropy = observation.policy.new_zeros(observation.policy.shape[0])
            else:
                action_actor, entropy = self.actor.sample(normalized_observation)
            action_sim = as_simulation_action(action_actor)
            with torch.no_grad():
                value = self.critic(normalized_critic_observation)
            transition = environment.step_diff(action_sim)
            with torch.no_grad():
                bootstrap_critic = critic_normalizer(transition.bootstrap_critic).detach()
                next_value = self.target_critic(bootstrap_critic).clamp(-20.0, 20.0)

            physics_actor_loss = physics_actor_loss + (transition.physics_loss * discount).sum()
            policy_actor_loss = policy_actor_loss + actor_action_delta_loss(
                action_actor,
                previous_action,
                is_alive_before,
                environment.spec.action_delta_weight,
                discount=discount,
            )
            previous_action = action_actor
            entropy_sum = entropy_sum + (entropy * is_alive_before).sum()
            reward_sum = reward_sum + transition.reward.sum()
            rollout.observations.append(normalized_critic_observation)
            rollout.losses.append(transition.critic_cost)
            rollout.values.append(value)
            rollout.next_values.append(next_value)
            rollout.dones.append(transition.done)
            rollout.terminated.append(transition.terminated)
            rollout.valid.append(is_alive_before)
            actor_actions.append(action_actor)
            sim_actions.append(action_sim)
            observation = transition.observation
            discount = discount * self.config.gamma

        # Short-horizon SHAC: L_π = Σ γ^t L_t + γ^H V(s_H). Freeze critic weights so the
        # actor optimizer cannot touch them, but keep ∂V/∂s_H through the observation.
        if self.config.use_terminal_value:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            terminal_observation = critic_normalizer(observation.critic)
            terminal_critic = self.target_critic if environment.spec.terminal_value_uses_target_critic else self.critic
            terminal_value = terminal_critic(terminal_observation)
            terminal_value = terminal_value * environment.is_alive.to(dtype=terminal_value.dtype)
            physics_actor_loss = physics_actor_loss + (terminal_value * discount).sum()

        denominator = tracked.sum().clamp_min(1) * steps
        physics_actor_loss = physics_actor_loss / denominator
        entropy = entropy_sum / denominator
        policy_actor_loss = policy_actor_loss / denominator - self.config.entropy_weight * entropy
        actor_loss = (physics_actor_loss + policy_actor_loss).detach()
        self.actor_optimizer.zero_grad(set_to_none=True)
        observation, action_gradients = environment.finish_window(physics_actor_loss, sim_actions)
        physics_actor_loss = physics_actor_loss.detach()
        apply_actor_action_gradients(actor_actions, action_gradients, policy_actor_loss)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.actor_max_grad_norm)
        self.actor_optimizer.step()
        if self.config.use_terminal_value:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)

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

        return observation, UpdateStats(
            actor_loss=actor_loss.item(),
            actor_grad_norm=actor_grad_norm.item(),
            mean_reward=(reward_sum / denominator).detach().item(),
            critic_loss=0.0 if critic_updates == 0 else critic_loss_sum / critic_updates,
            critic_grad_norm=0.0 if critic_updates == 0 else critic_grad_norm_sum / critic_updates,
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


DIFF_ALGORITHMS = {
    "apg": (ApgAgent, ApgConfig),
    "shac": (ShacAgent, ShacConfig),
}


def diff_algorithm_names() -> tuple[str, ...]:
    return tuple(DIFF_ALGORITHMS)


def build_diff_algorithm_config(algorithm: str, data: dict) -> ApgConfig | ShacConfig:
    if algorithm not in DIFF_ALGORITHMS:
        raise ValueError(f"unknown differentiable algorithm {algorithm!r}; valid: {', '.join(DIFF_ALGORITHMS)}")
    return DIFF_ALGORITHMS[algorithm][1](**data)


def make_diff_agent(
    algorithm: str,
    spec: DiffEnvSpec,
    network_config: NetworkConfig,
    config: ApgConfig | ShacConfig,
    device: torch.device,
) -> ApgAgent | ShacAgent:
    if algorithm not in DIFF_ALGORITHMS:
        raise ValueError(f"unknown differentiable algorithm {algorithm!r}; valid: {', '.join(DIFF_ALGORITHMS)}")
    agent_type, config_type = DIFF_ALGORITHMS[algorithm]
    if not isinstance(config, config_type):
        raise TypeError(f"{algorithm} requires {config_type.__name__}")
    return agent_type(spec, network_config, config, device)
