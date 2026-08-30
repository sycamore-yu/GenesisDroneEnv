import torch

from genesis_drones.algorithms.diff_rl import ApgConfig, NetworkConfig, RunningNormalizer, ShacConfig
from genesis_drones.algorithms.race_rl import RacingApgAgent, RacingShacAgent
from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig

import genesis as gs


def _ensure_genesis():
    if not gs._initialized:
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    return gs.device


def test_apg_keeps_gradient_on_next_state_observation():
    _ensure_genesis()
    environment = RaceEnv(RaceEnvConfig(horizon=2), num_envs=2, requires_grad=True)
    agent = RacingApgAgent(
        RaceEnv.policy_observation_dim,
        RaceEnv.action_dim,
        environment.controller.hover_action,
        NetworkConfig(),
        ApgConfig(horizon=2),
        gs.device,
    )
    observation, _ = environment.reset(seed=3)
    normalizer = RunningNormalizer(RaceEnv.policy_observation_dim).to(gs.device)
    action = agent.actor(normalizer(observation))
    action_sim = action.detach().requires_grad_(True)
    next_observation, (physics_loss, _, _), _, _ = environment.step(action_sim)
    assert next_observation.grad_fn is not None
    assert torch.isfinite(physics_loss.detach()).all()


def test_shac_target_critic_is_frozen_and_true_terminal_cost_is_zero():
    _ensure_genesis()
    environment = RaceEnv(RaceEnvConfig(horizon=2, max_episode_steps=4), num_envs=2, requires_grad=True)
    agent = RacingShacAgent(
        RaceEnv.policy_observation_dim,
        RaceEnv.critic_observation_dim,
        RaceEnv.action_dim,
        environment.controller.hover_action,
        NetworkConfig(),
        ShacConfig(horizon=2, gamma=0.999, td_lambda=0.95),
        gs.device,
    )
    for parameter in agent.target_critic.parameters():
        assert parameter.requires_grad is False
    observation, critic_observation = environment.reset(seed=4)
    critic_observation = critic_observation.detach().requires_grad_(True)
    cost = agent.target_critic(critic_observation)
    cost.sum().backward()
    assert critic_observation.grad is not None
    assert torch.isfinite(critic_observation.grad).all()
    terminated = torch.ones(2, device=gs.device)
    terminal_cost = cost.detach() * (1.0 - terminated)
    torch.testing.assert_close(terminal_cost, torch.zeros_like(terminal_cost))


def test_time_truncation_uses_critic_observation_from_before_reset():
    _ensure_genesis()
    environment = RaceEnv(RaceEnvConfig(horizon=1, max_episode_steps=1), num_envs=2, requires_grad=False)
    observation, critic_observation = environment.reset(seed=5)
    hover = torch.full((2, 4), environment.controller.hover_action, device=gs.device)
    hover[:, 1:] = 0.0
    next_observation, _, done, extras = environment.step(hover)
    assert extras["truncated"].any()
    assert extras["critic_observation"].shape[-1] == 41
    assert not torch.equal(extras["critic_observation"], extras["critic_observation_live"])
