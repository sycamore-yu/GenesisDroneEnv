import torch
from tensordict import TensorDict
from torch.distributions import Normal

from genesis_drones.algorithms.squashed_actor_critic import SquashedActorCritic


def _policy():
    obs = TensorDict({"policy": torch.zeros(8, 13)}, batch_size=8)
    return SquashedActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["policy"]},
        4,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        init_noise_std=0.2231301601,
        noise_std_type="log",
    )


def test_squashed_actions_stay_in_unit_box():
    policy = _policy()
    obs = TensorDict({"policy": torch.randn(8, 13)}, batch_size=8)
    sampled = policy.act(obs)
    inferred = policy.act_inference(obs)
    assert sampled.abs().max() <= 1.0 + 1e-6
    assert inferred.abs().max() <= 1.0 + 1e-6


def test_squashed_log_prob_matches_latent_plus_jacobian():
    policy = _policy()
    obs = TensorDict({"policy": torch.randn(8, 13)}, batch_size=8)
    policy.act(obs)
    actions = torch.tanh(policy.distribution.mean.detach())
    log_prob = policy.get_actions_log_prob(actions)
    latent = torch.atanh(actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    expected = policy.distribution.log_prob(latent) - torch.log(1.0 - actions.square() + 1e-8)
    torch.testing.assert_close(log_prob, expected.sum(dim=-1), atol=1e-5, rtol=0.0)
    assert torch.isfinite(log_prob).all()


def test_sample_then_ppo_update_log_prob_roundtrip():
    """Rollout stores tanh(z). PPO update recomputes log_prob from that stored action."""
    policy = _policy()
    obs = TensorDict({"policy": torch.randn(32, 13)}, batch_size=32)
    torch.manual_seed(0)
    actor_obs = policy.actor_obs_normalizer(policy.get_actor_obs(obs))
    mean = policy.actor(actor_obs).clamp(-5.0, 5.0)
    std = policy._mapped_std()
    noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype)
    latent = mean + std * noise
    action = torch.tanh(latent)
    policy.distribution = Normal(mean, std.expand_as(mean))
    log_prob_rollout = policy.get_actions_log_prob(action)

    policy._update_distribution(actor_obs)
    log_prob_update = policy.get_actions_log_prob(action)
    torch.testing.assert_close(log_prob_rollout, log_prob_update, atol=1e-5, rtol=0.0)
    assert torch.isfinite(log_prob_update).all()

    log_prob_from_z = (policy.distribution.log_prob(latent) - torch.log(1.0 - action.square() + 1e-8)).sum(dim=-1)
    unsaturated = latent.abs().max(dim=-1).values < 3.0
    if unsaturated.any():
        torch.testing.assert_close(
            log_prob_from_z[unsaturated],
            log_prob_update[unsaturated],
            atol=1e-4,
            rtol=0.0,
        )
