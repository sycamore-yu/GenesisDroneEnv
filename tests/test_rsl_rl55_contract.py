"""RSL-RL 5.5 migration contract: shapes, obs groups, timeout bootstrap."""

from __future__ import annotations

import importlib.metadata
import inspect

import pytest
import torch
import yaml
from tensordict import TensorDict

_rsl_version = importlib.metadata.version("rsl-rl-lib")
if tuple(int(part) for part in _rsl_version.split(".")[:2]) < (5, 0):
    pytest.skip(f"requires rsl-rl-lib>=5, got {_rsl_version}", allow_module_level=True)

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_class

from genesis_drones.experiment.builder import PROJECT_ROOT, build_ppo_actor


NUM_ENVS = 8
OBS_DIM = 13
NUM_ACTIONS = 4


class DummyEnv(VecEnv):
    def __init__(self, device: str = "cpu") -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = 50
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}
        self._obs = torch.randn(NUM_ENVS, OBS_DIM, device=device)

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": self._obs, "critic": self._obs.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor):
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, device=self.device)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return self.get_observations(), rewards, dones, extras


def _racing_cfg() -> dict:
    with (PROJECT_ROOT / "config/race/ppo_full_quad.yaml").open() as file:
        return yaml.safe_load(file)


def test_yaml_is_rslrl55_native():
    cfg = _racing_cfg()
    assert "actor" in cfg and "critic" in cfg
    assert "policy" not in cfg
    assert cfg["obs_groups"]["actor"] == ["policy"]
    assert cfg["obs_groups"]["critic"] == ["policy"]
    assert cfg["actor"]["class_name"] == "MLPModel"
    assert cfg["actor"]["hidden_dims"] == [256, 128]
    assert cfg["actor"]["activation"] == "elu"
    assert cfg["actor"]["obs_normalization"] is False
    dist = cfg["actor"]["distribution_cfg"]
    assert dist["class_name"] == "GaussianDistribution"
    assert dist["std_type"] == "log"
    assert dist["init_std"] == 0.4567092003128319
    assert cfg["num_steps_per_env"] == 16
    alg = cfg["algorithm"]
    assert alg["gamma"] == 0.99
    assert alg["lam"] == 0.95
    assert alg["clip_param"] == 0.2
    assert alg["num_learning_epochs"] == 4
    assert alg["num_mini_batches"] == 8
    assert alg["entropy_coef"] == 0.00875174422525385
    assert alg["value_loss_coef"] == 2.0
    assert alg["schedule"] == "adaptive"
    assert alg["desired_kl"] == 0.015906451881642775
    assert alg["max_grad_norm"] == 1.0
    assert alg["learning_rate"] == 0.0002110721108616641


def test_actor_critic_shapes_and_distribution():
    cfg = _racing_cfg()
    obs = TensorDict(
        {"policy": torch.randn(NUM_ENVS, OBS_DIM), "critic": torch.randn(NUM_ENVS, OBS_DIM)},
        batch_size=[NUM_ENVS],
    )
    actor = build_ppo_actor(cfg, OBS_DIM, NUM_ACTIONS, "cpu")
    critic_class, critic_cfg = resolve_class(cfg["critic"])
    critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg)
    det = actor(obs, stochastic_output=False)
    sto = actor(obs, stochastic_output=True)
    value = critic(obs)
    log_prob = actor.get_output_log_prob(sto)
    std = actor.output_std
    assert det.shape == (NUM_ENVS, NUM_ACTIONS)
    assert sto.shape == (NUM_ENVS, NUM_ACTIONS)
    assert value.shape[-1] == 1
    assert value.shape[0] == NUM_ENVS
    assert log_prob.shape == (NUM_ENVS,)
    assert std.shape[-1] == NUM_ACTIONS
    assert torch.isfinite(det).all()
    assert torch.isfinite(sto).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(std).all()


def test_adapter_observation_contract_keys():
    env = DummyEnv()
    obs = env.get_observations()
    assert "policy" in obs.keys()
    assert obs["policy"].shape == (NUM_ENVS, OBS_DIM)
    _, rewards, dones, extras = env.step(torch.zeros(NUM_ENVS, NUM_ACTIONS))
    assert rewards.shape == (NUM_ENVS,)
    assert dones.shape == (NUM_ENVS,)
    assert "time_outs" in extras
    assert extras["time_outs"].shape == (NUM_ENVS,)


def test_timeout_bootstrap_still_uses_current_transition_value():
    source = inspect.getsource(PPO.process_env_step)
    assert "time_outs" in source
    assert "self.transition.values" in source
    assert "self.gamma" in source
    cfg = _racing_cfg()
    cfg["num_steps_per_env"] = 4
    cfg["save_interval"] = 100
    env = DummyEnv()
    runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
    obs = env.get_observations()
    runner.alg.act(obs)
    values = runner.alg.transition.values.detach().clone()
    rewards = torch.zeros(NUM_ENVS)
    dones = torch.zeros(NUM_ENVS)
    extras = {"time_outs": torch.ones(NUM_ENVS)}
    runner.alg.process_env_step(obs, rewards, dones, extras)
    expected = runner.alg.gamma * values.view(-1)
    stored = runner.alg.storage.rewards[0].view(-1)
    torch.testing.assert_close(stored, expected, atol=1e-5, rtol=0.0)


def test_stock_ppo_clips_actor_and_critic_separately():
    source = inspect.getsource(PPO.update)
    assert "clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)" in source
    assert "clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)" in source
