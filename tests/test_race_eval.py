from pathlib import Path

import pytest
import yaml

import torch

import genesis as gs

from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
from genesis_drones.evaluation.race import RACING_CONTRACT, assert_shared_racing_contract
from genesis_drones.tasks.race_task import RaceTask


def _ensure_genesis():
    if not gs._initialized:
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")


def test_shared_contract_matches_spec():
    config = RaceEnvConfig()
    assert_shared_racing_contract(config)
    assert RACING_CONTRACT["policy_observation_size"] == 13
    assert RACING_CONTRACT["critic_observation_size"] == 34
    assert RACING_CONTRACT["action"] == "ctbr"


def test_ppo_task_exposes_observation_groups_timeouts_and_discount():
    _ensure_genesis()
    environment = RaceEnv(RaceEnvConfig(horizon=1, max_episode_steps=3), 2, requires_grad=False)
    with (Path(__file__).resolve().parents[1] / "config" / "race" / "ppo.yaml").open() as file:
        train_config = yaml.safe_load(file)
    assert train_config["algorithm"]["gamma"] == 0.99
    assert train_config["algorithm"]["lam"] == 0.95
    task = RaceTask(environment, train_config)
    observation = task.get_observations()
    assert observation["policy"].shape == (2, 13)
    assert observation["critic"].shape == (2, 34)
    hover = torch.full((2, 4), environment.controller.hover_action, device=gs.device)
    hover[:, 1:] = 0.0
    _, _, done, extras = task.step(hover)
    assert "time_outs" in extras
    assert train_config["algorithm"]["gamma"] == pytest.approx(0.99, rel=0, abs=1e-5)
    assert train_config["algorithm"]["lam"] == pytest.approx(0.95, rel=0, abs=1e-5)
