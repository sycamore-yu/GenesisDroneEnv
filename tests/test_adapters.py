import torch

from genesis_drones.adapters import DiffRLAdapter, RslRlAdapter
from genesis_drones.envs.contracts import EnvStep, ObservationBundle


def test_adapter_exports():
    assert DiffRLAdapter is not None
    assert RslRlAdapter is not None


def test_env_step_fields():
    zeros = torch.zeros(2)
    step = EnvStep(
        observation=ObservationBundle(policy=torch.zeros(2, 13), critic=torch.zeros(2, 13)),
        reward=zeros,
        physics_loss=zeros,
        policy_loss=zeros,
        terminated=zeros.bool(),
        truncated=zeros.bool(),
        done=zeros.bool(),
        events=None,
        metrics={},
        extras={},
    )
    assert step.observation.policy.shape[-1] == 13
