import torch

from genesis_drones.algorithms.diff_rl import DeterministicActor, NetworkConfig, RunningNormalizer
from genesis_drones.envs.track_diff_env import TrackDiffEnvConfig, smooth_safety_penalty
from genesis_drones.evaluation.track_diff import TrackScenarios, evaluate_diff_policy


def test_diff_tracking_math_contracts():
    distance = torch.tensor([0.0, 0.2, 1.0], requires_grad=True)
    penalty = smooth_safety_penalty(distance, warning_distance=0.2, temperature_ratio=0.1)
    torch.testing.assert_close(penalty[0], torch.tensor(1.0))
    assert penalty[0] > penalty[1] > penalty[2]
    penalty.sum().backward()
    assert torch.isfinite(distance.grad).all()
    assert (distance.grad < 0.0).all()

    normalizer = RunningNormalizer(2)
    normalizer.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([True, True]))
    torch.testing.assert_close(normalizer.mean, torch.tensor([2.0, 3.0]), atol=2e-4, rtol=0.0)
    torch.testing.assert_close(normalizer.variance, torch.tensor([1.0, 1.0]), atol=5e-4, rtol=0.0)
    normalized = normalizer(torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(normalized, torch.zeros_like(normalized), atol=2e-4, rtol=0.0)

    actor = DeterministicActor(17, 4, NetworkConfig(), hover_action=2.0 / 3.3 - 1.0)
    action = actor(torch.zeros((1, 17)))
    torch.testing.assert_close(action[:, 0], torch.tensor([2.0 / 3.3 - 1.0]))
    torch.testing.assert_close(action[:, 1:], torch.zeros((1, 3)))


def test_fixed_scenarios_are_reproducible():
    config = TrackDiffEnvConfig()
    first = TrackScenarios.generate(4, 8, config, seed=123)
    second = TrackScenarios.generate(4, 8, config, seed=123)
    different = TrackScenarios.generate(4, 8, config, seed=124)
    torch.testing.assert_close(first.initial_position, second.initial_position)
    torch.testing.assert_close(first.initial_quaternion, second.initial_quaternion)
    torch.testing.assert_close(first.waypoint_sequences, second.waypoint_sequences)
    assert not torch.equal(first.waypoint_sequences, different.waypoint_sequences)


def test_evaluation_releases_scene_state_cache():
    class FakeScene:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    class FakeEnvironment:
        def __init__(self):
            self.config = type("Config", (), {"max_episode_steps": 3})()
            self.scene = FakeScene()
            self.metric = torch.tensor([2.0])

        def reset(self, *_args):
            return torch.zeros((1, 17))

        def step(self, _action):
            return torch.zeros((1, 17)), None, None, None

        def episode_metrics(self):
            return {"waypoint_count": self.metric}

    class FakeAgent:
        def action(self, observation, _normalizer, deterministic=True):
            assert deterministic
            return torch.zeros((observation.shape[0], 4))

    environment = FakeEnvironment()
    scenarios = type(
        "Scenarios",
        (),
        {"initial_position": None, "initial_quaternion": None, "waypoint_sequences": None},
    )()
    metrics = evaluate_diff_policy(environment, FakeAgent(), object(), scenarios)

    assert environment.scene.reset_calls == 1
    torch.testing.assert_close(metrics["waypoint_count"], torch.tensor([2.0]))
