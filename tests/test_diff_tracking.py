import torch

import genesis as gs

from genesis_drones.algorithms.diff_rl import (
    ApgAgent,
    ApgConfig,
    DeterministicActor,
    NetworkConfig,
    RunningNormalizer,
    apply_actor_action_gradients,
    as_simulation_action,
    collect_simulation_action_gradients,
)
from genesis_drones.envs.track_diff_env import TrackDiffEnv, TrackDiffEnvConfig, smooth_safety_penalty
from genesis_drones.evaluation.track_diff import TrackScenarios, evaluate_diff_policy


def _ensure_genesis():
    if not gs._initialized:
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    return gs.device


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


def test_simulation_action_is_a_detached_leaf():
    observation = torch.zeros((2, 17), requires_grad=True)
    actor = DeterministicActor(17, 4, NetworkConfig(), hover_action=2.0 / 3.3 - 1.0)
    action_actor = actor(observation)
    action_sim = as_simulation_action(action_actor)
    assert action_actor.grad_fn is not None
    assert action_sim.is_leaf
    assert action_sim.requires_grad
    assert action_sim.grad_fn is None
    torch.testing.assert_close(action_sim, action_actor.detach())


def test_apg_bridges_action_gradients_across_windows():
    device = _ensure_genesis()
    agent = ApgAgent(17, 4, 3.3, NetworkConfig(), ApgConfig(horizon=2), device)
    environment = TrackDiffEnv(TrackDiffEnvConfig(horizon=2), 2, requires_grad=True)
    observation = environment.reset()
    actor_actions = []
    sim_actions = []
    physics_loss = observation.new_zeros(())
    for _ in range(2):
        actor_observation = observation.detach()
        assert actor_observation.grad_fn is None
        action_actor = agent.actor(actor_observation)
        action_sim = as_simulation_action(action_actor)
        observation, (step_physics_loss, _, _), _, _ = environment.step(action_sim)
        physics_loss = physics_loss + step_physics_loss.sum()
        actor_actions.append(action_actor)
        sim_actions.append(action_sim)

    for parameter in agent.actor.parameters():
        assert parameter.grad is None
    scaled_physics_loss = physics_loss / 4.0
    environment.scene.backward(scaled_physics_loss)
    for action_sim in sim_actions:
        assert action_sim.grad is not None
        assert torch.isfinite(action_sim.grad).all()
        assert action_sim.grad.abs().sum() > 0
    for parameter in agent.actor.parameters():
        assert parameter.grad is None
    action_gradients = collect_simulation_action_gradients(sim_actions)
    for action_sim in sim_actions:
        action_sim.grad = None
    environment.release_simulation_graphs(scaled_physics_loss)
    apply_actor_action_gradients(
        actor_actions,
        action_gradients,
        physics_loss.new_zeros(()),
    )
    actor_grad_norm = torch.zeros((), device=device)
    for parameter in agent.actor.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        actor_grad_norm = actor_grad_norm + parameter.grad.detach().square().sum()
    assert actor_grad_norm > 0

    observation = environment.reset()
    normalizer = RunningNormalizer(17).to(device)
    observation, stats = agent.update(environment, observation, normalizer)
    assert environment.episode_step == 2
    assert torch.isfinite(torch.tensor(stats.actor_loss))
    assert torch.isfinite(torch.tensor(stats.actor_grad_norm))
    observation, stats = agent.update(environment, observation, normalizer)
    assert environment.episode_step == 4
    assert torch.isfinite(observation).all()
    assert environment.is_alive.any()


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
