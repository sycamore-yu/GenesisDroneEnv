import torch

import genesis as gs

from genesis_drones.algorithms.diff_rl import (
    ApgConfig,
    DeterministicActor,
    NetworkConfig,
    RunningNormalizer,
    apply_actor_action_gradients,
    as_simulation_action,
    collect_simulation_action_gradients,
    make_diff_agent,
)
from genesis_drones.envs.track_diff_env import TrackDiffEnv, TrackDiffEnvConfig, smooth_safety_penalty, tracking_progress, closing_velocity, arrival_surrogate_bonus
from genesis_drones.evaluation.track_diff import TrackScenarios, classify_eval_step, evaluate_diff_policy, make_diff_observation


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

    hover_action = 2.0 / 3.3 - 1.0
    actor = DeterministicActor(TrackDiffEnv.observation_dim, 4, NetworkConfig(), hover_action=hover_action)
    action = actor(torch.zeros((1, TrackDiffEnv.observation_dim)))
    torch.testing.assert_close(action[:, 3], torch.tensor([hover_action]))
    torch.testing.assert_close(action[:, :3], torch.zeros((1, 3)))

    last_error = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    error = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    torch.testing.assert_close(tracking_progress(last_error, error, "l1"), torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(tracking_progress(last_error, error, "l2"), torch.tensor([1.0, torch.sqrt(torch.tensor(2.0)) - 1.0]))
    error_toward = torch.tensor([[1.0, 0.0, 0.0]])
    velocity = torch.tensor([[2.0, 0.0, 0.0]])
    torch.testing.assert_close(closing_velocity(velocity, error_toward), torch.tensor([2.0]))
    gauss_distance = torch.tensor([0.0, 0.1, 1.0], requires_grad=True)
    gauss = arrival_surrogate_bonus(gauss_distance, "gaussian", 0.15)
    assert gauss[0] > gauss[1] > gauss[2]
    gauss.sum().backward()
    assert gauss_distance.grad is not None
    from genesis_drones.envs.track_diff_env import quaternion_to_roll_pitch_yaw

    identity_euler = quaternion_to_roll_pitch_yaw(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(identity_euler, torch.zeros(1, 3))


def test_differentiable_tracking_reward_has_nonzero_gradients():
    from genesis_drones.envs.track_diff_env import attitude_error, differentiable_tracking_reward

    config = TrackDiffEnvConfig()
    position = torch.tensor([[0.3, 0.0, 0.6]], requires_grad=True)
    quaternion = torch.tensor([[0.98, 0.1, 0.0, 0.0]], requires_grad=True)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    quaternion = quaternion.detach().requires_grad_(True)
    linear_velocity = torch.tensor([[0.4, 0.0, 0.0]], requires_grad=True)
    angular_velocity = torch.tensor([[0.0, 0.2, 0.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 0.6]])
    hover = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    action = torch.zeros((1, 4))
    reward = differentiable_tracking_reward(
        position, target, linear_velocity, quaternion, hover, angular_velocity, action, action, torch.zeros(1), config
    )
    reward.sum().backward()
    assert position.grad.norm() > 0
    assert quaternion.grad.norm() > 0
    assert linear_velocity.grad.norm() > 0
    assert angular_velocity.grad.norm() > 0
    torch.testing.assert_close(attitude_error(hover, hover), torch.tensor([0.0]))

    distance = torch.tensor([0.101, 0.099], requires_grad=True)
    arrival_bonus = 20.0 * (distance < 0.1).to(distance.dtype)
    assert arrival_bonus.tolist() == [0.0, 20.0]
    assert arrival_bonus.grad_fn is None


def test_diff_observation_applies_training_scales():
    from genesis_drones.envs.track_diff_env import TrackDiffEnvConfig

    config = TrackDiffEnvConfig()
    position = torch.tensor([[1.0, 2.0, 0.6]])
    target = torch.tensor([[4.0, 5.0, 0.9]])
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    linear_velocity = torch.tensor([[0.3, 0.0, 0.0]])
    angular_velocity = torch.tensor([[0.0, 0.0, 3.14159]])
    last_action = torch.zeros((1, 4))
    observation = make_diff_observation(
        position,
        target,
        quaternion,
        linear_velocity,
        angular_velocity,
        last_action,
        torch.tensor([True]),
        config,
    )
    assert observation.shape == (1, TrackDiffEnv.observation_dim)
    torch.testing.assert_close(
        observation[0, 6:9],
        (target - position)[0] * config.obs_scale_position_error,
    )
    torch.testing.assert_close(
        observation[0, 13:16],
        linear_velocity[0] * config.obs_scale_linear_velocity,
    )
    torch.testing.assert_close(
        observation[0, 16:19],
        angular_velocity[0] * config.obs_scale_angular_velocity,
    )
    dead = make_diff_observation(
        position,
        target,
        quaternion,
        linear_velocity,
        angular_velocity,
        last_action,
        torch.tensor([False]),
        config,
    )
    torch.testing.assert_close(dead, torch.zeros_like(dead))

    config = TrackDiffEnvConfig()
    first = TrackScenarios.generate(4, 8, config, seed=123)
    second = TrackScenarios.generate(4, 8, config, seed=123)
    different = TrackScenarios.generate(4, 8, config, seed=124)
    torch.testing.assert_close(first.initial_position, second.initial_position)
    torch.testing.assert_close(first.initial_quaternion, second.initial_quaternion)
    torch.testing.assert_close(first.waypoint_sequences, second.waypoint_sequences)
    assert not torch.equal(first.waypoint_sequences, different.waypoint_sequences)


def test_simulation_action_is_a_detached_leaf():
    observation = torch.zeros((2, TrackDiffEnv.observation_dim), requires_grad=True)
    actor = DeterministicActor(TrackDiffEnv.observation_dim, 4, NetworkConfig(), hover_action=2.0 / 3.3 - 1.0)
    action_actor = actor(observation)
    action_sim = as_simulation_action(action_actor)
    assert action_actor.grad_fn is not None
    assert action_sim.is_leaf
    assert action_sim.requires_grad
    assert action_sim.grad_fn is None
    torch.testing.assert_close(action_sim, action_actor.detach())


def test_apg_bridges_action_gradients_across_windows():
    device = _ensure_genesis()
    config = TrackDiffEnvConfig(horizon=2)
    agent = make_diff_agent(
        "apg",
        TrackDiffEnv.spec_from_config(config),
        NetworkConfig(),
        ApgConfig(horizon=2),
        device,
    )
    environment = TrackDiffEnv(config, 2, requires_grad=True)
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

    observation = environment.reset_diff()
    normalizer = RunningNormalizer(TrackDiffEnv.observation_dim).to(device)
    observation, stats = agent.update(environment, observation, normalizer)
    assert environment.episode_step == 2
    assert torch.isfinite(torch.tensor(stats.actor_loss))
    assert torch.isfinite(torch.tensor(stats.actor_grad_norm))
    observation, stats = agent.update(environment, observation, normalizer)
    assert environment.episode_step == 4
    assert torch.isfinite(observation.policy).all()
    assert environment.is_alive.any()


def test_eval_step_does_not_count_z_error_as_crash():
    is_alive = torch.tensor([True, True, True, True])
    position = torch.tensor(
        [
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.05],
            [6.0, 0.0, 0.6],
            [0.0, 0.0, 2.0],
        ]
    )
    target = torch.tensor(
        [
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.6],
        ]
    )
    has_nan = torch.tensor([False, False, False, False])
    arrived, crashed, _, hit_ground, left_workspace = classify_eval_step(
        is_alive, position, target, has_nan, target_threshold=0.1, ground_height=0.1, horizontal_limit=5.0
    )
    torch.testing.assert_close(arrived, torch.tensor([True, False, False, False]))
    torch.testing.assert_close(crashed, torch.tensor([False, True, True, False]))
    torch.testing.assert_close(hit_ground, torch.tensor([False, True, False, False]))
    torch.testing.assert_close(left_workspace, torch.tensor([False, False, True, False]))


def test_evaluation_releases_scene_state_cache():
    device = _ensure_genesis()

    class FakeState:
        def __init__(self):
            self.position = torch.zeros((1, 3), device=device)
            self.linear_velocity = torch.zeros((1, 3), device=device)

    class FakeScene:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    class FakeEnvironment:
        def __init__(self):
            self.config = type(
                "Config",
                (),
                {"max_episode_steps": 3, "dt": 0.01, "target_threshold": 0.1},
            )()
            self.scene = FakeScene()
            self.metric = torch.tensor([2.0], device=device)
            self.num_envs = 1
            self.device = device
            self.action_dim = 4
            self.target_position = torch.zeros((1, 3), device=device)
            self.is_alive = torch.ones(1, device=device, dtype=torch.bool)
            self.end_on_vertical_error = True
            self.respawn_on_fail = True

        def reset(self, *_args):
            self.is_alive = torch.ones(1, device=device, dtype=torch.bool)
            return torch.zeros((1, TrackDiffEnv.observation_dim), device=device)

        def _read_state(self):
            return FakeState()

        def step(self, _action):
            extras = {"arrived": torch.tensor([False], device=device)}
            return torch.zeros((1, TrackDiffEnv.observation_dim), device=device), None, None, extras

        def episode_metrics(self):
            return {
                "waypoint_count": self.metric,
                "first_arrived": torch.tensor([True], device=device),
                "first_arrival_time": torch.tensor([0.5], device=device),
                "crashed": torch.tensor([False], device=device),
                "survival_time": torch.tensor([0.03], device=device),
                "mean_position_error": torch.tensor([0.1], device=device),
            }

    class FakeAgent:
        def action(self, observation, _normalizer, deterministic=True):
            assert deterministic
            return torch.zeros((observation.shape[0], 4), device=observation.device)

    environment = FakeEnvironment()
    scenarios = type(
        "Scenarios",
        (),
        {"initial_position": None, "initial_quaternion": None, "waypoint_sequences": None},
    )()
    metrics = evaluate_diff_policy(environment, FakeAgent(), object(), scenarios)

    assert environment.scene.reset_calls == 1
    torch.testing.assert_close(metrics["waypoint_count"], torch.tensor([2.0], device=device))
    assert "mean_speed" in metrics
    assert "path_efficiency" in metrics
    assert "action_total_variation" in metrics


def test_track_diff_env_visualize_shows_plane_and_waypoint():
    _ensure_genesis()
    environment = TrackDiffEnv(TrackDiffEnvConfig(horizon=2), 1, requires_grad=False, visualize=True)
    try:
        morphs = [type(entity.morph).__name__ for entity in environment.scene.entities]
        assert "Plane" in morphs
        assert "Mesh" in morphs
        assert "Drone" in morphs
        environment.reset()
        torch.testing.assert_close(environment.target_visual.get_pos(), environment.target_position)
    finally:
        environment.scene.reset()


def test_pid_hover_and_ppo_style_arrival_reward():
    _ensure_genesis()
    environment = TrackDiffEnv(TrackDiffEnvConfig(horizon=1), 1, requires_grad=False)
    try:
        hover = 2.0 / environment.config.thrust_to_weight_ratio - 1.0
        assert environment.spec.nominal_action == (0.0, 0.0, 0.0, hover)
        observation = environment.reset(
            torch.tensor([[0.0, 0.0, 0.6]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[[0.0, 0.0, 0.6], [0.4, 0.0, 0.8]]]),
        )
        assert observation.shape[-1] == TrackDiffEnv.observation_dim
        hover_action = torch.tensor([[0.0, 0.0, 0.0, hover]], device=environment.device)
        motor_thrust, wrench = environment.mix_action(hover_action)
        hover_thrust = environment.config.max_collective_thrust / environment.config.thrust_to_weight_ratio
        torch.testing.assert_close(wrench[0, 0], wrench.new_tensor(hover_thrust), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(wrench[0, 1:], torch.zeros(3, device=wrench.device), atol=1e-3, rtol=0.0)
        environment.reset(
            torch.tensor([[0.0, 0.0, 0.6]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[[0.0, 0.0, 0.6], [0.4, 0.0, 0.8]]]),
        )
        _, (_, _, reward), _, extras = environment.step(hover_action)
        assert bool(extras["arrived"][0])
        arrival_bonus = 20.0 * environment.config.reward_scales.target * environment.config.dt
        assert float(reward[0]) > 0.5 * arrival_bonus
        roll_action = hover_action.clone()
        roll_action[0, 0] = 0.4
        environment.reset(
            torch.tensor([[0.0, 0.0, 0.6]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[[0.0, 0.0, 0.8]]]),
        )
        rolled_thrust, _ = environment.mix_action(roll_action)
        assert rolled_thrust[0].max() - rolled_thrust[0].min() > 1e-4
    finally:
        environment.scene.reset()


def test_fully_differentiable_mode_keeps_target_and_drops_arrival_bonus():
    _ensure_genesis()
    environment = TrackDiffEnv(
        TrackDiffEnvConfig(horizon=1, fully_differentiable=True), 1, requires_grad=False
    )
    try:
        hover = 2.0 / environment.config.thrust_to_weight_ratio - 1.0
        environment.reset(
            torch.tensor([[0.0, 0.0, 0.6]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[[0.0, 0.0, 0.6], [0.4, 0.0, 0.8]]]),
        )
        target_before = environment.target_position.clone()
        hover_action = torch.tensor([[0.0, 0.0, 0.0, hover]], device=environment.device)
        _, (_, _, reward), _, extras = environment.step(hover_action)
        assert bool(extras["arrived"][0])
        torch.testing.assert_close(environment.target_position, target_before)
        arrival_bonus = 20.0 * environment.config.reward_scales.target * environment.config.dt
        assert float(reward[0]) < 0.25 * arrival_bonus
        assert int(environment.waypoint_count[0]) == 1
        _, _, _, extras = environment.step(hover_action)
        assert int(environment.waypoint_count[0]) == 1
        torch.testing.assert_close(environment.target_position, target_before)
    finally:
        environment.scene.reset()
