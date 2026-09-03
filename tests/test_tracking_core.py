import torch

from genesis_drones.tasks.tracking_core import (
    OBSERVATION_SIZE,
    TrackingCore,
    TrackingCoreConfig,
    arrival_surrogate_bonus,
    attitude_error,
    closing_velocity,
    differentiable_tracking_reward,
    smooth_safety_penalty,
    tracking_progress,
    tracking_safety_penalty,
)

DEVICE = torch.device("cpu")


class _State:
    def __init__(self, position, quaternion, linear_velocity, angular_velocity_body):
        self.position = position
        self.quaternion = quaternion
        self.linear_velocity = linear_velocity
        self.angular_velocity_body = angular_velocity_body


def _t(*args, **kwargs):
    kwargs.setdefault("device", DEVICE)
    return torch.tensor(*args, **kwargs)


def _state(position=None, quaternion=None, linear_velocity=None, angular_velocity_body=None):
    seed = position if position is not None else quaternion
    n = 2 if seed is None else seed.shape[0]
    return _State(
        position if position is not None else _t([[0.0, 0.0, 0.6], [0.5, 0.0, 0.6]])[:n],
        quaternion if quaternion is not None else _t([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1).clone(),
        linear_velocity if linear_velocity is not None else torch.zeros(n, 3, device=DEVICE),
        angular_velocity_body if angular_velocity_body is not None else torch.zeros(n, 3, device=DEVICE),
    )


def test_observation_shape_is_23():
    core = TrackingCore(TrackingCoreConfig(), num_envs=2, device=DEVICE)
    core.reset_task()
    core.target_position = _t([[1.0, 0.0, 0.6], [0.0, 1.0, 0.8]])
    policy, critic = core.observe(_state())
    assert policy.shape == (2, OBSERVATION_SIZE)
    assert critic.shape == (2, OBSERVATION_SIZE)
    torch.testing.assert_close(policy, critic)


def test_helper_formulas_match_known_values():
    distance = _t([0.0, 0.2, 1.0])
    penalty = smooth_safety_penalty(distance, warning_distance=0.2, temperature_ratio=0.1)
    torch.testing.assert_close(penalty[0], _t(1.0))
    assert penalty[0] > penalty[1] > penalty[2]

    last_error = _t([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    error = _t([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    torch.testing.assert_close(tracking_progress(last_error, error, "l1"), _t([1.0, 1.0]))
    torch.testing.assert_close(
        tracking_progress(last_error, error, "l2"),
        _t([1.0, (2.0**0.5) - 1.0]),
    )
    torch.testing.assert_close(
        closing_velocity(_t([[2.0, 0.0, 0.0]]), _t([[1.0, 0.0, 0.0]])),
        _t([2.0]),
    )
    gauss = arrival_surrogate_bonus(_t([0.0, 0.1, 1.0]), "gaussian", 0.15)
    assert gauss[0] > gauss[1] > gauss[2]
    hover = _t([[1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(attitude_error(hover, hover), _t([0.0]))


def test_discrete_and_differentiable_rewards():
    config = TrackingCoreConfig(dt=0.01, fully_differentiable=False)
    core = TrackingCore(config, num_envs=1, device=DEVICE)
    core.reset_task()
    core.target_position = _t([[0.0, 0.0, 0.6]])
    before = _state(
        position=_t([[0.2, 0.0, 0.6]]),
        linear_velocity=_t([[0.1, 0.0, 0.0]]),
    )
    after = _state(
        position=_t([[0.15, 0.0, 0.6]]),
        linear_velocity=_t([[0.1, 0.0, 0.0]]),
    )
    action = torch.zeros(1, 4, device=DEVICE)
    is_alive = torch.ones(1, dtype=torch.bool, device=DEVICE)
    result = core.evaluate(before, after, action, is_alive)
    assert torch.isfinite(result.reward).all()
    torch.testing.assert_close(result.physics_loss, -result.reward)

    position_error = core.target_position - after.position
    last_position_error = core.target_position - before.position
    expected_target = -torch.sum(torch.square(position_error), dim=-1) * 0.1
    expected_target = expected_target + tracking_progress(last_position_error, position_error, "l1")
    torch.testing.assert_close(result.loss_components["target"], expected_target)

    diff_config = TrackingCoreConfig(dt=0.01, fully_differentiable=True)
    diff_core = TrackingCore(diff_config, num_envs=1, device=DEVICE)
    diff_core.reset_task()
    diff_core.target_position = _t([[0.0, 0.0, 0.6]])
    hover = _t([[1.0, 0.0, 0.0, 0.0]])
    position_error = diff_core.target_position - after.position
    safety = tracking_safety_penalty(after.position, position_error, diff_config)
    expected = (
        differentiable_tracking_reward(
            after.position,
            diff_core.target_position,
            after.linear_velocity,
            after.quaternion,
            hover,
            after.angular_velocity_body,
            action,
            diff_core.last_action,
            safety,
            diff_config,
        )
        * diff_config.dt
    )
    result_diff = diff_core.evaluate(before, after, action, is_alive)
    torch.testing.assert_close(result_diff.reward, expected)


def test_crash_and_arrival_thresholds():
    config = TrackingCoreConfig(
        target_threshold=0.1,
        ground_termination_height=0.1,
        horizontal_termination_error=5.0,
        vertical_termination_error=1.2,
        roll_termination=180.0,
        pitch_termination=180.0,
    )
    core = TrackingCore(config, num_envs=3, device=DEVICE)
    core.reset_task()
    core.target_position = _t([[0.0, 0.0, 0.6], [0.0, 0.0, 0.6], [0.0, 0.0, 0.6]])

    after = _state(
        position=_t([[0.05, 0.0, 0.6], [0.0, 0.0, 0.05], [6.0, 0.0, 0.6]]),
    )
    distance = torch.linalg.vector_norm(core.target_position - after.position, dim=-1)
    arrived = core.arrival_mask(distance)
    assert arrived.tolist() == [True, False, False]

    crashed = core.crash_mask(
        after.position,
        after.quaternion,
        after.linear_velocity,
        after.angular_velocity_body,
        core.target_position - after.position,
    )
    assert crashed.tolist() == [False, True, True]

    before = _state(position=_t([[0.2, 0.0, 0.6], [0.0, 0.0, 0.6], [0.0, 0.0, 0.6]]))
    result = core.evaluate(before, after, torch.zeros(3, 4, device=DEVICE), torch.ones(3, dtype=torch.bool, device=DEVICE))
    assert result.events.arrived.tolist() == [True, False, False]
    assert result.terminated.tolist() == [False, True, True]
