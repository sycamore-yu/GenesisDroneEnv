from pathlib import Path

import numpy as np

from genesis_drones.evaluation.split_s import (
    GATE4_INDEX,
    GATE5_INDEX,
    analytic_split_s_events,
    assert_ctbr_within_limits,
    plan_split_s_trajectory,
    split_s_waypoints,
)
from genesis_drones.tasks.racing_tracks import FIXED_SEVEN_GATE_TRACK


def test_split_s_waypoints_are_before_and_after_each_gate():
    waypoints = split_s_waypoints(FIXED_SEVEN_GATE_TRACK)
    g4 = np.array(FIXED_SEVEN_GATE_TRACK.gates[GATE4_INDEX].position)
    g5 = np.array(FIXED_SEVEN_GATE_TRACK.gates[GATE5_INDEX].position)
    np.testing.assert_allclose(waypoints[1], g4)
    np.testing.assert_allclose(waypoints[4], g5)
    assert np.linalg.norm(waypoints[0] - g4) > 1.5
    assert np.linalg.norm(waypoints[2] - g4) > 1.5
    assert np.linalg.norm(waypoints[3] - g5) > 1.5
    assert np.linalg.norm(waypoints[5] - g5) > 1.5
    assert waypoints[2][2] > waypoints[3][2]


def test_planned_trajectory_passes_gate4_then_gate5_in_racing_core():
    plan = plan_split_s_trajectory()
    assert_ctbr_within_limits(plan)
    result = analytic_split_s_events(plan)
    assert result == {"passed4": True, "passed5": True, "collision": False, "wrong_way": False}


def test_training_code_does_not_import_split_s_waypoints():
    root = Path(__file__).resolve().parents[1] / "genesis_drones"
    banned = (
        root / "envs" / "race_env.py",
        root / "tasks" / "race_task.py",
        root / "algorithms" / "race_rl.py",
    )
    train = Path(__file__).resolve().parents[1] / "scripts" / "train" / "race_train.py"
    for path in (*banned, train):
        text = path.read_text()
        assert "split_s" not in text
        assert "hidden waypoint" not in text
