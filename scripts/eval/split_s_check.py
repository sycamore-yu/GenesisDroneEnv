"""Four-layer Split-S check. Not imported by PPO/APG/SHAC training."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import genesis as gs

from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
from genesis_drones.evaluation.split_s import (
    GATE4_INDEX,
    GATE5_INDEX,
    analytic_split_s_events,
    assert_ctbr_within_limits,
    gate_corners,
    plan_split_s_trajectory,
    replay_split_s,
    split_s_waypoints,
)
from genesis_drones.tasks.racing_tracks import FIXED_SEVEN_GATE_TRACK


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = PROJECT_ROOT / "logs" / "race" / "split_s"


def _draw_gate(ax, corners, normal, origin, color, label):
    closed = np.vstack((corners, corners[:1]))
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=color, label=label)
    ax.quiver(origin[0], origin[1], origin[2], normal[0], normal[1], normal[2], color=color, length=1.2)


def plot_plan(plan) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    waypoints = split_s_waypoints()
    g4 = waypoints[1]
    g5 = waypoints[4]
    n4 = waypoints[2] - waypoints[0]
    n4 = n4 / np.linalg.norm(n4)
    n5 = waypoints[5] - waypoints[3]
    n5 = n5 / np.linalg.norm(n5)
    c4 = gate_corners(FIXED_SEVEN_GATE_TRACK, GATE4_INDEX)
    c5 = gate_corners(FIXED_SEVEN_GATE_TRACK, GATE5_INDEX)
    figure = plt.figure(figsize=(12, 4))
    ax3d = figure.add_subplot(1, 3, 1, projection="3d")
    ax3d.plot(plan.position[:, 0], plan.position[:, 1], plan.position[:, 2], color="C0", label="planned trajectory")
    ax3d.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c="k", s=20)
    _draw_gate(ax3d, c4, n4, g4, "C1", "Gate 4")
    _draw_gate(ax3d, c5, n5, g5, "C2", "Gate 5")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax_xy = figure.add_subplot(1, 3, 2)
    ax_xy.plot(plan.position[:, 0], plan.position[:, 1], color="C0")
    ax_xy.scatter(waypoints[:, 0], waypoints[:, 1], c="k", s=16)
    ax_xy.arrow(g4[0], g4[1], n4[0], n4[1], color="C1", width=0.02)
    ax_xy.arrow(g5[0], g5[1], n5[0], n5[1], color="C2", width=0.02)
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_aspect("equal")
    ax_xy.set_title("XY")
    ax_xz = figure.add_subplot(1, 3, 3)
    ax_xz.plot(plan.position[:, 0], plan.position[:, 2], color="C0")
    ax_xz.scatter(waypoints[:, 0], waypoints[:, 2], c="k", s=16)
    ax_xz.arrow(g4[0], g4[2], n4[0], n4[2], color="C1", width=0.03)
    ax_xz.arrow(g5[0], g5[2], n5[0], n5[2], color="C2", width=0.03)
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_aspect("equal")
    ax_xz.set_title("XZ")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "split_s_plan.png", dpi=120)
    plt.close(figure)


def main() -> None:
    plan = plan_split_s_trajectory()
    plot_plan(plan)
    assert_ctbr_within_limits(plan)
    analytic = analytic_split_s_events(plan)
    if not (analytic["passed4"] and analytic["passed5"]) or analytic["collision"]:
        raise SystemExit(f"analytic Split-S failed: {analytic}")
    if not gs._initialized:
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    environment = RaceEnv(
        RaceEnvConfig(horizon=1, max_episode_steps=len(plan.times) + 50, enable_gate_contact=True),
        num_envs=1,
        requires_grad=False,
    )
    genesis_result = replay_split_s(environment, plan)
    print(
        {
            "analytic": analytic,
            "genesis": genesis_result,
            "steps": len(plan.times),
            "plots": str(PLOT_DIR / "split_s_plan.png"),
        }
    )
    if not (genesis_result["passed4"] and genesis_result["passed5"]) or genesis_result["collision"]:
        raise SystemExit(f"Genesis Split-S failed: {genesis_result}")


if __name__ == "__main__":
    main()
