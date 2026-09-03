"""Central registries for task / dynamics / algorithm names."""

from __future__ import annotations

from genesis_drones.dynamics import BACKENDS, QUAD_DYNAMICS
from genesis_drones.tasks.racing_core_task import RacingCore
from genesis_drones.tasks.tracking_core import TrackingCore

TASKS = {
    "racing": RacingCore,
    "tracking": TrackingCore,
}

DYNAMICS = dict(BACKENDS)

ALGORITHMS = ("ppo", "apg", "shac")

SENSORS = ("state", "relative_position")

NETWORKS = ("mlp",)


def require_task(name: str):
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; valid: {', '.join(TASKS)}")
    return TASKS[name]


def require_dynamics(name: str):
    if name not in DYNAMICS:
        raise ValueError(f"unknown dynamics {name!r}; valid: {', '.join(DYNAMICS)}")
    return DYNAMICS[name]


def require_algorithm(name: str) -> str:
    if name not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {name!r}; valid: {', '.join(ALGORITHMS)}")
    return name


__all__ = [
    "ALGORITHMS",
    "DYNAMICS",
    "NETWORKS",
    "QUAD_DYNAMICS",
    "SENSORS",
    "TASKS",
    "require_algorithm",
    "require_dynamics",
    "require_task",
]
