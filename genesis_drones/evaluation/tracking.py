"""Tracking evaluation: scenarios + metric aggregation. No Scene / algorithm construction."""

from genesis_drones.evaluation.track_diff import (
    EvaluationSummary,
    ScalarSummary,
    TrackScenarios,
    evaluate_diff_policy,
    paired_differences,
    success_against_ppo,
    summarize_metrics,
)

__all__ = [
    "EvaluationSummary",
    "ScalarSummary",
    "TrackScenarios",
    "evaluate_diff_policy",
    "paired_differences",
    "success_against_ppo",
    "summarize_metrics",
]
