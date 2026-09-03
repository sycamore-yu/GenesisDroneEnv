"""Racing evaluation metrics. Does not construct Scene / algorithms / agents."""

from __future__ import annotations

from genesis_drones.evaluation.race import (
    RACING_CONTRACT,
    RaceEpisodeResult,
    apply_network_hidden_sizes,
    assert_shared_racing_contract,
    evaluate_policy,
    evaluate_rolling,
    make_evaluation_states,
    racing_experiment,
    recorded_experiment,
    resolve_experiment,
    save_summary,
    summarize_race_results,
    write_experiment,
)

__all__ = [
    "RACING_CONTRACT",
    "RaceEpisodeResult",
    "apply_network_hidden_sizes",
    "assert_shared_racing_contract",
    "evaluate_policy",
    "evaluate_rolling",
    "make_evaluation_states",
    "racing_experiment",
    "recorded_experiment",
    "resolve_experiment",
    "save_summary",
    "summarize_race_results",
    "write_experiment",
]
