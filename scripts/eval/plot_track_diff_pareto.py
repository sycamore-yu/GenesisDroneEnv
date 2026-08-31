"""Phase 6–7: Pareto (crash vs speed) and efficiency (vs steps / wall time) from training summaries."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_runs(log_root: Path) -> list[dict]:
    points = []
    for summary_path in sorted(log_root.glob("*/training_summary.json")):
        with summary_path.open() as file:
            data = json.load(file)
        validation = data.get("validation") or []
        if not validation:
            continue
        name = summary_path.parent.name
        elapsed = data.get("elapsed_seconds") or 0.0
        env_steps = data.get("environment_steps") or 0
        updates = max(data.get("updates") or 1, 1)
        for entry in validation:
            if not isinstance(entry, dict) or "summary" not in entry:
                continue
            summary = entry["summary"]
            if not isinstance(summary, dict) or "waypoints_per_minute" not in summary:
                continue
            update = entry["update"]
            frac = 0.0 if updates == 0 else update / updates
            points.append(
                {
                    "name": name,
                    "algorithm": data.get("algorithm", name.split("_")[0]),
                    "horizon": data.get("horizon"),
                    "update": update,
                    "best_update": data.get("best_update"),
                    "is_best": update == data.get("best_update"),
                    "crash": summary["crash_rate"]["mean"],
                    "wpm": summary["waypoints_per_minute"]["mean"],
                    "arrival": summary["capped_first_arrival_time"]["mean"],
                    "speed": summary.get("mean_speed", {}).get("mean"),
                    "close": summary.get("mean_closing_velocity", {}).get("mean"),
                    "eta": summary.get("path_efficiency", {}).get("mean"),
                    "env_steps": env_steps * frac,
                    "elapsed": elapsed * frac,
                }
            )
    return points


def load_ppo(eval_path: Path) -> dict | None:
    if not eval_path.exists():
        return None
    with eval_path.open() as file:
        data = json.load(file)
    ppo = data.get("summaries", {}).get("ppo")
    if ppo is None:
        return None
    return {
        "crash": ppo["crash_rate"]["mean"],
        "wpm": ppo["waypoints_per_minute"]["mean"],
        "arrival": ppo["capped_first_arrival_time"]["mean"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=PROJECT_ROOT / "logs" / "track_diff")
    parser.add_argument(
        "--ppo-eval",
        type=Path,
        default=PROJECT_ROOT / "logs" / "track_diff" / "evaluation_onelife" / "evaluation.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "logs" / "track_diff")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_runs(args.log_root)
    ppo = load_ppo(args.ppo_eval)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = [point for point in points if point["is_best"] and point["update"] != 0]
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))

    crash_wpm = axes[0, 0]
    crash_arr = axes[0, 1]
    steps_wpm = axes[1, 0]
    time_arr = axes[1, 1]
    for point in best:
        crash_wpm.scatter(point["crash"], point["wpm"], label=point["name"][:28])
        crash_arr.scatter(point["crash"], point["arrival"])
    if ppo is not None:
        crash_wpm.scatter(ppo["crash"], ppo["wpm"], marker="*", s=180, color="black", label="PPO", zorder=5)
        crash_arr.scatter(ppo["crash"], ppo["arrival"], marker="*", s=180, color="black", zorder=5)
    crash_wpm.set_xlabel("Crash rate")
    crash_wpm.set_ylabel("Waypoints / min")
    crash_wpm.set_title("Speed–safety Pareto")
    crash_wpm.grid(True, alpha=0.3)
    crash_wpm.legend(fontsize=6, loc="best")
    crash_arr.set_xlabel("Crash rate")
    crash_arr.set_ylabel("Capped first arrival (s)")
    crash_arr.set_title("Arrival vs crash")
    crash_arr.grid(True, alpha=0.3)

    from collections import defaultdict

    by_name = defaultdict(list)
    for point in points:
        if point["update"] == 0:
            continue
        by_name[point["name"]].append(point)
    for name, series in by_name.items():
        series = sorted(series, key=lambda item: item["update"])
        steps_wpm.plot(
            [item["env_steps"] for item in series],
            [item["wpm"] for item in series],
            marker="o",
            alpha=0.7,
            label=name[:22],
        )
        time_arr.plot(
            [item["elapsed"] / 60.0 for item in series],
            [item["arrival"] for item in series],
            marker="o",
            alpha=0.7,
        )
    steps_wpm.set_xlabel("Environment steps (approx.)")
    steps_wpm.set_ylabel("Waypoints / min")
    steps_wpm.set_title("Sample efficiency")
    steps_wpm.grid(True, alpha=0.3)
    steps_wpm.legend(fontsize=6, loc="best")
    time_arr.set_xlabel("Wall time (min, approx.)")
    time_arr.set_ylabel("Capped first arrival (s)")
    time_arr.set_title("Compute efficiency")
    time_arr.grid(True, alpha=0.3)

    figure.tight_layout()
    output = args.output_dir / "pareto_and_efficiency.png"
    figure.savefig(output, dpi=120)
    table = args.output_dir / "pareto_best.json"
    with table.open("w") as file:
        json.dump(best, file, indent=2)
    print(output)
    print(f"best checkpoints: {len(best)}")


if __name__ == "__main__":
    main()
