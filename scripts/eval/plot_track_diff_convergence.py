import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_scalar(log_dir: Path, tag: str, max_step: int | None = None) -> tuple[list[float], list[float]]:
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    events = accumulator.Scalars(tag)
    if not events:
        raise ValueError(f"no scalar {tag} in {log_dir}")
    start = events[0].wall_time
    minutes = []
    values = []
    for event in events:
        if max_step is not None and event.step > max_step:
            break
        minutes.append((event.wall_time - start) / 60.0)
        values.append(event.value)
    return minutes, values


def combine_legend(axis, twin) -> None:
    handles, labels = axis.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    axis.legend(handles + extra_handles, labels + extra_labels, loc="best")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apg-logdir", type=Path, default=PROJECT_ROOT / "logs" / "track_diff" / "apg_300")
    parser.add_argument("--shac-logdir", type=Path, default=PROJECT_ROOT / "logs" / "track_diff" / "shac_300")
    parser.add_argument(
        "--ppo-logdir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "track_rl" / "track_2026-08-28_11:58:25",
    )
    parser.add_argument("--max-update", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "logs" / "track_diff" / "convergence_ppo_apg_shac.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apg_reward = load_scalar(args.apg_logdir, "train/mean_reward", args.max_update)
    apg_loss = load_scalar(args.apg_logdir, "train/actor_loss", args.max_update)
    shac_reward = load_scalar(args.shac_logdir, "train/mean_reward", args.max_update)
    shac_loss = load_scalar(args.shac_logdir, "train/actor_loss", args.max_update)
    ppo_reward = load_scalar(args.ppo_logdir, "Train/mean_reward")
    ppo_loss = load_scalar(args.ppo_logdir, "Loss/surrogate")

    figure, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)
    reward_axis = axes[0]
    reward_axis.plot(*apg_reward, color="C0", label="APG reward")
    reward_axis.plot(*shac_reward, color="C1", label="SHAC reward")
    reward_axis.set_ylabel("APG / SHAC mean reward")
    reward_axis.grid(True, alpha=0.3)
    ppo_reward_axis = reward_axis.twinx()
    ppo_reward_axis.plot(*ppo_reward, color="C2", linestyle="--", label="PPO reward")
    ppo_reward_axis.set_ylabel("PPO mean reward")
    combine_legend(reward_axis, ppo_reward_axis)
    reward_axis.set_title("Convergence vs wall-clock time")

    loss_axis = axes[1]
    loss_axis.plot(*apg_loss, color="C0", label="APG actor loss")
    loss_axis.plot(*shac_loss, color="C1", label="SHAC actor loss")
    loss_axis.set_ylabel("APG / SHAC actor loss")
    loss_axis.set_xlabel("Wall-clock time (minutes)")
    loss_axis.grid(True, alpha=0.3)
    ppo_loss_axis = loss_axis.twinx()
    ppo_loss_axis.plot(*ppo_loss, color="C2", linestyle="--", label="PPO surrogate loss")
    ppo_loss_axis.set_ylabel("PPO surrogate loss")
    combine_legend(loss_axis, ppo_loss_axis)

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(args.output)


if __name__ == "__main__":
    main()
