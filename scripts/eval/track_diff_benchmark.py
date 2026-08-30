import argparse
import gc
import json
from pathlib import Path

import torch

import genesis as gs

from genesis_drones.envs.track_diff_env import TrackDiffEnv
from genesis_drones.utils.track_diff_config import (
    load_track_diff_settings,
    make_track_diff_agent,
    make_track_diff_normalizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def linear_slope(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    count = len(values)
    x_mean = (count - 1) / 2.0
    y_mean = sum(values) / count
    numerator = 0.0
    denominator = 0.0
    for index, value in enumerate(values):
        deviation = index - x_mean
        numerator += deviation * (value - y_mean)
        denominator += deviation * deviation
    return numerator / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("apg", "shac"), required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--validation-scenarios", type=int, default=256)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--max-growth-bytes", type=int)
    parser.add_argument("--report-live-tensors", action="store_true")
    parser.add_argument("--collect-garbage", action="store_true")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "track_diff" / "train.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_track_diff_settings(args.config)
    gs.init(backend=gs.gpu, seed=settings.seed, logging_level="warning")
    torch.cuda.reset_peak_memory_stats()
    agent = make_track_diff_agent(args.algo, settings, gs.device)
    normalizer = make_track_diff_normalizer(gs.device)
    environment = TrackDiffEnv(settings.environment, args.num_envs, requires_grad=True)
    observation = environment.reset()
    if args.validation_scenarios > 0:
        validation_environment = TrackDiffEnv(
            settings.environment, args.validation_scenarios, requires_grad=False
        )
        validation_environment.reset()
    minimum_free_memory = torch.cuda.mem_get_info()[0]
    allocated_history = []
    for _ in range(args.updates):
        observation, stats = agent.update(environment, observation, normalizer)
        torch.cuda.synchronize()
        minimum_free_memory = min(minimum_free_memory, torch.cuda.mem_get_info()[0])
        if args.collect_garbage:
            gc.collect()
        allocated_history.append(torch.cuda.memory_allocated())
        if environment.episode_step >= settings.environment.max_episode_steps or not environment.is_alive.any():
            observation = environment.reset()
    free_memory, total_memory = torch.cuda.mem_get_info()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    current_reserved = torch.cuda.memory_reserved()
    non_torch_memory = max(0, total_memory - free_memory - current_reserved)
    estimated_peak_memory = non_torch_memory + peak_reserved
    measured_allocated = allocated_history[args.warmup_updates :]
    allocated_growth = max(measured_allocated) - min(measured_allocated) if measured_allocated else 0
    allocated_slope = linear_slope(measured_allocated)
    live_tensor_shapes = {}
    if args.report_live_tensors:
        for value in gc.get_objects():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                key = str(tuple(value.shape))
                count, elements = live_tensor_shapes.get(key, (0, 0))
                live_tensor_shapes[key] = (count + 1, elements + value.numel())
    result = {
        "algorithm": args.algo,
        "num_envs": args.num_envs,
        "steps": stats.steps,
        "updates": args.updates,
        "validation_scenarios": args.validation_scenarios,
        "actor_grad_norm": stats.actor_grad_norm,
        "allocated_history_bytes": allocated_history,
        "allocated_growth_after_warmup_bytes": allocated_growth,
        "allocated_slope_after_warmup_bytes_per_update": allocated_slope,
        "peak_allocated_bytes": peak_allocated,
        "live_tensor_shapes": live_tensor_shapes,
        "peak_reserved_bytes": peak_reserved,
        "non_torch_memory_bytes": non_torch_memory,
        "estimated_peak_memory_bytes": estimated_peak_memory,
        "total_memory_bytes": total_memory,
        "estimated_headroom": 1.0 - estimated_peak_memory / total_memory,
        "minimum_free_memory_bytes": minimum_free_memory,
        "meets_twenty_percent_headroom": estimated_peak_memory <= 0.8 * total_memory,
    }
    print(json.dumps(result, indent=2))
    if args.max_growth_bytes is not None and allocated_growth > args.max_growth_bytes:
        raise RuntimeError(
            f"allocated memory grew by {allocated_growth} bytes after warmup; "
            f"limit is {args.max_growth_bytes} bytes"
        )


if __name__ == "__main__":
    main()
