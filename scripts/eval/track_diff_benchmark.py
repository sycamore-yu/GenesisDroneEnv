import argparse
import json
from pathlib import Path

import torch

import genesis as gs

from genesis_drones.algorithms.diff_rl import ApgAgent, RunningNormalizer, ShacAgent
from genesis_drones.envs.track_diff_env import TrackDiffEnv
from genesis_drones.utils.track_diff_config import load_track_diff_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("apg", "shac"), required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--validation-scenarios", type=int, default=256)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "track_diff" / "train.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_track_diff_settings(args.config)
    gs.init(backend=gs.gpu, seed=settings.seed, logging_level="warning")
    torch.cuda.reset_peak_memory_stats()
    if args.algo == "apg":
        agent = ApgAgent(17, 4, 3.3, settings.network, settings.apg, gs.device)
    else:
        agent = ShacAgent(17, 4, 3.3, settings.network, settings.shac, gs.device)
    normalizer = RunningNormalizer(17).to(gs.device)
    environment = TrackDiffEnv(settings.environment, args.num_envs, requires_grad=True)
    validation_environment = TrackDiffEnv(
        settings.environment, args.validation_scenarios, requires_grad=False
    )
    observation = environment.reset()
    validation_environment.reset()
    minimum_free_memory = torch.cuda.mem_get_info()[0]
    for _ in range(args.updates):
        observation, stats = agent.update(environment, observation, normalizer)
        minimum_free_memory = min(minimum_free_memory, torch.cuda.mem_get_info()[0])
        if environment.episode_step >= settings.environment.max_episode_steps or not environment.is_alive.any():
            observation = environment.reset()
    free_memory, total_memory = torch.cuda.mem_get_info()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    current_reserved = torch.cuda.memory_reserved()
    non_torch_memory = max(0, total_memory - free_memory - current_reserved)
    estimated_peak_memory = non_torch_memory + peak_reserved
    result = {
        "algorithm": args.algo,
        "num_envs": args.num_envs,
        "steps": stats.steps,
        "updates": args.updates,
        "validation_scenarios": args.validation_scenarios,
        "actor_grad_norm": stats.actor_grad_norm,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "non_torch_memory_bytes": non_torch_memory,
        "estimated_peak_memory_bytes": estimated_peak_memory,
        "total_memory_bytes": total_memory,
        "estimated_headroom": 1.0 - estimated_peak_memory / total_memory,
        "minimum_free_memory_bytes": minimum_free_memory,
        "meets_twenty_percent_headroom": estimated_peak_memory <= 0.8 * total_memory,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
