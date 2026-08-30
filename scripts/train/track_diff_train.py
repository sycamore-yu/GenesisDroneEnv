import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import genesis as gs

from genesis_drones.algorithms.diff_rl import ApgAgent, RunningNormalizer, ShacAgent
from genesis_drones.envs.track_diff_env import TrackDiffEnv
from genesis_drones.evaluation.track_diff import TrackScenarios, evaluate_diff_policy, summarize_metrics
from genesis_drones.utils.track_diff_config import (
    build_track_diff_settings,
    load_track_diff_settings,
    make_track_diff_agent,
    make_track_diff_normalizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("apg", "shac"), required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "track_diff" / "train.yaml")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--validation-scenarios", type=int)
    parser.add_argument("--horizon", type=int, help="override environment/apg/shac horizon together")
    parser.add_argument(
        "--no-terminal-value",
        action="store_true",
        help="SHAC-old: actor loss without γ^H V(s_H) bootstrap",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-dir", type=Path)
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    algorithm: str,
    agent: ApgAgent | ShacAgent,
    normalizer: RunningNormalizer,
    environment: TrackDiffEnv,
    config: dict,
    update: int,
    environment_steps: int,
    best_key: tuple[float, float, float] | None,
    best_update: int,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": algorithm,
            "agent": agent.state_dict(),
            "normalizer": normalizer.state_dict(),
            "environment": environment.state_dict(),
            "config": config,
            "update": update,
            "environment_steps": environment_steps,
            "best_key": best_key,
            "best_update": best_update,
            "elapsed_seconds": elapsed_seconds,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    settings = load_track_diff_settings(args.config)
    if args.horizon is not None or args.no_terminal_value:
        data = dict(settings.raw)
        if args.horizon is not None:
            data["environment"] = {**data["environment"], "horizon": args.horizon}
            data["apg"] = {**data["apg"], "horizon": args.horizon}
            data["shac"] = {**data["shac"], "horizon": args.horizon}
        if args.no_terminal_value:
            data["shac"] = {**data["shac"], "use_terminal_value": False}
        settings = build_track_diff_settings(data, PROJECT_ROOT)
    gs.init(backend=gs.gpu, seed=settings.seed, logging_level="warning")
    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cuda:0", weights_only=False)
        if checkpoint["algorithm"] != args.algo:
            raise ValueError("resume checkpoint algorithm does not match --algo")
        settings = build_track_diff_settings(checkpoint["config"], PROJECT_ROOT)

    updates = settings.updates if args.updates is None else args.updates
    num_envs_default = settings.apg_num_envs if args.algo == "apg" else settings.shac_num_envs
    num_envs = num_envs_default if args.num_envs is None else args.num_envs
    validation_scenario_count = (
        settings.validation_scenarios if args.validation_scenarios is None else args.validation_scenarios
    )
    horizon = settings.environment.horizon
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    terminal_tag = ""
    if args.algo == "shac":
        terminal_tag = "_old" if not settings.shac.use_terminal_value else "_terminal"
    default_name = f"{args.algo}{terminal_tag}_H{horizon}_{timestamp}"
    log_dir = args.log_dir or settings.log_root / default_name
    log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)

    agent = make_track_diff_agent(args.algo, settings, gs.device)
    normalizer = make_track_diff_normalizer(gs.device)
    if checkpoint is not None:
        agent.load_state_dict(checkpoint["agent"])
        normalizer.load_state_dict(checkpoint["normalizer"])

    # Optimizers must exist before the differentiable scene initializes its gradient runtime.
    environment = TrackDiffEnv(settings.environment, num_envs, requires_grad=True)
    if checkpoint is None:
        observation = environment.reset()
        start_update = 0
        environment_steps = 0
        best_key = None
        best_update = 0
        elapsed_before_resume = 0.0
    else:
        observation = environment.load_state_dict(checkpoint["environment"])
        start_update = checkpoint["update"]
        environment_steps = checkpoint["environment_steps"]
        best_key = checkpoint["best_key"]
        best_update = checkpoint["best_update"]
        elapsed_before_resume = checkpoint["elapsed_seconds"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])

    validation_scenarios = TrackScenarios.generate(
        validation_scenario_count,
        settings.environment.max_episode_steps,
        settings.environment,
        settings.validation_seed,
    )
    validation_scenarios.save(log_dir / "validation_scenarios.pt")
    validation_environment = TrackDiffEnv(
        settings.environment,
        validation_scenario_count,
        requires_grad=False,
    )
    writer = SummaryWriter(log_dir)
    validation_history = []
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    if start_update == 0:
        metrics = evaluate_diff_policy(validation_environment, agent, normalizer, validation_scenarios)
        summary = summarize_metrics(metrics, settings.environment.max_episode_steps * settings.environment.dt)
        best_key = (
            summary.waypoint_count.mean,
            -summary.crash_rate.mean,
            -summary.capped_first_arrival_time.mean,
        )
        validation_history.append({"update": 0, "summary": summary.to_dict()})
        torch.save(metrics, log_dir / "validation_metrics_0000.pt")
        save_checkpoint(
            log_dir / "checkpoint_0000.pt",
            args.algo,
            agent,
            normalizer,
            environment,
            settings.raw,
            0,
            0,
            best_key,
            0,
            0.0,
        )
        shutil.copyfile(log_dir / "checkpoint_0000.pt", log_dir / "best.pt")

    for update in range(start_update + 1, updates + 1):
        observation, stats = agent.update(environment, observation, normalizer)
        environment_steps += num_envs * stats.steps
        if environment.episode_step >= settings.environment.max_episode_steps or not environment.is_alive.any():
            observation = environment.reset()

        writer.add_scalar("train/actor_loss", stats.actor_loss, update)
        writer.add_scalar("train/actor_grad_norm", stats.actor_grad_norm, update)
        writer.add_scalar("train/mean_reward", stats.mean_reward, update)
        writer.add_scalar("train/critic_loss", stats.critic_loss, update)
        writer.add_scalar("train/critic_grad_norm", stats.critic_grad_norm, update)
        writer.add_scalar("train/entropy", stats.entropy, update)
        writer.add_scalar("train/valid_transitions", stats.valid_transitions, update)
        writer.add_scalar("train/environment_steps", environment_steps, update)
        if update % 10 == 0:
            print(
                f"update={update} actor_loss={stats.actor_loss:.6f} reward={stats.mean_reward:.6f} "
                f"critic_loss={stats.critic_loss:.6f} actor_grad={stats.actor_grad_norm:.6f}",
                flush=True,
            )

        if update % settings.save_interval == 0 or update == updates:
            metrics = evaluate_diff_policy(validation_environment, agent, normalizer, validation_scenarios)
            summary = summarize_metrics(metrics, settings.environment.max_episode_steps * settings.environment.dt)
            torch.save(metrics, log_dir / f"validation_metrics_{update:04d}.pt")
            validation_history.append({"update": update, "summary": summary.to_dict()})
            writer.add_scalar("validation/waypoint_count", summary.waypoint_count.mean, update)
            writer.add_scalar("validation/first_arrival_rate", summary.first_arrival_rate.mean, update)
            writer.add_scalar("validation/crash_rate", summary.crash_rate.mean, update)
            writer.add_scalar("validation/mean_position_error", summary.mean_position_error.mean, update)
            writer.add_scalar("validation/mean_speed", summary.mean_speed.mean, update)
            writer.add_scalar("validation/mean_closing_velocity", summary.mean_closing_velocity.mean, update)
            writer.add_scalar("validation/path_efficiency", summary.path_efficiency.mean, update)
            writer.add_scalar(
                "validation/capped_first_arrival_time", summary.capped_first_arrival_time.mean, update
            )
            writer.add_scalar("validation/action_total_variation", summary.action_total_variation.mean, update)
            candidate_key = (
                summary.waypoint_count.mean,
                -summary.crash_rate.mean,
                -summary.capped_first_arrival_time.mean,
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_update = update
            elapsed_seconds = elapsed_before_resume + time.perf_counter() - start_time
            checkpoint_path = log_dir / f"checkpoint_{update:04d}.pt"
            save_checkpoint(
                checkpoint_path,
                args.algo,
                agent,
                normalizer,
                environment,
                settings.raw,
                update,
                environment_steps,
                best_key,
                best_update,
                elapsed_seconds,
            )
            if best_update == update:
                shutil.copyfile(checkpoint_path, log_dir / "best.pt")

    elapsed_seconds = elapsed_before_resume + time.perf_counter() - start_time
    free_memory, total_memory = torch.cuda.mem_get_info()
    peak_reserved = torch.cuda.max_memory_reserved()
    non_torch_memory = max(0, total_memory - free_memory - torch.cuda.memory_reserved())
    estimated_peak_memory = non_torch_memory + peak_reserved
    result = {
        "algorithm": args.algo,
        "horizon": horizon,
        "use_terminal_value": None if args.algo != "shac" else settings.shac.use_terminal_value,
        "dt": settings.environment.dt,
        "gradient_physical_time": horizon * settings.environment.dt,
        "updates": updates,
        "num_envs": num_envs,
        "environment_steps": environment_steps,
        "best_update": best_update,
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_bytes": estimated_peak_memory,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": peak_reserved,
        "validation": validation_history,
    }
    with (log_dir / "training_summary.json").open("w") as file:
        json.dump(result, file, indent=2)
    writer.close()
    print(json.dumps({key: value for key, value in result.items() if key != "validation"}, indent=2))


if __name__ == "__main__":
    main()
