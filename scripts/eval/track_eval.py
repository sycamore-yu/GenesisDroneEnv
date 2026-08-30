import argparse
from pathlib import Path

import torch
import yaml
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from genesis_drones.evaluation.track_diff import classify_eval_step
from genesis_drones.envs.genesis_env import Genesis_env
from genesis_drones.envs.track_diff_env import TrackDiffEnv
from genesis_drones.tasks.track_task import Track_task
from genesis_drones.utils.track_diff_config import (
    load_track_diff_settings,
    make_track_diff_agent,
    make_track_diff_normalizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "logs" / "track_rl" / "policy_demo" / "model_500.pt",
    )
    parser.add_argument(
        "--respawn",
        action="store_true",
        help="Old demo mode: teleport and continue after fail. Default is one life, matching the new eval.",
    )
    return parser.parse_args()


class OneLifeTrackTask(Track_task):
    def reset(self, env_idx=None):
        if env_idx is not None:
            return self.get_observations()
        return super().reset(env_idx)


def play_ppo(checkpoint: Path, max_sim_step: int, respawn: bool) -> None:
    with open("config/track_rl/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    with open("config/track_rl/rl_env.yaml", "r") as file:
        rl_config = yaml.load(file, Loader=yaml.FullLoader)
    with open("config/track_rl/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)

    genesis_env = Genesis_env(
        env_config=env_config,
        flight_config=flight_config,
        num_envs=1,
    )
    task_cls = Track_task if respawn else OneLifeTrackTask
    track_task = task_cls(
        genesis_env=genesis_env,
        env_config=env_config,
        task_config=rl_config["task"],
        train_config=rl_config["train"],
        num_envs=1,
    )
    runner = OnPolicyRunner(track_task, rl_config["train"], "", device="cuda:0")
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device="cuda:0")
    obs = track_task.reset()
    waypoint_count = 0
    task_cfg = rl_config["task"]

    with torch.no_grad():
        for step in range(max_sim_step):
            obs, _, _, _ = track_task.step(policy(obs))
            arrived, crashed, _, _, _ = classify_eval_step(
                torch.ones(1, dtype=torch.bool, device=gs.device),
                genesis_env.drone.odom.world_pos,
                track_task.command_buf,
                genesis_env.drone.odom.has_nan,
                task_cfg["target_thr"],
                task_cfg["termination_if_close_to_ground"],
                task_cfg["termination_if_x_greater_than"],
            )
            waypoint_count += int(arrived.item())
            if bool(crashed.item()) and not respawn:
                print(
                    f"one-life stop: steps={step + 1} waypoints={waypoint_count} "
                    f"z={float(genesis_env.drone.odom.world_pos[0, 2]):.3f}"
                )
                return
    print(f"finished {max_sim_step} steps waypoints={waypoint_count}")


def play_diff(checkpoint: Path, payload: dict, max_sim_step: int, respawn: bool) -> None:
    algorithm = payload["algorithm"]
    settings = load_track_diff_settings(PROJECT_ROOT / "config" / "track_diff" / "train.yaml")
    agent = make_track_diff_agent(algorithm, settings, gs.device)
    agent.actor.load_state_dict(payload["agent"]["actor"])
    normalizer = make_track_diff_normalizer(gs.device)
    normalizer.load_state_dict(payload["normalizer"])

    environment = TrackDiffEnv(settings.environment, num_envs=1, requires_grad=False, show_viewer=True)
    environment.end_on_vertical_error = False
    environment.respawn_on_fail = respawn
    observation = environment.reset()
    with torch.no_grad():
        for step in range(max_sim_step):
            action = agent.action(observation, normalizer, deterministic=True)
            observation, _, done, extras = environment.step(action)
            if bool(done.any()):
                print(
                    f"one-life stop: steps={step + 1} crashed={bool(extras['terminated'].any())} "
                    f"waypoints={int(environment.waypoint_count[0])}"
                )
                if not respawn:
                    return
                observation = environment.reset()


def main() -> None:
    args = parse_args()
    gs.init(logging_level="warning")
    max_sim_step = 10000
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("algorithm") in ("apg", "shac"):
        play_diff(args.checkpoint, payload, max_sim_step, args.respawn)
    else:
        play_ppo(args.checkpoint, max_sim_step, args.respawn)


if __name__ == "__main__":
    main()
