
import argparse
import os
import shutil
from datetime import datetime

import genesis as gs
import warp as wp
import yaml
from rsl_rl.runners import OnPolicyRunner

from genesis_drones.envs.genesis_env import Genesis_env
from genesis_drones.tasks.track_task import Track_task


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fully-differentiable", action="store_true")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--log-dir", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    gs.init(logging_level="warning")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_dir = args.log_dir or (
        f"logs/track_rl/track_fulldiff_{timestamp}" if args.fully_differentiable else f"logs/track_rl/track_{timestamp}"
    )
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    bash_content = f"""#!/bin/bash
    tensorboard --logdir="{log_dir}"
    """

    bash_path = "scripts/shell/launch_tb.bash"
    with open(bash_path, "w") as f:
        f.write(bash_content)

    with open("config/track_rl/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config/track_rl/rl_env.yaml", "r") as file:
        rl_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config/track_rl/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)

    task_config = rl_config["task"]
    train_config = rl_config["train"]
    if args.fully_differentiable:
        task_config["fully_differentiable"] = True
        env_config["show_viewer"] = False
        env_config["render_cam"] = False
        env_config["vis_waypoints"] = False
    if args.num_envs is not None:
        env_config["num_envs"] = args.num_envs
    if args.max_iterations is not None:
        train_config["max_iterations"] = args.max_iterations

    genesis_env = Genesis_env(
        env_config = env_config, 
        flight_config = flight_config,
    )

    track_task = Track_task(
        genesis_env = genesis_env, 
        env_config = env_config, 
        task_config = task_config,
        train_config = train_config,
    )

    runner = OnPolicyRunner(track_task, train_config, log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=train_config["max_iterations"], init_at_random_ep_len=True)

if __name__ == "__main__" :
    wp.config.enable_backward_log = True
    main()
