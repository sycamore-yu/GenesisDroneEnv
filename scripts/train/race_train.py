import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

import genesis as gs

from genesis_drones.algorithms.diff_rl import (
    NetworkConfig,
    RunningNormalizer,
    build_diff_algorithm_config,
    diff_algorithm_names,
    make_diff_agent,
)
from genesis_drones.controllers.quad_plant import QUAD_DYNAMICS
from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
from genesis_drones.evaluation.race import (
    apply_network_hidden_sizes,
    assert_shared_racing_contract,
    racing_experiment,
    write_experiment,
)
from genesis_drones.tasks.race_task import RaceTask


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("ppo", *diff_algorithm_names()), required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "race" / "train.yaml")
    parser.add_argument("--dynamics", choices=QUAD_DYNAMICS)
    parser.add_argument("--horizon", type=int, choices=(32, 64, 96))
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def load_settings(path: Path) -> dict:
    with path.open() as file:
        return yaml.safe_load(file)


def make_env(data: dict, num_envs: int, requires_grad: bool) -> RaceEnv:
    environment_data = data["environment"]
    config = RaceEnvConfig(
        dt=environment_data["dt"],
        horizon=environment_data["horizon"],
        max_episode_steps=environment_data["max_episode_steps"],
        gamma=environment_data["gamma"],
        td_lambda=environment_data["td_lambda"],
        dynamics=environment_data.get("dynamics", "native_quad"),
    )
    assert_shared_racing_contract(config)
    return RaceEnv(config, num_envs=num_envs, requires_grad=requires_grad)


def ppo_config_path(dynamics: str) -> Path:
    if dynamics == "full_quad":
        return PROJECT_ROOT / "config" / "race" / "ppo_full_quad.yaml"
    return PROJECT_ROOT / "config" / "race" / "ppo.yaml"


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_ppo(args: argparse.Namespace, data: dict, log_dir: Path) -> None:
    from rsl_rl.runners import OnPolicyRunner

    from genesis_drones.algorithms.squashed_actor_critic import register_squashed_actor_critic

    register_squashed_actor_critic()
    dynamics = data["environment"]["dynamics"]
    experiment = racing_experiment("ppo", dynamics, data["seed"])
    write_experiment(log_dir / "experiment.json", experiment)
    with ppo_config_path(dynamics).open() as file:
        train_config = yaml.safe_load(file)
    train_config["seed"] = data["seed"]
    apply_network_hidden_sizes(train_config, data["network"]["hidden_sizes"])
    num_envs = data["num_envs"]["ppo"] if args.num_envs is None else args.num_envs
    environment = make_env(data, num_envs, requires_grad=False)
    task = RaceTask(environment, train_config)
    runner = OnPolicyRunner(task, train_config, str(log_dir), device=str(gs.device))
    if args.resume is not None:
        runner.load(str(args.resume))
    iterations = train_config["max_iterations"] if args.updates is None else args.updates
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    write_experiment(log_dir / "contract.json", experiment)


def train_diff(args: argparse.Namespace, data: dict, log_dir: Path) -> None:
    num_envs = data["num_envs"][args.algo] if args.num_envs is None else args.num_envs
    if args.horizon is not None:
        data["environment"]["horizon"] = args.horizon
        data[args.algo]["horizon"] = args.horizon
    environment = make_env(data, num_envs, requires_grad=True)
    experiment = racing_experiment(args.algo, data["environment"]["dynamics"], data["seed"])
    write_experiment(log_dir / "experiment.json", experiment)
    network = NetworkConfig(hidden_sizes=tuple(data["network"]["hidden_sizes"]))
    policy_normalizer = RunningNormalizer(RaceEnv.policy_observation_dim).to(gs.device)
    algorithm_config = build_diff_algorithm_config(args.algo, data[args.algo])
    agent = make_diff_agent(args.algo, environment.spec, network, algorithm_config, gs.device)
    observation = environment.reset_diff(seed=data["seed"])
    writer = SummaryWriter(log_dir)
    updates = data["updates"] if args.updates is None else args.updates
    for update in range(1, updates + 1):
        observation, stats = agent.update(environment, observation, policy_normalizer)
        writer.add_scalar("loss/actor", stats.actor_loss, update)
        writer.add_scalar("reward/mean", stats.mean_reward, update)
        writer.add_scalar("train/gamma", environment.config.gamma, update)
        writer.add_scalar("train/lambda", environment.config.td_lambda, update)
        if update % data["save_interval"] == 0:
            save_checkpoint(
                log_dir / f"model_{update}.pt",
                {
                    **experiment,
                    "legacy": False,
                    "agent": agent.state_dict(),
                    "policy_normalizer": policy_normalizer.state_dict(),
                    "config": data,
                    "update": update,
                },
            )
    writer.close()
    write_experiment(
        log_dir / "contract.json",
        {**experiment, "horizon": data["environment"]["horizon"]},
    )


def main() -> None:
    args = parse_args()
    data = load_settings(args.config)
    if args.seed is not None:
        data["seed"] = args.seed
    if args.dynamics is not None:
        data["environment"]["dynamics"] = args.dynamics
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, seed=data["seed"], logging_level="warning")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    horizon = data["environment"]["horizon"] if args.horizon is None else args.horizon
    dynamics = data["environment"]["dynamics"]
    log_dir = args.log_dir or PROJECT_ROOT / data["log_root"] / f"{args.algo}_{dynamics}_H{horizon}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.algo == "ppo":
        train_ppo(args, data, log_dir)
    else:
        train_diff(args, data, log_dir)


if __name__ == "__main__":
    main()
