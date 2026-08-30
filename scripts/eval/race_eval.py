import argparse
import json
from pathlib import Path

import torch
import yaml

import genesis as gs

from genesis_drones.algorithms.diff_rl import NetworkConfig, RunningNormalizer
from genesis_drones.algorithms.race_rl import RacingApgAgent, RacingShacAgent
from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
from genesis_drones.evaluation.race import (
    RACING_CONTRACT,
    assert_shared_racing_contract,
    evaluate_policy,
    make_evaluation_states,
    save_summary,
    summarize_race_results,
)
from genesis_drones.tasks.racing_core import load_evaluation_initial_states, save_evaluation_initial_states


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("ppo", "apg", "shac"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "race" / "train.yaml")
    parser.add_argument("--states", type=Path, default=PROJECT_ROOT / "config" / "race" / "eval_states.pt")
    parser.add_argument("--write-states", action="store_true")
    parser.add_argument("--enable-gate-contact", action="store_true", default=True)
    parser.add_argument("--no-gate-contact", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as file:
        data = yaml.safe_load(file)
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    config = RaceEnvConfig(
        dt=data["environment"]["dt"],
        horizon=1,
        max_episode_steps=data["environment"]["max_episode_steps"],
        enable_gate_contact=args.enable_gate_contact and not args.no_gate_contact,
        gamma=data["environment"]["gamma"],
        td_lambda=environment_lambda(data),
    )
    assert_shared_racing_contract(config)
    environment = RaceEnv(config, num_envs=1, requires_grad=False)
    if args.write_states:
        states = make_evaluation_states(environment, count=100, seed=data["eval_seed"])
        checksum = save_evaluation_initial_states(states, args.states)
        print(json.dumps({"checksum": checksum, "path": str(args.states)}))
        return
    states, checksum = load_evaluation_initial_states(args.states)
    action_fn = load_action_fn(args, data, environment)
    results = evaluate_policy(environment, action_fn, states)
    summary = {
        **summarize_race_results(results),
        "checksum": checksum,
        "algorithm": args.algo,
        "gate_contact": config.enable_gate_contact,
        "contract": RACING_CONTRACT,
    }
    output = args.output or PROJECT_ROOT / "logs" / "race" / f"{args.algo}_eval.json"
    save_summary(output, summary)
    print(json.dumps(summary, indent=2))


def environment_lambda(data: dict) -> float:
    return data["environment"]["td_lambda"]


def load_action_fn(args: argparse.Namespace, data: dict, environment: RaceEnv):
    hover = environment.controller.hover_action
    if args.checkpoint is None:
        return lambda observation: torch.tensor([[hover, 0.0, 0.0, 0.0]], device=observation.device)
    payload = torch.load(args.checkpoint, map_location=gs.device, weights_only=False)
    algorithm = payload.get("algorithm", args.algo)
    if payload.get("legacy"):
        raise ValueError("legacy decoupled checkpoint has no racing algorithm branch")
    if algorithm != args.algo:
        raise ValueError("checkpoint algorithm does not match --algo")
    network = NetworkConfig(hidden_sizes=tuple(data["network"]["hidden_sizes"]))
    if args.algo == "apg":
        from genesis_drones.algorithms.diff_rl import ApgConfig

        agent = RacingApgAgent(
            RaceEnv.policy_observation_dim, RaceEnv.action_dim, hover, network, ApgConfig(**data["apg"]), gs.device
        )
        agent.load_state_dict(payload["agent"])
        normalizer = RunningNormalizer(RaceEnv.policy_observation_dim).to(gs.device)
        normalizer.load_state_dict(payload["policy_normalizer"])
        return lambda observation: agent.action(observation, normalizer, deterministic=True)
    if args.algo == "shac":
        from genesis_drones.algorithms.diff_rl import ShacConfig

        agent = RacingShacAgent(
            RaceEnv.policy_observation_dim,
            RaceEnv.critic_observation_dim,
            RaceEnv.action_dim,
            hover,
            network,
            ShacConfig(**data["shac"]),
            gs.device,
        )
        agent.load_state_dict(payload["agent"])
        normalizer = RunningNormalizer(RaceEnv.policy_observation_dim).to(gs.device)
        normalizer.load_state_dict(payload["policy_normalizer"])
        return lambda observation: agent.action(observation, normalizer, deterministic=True)
    from rsl_rl.runners import OnPolicyRunner
    from genesis_drones.tasks.race_task import RaceTask

    with (PROJECT_ROOT / "config" / "race" / "ppo.yaml").open() as file:
        train_config = yaml.safe_load(file)
    task = RaceTask(environment, train_config)
    runner = OnPolicyRunner(task, train_config, str(args.checkpoint.parent), device=str(gs.device))
    runner.load(str(args.checkpoint))

    def ppo_action(observation: torch.Tensor) -> torch.Tensor:
        return runner.alg.act_inference(observation)

    return ppo_action


if __name__ == "__main__":
    main()
