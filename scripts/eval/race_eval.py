import argparse
import json
from pathlib import Path

import torch
import yaml

import genesis as gs

from genesis_drones.algorithms.diff_rl import (
    NetworkConfig,
    RunningNormalizer,
    build_diff_algorithm_config,
    diff_algorithm_names,
    make_diff_agent,
)
from genesis_drones.controllers.ctbr_controller import CtbrControllerConfig
from genesis_drones.controllers.quad_plant import QUAD_DYNAMICS
from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
from genesis_drones.evaluation.race import (
    RACING_CONTRACT,
    apply_network_hidden_sizes,
    assert_shared_racing_contract,
    evaluate_policy,
    evaluate_rolling,
    make_evaluation_states,
    recorded_experiment,
    resolve_experiment,
    save_summary,
    summarize_race_results,
)
from genesis_drones.tasks.racing_core import load_evaluation_initial_states, save_evaluation_initial_states


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("ppo", *diff_algorithm_names()), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "race" / "train.yaml")
    parser.add_argument("--dynamics", choices=QUAD_DYNAMICS)
    parser.add_argument("--states", type=Path, default=PROJECT_ROOT / "config" / "race" / "eval_states.pt")
    parser.add_argument("--write-states", action="store_true")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as file:
        data = yaml.safe_load(file)
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    payload = None
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location=gs.device, weights_only=False)
        if payload.get("legacy"):
            raise ValueError("legacy decoupled checkpoint has no racing algorithm branch")
    experiment = resolve_experiment(
        args.algo,
        args.dynamics,
        recorded_experiment(args.checkpoint, payload),
        data["environment"].get("dynamics", "native_quad"),
    )
    dynamics = experiment["dynamics"]
    config = RaceEnvConfig(
        dt=data["environment"]["dt"],
        horizon=1,
        max_episode_steps=data["environment"]["max_episode_steps"],
        min_target_velocity=5.0,
        max_target_velocity=5.0,
        gamma=data["environment"]["gamma"],
        td_lambda=environment_lambda(data),
        dynamics=dynamics,
        controller=CtbrControllerConfig(randomize=False),
    )
    assert_shared_racing_contract(config)
    rolling = args.steps is not None
    if rolling and args.num_envs is None:
        raise SystemExit("--num-envs is required with --steps")
    environment = RaceEnv(config, num_envs=args.num_envs or 1, requires_grad=False)
    if args.write_states:
        states = make_evaluation_states(environment, count=100, seed=data["eval_seed"])
        checksum = save_evaluation_initial_states(states, args.states)
        print(json.dumps({"checksum": checksum, "path": str(args.states)}))
        return
    action_fn = load_action_fn(args, data, environment, payload)
    if rolling:
        metrics = evaluate_rolling(environment, action_fn, args.steps)
        checksum = None
    else:
        states, checksum = load_evaluation_initial_states(args.states)
        metrics = summarize_race_results(evaluate_policy(environment, action_fn, states))
    summary = {
        **metrics,
        "checksum": checksum,
        "algorithm": experiment["algorithm"],
        "dynamics": dynamics,
        "task": experiment["task"],
        "network": experiment["network"],
        "sensor": experiment["sensor"],
        "contract": RACING_CONTRACT,
        "protocol": "rolling" if rolling else "fixed_states",
    }
    output = args.output or PROJECT_ROOT / "logs" / "race" / f"{args.algo}_eval.json"
    save_summary(output, summary)
    print(json.dumps(summary, indent=2))


def environment_lambda(data: dict) -> float:
    return data["environment"]["td_lambda"]


def load_action_fn(args: argparse.Namespace, data: dict, environment: RaceEnv, payload: dict | None):
    if args.checkpoint is None:
        return lambda observation: environment.hover_command(observation.shape[0])
    algorithm = payload.get("algorithm", args.algo)
    if algorithm != args.algo:
        raise ValueError("checkpoint algorithm does not match --algo")
    network = NetworkConfig(hidden_sizes=tuple(data["network"]["hidden_sizes"]))
    if args.algo in diff_algorithm_names():
        algorithm_config = build_diff_algorithm_config(args.algo, data[args.algo])
        agent = make_diff_agent(args.algo, environment.spec, network, algorithm_config, gs.device)
        agent.load_state_dict(payload["agent"])
        normalizer = RunningNormalizer(RaceEnv.policy_observation_dim).to(gs.device)
        normalizer.load_state_dict(payload["policy_normalizer"])
        return lambda observation: agent.action(observation, normalizer, deterministic=True)
    from tensordict import TensorDict

    from genesis_drones.algorithms.squashed_actor_critic import SquashedActorCritic
    from genesis_drones.envs.race_env import detached_torch_tensor
    from rsl_rl.modules import ActorCritic

    dynamics = environment.dynamics
    config_name = "ppo_full_quad.yaml" if dynamics == "full_quad" else "ppo.yaml"
    with (PROJECT_ROOT / "config" / "race" / config_name).open() as file:
        train_config = yaml.safe_load(file)
    apply_network_hidden_sizes(train_config, data["network"]["hidden_sizes"])
    policy_cfg = dict(train_config["policy"])
    class_name = policy_cfg.pop("class_name")
    policy_class = SquashedActorCritic if class_name == "SquashedActorCritic" else ActorCritic
    dummy = TensorDict(
        {"policy": torch.zeros(1, RaceEnv.policy_observation_dim, device=gs.device)},
        batch_size=1,
    )
    policy = policy_class(dummy, train_config["obs_groups"], environment.action_dim, **policy_cfg)
    policy.load_state_dict(payload["model_state_dict"])
    policy.to(gs.device)
    policy.eval()
    return lambda observation: policy.act_inference({"policy": detached_torch_tensor(observation)})


if __name__ == "__main__":
    main()
