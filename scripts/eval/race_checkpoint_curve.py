"""Deterministic vs stochastic eval on every full_quad PPO checkpoint. Diagnosis only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tensordict import TensorDict
from torch.distributions import Normal

import genesis as gs

from genesis_drones.algorithms.squashed_actor_critic import SquashedActorCritic
from genesis_drones.controllers.ctbr_controller import CtbrControllerConfig
from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig, detached_torch_tensor
from genesis_drones.evaluation.race import apply_network_hidden_sizes, make_evaluation_states
from genesis_drones.tasks.racing_core import load_evaluation_initial_states, save_evaluation_initial_states


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "race" / "ppo_full_quad_squash_seed0",
    )
    parser.add_argument("--states", type=Path, default=PROJECT_ROOT / "config" / "race" / "eval_states.pt")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "race" / "train.yaml")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_policy(environment: RaceEnv, train_data: dict, payload: dict) -> SquashedActorCritic:
    with (PROJECT_ROOT / "config" / "race" / "ppo_full_quad.yaml").open() as file:
        train_config = yaml.safe_load(file)
    apply_network_hidden_sizes(train_config, train_data["network"]["hidden_sizes"])
    policy_cfg = dict(train_config["policy"])
    policy_cfg.pop("class_name")
    dummy = TensorDict(
        {"policy": torch.zeros(1, RaceEnv.policy_observation_dim, device=gs.device)},
        batch_size=1,
    )
    policy = SquashedActorCritic(dummy, train_config["obs_groups"], environment.action_dim, **policy_cfg)
    policy.load_state_dict(payload["model_state_dict"])
    policy.to(gs.device)
    policy.eval()
    return policy


def _stats(policy: SquashedActorCritic, observation: torch.Tensor, deterministic: bool):
    obs = TensorDict({"policy": detached_torch_tensor(observation)}, batch_size=observation.shape[0])
    actor_obs = policy.actor_obs_normalizer(policy.get_actor_obs(obs))
    mu_raw = policy.actor(actor_obs)
    mu = mu_raw.clamp(-5.0, 5.0)
    std = policy._mapped_std().expand_as(mu)
    if deterministic:
        action = policy.act_inference(obs)
    else:
        action = policy.act(obs)
        mu = policy.distribution.mean
        std = policy.distribution.stddev
    value = policy.evaluate(obs).squeeze(-1)
    entropy = Normal(mu, std).entropy().sum(dim=-1)
    squashed = torch.tanh(mu)
    return action, {
        "mu": mu,
        "mu_raw": mu_raw,
        "std": std,
        "action": action,
        "squashed_mean": squashed,
        "value": value,
        "entropy": entropy,
    }


def evaluate(environment: RaceEnv, policy: SquashedActorCritic, states, deterministic: bool) -> dict:
    environment.respawn_on_fail = False
    observation, _ = environment.reset(initial_states=states)
    n_envs = environment.num_envs
    device = environment.device
    alive = torch.ones(n_envs, dtype=torch.bool, device=device)
    episode_return = torch.zeros(n_envs, device=device)
    passed = torch.zeros(n_envs, device=device)
    success = torch.zeros(n_envs, dtype=torch.bool, device=device)
    truncated = torch.zeros(n_envs, dtype=torch.bool, device=device)
    terminated = torch.zeros(n_envs, dtype=torch.bool, device=device)
    duration = torch.zeros(n_envs, device=device)
    n_alive = 0
    step_reward_sum = torch.tensor(0.0, device=device)
    sums = {
        "mu": torch.zeros(4, device=device),
        "mu_abs": torch.zeros(4, device=device),
        "mu_raw_abs": torch.zeros(4, device=device),
        "std": torch.zeros(4, device=device),
        "action_abs": torch.zeros(4, device=device),
        "squashed_abs": torch.zeros(4, device=device),
        "entropy": torch.tensor(0.0, device=device),
        "value": torch.tensor(0.0, device=device),
        "value_sq": torch.tensor(0.0, device=device),
        "near_bound": torch.tensor(0.0, device=device),
        "clamp_change": torch.tensor(0.0, device=device),
    }
    mu_min = torch.full((4,), float("inf"), device=device)
    mu_max = torch.full((4,), float("-inf"), device=device)
    action_min = torch.full((4,), float("inf"), device=device)
    action_max = torch.full((4,), float("-inf"), device=device)
    max_steps = environment.config.max_episode_steps + 2
    with torch.no_grad():
        for _ in range(max_steps):
            if not alive.any():
                break
            action, info = _stats(policy, observation, deterministic)
            mask = alive.float()[:, None]
            n = float(alive.sum().item())
            n_alive += n
            sums["mu"] += (info["mu"] * mask).sum(dim=0)
            sums["mu_abs"] += (info["mu"].abs() * mask).sum(dim=0)
            sums["mu_raw_abs"] += (info["mu_raw"].abs() * mask).sum(dim=0)
            sums["std"] += (info["std"] * mask).sum(dim=0)
            sums["action_abs"] += (action.abs() * mask).sum(dim=0)
            sums["squashed_abs"] += (info["squashed_mean"].abs() * mask).sum(dim=0)
            sums["entropy"] += (info["entropy"] * alive).sum()
            sums["value"] += (info["value"] * alive).sum()
            sums["value_sq"] += (info["value"].square() * alive).sum()
            sums["near_bound"] += ((action.abs() > 0.99).float() * mask).sum()
            clamped = action.clamp(-1.0, 1.0)
            sums["clamp_change"] += ((clamped - action).abs() * mask).sum()
            mu_min = torch.minimum(mu_min, info["mu"].masked_fill(~alive[:, None], float("inf")).amin(dim=0))
            mu_max = torch.maximum(mu_max, info["mu"].masked_fill(~alive[:, None], float("-inf")).amax(dim=0))
            action_min = torch.minimum(action_min, action.masked_fill(~alive[:, None], float("inf")).amin(dim=0))
            action_max = torch.maximum(action_max, action.masked_fill(~alive[:, None], float("-inf")).amax(dim=0))
            observation, (_, _, reward), done, extras = environment.step(action)
            step_reward_sum = step_reward_sum + (reward * alive).sum()
            episode_return = episode_return + reward * alive
            finished = done & alive
            if finished.any():
                passed[finished] = extras["n_passed_gates"][finished].float()
                success[finished] = extras["success"][finished]
                truncated[finished] = extras["truncated"][finished]
                terminated[finished] = extras["terminated"][finished]
                duration[finished] = (extras["episode_length"][finished] - 1).float() * environment.config.dt
            alive = alive & ~done
    scale = max(n_alive, 1.0)
    value_mean = float((sums["value"] / scale).item())
    return {
        "deterministic": deterministic,
        "mean_passed_gates": float(passed.mean().item()),
        "survival": float(truncated.float().mean().item()),
        "timeout_success": float(success.float().mean().item()),
        "collision": float(terminated.float().mean().item()),
        "mean_episode_time": float(duration.mean().item()),
        "mean_episode_return": float(episode_return.mean().item()),
        "mean_reward": float((step_reward_sum / scale).item()),
        "n_envs": n_envs,
        "n_still_alive": int(alive.sum().item()),
        "policy": {
            "mu_mean": [float(x) for x in (sums["mu"] / scale).tolist()],
            "mu_abs_mean": [float(x) for x in (sums["mu_abs"] / scale).tolist()],
            "mu_raw_abs_mean": [float(x) for x in (sums["mu_raw_abs"] / scale).tolist()],
            "mu_min": [float(x) for x in mu_min.tolist()],
            "mu_max": [float(x) for x in mu_max.tolist()],
            "std": [float(x) for x in (sums["std"] / scale).tolist()],
            "action_abs_mean": [float(x) for x in (sums["action_abs"] / scale).tolist()],
            "squashed_mean_abs": [float(x) for x in (sums["squashed_abs"] / scale).tolist()],
            "action_min": [float(x) for x in action_min.tolist()],
            "action_max": [float(x) for x in action_max.tolist()],
            "near_bound_fraction": float((sums["near_bound"] / (scale * 4.0)).item()),
            "env_clamp_change_mean": float((sums["clamp_change"] / (scale * 4.0)).item()),
            "entropy": float((sums["entropy"] / scale).item()),
            "value_mean": value_mean,
            "value_std": float(torch.sqrt((sums["value_sq"] / scale) - value_mean**2).clamp_min(0.0).item()),
            "mapped_std": [float(x) for x in policy._mapped_std().detach().cpu().tolist()],
        },
    }


def main() -> None:
    args = parse_args()
    with args.config.open() as file:
        train_data = yaml.safe_load(file)
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    n_states = 100
    environment = RaceEnv(
        RaceEnvConfig(
            dt=train_data["environment"]["dt"],
            horizon=1,
            max_episode_steps=train_data["environment"]["max_episode_steps"],
            min_target_velocity=5.0,
            max_target_velocity=5.0,
            gamma=train_data["environment"]["gamma"],
            td_lambda=train_data["environment"]["td_lambda"],
            dynamics="full_quad",
            controller=CtbrControllerConfig(randomize=False),
        ),
        num_envs=n_states,
        requires_grad=False,
    )
    try:
        states, checksum = load_evaluation_initial_states(args.states)
        states_source = str(args.states)
    except (KeyError, ValueError, FileNotFoundError) as error:
        # config/race/eval_states.pt is a leftover 7-gate fixture without target_gate.
        states = make_evaluation_states(environment, count=n_states, seed=20250830)
        used = args.log_dir / "eval_states_8gate_seed20250830.pt"
        checksum = save_evaluation_initial_states(states, used)
        states_source = f"generated:{used} ({error})"
    checkpoints = sorted(args.log_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    output = args.output or args.log_dir / "checkpoint_curve.json"
    results = {
        "checksum": checksum,
        "states_source": states_source,
        "n_states": int(states.position.shape[0]),
        "checkpoints": [],
    }
    policy = None
    for path in checkpoints:
        payload = torch.load(path, map_location=gs.device, weights_only=False)
        if policy is None:
            policy = load_policy(environment, train_data, payload)
        else:
            policy.load_state_dict(payload["model_state_dict"])
            policy.eval()
        iteration = int(path.stem.split("_")[1])
        torch.manual_seed(0)
        det = evaluate(environment, policy, states, deterministic=True)
        torch.manual_seed(0)
        sto = evaluate(environment, policy, states, deterministic=False)
        row = {"iteration": iteration, "path": str(path), "deterministic": det, "stochastic": sto}
        results["checkpoints"].append(row)
        output.write_text(json.dumps(results, indent=2))
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "det_gates": det["mean_passed_gates"],
                    "sto_gates": sto["mean_passed_gates"],
                    "det_survive": det["survival"],
                    "sto_survive": sto["survival"],
                    "det_near_bound": det["policy"]["near_bound_fraction"],
                    "sto_std": sto["policy"]["std"],
                }
            ),
            flush=True,
        )
    print(json.dumps({"wrote": str(output), "n": len(results["checkpoints"])}))


if __name__ == "__main__":
    main()
