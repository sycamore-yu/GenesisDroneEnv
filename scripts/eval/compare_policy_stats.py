"""Same policy-action stats for Track PPO and Racing PPO checkpoints. No retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tensordict import TensorDict
from torch.distributions import Normal

import genesis as gs

from rsl_rl.modules import ActorCritic


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def summarize(raw: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, entropy: torch.Tensor) -> dict:
    clipped = raw.clamp(-1.0, 1.0)
    return {
        "mu_mean": [float(x) for x in mu.mean(dim=0).tolist()],
        "mu_abs_mean": [float(x) for x in mu.abs().mean(dim=0).tolist()],
        "mu_abs_max": [float(x) for x in mu.abs().amax(dim=0).tolist()],
        "std": [float(x) for x in std.mean(dim=0).tolist()],
        "raw_abs_mean": [float(x) for x in raw.abs().mean(dim=0).tolist()],
        "raw_abs_max": [float(x) for x in raw.abs().amax(dim=0).tolist()],
        "sat_frac_|raw|>1": [float(x) for x in (raw.abs() > 1.0).float().mean(dim=0).tolist()],
        "sat_frac_|raw|>1_any": float((raw.abs() > 1.0).any(dim=-1).float().mean().item()),
        "clipped_abs_mean": [float(x) for x in clipped.abs().mean(dim=0).tolist()],
        "clip_change_mean": float((clipped - raw).abs().mean().item()),
        "entropy": float(entropy.mean().item()),
        "n": int(raw.shape[0]),
    }


def accumulate(policy: ActorCritic, obs: TensorDict, rows: list) -> torch.Tensor:
    raw = policy.act(obs)
    mu = policy.action_mean
    std = policy.action_std
    entropy = Normal(mu, std).entropy().sum(dim=-1)
    rows.append((mu.detach(), std.detach(), raw.detach(), entropy.detach()))
    return raw


def stack_rows(rows: list) -> dict:
    mu = torch.cat([r[0] for r in rows], dim=0)
    std = torch.cat([r[1] for r in rows], dim=0)
    raw = torch.cat([r[2] for r in rows], dim=0)
    entropy = torch.cat([r[3] for r in rows], dim=0)
    return summarize(raw, mu, std, entropy)


def load_actor_critic(obs: TensorDict, obs_groups: dict, num_actions: int, policy_cfg: dict, payload: dict) -> ActorCritic:
    cfg = dict(policy_cfg)
    cfg.pop("class_name", None)
    policy = ActorCritic(obs, obs_groups, num_actions, **cfg)
    policy.load_state_dict(payload["model_state_dict"])
    policy.to(gs.device)
    policy.eval()
    return policy


def run_track(checkpoints: list[Path], steps: int) -> dict:
    with (PROJECT_ROOT / "config" / "track_rl" / "genesis_env.yaml").open() as file:
        env_config = yaml.safe_load(file)
    with (PROJECT_ROOT / "config" / "track_rl" / "rl_env.yaml").open() as file:
        rl_config = yaml.safe_load(file)
    with (PROJECT_ROOT / "config" / "track_rl" / "flight.yaml").open() as file:
        flight_config = yaml.safe_load(file)
    env_config.update({"show_viewer": False, "render_cam": False, "vis_waypoints": False, "use_FPV_camera": False})
    from genesis_drones.envs.genesis_env import Genesis_env
    from genesis_drones.tasks.track_task import Track_task

    genesis_env = Genesis_env(env_config=env_config, flight_config=flight_config, num_envs=32)
    task = Track_task(genesis_env, env_config, rl_config["train"], rl_config["task"], num_envs=32)
    obs = task.reset()
    policy_cfg = dict(rl_config["train"]["policy"])
    results = []
    policy = None
    for path in checkpoints:
        payload = torch.load(path, map_location=gs.device, weights_only=False)
        if policy is None:
            policy = load_actor_critic(obs, rl_config["train"]["obs_groups"], 4, policy_cfg, payload)
        else:
            policy.load_state_dict(payload["model_state_dict"])
            policy.eval()
        obs = task.reset()
        task._update_obs()
        obs = task.get_observations()
        rows = []
        with torch.no_grad():
            for _ in range(steps):
                raw = accumulate(policy, obs, rows)
                obs, _, _, _ = task.step(raw)
        results.append({"path": str(path), "iter": payload.get("iter"), **stack_rows(rows)})
        print(json.dumps({"task": "track", "iter": payload.get("iter"), "sat": results[-1]["sat_frac_|raw|>1_any"], "std": results[-1]["std"]}), flush=True)
    return {"task": "track", "obs_scales_in_yaml": rl_config["task"]["obs_scales"], "obs_scales_applied_in__update_obs": False, "checkpoints": results}


def run_race(checkpoints: list[Path], steps: int, dynamics: str) -> dict:
    from genesis_drones.controllers.ctbr_controller import CtbrControllerConfig
    from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig, detached_torch_tensor
    from genesis_drones.evaluation.race import apply_network_hidden_sizes

    with (PROJECT_ROOT / "config" / "race" / "train.yaml").open() as file:
        train_data = yaml.safe_load(file)
    cfg_name = "ppo_full_quad.yaml" if dynamics == "full_quad" else "ppo.yaml"
    with (PROJECT_ROOT / "config" / "race" / cfg_name).open() as file:
        ppo_cfg = yaml.safe_load(file)
    apply_network_hidden_sizes(ppo_cfg, train_data["network"]["hidden_sizes"])
    environment = RaceEnv(
        RaceEnvConfig(
            dt=train_data["environment"]["dt"],
            horizon=1,
            max_episode_steps=train_data["environment"]["max_episode_steps"],
            min_target_velocity=5.0,
            max_target_velocity=5.0,
            gamma=train_data["environment"]["gamma"],
            td_lambda=train_data["environment"]["td_lambda"],
            dynamics=dynamics,
            controller=CtbrControllerConfig(randomize=False),
        ),
        num_envs=32,
        requires_grad=False,
    )
    environment.respawn_on_fail = True
    dummy = TensorDict({"policy": torch.zeros(32, 13, device=gs.device)}, batch_size=32)
    policy_cfg = dict(ppo_cfg["policy"])
    results = []
    policy = None
    for path in checkpoints:
        payload = torch.load(path, map_location=gs.device, weights_only=False)
        if policy is None:
            policy = load_actor_critic(dummy, ppo_cfg["obs_groups"], 4, policy_cfg, payload)
        else:
            policy.load_state_dict(payload["model_state_dict"])
            policy.eval()
        observation, _ = environment.reset(seed=0)
        rows = []
        with torch.no_grad():
            for _ in range(steps):
                obs = TensorDict({"policy": detached_torch_tensor(observation)}, batch_size=32)
                raw = accumulate(policy, obs, rows)
                observation, _, _, _ = environment.step(raw)
        results.append({"path": str(path), "iter": payload.get("iter"), **stack_rows(rows)})
        print(json.dumps({"task": "race", "iter": payload.get("iter"), "sat": results[-1]["sat_frac_|raw|>1_any"], "std": results[-1]["std"]}), flush=True)
    return {"task": "race", "dynamics": dynamics, "checkpoints": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("track", "race"), required=True)
    parser.add_argument("--dynamics", choices=("native_quad", "full_quad"), default="native_quad")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    payload = run_track(args.checkpoints, args.steps) if args.task == "track" else run_race(args.checkpoints, args.steps, args.dynamics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"wrote": str(args.output)}))


if __name__ == "__main__":
    main()
