"""Fixed-state deterministic vs stochastic eval for RSL-RL 5.5 MLPModel actors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tensordict import TensorDict

import genesis as gs

from genesis_drones.controllers.ctbr_controller import CtbrControllerConfig
from genesis_drones.envs.genesis_task_env import detached_torch_tensor
from genesis_drones.envs.race_env import RaceEnvConfig, make_racing_env
from genesis_drones.evaluation.race import apply_network_hidden_sizes, make_evaluation_states
from genesis_drones.experiment.builder import PROJECT_ROOT, build_ppo_actor
from genesis_drones.tasks.racing_core import load_evaluation_initial_states, save_evaluation_initial_states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--states", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "race" / "train.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-states", type=int, default=100)
    parser.add_argument("--latest", action="store_true")
    return parser.parse_args()


def load_actor(payload: dict, train_config: dict, action_dim: int, obs_dim: int, device):
    actor = build_ppo_actor(train_config, obs_dim, action_dim, device)
    actor.load_state_dict(payload["actor_state_dict"])
    actor.to(device)
    actor.eval()
    return actor


def evaluate(environment, actor, states, deterministic: bool) -> dict:
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
    sums = {
        "mu_abs": torch.zeros(4, device=device),
        "std": torch.zeros(4, device=device),
        "raw_abs": torch.zeros(4, device=device),
        "raw_gt1": torch.tensor(0.0, device=device),
        "near_bound": torch.tensor(0.0, device=device),
        "clip_change": torch.tensor(0.0, device=device),
    }
    max_steps = environment.task.config.max_episode_steps + 2
    with torch.no_grad():
        for _ in range(max_steps):
            if not alive.any():
                break
            packed = TensorDict(
                {"policy": detached_torch_tensor(observation)},
                batch_size=[n_envs],
            )
            latent = actor.get_latent(packed)
            mlp_out = actor.mlp(latent)
            actor.distribution.update(mlp_out)
            mu = actor.distribution.mean
            std = actor.distribution.std
            raw = mu if deterministic else actor.distribution.sample()
            clipped = raw.clamp(-1.0, 1.0)
            mask = alive.float()[:, None]
            n = float(alive.sum().item())
            n_alive += n
            sums["mu_abs"] += (mu.abs() * mask).sum(dim=0)
            sums["std"] += (std.expand_as(mu) * mask).sum(dim=0)
            sums["raw_abs"] += (raw.abs() * mask).sum(dim=0)
            sums["raw_gt1"] += ((raw.abs() > 1.0).float() * mask).sum()
            sums["near_bound"] += ((clipped.abs() > 0.99).float() * mask).sum()
            sums["clip_change"] += ((clipped - raw).abs() * mask).sum()
            step = environment.step(raw)
            observation = step.observation.policy
            episode_return = episode_return + step.reward * alive
            finished = step.done & alive
            if finished.any():
                passed[finished] = step.extras["n_passed_gates"][finished].float()
                success[finished] = step.extras["success"][finished]
                truncated[finished] = step.extras["truncated"][finished]
                terminated[finished] = step.extras["terminated"][finished]
                duration[finished] = (step.extras["episode_length"][finished] - 1).float() * environment.dt
            alive = alive & ~step.done
    scale = max(n_alive, 1.0)
    return {
        "deterministic": deterministic,
        "mean_passed_gates": float(passed.mean().item()),
        "survival": float(truncated.float().mean().item()),
        "timeout_success": float(success.float().mean().item()),
        "collision": float(terminated.float().mean().item()),
        "mean_episode_time": float(duration.mean().item()),
        "mean_episode_return": float(episode_return.mean().item()),
        "n_envs": n_envs,
        "n_still_alive": int(alive.sum().item()),
        "policy": {
            "mu_abs_mean": [float(x) for x in (sums["mu_abs"] / scale).tolist()],
            "std": [float(x) for x in (sums["std"] / scale).tolist()],
            "raw_abs_mean": [float(x) for x in (sums["raw_abs"] / scale).tolist()],
            "raw_gt1_fraction": float((sums["raw_gt1"] / (scale * 4.0)).item()),
            "near_bound_fraction": float((sums["near_bound"] / (scale * 4.0)).item()),
            "env_clip_change_mean": float((sums["clip_change"] / (scale * 4.0)).item()),
        },
    }


def load_states(environment, path: Path | None, n_states: int, log_dir: Path):
    if path is not None and path.exists():
        try:
            states, checksum = load_evaluation_initial_states(path)
            if int(states.position.shape[0]) == n_states and hasattr(states, "target_gate"):
                return states, checksum, str(path)
        except (KeyError, ValueError, FileNotFoundError):
            pass
    states = make_evaluation_states(environment, count=n_states, seed=20250830)
    used = log_dir / "eval_states_8gate_seed20250830.pt"
    checksum = save_evaluation_initial_states(states, used)
    return states, checksum, f"generated:{used}"


def evaluate_log_dir(log_dir: Path, output: Path | None = None, states_path: Path | None = None, n_states: int = 100, latest: bool = False) -> dict:
    with (PROJECT_ROOT / "config/race/train.yaml").open() as file:
        train_data = yaml.safe_load(file)
    saved = log_dir / "train_config.yaml"
    if saved.exists():
        with saved.open() as file:
            train_config = yaml.safe_load(file)
    else:
        with (PROJECT_ROOT / "config/race/ppo_full_quad.yaml").open() as file:
            train_config = yaml.safe_load(file)
        probes_path = log_dir / "train_probes.json"
        if probes_path.exists():
            probes = json.loads(probes_path.read_text())
            if probes.get("actor_class"):
                train_config["actor"]["class_name"] = probes["actor_class"]
            if probes.get("critic_class"):
                train_config["critic"]["class_name"] = probes["critic_class"]
    apply_network_hidden_sizes(train_config, train_data["network"]["hidden_sizes"])
    environment = make_racing_env(
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
    states, checksum, states_source = load_states(environment, states_path, n_states, log_dir)
    checkpoints = []
    init_path = log_dir / "model_init.pt"
    if init_path.exists() and not latest:
        checkpoints.append((-1, init_path))
    wanted = {50, 100, 150, 199, 200, 300, 350, 400, 499, 500, 999, 1000}
    numeric = []
    for path in log_dir.glob("model_*.pt"):
        suffix = path.stem.split("_")[1]
        if suffix.isdigit():
            numeric.append((int(suffix), path))
    if latest:
        if not numeric:
            final = log_dir / "model.pt"
            if not final.exists():
                raise FileNotFoundError(f"no checkpoint in {log_dir}")
            checkpoints.append((0, final))
        else:
            checkpoints.append(max(numeric, key=lambda item: item[0]))
    else:
        checkpoints.extend(item for item in numeric if item[0] in wanted)
        checkpoints.sort(key=lambda item: 0 if item[0] < 0 else item[0])
    results = {"checksum": checksum, "states_source": states_source, "n_states": n_states, "checkpoints": []}
    dest = output or log_dir / "det_sto_eval.json"
    actor = None
    for iteration, path in checkpoints:
        payload = torch.load(path, map_location=gs.device, weights_only=False)
        if actor is None:
            actor = load_actor(
                payload,
                train_config,
                environment.action_dim,
                environment.task.policy_observation_dim,
                gs.device,
            )
        else:
            actor.load_state_dict(payload["actor_state_dict"])
            actor.eval()
        label = 0 if iteration < 0 else iteration
        torch.manual_seed(0)
        det = evaluate(environment, actor, states, deterministic=True)
        torch.manual_seed(0)
        sto = evaluate(environment, actor, states, deterministic=False)
        row = {"iteration": label, "path": str(path), "deterministic": det, "stochastic": sto}
        results["checkpoints"].append(row)
        dest.write_text(json.dumps(results, indent=2))
        print(
            json.dumps(
                {
                    "iteration": label,
                    "det_gates": det["mean_passed_gates"],
                    "sto_gates": sto["mean_passed_gates"],
                    "det_collision": det["collision"],
                    "sto_collision": sto["collision"],
                    "raw_gt1_det": det["policy"]["raw_gt1_fraction"],
                    "raw_gt1_sto": sto["policy"]["raw_gt1_fraction"],
                }
            ),
            flush=True,
        )
    print(json.dumps({"wrote": str(dest), "n": len(results["checkpoints"])}))
    return results


def main() -> None:
    args = parse_args()
    if not gs._initialized:
        gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    evaluate_log_dir(args.log_dir, args.output, args.states, args.n_states, latest=args.latest)


if __name__ == "__main__":
    main()
