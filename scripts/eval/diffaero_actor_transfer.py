"""Frozen DiffAero Racing PPO actor closed-loop on Genesis / DiffAero.

Diagnostic only. Does not change official Racing benchmark or training YAML.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFAERO_ROOT = Path("/home/tong/tongworkspace/refer/diffaero")
DIFFAERO_PARENT = DIFFAERO_ROOT.parent
DEFAULT_ACTOR = Path(
    "/home/tong/tongworkspace/experiments/GenesisDroneEnv/diffaero_oracle/"
    "racing_quad_ppo_seed0/full/best/actor.pth"
)
DEFAULT_STATES = PROJECT_ROOT / "logs/race/ppo_full_quad_squash_seed0/eval_states_8gate_seed20250830.pt"
DEFAULT_HYDRA = Path(
    "/home/tong/tongworkspace/experiments/GenesisDroneEnv/diffaero_oracle/"
    "racing_quad_ppo_seed0/full/.hydra/config.yaml"
)
OBS_DIM = 13
ACTION_DIM = 4
HIDDEN = [256, 128]


class _NormedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ln(self.linear(x)))


class DiffAeroActorMean(nn.Module):
    """MLP [13] → [256] → [128] → [4] with LayerNorm+ELU, matching DiffAero StochasticActor."""

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            _NormedLinear(OBS_DIM, HIDDEN[0]),
            _NormedLinear(HIDDEN[0], HIDDEN[1]),
            nn.Linear(HIDDEN[1], ACTION_DIM),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(obs)


def load_diffaero_actor(path: Path, device: torch.device) -> DiffAeroActorMean:
    payload = torch.load(path, map_location=device, weights_only=False)
    actor = DiffAeroActorMean().to(device)
    actor.load_state_dict(payload["actor_mean"])
    actor.eval()
    return actor


def det_action(actor: DiffAeroActorMean, obs: torch.Tensor) -> torch.Tensor:
    return torch.tanh(actor(obs))


def _moments(x: torch.Tensor) -> dict:
    x = x.detach().float().reshape(-1)
    if x.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _histogram(values: torch.Tensor, max_bin: int = 12) -> dict[str, int]:
    counts = torch.bincount(values.clamp(0, max_bin).to(torch.int64), minlength=max_bin + 1)
    return {str(i): int(counts[i]) for i in range(max_bin + 1)}


def eval_genesis(actor_path: Path, states_path: Path, output: Path) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT))
    import genesis as gs

    from genesis_drones.controllers.ctbr_controller import CtbrControllerConfig
    from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
    from genesis_drones.evaluation.race import assert_shared_racing_contract
    from genesis_drones.tasks.racing_core import load_evaluation_initial_states

    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    device = gs.device
    actor = load_diffaero_actor(actor_path, device)
    states, checksum = load_evaluation_initial_states(states_path)
    n = int(states.position.shape[0])
    config = RaceEnvConfig(
        dt=0.0333,
        horizon=1,
        max_episode_steps=1201,
        min_target_velocity=5.0,
        max_target_velocity=5.0,
        gamma=0.99,
        td_lambda=0.95,
        dynamics="full_quad",
        controller=CtbrControllerConfig(randomize=False),
    )
    assert_shared_racing_contract(config)
    env = RaceEnv(config, num_envs=n, requires_grad=False)
    env.respawn_on_fail = False
    obs, _ = env.reset(initial_states=states)
    obs0 = obs.detach().clone()
    action0 = det_action(actor, obs0)
    alive = torch.ones(n, device=device, dtype=torch.bool)
    gates = torch.zeros(n, device=device)
    duration = torch.zeros(n, device=device)
    collision = torch.zeros(n, device=device, dtype=torch.bool)
    oob = torch.zeros(n, device=device, dtype=torch.bool)
    timeout = torch.zeros(n, device=device, dtype=torch.bool)
    success = torch.zeros(n, device=device, dtype=torch.bool)
    action_sum = torch.zeros(ACTION_DIM, device=device)
    action_sq = torch.zeros(ACTION_DIM, device=device)
    action_abs = torch.zeros(ACTION_DIM, device=device)
    n_alive_steps = 0
    max_steps = config.max_episode_steps + 1
    with torch.no_grad():
        for _ in range(max_steps):
            if not bool(alive.any()):
                break
            action = det_action(actor, obs)
            n_now = int(alive.sum())
            action_sum += action[alive].sum(0)
            action_sq += action[alive].square().sum(0)
            action_abs += action[alive].abs().sum(0)
            n_alive_steps += n_now
            obs, _, done, extras = env.step(action)
            newly = done & alive
            if newly.any():
                pos = env.env._read_state().position
                xy_oob = torch.any(pos[:, :2].abs() > 5.0, dim=-1)
                z_oob = pos[:, 2] > 7.0
                hit = extras["analytic_collision"].bool()
                time_up = extras["truncated"].bool() & ~xy_oob & ~z_oob & ~hit
                collision[newly] = hit[newly]
                oob[newly] = (xy_oob | z_oob)[newly] & ~hit[newly]
                timeout[newly] = time_up[newly]
                success[newly] = extras["success"].bool()[newly]
                gates[newly] = extras["n_passed_gates"][newly].to(gates.dtype)
                duration[newly] = extras["episode_length"][newly].to(duration.dtype) * config.dt
            alive = alive & ~done
    mean_action = (action_sum / max(n_alive_steps, 1)).tolist()
    std_action = torch.sqrt(action_sq / max(n_alive_steps, 1) - (action_sum / max(n_alive_steps, 1)).square()).tolist()
    report = {
        "sim": "genesis_full_quad",
        "actor": str(actor_path),
        "states": str(states_path),
        "checksum": checksum,
        "count": n,
        "mean_gates_passed": float(gates.mean()),
        "mean_gates_by_start_gate": {
            str(g): float(gates[states.target_gate.to(device) == g].mean())
            if int((states.target_gate.to(device) == g).sum())
            else None
            for g in range(8)
        },
        "n_by_start_gate": {str(g): int((states.target_gate.to(device) == g).sum()) for g in range(8)},
        "gates_histogram": _histogram(gates),
        "collision_rate": float(collision.float().mean()),
        "oob_rate": float(oob.float().mean()),
        "timeout_rate": float(timeout.float().mean()),
        "success_rate": float(success.float().mean()),
        "survive_rate": float((~collision).float().mean()),
        "mean_episode_duration_s": float(duration.mean()),
        "action0": _moments(action0),
        "action0_per_dim": [_moments(action0[:, i]) for i in range(ACTION_DIM)],
        "action_mean": mean_action,
        "action_std": std_action,
        "action_abs_mean": (action_abs / max(n_alive_steps, 1)).tolist(),
        "obs0": {f"dim_{i}": _moments(obs0[:, i]) for i in range(OBS_DIM)},
        "obs0_all": _moments(obs0),
        "hover_action": float(env.hover_command(1)[0, 0]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def eval_diffaero(actor_path: Path, states_path: Path, hydra_path: Path, output: Path) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(DIFFAERO_PARENT))
    from omegaconf import OmegaConf

    from diffaero.env import build_env
    from genesis_drones.tasks.racing_core import load_evaluation_initial_states

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = load_diffaero_actor(actor_path, device)
    states, checksum = load_evaluation_initial_states(states_path)
    n = int(states.position.shape[0])
    cfg = OmegaConf.load(hydra_path)
    OmegaConf.set_struct(cfg, False)
    cfg.n_envs = n
    cfg.n_agents = 1
    cfg.headless = True
    cfg.ref_path = None
    cfg.record_video = False
    cfg.env.n_obstacles = 0
    cfg.env.obstacles = {"walls": False, "ceiling": False}
    cfg.env.randomizer.enabled = False
    cfg.env.render.headless = True
    cfg.env.n_envs = n
    cfg.dynamics.n_envs = n
    cfg.dynamics.m.enabled = False
    cfg.dynamics.J.xy.enabled = False
    cfg.dynamics.J.z.enabled = False
    cfg.dynamics.D.xy.enabled = False
    cfg.dynamics.D.z.enabled = False
    OmegaConf.resolve(cfg)
    env = build_env(cfg.env, device)
    env.reset()
    pos = states.position.to(device)
    quat_wxyz = states.quaternion.to(device)
    q_xyzw = torch.cat((quat_wxyz[..., 1:], quat_wxyz[..., :1]), dim=-1)
    vel = states.linear_velocity.to(device)
    env.dynamics._state = torch.cat((pos, q_xyzw, vel, torch.zeros_like(vel)), dim=-1)
    env.target_gates = states.target_gate.to(device=device, dtype=torch.int32)
    env.n_passed_gates.zero_()
    env.progress.zero_()
    env.last_action.zero_()
    env.max_vel.fill_(5.0)
    env.reset_idx = lambda _idx: None
    obs = env.get_observations()
    obs0 = obs.detach().clone()
    action0_norm = det_action(actor, obs0)
    alive = torch.ones(n, device=device, dtype=torch.bool)
    gates = torch.zeros(n, device=device)
    duration = torch.zeros(n, device=device)
    collision = torch.zeros(n, device=device, dtype=torch.bool)
    oob = torch.zeros(n, device=device, dtype=torch.bool)
    timeout = torch.zeros(n, device=device, dtype=torch.bool)
    success = torch.zeros(n, device=device, dtype=torch.bool)
    action_sum = torch.zeros(ACTION_DIM, device=device)
    n_alive_steps = 0
    dt = float(cfg.env.dt)
    max_steps = int(cfg.env.max_time / dt) + 1
    with torch.no_grad():
        for _ in range(max_steps):
            if not bool(alive.any()):
                break
            normed = det_action(actor, obs)
            n_now = int(alive.sum())
            action_sum += normed[alive].sum(0)
            n_alive_steps += n_now
            physical = env.rescale_action(normed)
            obs, _, terminated, extra = env.step(physical)
            done = extra["reset"]
            newly = done & alive
            if newly.any():
                pos_now = env.p
                xy_oob = torch.any(pos_now[:, :2].abs() > 5.0, dim=-1)
                z_oob = pos_now[:, 2] > 7.0
                hit = terminated.bool()
                time_up = extra["truncated"].bool() & ~xy_oob & ~z_oob
                collision[newly] = hit[newly]
                oob[newly] = (xy_oob | z_oob)[newly] & ~hit[newly]
                timeout[newly] = time_up[newly] & ~hit[newly]
                success[newly] = extra["success"].bool()[newly]
                gates[newly] = env.n_passed_gates[newly].to(gates.dtype)
                duration[newly] = extra["l"][newly].to(duration.dtype) * dt
            alive = alive & ~done
    report = {
        "sim": "diffaero_quadrotor",
        "actor": str(actor_path),
        "states": str(states_path),
        "checksum": checksum,
        "count": n,
        "mean_gates_passed": float(gates.mean()),
        "gates_histogram": _histogram(gates),
        "collision_rate": float(collision.float().mean()),
        "oob_rate": float(oob.float().mean()),
        "timeout_rate": float(timeout.float().mean()),
        "success_rate": float(success.float().mean()),
        "survive_rate": float((~collision).float().mean()),
        "mean_episode_duration_s": float(duration.mean()),
        "action0": _moments(action0_norm),
        "action0_per_dim": [_moments(action0_norm[:, i]) for i in range(ACTION_DIM)],
        "action_mean": (action_sum / max(n_alive_steps, 1)).tolist(),
        "obs0": {f"dim_{i}": _moments(obs0[:, i]) for i in range(OBS_DIM)},
        "obs0_all": _moments(obs0),
        "rescale_min": env.dynamics.min_action.detach().cpu().tolist(),
        "rescale_max": env.dynamics.max_action.detach().cpu().tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=("genesis", "diffaero"), required=True)
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--hydra", type=Path, default=DEFAULT_HYDRA)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sim == "genesis":
        out = args.output or PROJECT_ROOT / "logs/race/diag_diffaero_actor_on_genesis.json"
        eval_genesis(args.actor, args.states, out)
        return
    out = args.output or PROJECT_ROOT / "logs/race/diag_diffaero_actor_on_diffaero.json"
    eval_diffaero(args.actor, args.states, args.hydra, out)


if __name__ == "__main__":
    main()
