"""Open-loop replay of a DiffAero success trajectory on Genesis FullQuad.

Phase record (diffaero-oracle): compare obs0, save one high-gate trajectory.
Phase replay (genesis): execute the same normalized actions from the same s0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSFER = PROJECT_ROOT / "scripts/eval/diffaero_actor_transfer.py"
DEFAULT_TRAJ = PROJECT_ROOT / "logs/race/diag_diffaero_success_traj.pt"
DEFAULT_OUT = PROJECT_ROOT / "logs/race/diag_openloop_replay.json"


def _load_transfer():
    spec = importlib.util.spec_from_file_location("diffaero_actor_transfer", TRANSFER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 3:4], q[..., :3]), dim=-1)


def _wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 1:], q[..., :1]), dim=-1)


def _quat_angle(q1_wxyz: torch.Tensor, q2_wxyz: torch.Tensor) -> torch.Tensor:
    dot = (q1_wxyz * q2_wxyz).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def record(output: Path) -> dict:
    xfer = _load_transfer()
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(xfer.DIFFAERO_PARENT))
    from omegaconf import OmegaConf

    from diffaero.env import build_env
    from genesis_drones.tasks.racing_core import racing_observations, track_to_tensors
    from genesis_drones.tasks.racing_core import load_evaluation_initial_states
    from genesis_drones.tasks.racing_tracks import RACING_TRACK

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = xfer.load_diffaero_actor(xfer.DEFAULT_ACTOR, device)
    states, checksum = load_evaluation_initial_states(xfer.DEFAULT_STATES)
    n = int(states.position.shape[0])
    cfg = OmegaConf.load(xfer.DEFAULT_HYDRA)
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
    OmegaConf.resolve(cfg)
    env = build_env(cfg.env, device)
    env.reset()
    pos = states.position.to(device)
    quat_wxyz = states.quaternion.to(device)
    vel = states.linear_velocity.to(device)
    env.dynamics._state = torch.cat((pos, _wxyz_to_xyzw(quat_wxyz), vel, torch.zeros_like(vel)), dim=-1)
    env.target_gates = states.target_gate.to(device=device, dtype=torch.int32)
    env.n_passed_gates.zero_()
    env.progress.zero_()
    env.last_action.zero_()
    env.max_vel.fill_(5.0)
    env.reset_idx = lambda _idx: None
    da_obs = env.get_observations()
    track = track_to_tensors(RACING_TRACK, device, torch.float32)
    gen_obs, _ = racing_observations(pos, quat_wxyz, vel, track, states.target_gate.to(device))
    obs_err = (da_obs - gen_obs).abs()
    per_dim = obs_err.max(0).values.cpu().tolist()
    per_gate = {}
    for gate in range(8):
        mask = states.target_gate.to(device) == gate
        if mask.any():
            per_gate[str(gate)] = float(obs_err[mask].max())
    env.reset_idx = lambda _idx: None
    alive = torch.ones(n, device=device, dtype=torch.bool)
    gates = torch.zeros(n, device=device)
    dt = float(cfg.env.dt)
    max_steps = int(cfg.env.max_time / dt) + 1
    with torch.no_grad():
        obs = da_obs
        for _ in range(max_steps):
            if not bool(alive.any()):
                break
            normed = xfer.det_action(actor, obs)
            physical = env.rescale_action(normed)
            obs, _, terminated, extra = env.step(physical)
            newly = extra["reset"] & alive
            if newly.any():
                gates[newly] = env.n_passed_gates[newly].to(gates.dtype)
            alive = alive & ~extra["reset"]
    best = int(gates.argmax().item())
    from diffaero.utils.randomizer import RandomizerManager

    RandomizerManager.randomizers = []
    cfg.n_envs = 1
    cfg.env.n_envs = 1
    cfg.dynamics.n_envs = 1
    if hasattr(cfg.env, "dynamics"):
        cfg.env.dynamics.n_envs = 1
    one = build_env(cfg.env, device)
    one.reset()
    p0 = states.position[best : best + 1].to(device)
    q0 = states.quaternion[best : best + 1].to(device)
    v0 = states.linear_velocity[best : best + 1].to(device)
    g0 = states.target_gate[best : best + 1].to(device=device, dtype=torch.int32)
    one.dynamics._state = torch.cat((p0, _wxyz_to_xyzw(q0), v0, torch.zeros_like(v0)), dim=-1)
    one.target_gates = g0
    one.n_passed_gates.zero_()
    one.progress.zero_()
    one.last_action.zero_()
    one.max_vel.fill_(5.0)
    one.reset_idx = lambda _idx: None
    log = {
        "position": [],
        "quaternion_wxyz": [],
        "lin_vel": [],
        "ang_vel": [],
        "obs": [],
        "action_norm": [],
        "action_physical": [],
        "thrust": [],
        "torque": [],
        "lin_acc": [],
        "ang_acc": [],
        "target_gate": [],
        "n_passed": [],
    }

    def snapshot():
        st = one.dynamics._state[0]
        log["position"].append(st[:3].detach().cpu())
        log["quaternion_wxyz"].append(_xyzw_to_wxyz(st[3:7]).detach().cpu())
        log["lin_vel"].append(st[7:10].detach().cpu())
        log["ang_vel"].append(st[10:13].detach().cpu())
        log["obs"].append(one.get_observations()[0].detach().cpu())
        log["target_gate"].append(int(one.target_gates[0].item()))
        log["n_passed"].append(int(one.n_passed_gates[0].item()))

    snapshot()
    done = False
    with torch.no_grad():
        for _ in range(max_steps):
            obs = one.get_observations()
            normed = xfer.det_action(actor, obs)
            physical = one.rescale_action(normed)
            q = one.dynamics._q
            w = one.dynamics._w
            thrust, torque = one.dynamics.controller(q, w, physical)
            log["action_norm"].append(normed[0].detach().cpu())
            log["action_physical"].append(physical[0].detach().cpu())
            log["thrust"].append(thrust[0].detach().cpu())
            log["torque"].append(torque[0].detach().cpu())
            _, _, terminated, extra = one.step(physical)
            log["lin_acc"].append(one.dynamics._acc[0].detach().cpu())
            log["ang_acc"].append(
                ((one.dynamics._state[0, 10:13] - w[0]) / dt).detach().cpu()
            )
            snapshot()
            if bool(extra["reset"][0]):
                done = True
                break
    packed = {key: torch.stack(val) if key not in ("target_gate", "n_passed") else torch.tensor(val) for key, val in log.items()}
    packed.update(
        {
            "best_index": best,
            "gates_at_done": int(gates[best].item()),
            "dt": dt,
            "checksum": checksum,
            "s0_position": p0.cpu(),
            "s0_quaternion": q0.cpu(),
            "s0_lin_vel": v0.cpu(),
            "s0_target_gate": g0.cpu(),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, output)
    summary = {
        "phase": "record",
        "best_index": best,
        "best_gates": int(gates[best].item()),
        "mean_gates_100": float(gates.mean()),
        "traj_steps": int(packed["action_norm"].shape[0]),
        "traj_final_gates": int(packed["n_passed"][-1].item()),
        "obs0_max_abs_err_per_dim": per_dim,
        "obs0_max_abs_err": float(obs_err.max()),
        "obs0_max_abs_err_per_gate": per_gate,
        "path": str(output),
    }
    print(json.dumps(summary, indent=2))
    return summary


def replay(traj_path: Path, output: Path) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT))
    import genesis as gs

    from genesis_drones.controllers.ctbr_controller import CtbrControllerConfig
    from genesis_drones.envs.race_env import RaceEnv, RaceEnvConfig
    from genesis_drones.evaluation.race import assert_shared_racing_contract
    from genesis_drones.tasks.racing_core import EvaluationInitialStates

    traj = torch.load(traj_path, map_location="cpu", weights_only=False)
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    config = RaceEnvConfig(
        dt=float(traj["dt"]),
        horizon=1,
        max_episode_steps=1201,
        min_target_velocity=5.0,
        max_target_velocity=5.0,
        dynamics="full_quad",
        controller=CtbrControllerConfig(randomize=False),
    )
    assert_shared_racing_contract(config)
    env = RaceEnv(config, num_envs=1, requires_grad=False)
    env.respawn_on_fail = False
    s0 = EvaluationInitialStates(
        traj["s0_position"],
        traj["s0_quaternion"],
        traj["s0_lin_vel"],
        traj["s0_target_gate"],
    )
    env.reset(initial_states=s0)
    dt = float(traj["dt"])
    n_steps = int(traj["action_norm"].shape[0])
    rows = []
    first_bad = None
    thresholds = {
        "position_m": 0.05,
        "lin_vel_mps": 0.2,
        "ang_vel_radps": 0.2,
        "attitude_deg": 5.0,
        "thrust_n": 0.5,
        "torque_nm": 0.05,
    }
    with torch.no_grad():
        for t in range(n_steps):
            state_before = env.env._read_state()
            action = traj["action_norm"][t : t + 1].to(env.device)
            control = env.env.plant.control(action, state_before, env.is_alive)
            obs, _, done, extras = env.step(action)
            state = env.env._read_state()
            da_pos = traj["position"][t + 1]
            da_quat = traj["quaternion_wxyz"][t + 1]
            da_v = traj["lin_vel"][t + 1]
            da_w = traj["ang_vel"][t + 1]
            pos_err = float((state.position[0].cpu() - da_pos).norm())
            vel_err = float((state.linear_velocity[0].cpu() - da_v).norm())
            # DiffAero w vs Genesis solver omega: compare both world and body-mapped later
            omega_err = float((state.angular_velocity[0].cpu() - da_w).norm())
            att_deg = float(_quat_angle(state.quaternion[0].cpu(), da_quat) * 180.0 / 3.14159265)
            thrust_g = float(control.wrench[0, 0].cpu())
            torque_g = control.wrench[0, 1:].cpu()
            thrust_err = abs(thrust_g - float(traj["thrust"][t]))
            torque_err = float((torque_g - traj["torque"][t]).norm())
            lin_acc_g = (state.linear_velocity[0] - state_before.linear_velocity[0]) / dt
            lin_acc_err = float((lin_acc_g.cpu() - traj["lin_acc"][t]).norm())
            row = {
                "t": t + 1,
                "position_err": pos_err,
                "lin_vel_err": vel_err,
                "ang_vel_err": omega_err,
                "attitude_deg": att_deg,
                "thrust_genesis": thrust_g,
                "thrust_diffaero": float(traj["thrust"][t]),
                "thrust_err": thrust_err,
                "torque_err": torque_err,
                "lin_acc_err": lin_acc_err,
                "genesis_pos": state.position[0].cpu().tolist(),
                "diffaero_pos": da_pos.tolist(),
                "genesis_gates": int(extras["n_passed_gates"][0].item()),
                "diffaero_gates": int(traj["n_passed"][t + 1].item()),
                "done": bool(done[0]),
            }
            rows.append(row)
            if first_bad is None:
                if pos_err > thresholds["position_m"]:
                    first_bad = ("position", t + 1, pos_err)
                elif att_deg > thresholds["attitude_deg"]:
                    first_bad = ("attitude", t + 1, att_deg)
                elif vel_err > thresholds["lin_vel_mps"]:
                    first_bad = ("lin_vel", t + 1, vel_err)
                elif omega_err > thresholds["ang_vel_radps"]:
                    first_bad = ("ang_vel", t + 1, omega_err)
                elif thrust_err > thresholds["thrust_n"]:
                    first_bad = ("thrust", t + 1, thrust_err)
                elif torque_err > thresholds["torque_nm"]:
                    first_bad = ("torque", t + 1, torque_err)
            if bool(done[0]):
                break
    checkpoints = [1, 5, 10, 30]
    report = {
        "phase": "replay",
        "traj": str(traj_path),
        "diffaero_gates": int(traj["gates_at_done"]),
        "genesis_openloop_gates": rows[-1]["genesis_gates"] if rows else 0,
        "steps_compared": len(rows),
        "first_divergence": None
        if first_bad is None
        else {"quantity": first_bad[0], "step": first_bad[1], "value": first_bad[2]},
        "checkpoints": [row for row in rows if row["t"] in checkpoints],
        "step1": rows[0] if rows else None,
        "thresholds": thresholds,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("record", "replay"), required=True)
    parser.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "record":
        record(args.traj)
        return
    replay(args.traj, args.output)


if __name__ == "__main__":
    main()
