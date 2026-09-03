"""Phase-2 smoke: train one-update + eval checkpoint= only."""
import json
import subprocess
import sys
from pathlib import Path

import torch
import genesis as gs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis_drones.algorithms.diff_rl import NetworkConfig, RunningNormalizer, ApgConfig, ShacConfig, make_diff_agent
from genesis_drones.adapters import DiffRLAdapter, RslRlAdapter
from genesis_drones.adapters.policy import DiffRLPolicyAdapter, CallablePolicyAdapter
from genesis_drones.experiment.builder import build_environment, build_training_stack, load_policy, run_spec_from_cfg
from genesis_drones.experiment.checkpoint import save_checkpoint
from genesis_drones.experiment.spec import RunSpec
from genesis_drones.evaluation.runner import EvaluationRunner
from genesis_drones.evaluation.race import evaluate_policy, make_evaluation_states, summarize_race_results
from genesis_drones.envs.track_diff_env import TrackDiffEnv
from genesis_drones.experiment.builder import tracking_env_config_from
from genesis_drones.evaluation.tracking import TrackScenarios, evaluate_diff_policy


def check(name, fn, results):
    try:
        fn()
        results.append((name, "PASS", ""))
        print("PASS", name)
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"[:240]))
        print("FAIL", name, e)


def main():
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")
    results = []
    out = ROOT / "logs" / "phase2_smoke"
    out.mkdir(parents=True, exist_ok=True)

    # Phase-1 train matrix (construct + one update)
    for dyn in ("native_quad", "full_quad"):
        for algo in ("ppo", "apg", "shac"):
            def race(d=dyn, a=algo):
                data = {
                    "task": "racing",
                    "dynamics": d,
                    "algorithm": a,
                    "sensor": "relative_position",
                    "seed": 0,
                    "updates": 1,
                    "save_interval": 1,
                    "network": {"hidden_sizes": [64, 64]},
                    "environment": {"dt": 0.0333, "horizon": 2, "max_episode_steps": 8, "gamma": 0.99, "td_lambda": 0.95},
                    "apg": {"horizon": 2, "learning_rate": 1e-3, "max_grad_norm": 1.0},
                    "shac": {"horizon": 2, "actor_learning_rate": 1e-3, "critic_learning_rate": 1e-3, "gamma": 0.99, "td_lambda": 0.95, "entropy_weight": 0.0, "actor_max_grad_norm": 1.0, "critic_max_grad_norm": 1.0, "critic_minibatches": 2, "critic_iterations": 1, "target_update_rate": 0.005, "log_standard_deviation_min": -5.0, "log_standard_deviation_max": -1.0},
                    "num_envs": {"ppo": 2, "apg": 2, "shac": 2},
                }
                spec = run_spec_from_cfg(data)
                log_dir = out / f"race_{d}_{a}"
                log_dir.mkdir(exist_ok=True)
                stack = build_training_stack(spec, data, 2, log_dir=log_dir)
                if a == "ppo":
                    from genesis_drones.experiment.train_loop import train_ppo_stack
                    train_ppo_stack(stack, spec, 1, log_dir)
                else:
                    from genesis_drones.experiment.train_loop import train_diff_stack
                    train_diff_stack(stack, spec, data, 1, 1, log_dir)
                assert (log_dir / "model.pt").exists() and (log_dir / "experiment.json").exists()
            check(f"train Racing+{dyn}+{algo}", race, results)

    for algo in ("ppo", "apg", "shac"):
        def track(a=algo):
            data = {
                "task": "tracking",
                "dynamics": "native_quad",
                "algorithm": a,
                "sensor": "state",
                "seed": 0,
                "updates": 1,
                "save_interval": 1,
                "network": {"hidden_sizes": [64, 64]},
                "environment": {"dt": 0.01, "horizon": 2, "max_episode_steps": 8},
                "apg": {"horizon": 2, "learning_rate": 1e-3, "max_grad_norm": 1.0},
                "shac": {"horizon": 2, "actor_learning_rate": 1e-3, "critic_learning_rate": 1e-3, "gamma": 0.99, "td_lambda": 0.95, "entropy_weight": 0.0, "actor_max_grad_norm": 1.0, "critic_max_grad_norm": 1.0, "critic_minibatches": 2, "critic_iterations": 1, "target_update_rate": 0.005, "log_standard_deviation_min": -5.0, "log_standard_deviation_max": -1.0},
                "num_envs": {"ppo": 2, "apg": 2, "shac": 2},
            }
            spec = run_spec_from_cfg(data)
            log_dir = out / f"track_native_{a}"
            log_dir.mkdir(exist_ok=True)
            stack = build_training_stack(spec, data, 2, log_dir=log_dir)
            if a == "ppo":
                from genesis_drones.experiment.train_loop import train_ppo_stack
                train_ppo_stack(stack, spec, 1, log_dir)
            else:
                from genesis_drones.experiment.train_loop import train_diff_stack
                train_diff_stack(stack, spec, data, 1, 1, log_dir)
            assert (log_dir / "model.pt").exists()
        check(f"train Tracking+native+{algo}", track, results)

    def track_full():
        data = {"task": "tracking", "dynamics": "full_quad", "algorithm": "ppo", "sensor": "state", "seed": 0, "environment": {"dt": 0.01, "horizon": 1}, "network": {"hidden_sizes": [64, 64]}}
        env = build_environment(run_spec_from_cfg(data), 2, requires_grad=False)
        step = env.step(env.hover_command())
        assert torch.isfinite(step.reward).all()
    check("Tracking+full construct/step", track_full, results)

    # Eval smokes using new checkpoints
    for task_dir, task in (("race_native_quad_apg", "racing"), ("race_native_quad_shac", "racing"), ("race_native_quad_ppo", "racing"), ("track_native_apg", "tracking"), ("track_native_shac", "tracking"), ("track_native_ppo", "tracking")):
        ckpt = out / task_dir / "model.pt"
        if not ckpt.exists():
            results.append((f"eval {task_dir}", "SKIP", "no ckpt"))
            continue
        def ev(c=ckpt, t=task, name=task_dir):
            payload = torch.load(c, map_location=gs.device, weights_only=False)
            from genesis_drones.experiment.checkpoint import extract_run_spec
            spec = extract_run_spec(c, payload)
            env = build_environment(spec, 2 if t == "tracking" else 1, requires_grad=False)
            policy, spec2, _ = load_policy(c, env, cfg=payload.get("config") or {})
            if t == "racing":
                states = make_evaluation_states(env, count=2, seed=0)
                metrics = summarize_race_results(evaluate_policy(env, lambda o: policy.act(o, True), states))
                assert "success_rate" in metrics
            else:
                cfg = tracking_env_config_from({"environment": spec.environment}, spec.dynamics)
                facade = TrackDiffEnv(cfg, num_envs=2, requires_grad=False)
                policy2, _, _ = load_policy(c, facade.env, cfg=payload.get("config") or {})
                scenarios = TrackScenarios.generate(2, 4, cfg, seed=0)
                # shorten loop: temporarily patch max steps via config is frozen — use EvaluationRunner briefly
                runner = EvaluationRunner(facade.env, policy2, max_steps=3)
                runner.run(reset_kwargs={"seed": None})
            # checkpoint-only identity
            assert spec2.task and spec2.algorithm and spec2.dynamics
        check(f"eval {task_dir}", ev, results)

    # Hydra eval CLI: checkpoint only (new format)
    ckpt = out / "race_native_quad_apg" / "model.pt"
    if ckpt.exists():
        def hydra_eval():
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "eval.py"),
                f"checkpoint={ckpt}",
                "eval.num_envs=1",
                f"eval.output={out / 'hydra_eval.json'}",
                "hydra.run.dir=.",
                "hydra.output_subdir=null",
                "hydra/job_logging=disabled",
                "hydra/hydra_logging=disabled",
            ]
            # Avoid CLI override clash: don't pass task/algorithm when checkpoint has them
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-1500:] or r.stdout[-1500:])
            assert (out / "hydra_eval.json").exists()
        check("eval.py checkpoint-only", hydra_eval, results)

    print("---SUMMARY---")
    for name, status, msg in results:
        print(status, name, msg)
    raise SystemExit(sum(1 for _, s, _ in results if s == "FAIL"))


if __name__ == "__main__":
    main()
