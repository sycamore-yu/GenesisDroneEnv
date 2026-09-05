"""Orchestrate Racing PPO learning-rate trials. Does not implement PPO."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PPO_YAML = PROJECT_ROOT / "config" / "race" / "ppo_full_quad.yaml"
SOURCE_TRAIN_YAML = PROJECT_ROOT / "config" / "race" / "train.yaml"
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "logs" / "tuning" / "racing_ppo_opt55"
DEFAULT_EVAL_STATES = PROJECT_ROOT / "logs" / "race" / "eval_states_8gate_seed20250830.pt"
DESIRED_KL = 0.01
LR_FLOOR = 1.0e-5
BASELINE_DET_200 = 1.38
STABILITY_GRID = (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3)
LEARNABILITY_UPDATES = 150
TB_TAGS = {
    "lr": ("Loss/learning_rate",),
    "kl": ("Loss/kl",),
    "mu": ("Loss/mu_abs_mean",),
    "std": ("Policy/mean_std", "Policy/mean_noise_std"),
    "sat": ("Loss/sat_frac", "Policy/sat_frac"),
    "entropy": ("Loss/entropy",),
    "policy_loss": ("Loss/surrogate",),
    "value_loss": ("Loss/value", "Loss/value_function"),
    "gates": ("Episode/n_passed_gates",),
}
REQUIRED = ("lr", "std", "policy_loss", "value_loss")
ENGINE_MARKERS = (
    "ModuleNotFoundError",
    "ImportError",
    "AttributeError",
    "FileNotFoundError",
    "YAML",
    "gs.init",
    "TypeError",
    "No such file",
    "missing TensorBoard",
)


class EngineeringError(RuntimeError):
    """Shared tooling/code error. Stop the study; this is not a bad hyperparameter."""


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def source_ppo_digest() -> str:
    import hashlib

    return hashlib.sha256(SOURCE_PPO_YAML.read_bytes()).hexdigest()


def write_trial_ppo_config(
    learning_rate: float,
    dest: Path,
    *,
    desired_kl: float | None = None,
    entropy_coef: float | None = None,
    init_std: float | None = None,
) -> Path:
    with SOURCE_PPO_YAML.open() as file:
        data = yaml.safe_load(file)
    data["algorithm"]["learning_rate"] = float(learning_rate)
    if desired_kl is not None:
        data["algorithm"]["desired_kl"] = float(desired_kl)
    if entropy_coef is not None:
        data["algorithm"]["entropy_coef"] = float(entropy_coef)
    if init_std is not None:
        data["actor"]["distribution_cfg"]["init_std"] = float(init_std)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(data, sort_keys=False))
    return dest


def build_train_command(
    *,
    dynamics: str,
    log_dir: Path,
    ppo_config: Path,
    learning_rate: float,
    seed: int,
    updates: int,
    save_interval: int,
    num_envs: int | None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train" / "race_train.py"),
        "--algo",
        "ppo",
        "--dynamics",
        dynamics,
        "--seed",
        str(seed),
        "--updates",
        str(updates),
        "--save-interval",
        str(save_interval),
        "--log-dir",
        str(log_dir),
        "--ppo-config",
        str(ppo_config),
        "--learning-rate",
        str(learning_rate),
    ]
    if num_envs is not None:
        command.extend(["--num-envs", str(num_envs)])
    return command


def build_eval_command(checkpoint: Path, output: Path, states: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval" / "race_eval.py"),
        "--algo",
        "ppo",
        "--dynamics",
        "full_quad",
        "--checkpoint",
        str(checkpoint),
        "--states",
        str(states),
        "--output",
        str(output),
    ]


def read_tensorboard_scalars(log_dir: Path) -> dict[str, list[float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = sorted(log_dir.glob("events.out.tfevents.*"))
    if not events:
        raise EngineeringError(f"missing TensorBoard events in {log_dir}")
    accumulator = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    series: dict[str, list[float]] = {}
    missing = []
    for name, candidates in TB_TAGS.items():
        hit = next((tag for tag in candidates if tag in tags), None)
        if hit is None:
            if name in REQUIRED:
                missing.append(name)
            else:
                series[name] = []
            continue
        series[name] = [float(event.value) for event in accumulator.Scalars(hit)]
    if missing:
        raise EngineeringError(f"missing TensorBoard tags {missing} in {log_dir}")
    return series


def summarize_series(series: dict[str, list[float]], initial_lr: float) -> dict:
    def _get(name: str, index: int) -> float | None:
        values = series.get(name) or []
        if not values:
            return None
        return float(values[index])

    actual_lr_0 = _get("lr", 0)
    first_kl = _get("kl", 0)
    std_vals = series.get("std") or []
    mu_vals = series.get("mu") or []
    sat = series.get("sat") or []
    entropy = series.get("entropy") or []
    gates = series.get("gates") or []
    finite = all(
        math.isfinite(value)
        for name, values in series.items()
        if name not in {"sat", "gates"}
        for value in values
    )
    crushed = (
        actual_lr_0 is not None
        and initial_lr > 2.0 * LR_FLOOR
        and actual_lr_0 <= LR_FLOOR * 1.0001
    )
    return {
        "initial_learning_rate": float(initial_lr),
        "actual_learning_rate_0": actual_lr_0,
        "actual_learning_rate_last": _get("lr", -1),
        "actual_learning_rate_max": max(series["lr"]) if series.get("lr") else None,
        "kl_0": first_kl,
        "kl_last": _get("kl", -1),
        "mu_0": _get("mu", 0),
        "mu_last": _get("mu", -1),
        "mu_max": max(mu_vals) if mu_vals else None,
        "std_0": _get("std", 0),
        "std_last": _get("std", -1),
        "std_max": max(std_vals) if std_vals else None,
        "entropy_last": _get("entropy", -1),
        "train_gates_last": _get("gates", -1),
        "sat_0": sat[0] if sat else None,
        "sat_last": sat[-1] if sat else None,
        "sat_max": max(sat) if sat else None,
        "policy_loss_0": _get("policy_loss", 0),
        "value_loss_0": _get("value_loss", 0),
        "n_updates": len(series.get("lr") or []),
        "finite": finite,
        "scheduler_crushed": crushed,
        "lr_series": series.get("lr") or [],
        "kl_series": series.get("kl") or [],
        "std_series": std_vals,
        "entropy_series": entropy,
        "gates_series": gates,
    }


def prune_reason(summary: dict) -> str | None:
    if not summary["finite"]:
        return "non_finite"
    kl_0 = summary.get("kl_0")
    if kl_0 is not None and kl_0 > 1.0:
        return "kl_jump"
    std_max = summary.get("std_max")
    if std_max is not None and std_max > 10.0:
        return "std_explode"
    mu_max = summary.get("mu_max")
    if mu_max is not None and mu_max > 20.0:
        return "mu_explode"
    return None


def screening_score(summary: dict) -> float:
    kl_0 = summary.get("kl_0")
    if kl_0 is None:
        kl_0 = DESIRED_KL
    kl_term = abs(math.log(max(kl_0, 1.0e-12) / DESIRED_KL))
    crushed = 2.0 if summary["scheduler_crushed"] else 0.0
    return kl_term + crushed


def is_engineering_failure(text: str) -> bool:
    return any(marker in text for marker in ENGINE_MARKERS)


def latest_checkpoint(log_dir: Path) -> Path:
    numbered = []
    for path in log_dir.glob("model_*.pt"):
        suffix = path.stem.split("_")[1]
        if suffix.isdigit():
            numbered.append((int(suffix), path))
    if numbered:
        return max(numbered)[1]
    final = log_dir / "model.pt"
    if final.exists():
        return final
    raise EngineeringError(f"no checkpoint in {log_dir}")


def build_det_sto_eval_command(log_dir: Path, output: Path, states: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval" / "race_rslrl55_curve.py"),
        "--log-dir",
        str(log_dir),
        "--output",
        str(output),
        "--states",
        str(states),
        "--latest",
    ]


def run_command(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def stage1_objective(trial, state: _StudyState):
    import optuna

    learning_rate = trial.suggest_float("learning_rate", 3.0e-5, 3.0e-3, log=True)
    desired_kl = trial.suggest_float("desired_kl", 0.005, 0.03, log=True)
    entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.02)
    init_std = trial.suggest_float("init_std", 0.10, 0.50)
    run_dir = state.storage_dir / "runs" / f"{state.phase}_{trial.number}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ppo_config = write_trial_ppo_config(
        learning_rate,
        run_dir / "ppo.yaml",
        desired_kl=desired_kl,
        entropy_coef=entropy_coef,
        init_std=init_std,
    )
    command = build_train_command(
        dynamics=state.args.dynamics,
        log_dir=run_dir,
        ppo_config=ppo_config,
        learning_rate=learning_rate,
        seed=state.args.seed,
        updates=state.args.updates,
        save_interval=max(state.args.updates, 1) if state.args.updates < 50 else 50,
        num_envs=state.args.num_envs,
    )
    (run_dir / "train_command.json").write_text(json.dumps(command, indent=2))
    result = run_command(command, timeout=state.timeout)
    trial.set_user_attr("run_directory", str(run_dir))
    trial.set_user_attr("git_commit", state.git)
    trial.set_user_attr("seed", state.args.seed)
    trial.set_user_attr("phase", state.phase)
    trial.set_user_attr("exit_code", int(result.returncode))
    trial.set_user_attr("learning_rate", learning_rate)
    trial.set_user_attr("desired_kl", desired_kl)
    trial.set_user_attr("entropy_coef", entropy_coef)
    trial.set_user_attr("init_std", init_std)
    if result.returncode != 0:
        crashed = "nan" in ((result.stderr or "") + (result.stdout or "")).lower()
        trial.set_user_attr("nan_or_crash", True)
        _record_failure(trial, run_dir, result, state)
        raise optuna.TrialPruned("nan" if crashed else f"train_exit_{result.returncode}")
    series = read_tensorboard_scalars(run_dir)
    summary = summarize_series(series, learning_rate)
    reason = prune_reason(summary)
    summary["prune_reason"] = reason
    summary["params"] = {
        "learning_rate": learning_rate,
        "desired_kl": desired_kl,
        "entropy_coef": entropy_coef,
        "init_std": init_std,
    }
    if reason:
        summary["nan_or_crash"] = reason == "non_finite"
        (run_dir / "trial.json").write_text(json.dumps(summary, indent=2))
        trial.set_user_attr("fail_reason", reason)
        trial.set_user_attr("std_last", summary.get("std_last"))
        trial.set_user_attr("kl_last", summary.get("kl_last"))
        raise optuna.TrialPruned(reason)
    if not state.args.eval_states.exists():
        raise EngineeringError(f"missing eval states {state.args.eval_states}")
    eval_path = run_dir / "det_sto_eval.json"
    eval_result = run_command(
        build_det_sto_eval_command(run_dir, eval_path, state.args.eval_states),
        timeout=1800,
    )
    if eval_result.returncode != 0:
        _record_failure(trial, run_dir, eval_result, state)
        raise optuna.TrialPruned(f"eval_exit_{eval_result.returncode}")
    eval_metrics = json.loads(eval_path.read_text())
    row = eval_metrics["checkpoints"][-1]
    det = row["deterministic"]
    sto = row["stochastic"]
    det_gates = float(det["mean_passed_gates"])
    sto_gates = float(sto["mean_passed_gates"])
    summary["deterministic_eval"] = det
    summary["stochastic_eval"] = sto
    summary["det_gates"] = det_gates
    summary["sto_gates"] = sto_gates
    (run_dir / "trial.json").write_text(json.dumps(summary, indent=2))
    trial.set_user_attr("det_gates", det_gates)
    trial.set_user_attr("sto_gates", sto_gates)
    trial.set_user_attr("collision", det.get("collision"))
    trial.set_user_attr("episode_duration", det.get("mean_episode_time"))
    trial.set_user_attr("return", det.get("mean_episode_return"))
    trial.set_user_attr("std_last", summary.get("std_last"))
    trial.set_user_attr("entropy_last", summary.get("entropy_last"))
    trial.set_user_attr("lr_last", summary.get("actual_learning_rate_last"))
    trial.set_user_attr("kl_last", summary.get("kl_last"))
    trial.set_user_attr("raw_gt1_det", det.get("policy", {}).get("raw_gt1_fraction"))
    trial.set_user_attr("raw_gt1_sto", sto.get("policy", {}).get("raw_gt1_fraction"))
    trial.set_user_attr("nan_or_crash", False)
    return det_gates


def run_stage1(args: argparse.Namespace, storage: str, storage_dir: Path) -> None:
    study = load_study(args.study_name, storage, "maximize", args.seed)
    if not any(trial.params for trial in study.trials):
        study.enqueue_trial(
            {
                "learning_rate": 3.0e-4,
                "desired_kl": 0.01,
                "entropy_coef": 0.01,
                "init_std": 0.2231301601,
            }
        )
        study.enqueue_trial(
            {
                "learning_rate": 3.0e-4,
                "desired_kl": 0.01,
                "entropy_coef": 0.0,
                "init_std": 0.2231301601,
            }
        )
    state = _StudyState(args, storage_dir, "stage1", timeout=2400)
    n = remaining_trials(study, args.n_trials)
    if n:
        study.optimize(lambda trial: stage1_objective(trial, state), n_trials=n, catch=())
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE" and trial.value is not None]
    best = max(complete, key=lambda trial: trial.value) if complete else None
    print(
        json.dumps(
            {
                "phase": "stage1",
                "n_complete": len(complete),
                "best_det_gates": None if best is None else best.value,
                "best_params": None if best is None else best.params,
                "best_dir": None if best is None else best.user_attrs.get("run_directory"),
            },
            indent=2,
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamics", default="full_quad", choices=("full_quad", "native_quad"))
    parser.add_argument("--study-name", default="racing-full-quad-opt55")
    parser.add_argument("--storage", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument(
        "--phase",
        choices=("stage1", "stability", "learnability", "train", "all"),
        default="stage1",
    )
    parser.add_argument("--lrs", type=str, help="comma-separated LRs for learnability/train")
    parser.add_argument("--eval-states", type=Path, default=DEFAULT_EVAL_STATES)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args(argv)


def load_study(name: str, storage: str, direction: str, seed: int):
    import optuna

    sampler = optuna.samplers.TPESampler(seed=seed)
    return optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=sampler,
    )


def remaining_trials(study, n_trials: int) -> int:
    done = sum(trial.state.name in {"COMPLETE", "PRUNED"} for trial in study.trials)
    return max(0, n_trials - done)


class _StudyState:
    def __init__(self, args: argparse.Namespace, storage_dir: Path, phase: str, timeout: int):
        self.args = args
        self.storage_dir = storage_dir
        self.phase = phase
        self.timeout = timeout
        self.engine_fails: list[str] = []
        self.git = git_commit()
        self.source_digest = source_ppo_digest()


def _record_failure(trial, run_dir: Path, result: subprocess.CompletedProcess, state: _StudyState) -> None:
    text = (result.stderr or "") + "\n" + (result.stdout or "")
    trial.set_user_attr("run_directory", str(run_dir))
    trial.set_user_attr("exit_code", int(result.returncode))
    trial.set_user_attr("fail_reason", text[-4000:])
    trial.set_user_attr("git_commit", state.git)
    trial.set_user_attr("phase", state.phase)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fail.txt").write_text(text[-8000:])
    if is_engineering_failure(text):
        state.engine_fails.append(text[-500:])
        raise EngineeringError(text[-2000:])


def stability_objective(trial, state: _StudyState):
    import optuna

    learning_rate = trial.suggest_float("learning_rate", 1.0e-5, 1.0e-3, log=True)
    run_dir = state.storage_dir / "runs" / f"{state.phase}_{trial.number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ppo_config = write_trial_ppo_config(learning_rate, run_dir / "ppo.yaml")
    if source_ppo_digest() != state.source_digest:
        raise EngineeringError("source ppo yaml changed during tuning")
    command = build_train_command(
        dynamics=state.args.dynamics,
        log_dir=run_dir,
        ppo_config=ppo_config,
        learning_rate=learning_rate,
        seed=state.args.seed,
        updates=state.args.updates,
        save_interval=max(state.args.updates, 1),
        num_envs=state.args.num_envs,
    )
    (run_dir / "train_command.json").write_text(json.dumps(command, indent=2))
    result = run_command(command, timeout=state.timeout)
    trial.set_user_attr("run_directory", str(run_dir))
    trial.set_user_attr("git_commit", state.git)
    trial.set_user_attr("seed", state.args.seed)
    trial.set_user_attr("phase", state.phase)
    trial.set_user_attr("exit_code", int(result.returncode))
    if result.returncode != 0:
        _record_failure(trial, run_dir, result, state)
        raise optuna.TrialPruned(f"train_exit_{result.returncode}")
    series = read_tensorboard_scalars(run_dir)
    summary = summarize_series(series, learning_rate)
    reason = prune_reason(summary)
    summary["prune_reason"] = reason
    summary["screening_score"] = None if reason else screening_score(summary)
    summary["command"] = command
    (run_dir / "trial.json").write_text(json.dumps(summary, indent=2))
    for key in (
        "kl_0",
        "actual_learning_rate_0",
        "actual_learning_rate_last",
        "mu_max",
        "std_max",
        "sat_last",
        "scheduler_crushed",
        "prune_reason",
    ):
        trial.set_user_attr(key, summary[key])
    if reason:
        trial.set_user_attr("fail_reason", reason)
        raise optuna.TrialPruned(reason)
    return screening_score(summary)


def learnability_objective(trial, state: _StudyState, space: tuple[float, ...]):
    import optuna

    learning_rate = trial.suggest_categorical("learning_rate", space)
    run_dir = state.storage_dir / "runs" / f"{state.phase}_{trial.number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ppo_config = write_trial_ppo_config(learning_rate, run_dir / "ppo.yaml")
    command = build_train_command(
        dynamics=state.args.dynamics,
        log_dir=run_dir,
        ppo_config=ppo_config,
        learning_rate=learning_rate,
        seed=state.args.seed,
        updates=state.args.updates,
        save_interval=50,
        num_envs=state.args.num_envs,
    )
    (run_dir / "train_command.json").write_text(json.dumps(command, indent=2))
    result = run_command(command, timeout=state.timeout)
    trial.set_user_attr("run_directory", str(run_dir))
    trial.set_user_attr("git_commit", state.git)
    trial.set_user_attr("seed", state.args.seed)
    trial.set_user_attr("phase", state.phase)
    trial.set_user_attr("exit_code", int(result.returncode))
    if result.returncode != 0:
        _record_failure(trial, run_dir, result, state)
        raise optuna.TrialPruned(f"train_exit_{result.returncode}")
    series = read_tensorboard_scalars(run_dir)
    summary = summarize_series(series, learning_rate)
    reason = prune_reason(summary)
    summary["prune_reason"] = reason
    eval_metrics = None
    if not state.args.skip_eval:
        if not state.args.eval_states.exists():
            raise EngineeringError(f"missing eval states {state.args.eval_states}")
        checkpoint = latest_checkpoint(run_dir)
        eval_path = run_dir / "det_eval.json"
        eval_result = run_command(
            build_eval_command(checkpoint, eval_path, state.args.eval_states),
            timeout=1800,
        )
        if eval_result.returncode != 0:
            _record_failure(trial, run_dir, eval_result, state)
            raise optuna.TrialPruned(f"eval_exit_{eval_result.returncode}")
        eval_metrics = json.loads(eval_path.read_text())
        summary["deterministic_eval"] = eval_metrics
        gates = eval_metrics.get("mean_gates_passed", eval_metrics.get("mean_passed_gates"))
        trial.set_user_attr("mean_gates_passed", gates)
        trial.set_user_attr("collision_rate", eval_metrics.get("collision_rate"))
        trial.set_user_attr("success_rate", eval_metrics.get("success_rate"))
        trial.set_user_attr("mean_completion_time", eval_metrics.get("mean_completion_time"))
        trial.set_user_attr("checkpoint", str(checkpoint))
    (run_dir / "trial.json").write_text(json.dumps(summary, indent=2))
    for key in ("kl_0", "actual_learning_rate_0", "mu_max", "std_max", "sat_last", "scheduler_crushed"):
        trial.set_user_attr(key, summary[key])
    if reason:
        trial.set_user_attr("fail_reason", reason)
        raise optuna.TrialPruned(reason)
    if eval_metrics is None:
        raise EngineeringError("learnability trial has no deterministic eval")
    return float(trial.user_attrs["mean_gates_passed"])


def select_stable_lrs(study, limit: int = 3) -> list[float]:
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    completed.sort(key=lambda trial: trial.value if trial.value is not None else 1.0e9)
    values = []
    for trial in completed:
        learning_rate = float(trial.params["learning_rate"])
        if learning_rate not in values:
            values.append(learning_rate)
        if len(values) >= limit:
            break
    return values


def run_stability(args: argparse.Namespace, storage: str, storage_dir: Path) -> list[float]:
    study = load_study(args.study_name, storage, "minimize", args.seed)
    existing = {trial.params.get("learning_rate") for trial in study.trials if trial.params}
    if remaining_trials(study, args.n_trials) > 0:
        for learning_rate in STABILITY_GRID:
            if learning_rate not in existing:
                study.enqueue_trial({"learning_rate": learning_rate})
    state = _StudyState(args, storage_dir, "stability", timeout=900)
    n = remaining_trials(study, args.n_trials)
    if n:
        study.optimize(lambda trial: stability_objective(trial, state), n_trials=n, catch=())
    return select_stable_lrs(study)


def run_learnability(args: argparse.Namespace, storage: str, storage_dir: Path, lrs: list[float]) -> float | None:
    if not lrs:
        return None
    study = load_study(f"{args.study_name}-gates", storage, "maximize", args.seed)
    allowed = tuple(lrs)
    finished = {
        trial.params.get("learning_rate")
        for trial in study.trials
        if trial.state.name in {"COMPLETE", "PRUNED"}
    }
    space = allowed
    for trial in study.trials:
        dist = trial.distributions.get("learning_rate")
        if dist is not None and getattr(dist, "choices", None):
            space = tuple(dist.choices)
            break
    for learning_rate in allowed:
        if learning_rate not in finished:
            study.enqueue_trial({"learning_rate": learning_rate})
    state = _StudyState(args, storage_dir, "learnability", timeout=2400)
    state.args.updates = LEARNABILITY_UPDATES if args.phase == "all" else args.updates
    n = sum(1 for learning_rate in allowed if learning_rate not in finished)
    if n:
        study.optimize(lambda trial: learnability_objective(trial, state, space), n_trials=n, catch=())
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not complete:
        return None
    best = max(complete, key=lambda trial: trial.value or 0.0)
    return float(best.params["learning_rate"])


def run_formal(args: argparse.Namespace, storage_dir: Path, learning_rate: float) -> Path:
    run_dir = storage_dir / "runs" / "formal_1000"
    run_dir.mkdir(parents=True, exist_ok=True)
    ppo_config = write_trial_ppo_config(learning_rate, run_dir / "ppo.yaml")
    command = build_train_command(
        dynamics=args.dynamics,
        log_dir=run_dir,
        ppo_config=ppo_config,
        learning_rate=learning_rate,
        seed=args.seed,
        updates=1000,
        save_interval=100,
        num_envs=args.num_envs,
    )
    (run_dir / "train_command.json").write_text(json.dumps(command, indent=2))
    result = run_command(command, timeout=7200)
    if result.returncode != 0:
        (run_dir / "fail.txt").write_text((result.stderr or "") + (result.stdout or ""))
        raise EngineeringError(f"formal training failed: {(result.stderr or '')[-2000:]}")
    series = read_tensorboard_scalars(run_dir)
    summary = summarize_series(series, learning_rate)
    checkpoint = latest_checkpoint(run_dir)
    eval_path = run_dir / "det_eval.json"
    eval_result = run_command(build_eval_command(checkpoint, eval_path, args.eval_states), timeout=1800)
    if eval_result.returncode != 0:
        raise EngineeringError(f"formal eval failed: {(eval_result.stderr or '')[-2000:]}")
    summary["deterministic_eval"] = json.loads(eval_path.read_text())
    summary["checkpoint"] = str(checkpoint)
    (run_dir / "trial.json").write_text(json.dumps(summary, indent=2))
    return run_dir


def gates_grew(storage: str, study_name: str) -> bool:
    import optuna

    try:
        study = optuna.load_study(study_name=f"{study_name}-gates", storage=storage)
    except KeyError:
        return False
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE" and trial.value is not None]
    return any(trial.value >= 1.0 for trial in complete)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    storage_dir = args.storage.parent if args.storage else DEFAULT_STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.storage or (storage_dir / "study.db")
    storage = f"sqlite:///{db_path}"
    chosen: list[float] = []
    if args.lrs:
        chosen = [float(item) for item in args.lrs.split(",")]
    try:
        if args.phase == "stage1":
            run_stage1(args, storage, storage_dir)
            return 0
        if args.phase in {"stability", "all"}:
            updates = args.updates
            if args.phase == "all":
                args.updates = 8
            chosen = run_stability(args, storage, storage_dir) or chosen
            args.updates = updates
            print(json.dumps({"phase": "stability", "stable_lrs": chosen}, indent=2))
        if args.phase in {"learnability", "all"}:
            if not chosen:
                raise SystemExit("no stable LR for learnability")
            best = run_learnability(args, storage, storage_dir, chosen)
            print(json.dumps({"phase": "learnability", "best_lr": best}, indent=2))
            chosen = [best] if best is not None else []
        if args.phase == "train" or (args.phase == "all" and chosen and gates_grew(storage, args.study_name)):
            run_dir = run_formal(args, storage_dir, chosen[0])
            print(json.dumps({"phase": "train", "run_directory": str(run_dir)}, indent=2))
        elif args.phase == "all":
            print(json.dumps({"phase": "train", "skipped": "no sustained deterministic gates"}, indent=2))
    except EngineeringError as error:
        print(f"ENGINEERING_STOP: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
