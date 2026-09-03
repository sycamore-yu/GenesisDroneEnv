import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "tune"))
import race_ppo_optuna as tune  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tb(log_dir: Path, *, kl=0.01, lr=1.0e-4, mu=0.5, std=0.23, sat=0.1):
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    writer.add_scalar("Loss/learning_rate", lr, 0)
    writer.add_scalar("Loss/kl", kl, 0)
    writer.add_scalar("Loss/mu_abs_mean", mu, 0)
    writer.add_scalar("Policy/mean_noise_std", std, 0)
    writer.add_scalar("Policy/sat_frac", sat, 0)
    writer.add_scalar("Loss/surrogate", 0.01, 0)
    writer.add_scalar("Loss/value_function", 10.0, 0)
    writer.close()


def test_trial_builds_independent_config_and_command(tmp_path):
    before = _digest(tune.SOURCE_PPO_YAML)
    dest = tmp_path / "ppo.yaml"
    tune.write_trial_ppo_config(1.0e-4, dest)
    after = _digest(tune.SOURCE_PPO_YAML)
    assert after == before
    data = yaml.safe_load(dest.read_text())
    assert data["algorithm"]["learning_rate"] == pytest.approx(1.0e-4)
    command = tune.build_train_command(
        dynamics="full_quad",
        log_dir=tmp_path / "run",
        ppo_config=dest,
        learning_rate=1.0e-4,
        seed=0,
        updates=8,
        save_interval=8,
        num_envs=16,
    )
    assert command[1].endswith("scripts/train/race_train.py")
    assert "--ppo-config" in command
    assert str(dest) in command
    assert "0.0001" in command or "1e-4" in command or "0.0001" in "".join(command)


def test_trial_does_not_modify_official_yaml(tmp_path):
    before = tune.SOURCE_PPO_YAML.read_bytes()
    tune.write_trial_ppo_config(3.0e-4, tmp_path / "ppo.yaml")
    assert tune.SOURCE_PPO_YAML.read_bytes() == before
    source = yaml.safe_load(before)
    assert source["algorithm"]["learning_rate"] == pytest.approx(0.0026)


def test_reads_metrics_from_completed_run():
    log_dir = ROOT / "logs" / "race" / "ppo_full_quad_adaptkl_seed0"
    series = tune.read_tensorboard_scalars(log_dir)
    assert series["kl"][0] == pytest.approx(145.495, rel=0.01)
    summary = tune.summarize_series(series, 0.0026)
    assert summary["scheduler_crushed"] is True
    assert tune.prune_reason(summary) == "kl_jump"


def test_study_records_completed_pruned_and_failed(tmp_path, monkeypatch):
    import optuna

    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name="status", storage=storage, direction="minimize")

    def fake_run(command, timeout):
        log_dir = Path(command[command.index("--log-dir") + 1])
        log_dir.mkdir(parents=True, exist_ok=True)
        lr = float(command[command.index("--learning-rate") + 1])
        if lr < 2.0e-5:
            _write_tb(log_dir, kl=0.01, lr=lr, mu=0.5, std=0.23, sat=0.1)
            return subprocess.CompletedProcess(command, 0, "", "")
        if lr < 2.0e-4:
            _write_tb(log_dir, kl=145.0, lr=1.0e-5, mu=10.0, std=0.3, sat=0.2)
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 1, "", "policy diverged")

    monkeypatch.setattr(tune, "run_command", fake_run)
    args = tune.parse_args(
        [
            "--phase",
            "stability",
            "--n-trials",
            "3",
            "--updates",
            "8",
            "--study-name",
            "status",
        ]
    )
    state = tune._StudyState(args, tmp_path, "stability", timeout=30)

    def objective(trial):
        return tune.stability_objective(trial, state)

    study.enqueue_trial({"learning_rate": 1.0e-5})
    study.enqueue_trial({"learning_rate": 1.0e-4})
    study.enqueue_trial({"learning_rate": 1.0e-3})
    study.optimize(objective, n_trials=3, catch=())
    states = {trial.params["learning_rate"]: trial.state.name for trial in study.trials}
    assert states[1.0e-5] == "COMPLETE"
    assert states[1.0e-4] == "PRUNED"
    assert states[1.0e-3] == "PRUNED"
    failed = [trial for trial in study.trials if trial.params["learning_rate"] == 1.0e-3][0]
    assert failed.user_attrs["exit_code"] == 1
    assert "diverged" in failed.user_attrs["fail_reason"]


def test_sqlite_study_resumes(tmp_path, monkeypatch):
    import optuna

    db = tmp_path / "study.db"
    storage = f"sqlite:///{db}"
    args = tune.parse_args(["--phase", "stability", "--n-trials", "2", "--study-name", "resume-lr"])
    monkeypatch.setattr(
        tune,
        "run_command",
        lambda command, timeout: subprocess.CompletedProcess(command, 1, "", "policy diverged"),
    )
    state = tune._StudyState(args, tmp_path, "stability", timeout=30)
    first = tune.load_study("resume-lr", storage, "minimize", 0)
    first.enqueue_trial({"learning_rate": 1.0e-4})
    first.optimize(lambda trial: tune.stability_objective(trial, state), n_trials=1, catch=())
    assert len(first.trials) == 1
    second = tune.load_study("resume-lr", storage, "minimize", 0)
    assert len(second.trials) == 1
    assert second.trials[0].state.name == "PRUNED"
    assert tune.remaining_trials(second, 2) == 1


def test_smoke_calls_existing_race_train(tmp_path):
    dest = tmp_path / "run"
    dest.mkdir()
    config = tune.write_trial_ppo_config(1.0e-4, dest / "ppo.yaml")
    command = tune.build_train_command(
        dynamics="full_quad",
        log_dir=dest,
        ppo_config=config,
        learning_rate=1.0e-4,
        seed=0,
        updates=1,
        save_interval=1,
        num_envs=16,
    )
    before = tune.SOURCE_PPO_YAML.read_bytes()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert tune.SOURCE_PPO_YAML.read_bytes() == before
    assert result.returncode == 0, result.stderr[-2000:]
    assert (dest / "experiment.json").exists()
    experiment = json.loads((dest / "experiment.json").read_text())
    assert experiment["algorithm"] == "ppo"
    assert experiment["dynamics"] == "full_quad"
