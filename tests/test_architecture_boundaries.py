"""Architecture boundary checks — no illegal cross-layer imports."""

from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1] / "genesis_drones"


def _imports_from(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
    return found


def _modules_under(relative: str) -> list[Path]:
    base = ROOT / relative
    return sorted(base.rglob("*.py"))


def test_dynamics_does_not_import_tasks():
    for path in _modules_under("dynamics"):
        imports = _imports_from(path)
        assert not any(name.startswith("genesis_drones.tasks") for name in imports), path


def test_controllers_do_not_import_tasks_or_envs():
    for path in _modules_under("controllers"):
        imports = _imports_from(path)
        assert not any(name.startswith("genesis_drones.tasks") for name in imports), path
        assert not any(name.startswith("genesis_drones.envs") for name in imports), path


def test_tasks_do_not_import_algorithms_or_rsl_rl():
    for path in _modules_under("tasks"):
        imports = _imports_from(path)
        assert not any(name.startswith("genesis_drones.algorithms") for name in imports), path
        assert "rsl_rl" not in imports and not any(name.startswith("rsl_rl") for name in imports), path


def test_envs_do_not_import_algorithms_or_rsl_rl():
    for path in _modules_under("envs"):
        imports = _imports_from(path)
        assert not any(name.startswith("genesis_drones.algorithms") for name in imports), path
        assert "rsl_rl" not in imports and not any(name.startswith("rsl_rl") for name in imports), path


def test_evaluation_does_not_import_concrete_algorithms():
    banned = (
        "genesis_drones.algorithms.diff_rl",
        "genesis_drones.algorithms.squashed_actor_critic",
        "rsl_rl",
        "rsl_rl.modules",
        "rsl_rl.runners",
    )
    for path in _modules_under("evaluation"):
        if path.name.startswith("__"):
            continue
        imports = _imports_from(path)
        for name in imports:
            assert name not in banned and not name.startswith("rsl_rl"), f"{path}: {name}"
            # ApgAgent / ShacAgent must not appear via from-import names either
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("genesis_drones.algorithms"):
                raise AssertionError(f"{path} imports algorithms")
            if isinstance(node, ast.ImportFrom) and node.names:
                for alias in node.names:
                    assert alias.name not in {"ApgAgent", "ShacAgent", "ActorCritic", "OnPolicyRunner", "SquashedActorCritic"}, path


def test_core_layers_do_not_import_experiment_or_scripts():
    for relative in ("tasks", "dynamics", "controllers", "envs"):
        for path in _modules_under(relative):
            imports = _imports_from(path)
            assert not any(name.startswith("genesis_drones.experiment") for name in imports), path
            assert not any(name.startswith("scripts") for name in imports), path


def test_task_evaluators_do_not_construct_genesis_scene():
    for name in ("racing.py", "tracking.py", "race.py", "track_diff.py", "runner.py"):
        path = ROOT / "evaluation" / name
        if not path.exists():
            continue
        text = path.read_text()
        assert "gs.Scene(" not in text, path
        assert "genesis.Scene(" not in text, path
