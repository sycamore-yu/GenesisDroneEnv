from pathlib import Path


def test_training_code_does_not_import_split_s_waypoints():
    root = Path(__file__).resolve().parents[1] / "genesis_drones"
    banned = (
        root / "envs" / "race_env.py",
        root / "tasks" / "race_task.py",
        root / "algorithms" / "race_rl.py",
    )
    train = Path(__file__).resolve().parents[1] / "scripts" / "train" / "race_train.py"
    for path in (*banned, train):
        if not path.exists():
            continue
        text = path.read_text()
        assert "split_s" not in text
        assert "hidden waypoint" not in text
