"""Record seeds, code version, and evaluate the 90%/5% reliability gate.

Full 3-seed training is a GPU job. Use --smoke to check the harness only.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUCCESS_THRESHOLD = 0.90
COLLISION_THRESHOLD = 0.05
SEEDS = (1, 2, 3)
APG_HORIZONS = (32, 64, 96)


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "logs" / "race" / "acceptance.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with (PROJECT_ROOT / "config" / "race" / "train.yaml").open() as file:
        train = yaml.safe_load(file)
    with (PROJECT_ROOT / "config" / "race" / "ppo.yaml").open() as file:
        ppo = yaml.safe_load(file)
    report = {
        "code_version": git_revision(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "eval_states": 100,
        "success_threshold": SUCCESS_THRESHOLD,
        "collision_threshold": COLLISION_THRESHOLD,
        "gamma": train["environment"]["gamma"],
        "lambda": train["environment"]["td_lambda"],
        "ppo_gamma": ppo["algorithm"]["gamma"],
        "ppo_lambda": ppo["algorithm"]["lam"],
        "apg_horizons": list(APG_HORIZONS),
        "algorithms": ("ppo", "apg", "shac"),
        "smoke": args.smoke,
        "results": [],
        "passed": False,
    }
    if args.smoke:
        report["results"] = [
            {"algorithm": name, "seed": seed, "success_rate": None, "collision_rate": None, "note": "smoke"}
            for name in report["algorithms"]
            for seed in SEEDS
        ]
        report["note"] = "harness only; run full training on GPU to fill success/collision rates"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not args.smoke:
        raise SystemExit("full 3-seed training is not started by this script; pass --smoke or launch race_train.py per seed")


if __name__ == "__main__":
    main()
