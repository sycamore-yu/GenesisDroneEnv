"""Pick the Phase-5 objective winner. Prints EXTRA flags. Writes objective_winner.json."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "logs" / "track_diff"
BASELINE = ROOT / "apg_H32_2026-08-30_14-55-38"


def score(summary_path: Path):
    data = json.loads(summary_path.read_text())
    best_update = data.get("best_update")
    for entry in data.get("validation") or []:
        if entry.get("update") != best_update or "summary" not in entry:
            continue
        summary = entry["summary"]
        key = (
            summary["waypoint_count"]["mean"],
            -summary["crash_rate"]["mean"],
            -summary["capped_first_arrival_time"]["mean"],
        )
        return key, summary, data
    return None, None, data


def newest(pattern: str) -> Path | None:
    matches = sorted(ROOT.glob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def main() -> None:
    candidates = [("baseline", BASELINE, "")]
    for pattern, extra in (
        ("apg_H32_l2_s*", "--progress-norm l2"),
        ("apg_H32_close1_s*", "--closing-velocity-weight 1.0"),
        ("apg_H32_gaussian_s*", "--arrival-surrogate gaussian"),
    ):
        path = newest(pattern)
        if path is not None:
            candidates.append((path.name, path, extra))

    ranked = []
    for name, path, extra in candidates:
        key, summary, data = score(path / "training_summary.json")
        if key is None:
            continue
        ranked.append(
            {
                "name": name,
                "path": str(path),
                "extra": extra,
                "key": list(key),
                "waypoint_count": summary["waypoint_count"]["mean"],
                "crash_rate": summary["crash_rate"]["mean"],
                "capped_first_arrival_time": summary["capped_first_arrival_time"]["mean"],
                "best_update": data.get("best_update"),
            }
        )
    if not ranked:
        raise SystemExit("no scored runs")
    ranked.sort(key=lambda item: tuple(item["key"]), reverse=True)
    winner = ranked[0]
    extra = winner["extra"]
    skip_h96 = extra == ""
    promote = "--horizon 96" if extra == "" else f"--horizon 96 {extra}"
    payload = {
        "winner": winner,
        "ranked": ranked,
        "skip_h96_retrain": skip_h96,
        "promote_extra": promote,
        "seed_extra": "--horizon 96" if skip_h96 else promote,
        "apg_eval_if_skip": str(ROOT / "apg_H96_2026-08-30_16-32-45" / "best.pt"),
        "shac_eval": str(ROOT / "shac_old_H32_2026-08-30_18-42-19" / "best.pt"),
    }
    output = ROOT / "objective_winner.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
