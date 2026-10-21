"""Freeze the best complete policy using 2024 results, including baselines.

The core experiment separately selects an independent model. This reporting
extension prevents that narrower choice from hiding a better simple baseline.
No predictor, model parameter or core manifest is changed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def freeze_policy(run: Path):
    output = run / "development_policy_selection.json"
    if output.exists():
        return json.loads(output.read_text())
    if any((run / "forecasts").glob("*__test__*")) or (run / "forecasts/all_test.csv.gz").exists():
        raise RuntimeError("Complete-policy choice must be frozen before test replay")
    selected_path = run / "selected_parameters.json"
    baseline_path = run / "tables/development_baselines.csv"
    selected = json.loads(selected_path.read_text())
    if len(selected) != 23 or any(v["status"] != "selected" for v in selected.values()):
        raise RuntimeError("All 23 model variants must finish validation first")
    models = [{"model": key, "MAE_MW": value["MAE_MW"], "kind": "independent_model"}
              for key, value in selected.items()]
    with baseline_path.open() as stream:
        baselines = [{"model": row["model"], "MAE_MW": float(row["MAE_MW"]),
                      "kind": "predeclared_baseline"} for row in csv.DictReader(stream)]
    if len(baselines) != 4:
        raise RuntimeError("All four predeclared baseline scores are required")
    entries = sorted(models + baselines, key=lambda item: item["MAE_MW"])
    payload = {
        "selection_period": "2024 delivery intervals only",
        "criterion": "pooled national MAE_MW; include all four predeclared baselines",
        "selected_policy": entries[0]["model"],
        "selected_independent_model": min(models, key=lambda item: item["MAE_MW"])["model"],
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "test_previously_explored": True,
        "selection_inputs_sha256": {str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (selected_path, baseline_path)},
        "freeze_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "entries": entries,
        "scope": "additional development-only policy choice; no model/core-selection changes",
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir or Path(json.loads((ROOT / ".runtime/latest_research_run.json").read_text())["run_dir"])
    if not run.is_absolute():
        run = ROOT / run
    result = freeze_policy(run)
    print(json.dumps({k: v for k, v in result.items() if k != "entries"}, indent=2))
