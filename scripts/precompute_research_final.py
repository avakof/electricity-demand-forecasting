"""Replay disjoint frozen model groups in bounded CPU worker processes.

This changes execution scheduling only. Calls are identical to final_replay and
use the same frozen parameters, daily origins, artifact signatures and ledgers.
Run only after all validation and development diagnostics have finished.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import write_json
from src.research_experiment import ResearchRun, POINT_FAMILIES

GROUPS = {"point": POINT_FAMILIES, "A": ["nbeats", "nhits"],
          "B": ["tft", "gru"], "C": ["patchtst", "dlinear"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=GROUPS, required=True)
    args = parser.parse_args()
    run = ResearchRun.latest(ROOT)
    if not (run.path/"frozen_selection.json").exists():
        raise RuntimeError("Finish notebook 08 development and freeze first")
    done = run.path/f"final_worker_{args.group}.done.json"
    error = run.path/f"final_worker_{args.group}.error.json"
    done.unlink(missing_ok=True)
    error.unlink(missing_ok=True)
    try:
        selected = run.freeze()
        print(f"FINAL WORKER {args.group} PID {os.getpid()} RUN {run.path.name}", flush=True)
        counts = {}
        for phase in ("calibration", "test"):
            dates = run.contract[phase]
            if args.group == "point":
                run.baselines(dates["start"], dates["end"], phase)
            for key, entry in selected.items():
                if entry["status"] != "selected" or entry["family"] not in GROUPS[args.group]:
                    continue
                out = run.replay(entry["family"], entry["params"], dates["start"], dates["end"],
                                 arm=entry["arm"], tag=key, phase=phase,
                                 quantiles=entry["family"] == "lightgbm_quantile")
                counts[f"{phase}:{key}"] = len(out)
        write_json(done, {"status": "complete", "group": args.group,
                          "run": run.path.name, "prediction_rows": counts})
        print(f"COMPLETE FINAL WORKER {args.group}", flush=True)
    except Exception:
        write_json(error, {"group": args.group, "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
