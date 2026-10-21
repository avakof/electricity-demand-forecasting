"""Sequential, resumable research execution; each sealed replay is reusable."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.common import write_json
from src.rte_risk_lab import RiskLabRun
from src.rte_risk_uncertainty import uncertainty_replay


def state(run,stage,status='running'):
    record = dict(status=status,stage=stage,run_dir=str(run.path),updated_utc=datetime.now(timezone.utc).isoformat(),
                  no_production=True, test_previously_explored=True)
    write_json(run.path/'execution_state.json',record)
    write_json(ROOT/'.runtime/rte_risk_execution_state.json',record)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--new',action='store_true')
    parser.add_argument('--run-dir')
    parser.add_argument('--stage',choices=['prepare','development','evaluation','report','all'],default='all')
    args=parser.parse_args()
    run=RiskLabRun.create(ROOT) if args.new else RiskLabRun(ROOT,args.run_dir) if args.run_dir else RiskLabRun.latest(ROOT)
    print('RUN_DIRECTORY',run.path,flush=True)
    if args.stage in ('prepare','development','all'):
        state(run,'causal feature banks')
        _=run.frame
        print('VARIANTS',len(json.loads((run.path/'variant_register.json').read_text())),flush=True)
    if args.stage in ('development','all'):
        state(run,'chronological development tuning')
        run.tune()
        state(run,'development probability comparison and freeze')
        frozen=run.freeze()
        print('FROZEN',frozen['selected_point'],frozen['selected_quantile'],flush=True)
    if args.stage in ('evaluation','all'):
        state(run,'frozen calibration and retrospective test')
        run.evaluate()
        state(run,'quantile and fixed-model uncertainty comparisons')
        uncertainty_replay(run)
    if args.stage in ('report','all'):
        from scripts.rte_risk_reporting import report
        state(run,'matched scores and risk report')
        report(run)
        state(run,'awaiting notebook rendering and independent verification','computed')
    print('STAGE_COMPLETE',args.stage,flush=True)


if __name__=='__main__':
    main()
