"""Read-only validation of executed research-v2 artifacts.

Default mode requires a complete run. ``--allow-incomplete`` reports unfinished
steps separately while still failing verified integrity violations. Forecast
files are checked independently of the scoring engine; national losses are
recomputed from the immutable source CSV on common eligible origin/target pairs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.common import fingerprint, forecast_origin, local_day_grid

FAMILIES = {
    'arima', 'sarima', 'sarimax', 'hist_gradient_boosting', 'lightgbm',
    'lightgbm_quantile', 'gam', 'nbeats', 'nhits', 'tft', 'gru', 'patchtst', 'dlinear',
}
NOTEBOOKS = ('07_statistical_and_boosting_models.ipynb',
             '08_deep_learning_and_uncertainty.ipynb')
KEY = ['model', 'origin', 'delivery_time']


class ArtifactChecker:
    def __init__(self, root: Path, run: Path, allow_incomplete=False):
        self.root, self.run = root, run
        self.allow_incomplete = allow_incomplete
        self.errors, self.pending, self.warnings = [], [], []
        self.report = {'run_dir': str(run), 'allow_incomplete': allow_incomplete}
        self.contract = None
        self.completed = {}
        self.fits = Counter()
        self.forecast_counts = Counter()

    def require(self, condition, message):
        if not condition:
            self.errors.append(message)
        return bool(condition)

    def unfinished(self, message):
        (self.pending if self.allow_incomplete else self.errors).append(message)

    def read_json(self, path, required=True):
        if not path.exists():
            if required:
                self.unfinished(f'Missing {path.relative_to(self.root) if path.is_relative_to(self.root) else path}')
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            self.unfinished(f'Unreadable/incomplete JSON {path.name}: {exc}')
            return None

    def manifest(self):
        manifest = self.read_json(self.run/'manifest.json')
        if manifest is None:
            return
        self.contract = manifest.get('contract')
        self.require(isinstance(self.contract, dict), 'Manifest is missing its research contract')
        raws = manifest.get('raw_sha256', {})
        self.require(len(raws) == 2, 'Manifest must fingerprint both source CSVs')
        for name, expected in raws.items():
            path = self.root/'data'/name
            self.require(path.exists() and fingerprint(path) == expected, f'Raw source changed: {name}')
        source_changes = []
        for name, expected in manifest.get('source_sha256', {}).items():
            path = self.root/name
            if not path.exists() or fingerprint(path) != expected:
                source_changes.append(name)
        if source_changes:
            self.unfinished('Run source/config fingerprint drift: '+', '.join(source_changes))
        self.require(manifest.get('predictions_include_outcomes') is False,
                     'Manifest must explicitly exclude outcomes from predictions')
        self.report['manifest_raw_files_checked'] = len(raws)
        self.report['manifest_source_files_checked'] = len(manifest.get('source_sha256', {}))

    def check_frame(self, frame, name):
        self.require('actual_mw' not in frame, f'Outcomes leaked into prediction artifact: {name}')
        if not self.require(set(KEY).issubset(frame), f'Prediction keys missing: {name}'):
            return
        keys = KEY+(['method'] if 'method' in frame else [])
        self.require(not frame.duplicated(keys).any(), f'Duplicate model/method/origin/delivery keys: {name}')
        frame = frame.copy()
        frame['origin'] = pd.to_datetime(frame.origin, utc=True, errors='raise')
        frame['delivery_time'] = pd.to_datetime(frame.delivery_time, utc=True, errors='raise')
        self.require(frame[KEY].notna().all().all(), f'Missing prediction keys: {name}')
        self.require(frame.delivery_time.gt(frame.origin).all(), f'Nonfuture delivery interval: {name}')
        if self.contract is None:
            return
        groups = ['model', 'origin']+(['method'] if 'method' in frame else [])
        for identity, group in frame.groupby(groups, sort=False):
            model, origin = identity[:2]
            days = group.delivery_time.dt.tz_convert(self.contract['timezone']).dt.strftime('%Y-%m-%d')
            if not self.require(days.nunique() == 1, f'Multiple delivery days at {model}/{origin}: {name}'):
                continue
            day = days.iloc[0]
            expected = local_day_grid(day)
            actual = pd.DatetimeIndex(group.delivery_time).sort_values()
            self.require(actual.equals(expected), f'Incomplete or wrong DST delivery grid {model}/{day}: {name}')
            self.require(origin == forecast_origin(day, self.contract), f'Wrong forecast origin {model}/{day}: {name}')
            if 'local_day' in group:
                self.require(group.local_day.astype(str).eq(day).all(), f'Wrong local_day metadata: {name}')
        if 'raw_prediction_mw' in frame and 'fallback_used' in frame:
            fallback = frame.fallback_used.astype(str).str.lower().isin(['true', '1'])
            self.require(fallback.eq(frame.raw_prediction_mw.isna()).all(), f'Fallback flag/raw prediction disagreement: {name}')

    def forecasts_and_ledgers(self):
        forecast_dir = self.run/'forecasts'
        files = sorted(forecast_dir.glob('*.csv.gz'))
        if not files:
            self.unfinished('No completed forecast CSVs yet')
        ledger_count, total_rows = 0, 0
        for path in files:
            meta_path = path.with_suffix('').with_suffix('.json')
            combined = path.name.startswith('all_') or path.name in {
                'fixed_conformal_test.csv.gz', 'uncertainty_test.csv.gz', 'peak_uncertainty_test.csv.gz'}
            meta = self.read_json(meta_path, required=not combined)
            if not combined and meta is None:
                continue
            try:
                frame = pd.read_csv(path)
                self.check_frame(frame, path.name)
            except (OSError, EOFError, ValueError, KeyError) as exc:
                self.require(False, f'Invalid forecast CSV {path.name}: {exc}')
                continue
            total_rows += len(frame)
            if combined:
                self.completed[path.name] = frame
                continue
            self.require(fingerprint(path) == meta.get('sha256'), f'Forecast checksum mismatch: {path.name}')
            self.require(len(frame) == meta.get('rows'), f'Forecast row count mismatch: {path.name}')
            self.require(meta.get('predictions_archived_before_scoring') is True,
                         f'Missing pre-outcome archive declaration: {path.name}')
            ledger = self.run/'ledgers'/(path.name.removesuffix('.csv.gz')+'.jsonl.gz')
            if not self.require(ledger.exists(), f'Missing origin ledger: {ledger.name}'):
                continue
            self.require(fingerprint(ledger) == meta.get('ledger_sha256'), f'Ledger checksum mismatch: {ledger.name}')
            records, event_origins, previous, ledger_rows = [], [], None, 0
            try:
                with gzip.open(ledger, 'rt') as stream:
                    for line in stream:
                        event = json.loads(line)
                        origin = pd.to_datetime(event['origin'], utc=True)
                        self.require(previous is None or origin > previous, f'Nonchronological/repeated ledger origin: {ledger.name}')
                        values = event['predictions']
                        self.require(all('actual_mw' not in value for value in values),
                                     f'Outcomes leaked into origin ledger: {ledger.name}')
                        records.extend(values)
                        event_origins.extend([origin]*len(values))
                        ledger_rows += len(values)
                        previous = origin
            except (OSError, EOFError, ValueError, KeyError) as exc:
                self.require(False, f'Invalid origin ledger {ledger.name}: {exc}')
                continue
            self.require(ledger_rows == len(frame), f'Ledger/CSV row count mismatch: {ledger.name}')
            if records:
                ledger_frame = pd.DataFrame(records)
                self.check_frame(ledger_frame, ledger.name)
                ledger_origins = pd.to_datetime(ledger_frame.origin, utc=True)
                self.require(ledger_origins.eq(pd.Series(event_origins)).all(),
                             f'Ledger event/row origin mismatch: {ledger.name}')
                # Compare prediction values and keys, allowing only serialization precision.
                for col in ('origin', 'delivery_time'):
                    ledger_frame[col] = pd.to_datetime(ledger_frame[col], utc=True)
                    frame[col] = pd.to_datetime(frame[col], utc=True)
                left = frame.sort_values(KEY).reset_index(drop=True)
                right = ledger_frame.sort_values(KEY).reset_index(drop=True)
                self.require(left[KEY].equals(right[KEY]), f'Ledger/CSV key disagreement: {ledger.name}')
                for column in ('prediction_mw', 'raw_prediction_mw', 'q05', 'q50', 'q95'):
                    if column in left and column in right:
                        left_numeric = pd.to_numeric(left[column], errors='raise').to_numpy(dtype=float)
                        right_numeric = pd.to_numeric(right[column], errors='raise').to_numpy(dtype=float)
                        self.require(np.allclose(left_numeric, right_numeric, rtol=1e-9, atol=1e-6, equal_nan=True),
                                     f'Ledger/CSV {column} disagreement: {ledger.name}')
            ledger_count += 1
        # In-progress ledgers without a completion marker are explicitly unfinished.
        for path in (self.run/'ledgers').glob('*.jsonl.gz'):
            stem = path.name.removesuffix('.jsonl.gz')
            if not (forecast_dir/f'{stem}.json').exists():
                self.unfinished(f'Incomplete forecast replay: {stem}')
        self.report.update(forecast_csvs_checked=len(files), forecast_rows_checked=total_rows,
                           verified_origin_ledgers=ledger_count)

    def fit_records(self):
        paths = sorted((self.run/'fits').glob('*.jsonl.gz'))
        for path in paths:
            try:
                with gzip.open(path, 'rt') as stream:
                    for line in stream:
                        record = json.loads(line)
                        family = record.get('family') or record.get('model', '').split('__')[0]
                        status = record.get('status', 'unknown')
                        self.fits[(family, status)] += 1
                        if status == 'fit':
                            available = pd.to_datetime(record.get('latest_label_available_at'), utc=True)
                            origin = pd.to_datetime(record.get('origin'), utc=True)
                            self.require(pd.notna(available) and pd.notna(origin) and available < origin,
                                         f'Training label not strictly before fit origin: {path.name}')
                        elif status == 'failed':
                            self.require(bool(record.get('error')), f'Failed fit lacks explanation: {path.name}')
            except (OSError, EOFError, ValueError, KeyError) as exc:
                self.unfinished(f'Incomplete fit ledger {path.name}: {exc}')
        self.report['fit_counts_by_family'] = {
            family: {'trained_fits': self.fits[(family, 'fit')], 'failed_fits': self.fits[(family, 'failed')]}
            for family in sorted(FAMILIES)
        }
        for family in sorted(FAMILIES):
            if self.fits[(family, 'fit')] == 0 and self.fits[(family, 'failed')] > 0:
                self.warnings.append(f'{family}: no successful fit; failed attempts must not be described as a trained model')

    def frozen_selection(self):
        frozen = self.read_json(self.run/'frozen_selection.json')
        if frozen is None:
            return
        selections = frozen.get('selections', {})
        observed = {entry.get('family') for entry in selections.values()}
        self.require(FAMILIES.issubset(observed), f'Frozen selection misses families: {sorted(FAMILIES-observed)}')
        current = self.read_json(self.run/'selected_parameters.json')
        if current is not None:
            self.require(current == selections, 'Selection changed after freeze')
        contract_path = self.root/'config/research_experiment.json'
        self.require(contract_path.exists() and fingerprint(contract_path) == frozen.get('contract_sha256'),
                     'Contract changed after selection freeze')
        trials_path = self.run/'tables/tuning_trials.csv'
        if trials_path.exists():
            trials = pd.read_csv(trials_path)
            self.require('family' in trials, 'Tuning trial table lacks family')
            if 'family' in trials:
                missing = FAMILIES-set(trials.family)
                self.require(not missing, f'No recorded tuning trials for families: {sorted(missing)}')
            if self.contract and {'model', 'trial', 'fold'}.issubset(trials):
                expected = {fold['name'] for fold in self.contract['validation_folds']}
                for (model, trial), rows in trials.groupby(['model', 'trial']):
                    self.require(expected.issubset(set(rows.fold)), f'Incomplete validation folds for {model}/trial{trial}')
        else:
            self.unfinished('Missing tuning_trials.csv')
        for key, entry in selections.items():
            self.require(entry.get('status') in {'selected', 'no_eligible_configuration'}, f'Unknown selection status: {key}')
            if entry.get('status') == 'selected':
                self.require(self.fits[(entry['family'], 'fit')] > 0, f'Selected family has no successful recorded fit: {key}')
        frozen_time = pd.to_datetime(frozen.get('freeze_time_utc'), utc=True)
        self.require(pd.notna(frozen_time), 'Selection freeze has no valid wall-clock timestamp')
        if pd.notna(frozen_time):
            for path in (self.run/'forecasts').glob('*__test__*.json'):
                self.require(pd.Timestamp(path.stat().st_mtime, unit='s', tz='UTC') >= frozen_time,
                             f'Test completion predates selection freeze: {path.name}')
        self.report['investigated_families'] = sorted(observed.intersection(FAMILIES))

    def notebooks(self):
        notebooks = []
        for name in NOTEBOOKS:
            path = self.root/'notebooks'/name
            if not path.exists():
                self.unfinished(f'Missing notebook: {name}')
                continue
            try:
                nb = nbformat.read(path, as_version=4)
                nbformat.validate(nb)
                cells = [cell for cell in nb.cells if cell.cell_type == 'code']
                unexecuted = [i for i, cell in enumerate(nb.cells)
                              if cell.cell_type == 'code' and cell.execution_count is None]
                errors = [i for i, cell in enumerate(nb.cells) if any(
                    output.get('output_type') == 'error' for output in cell.get('outputs', []))]
                self.require(not errors, f'Notebook execution errors: {name}, cells {errors}')
                if unexecuted:
                    self.unfinished(f'Unexecuted notebook cells: {name}, cells {unexecuted}')
                notebooks.append({'notebook': name, 'code_cells': len(cells),
                                  'unexecuted_cells': unexecuted, 'error_cells': errors})
            except (OSError, ValueError, nbformat.ValidationError) as exc:
                self.require(False, f'Invalid notebook {name}: {exc}')
        self.report['notebooks'] = notebooks

    def scorecards(self):
        for name in ('test_scorecard.csv', 'uncertainty_scorecard.csv', 'bootstrap.csv'):
            if not (self.run/'tables'/name).exists():
                self.unfinished(f'Missing final table: {name}')
        path = self.run/'tables/test_scorecard.csv'
        prediction_path = self.run/'forecasts/all_test.csv.gz'
        if not path.exists() or not prediction_path.exists():
            return
        scorecard = pd.read_csv(path)
        predictions = pd.read_csv(prediction_path)
        if not self.require({'model', 'MAE_MW'}.issubset(scorecard), 'Final scorecard needs model and MAE_MW'):
            return
        for col in ('origin', 'delivery_time'):
            predictions[col] = pd.to_datetime(predictions[col], utc=True)
        source = pd.read_csv(self.root/'data/france_electricity_hourly.csv', usecols=['datetime_utc', 'load_mw'])
        source = source.rename(columns={'datetime_utc': 'delivery_time', 'load_mw': 'actual_mw'})
        source['delivery_time'] = pd.to_datetime(source.delivery_time, utc=True)
        selected_models = scorecard.model.unique()
        pivot = predictions.loc[predictions.model.isin(selected_models)].pivot(
            index=['origin', 'delivery_time'], columns='model', values='prediction_mw')
        if not self.require(set(selected_models).issubset(pivot.columns), 'Scorecard model absent from final predictions'):
            return
        common = pivot.dropna().reset_index().merge(source, on='delivery_time', how='left', validate='many_to_one')
        eligible = common.actual_mw.notna()
        y = common.loc[eligible, 'actual_mw'].to_numpy()
        if not self.require(len(y) > 0, 'No common eligible targets in final scorecard'):
            return
        for _, row in scorecard.iterrows():
            p = common.loc[eligible, row.model].to_numpy()
            error = y-p
            values = {'eligible_targets': int(eligible.sum()), 'scored': len(y),
                      'missing_targets': int((~eligible).sum()),
                      'MAE_MW': np.mean(np.abs(error)), 'RMSE_MW': np.sqrt(np.mean(error**2)),
                      'bias_actual_minus_forecast_MW': np.mean(error),
                      'WAPE_percent': 100*np.sum(np.abs(error))/np.sum(np.abs(y))}
            for column, expected in values.items():
                if column in row and pd.notna(row[column]):
                    self.require(np.isclose(float(row[column]), expected, rtol=1e-8, atol=1e-6, equal_nan=True),
                                 f'Common-eligible score mismatch {row.model}/{column}: {row[column]} vs {expected}')
        if 'rte_proxy' in selected_models:
            rte = float(scorecard.loc[scorecard.model.eq('rte_proxy'), 'MAE_MW'].iloc[0])
            for _, row in scorecard.iterrows():
                for column in ('relative_improvement_percent', 'RTE_relative_improvement_percent', 'MAE_improvement_vs_RTE_percent'):
                    if column in row and pd.notna(row[column]) and rte:
                        expected = 100*(1-float(row.MAE_MW)/rte)
                        self.require(np.isclose(float(row[column]), expected, atol=1e-6),
                                     f'RTE-relative gain mismatch: {row.model}')
        self.report['common_eligible_test_hours'] = int(eligible.sum())
        self.report['scorecard_models_checked'] = len(selected_models)

    def development_policy(self):
        """Check the separate development-only choice across models AND baselines."""
        path = self.run/'development_policy_selection.json'
        if not path.exists():
            return
        policy = self.read_json(path)
        if policy is None:
            return
        script = self.root/'scripts/freeze_research_policy.py'
        self.require(script.exists() and fingerprint(script) == policy.get('freeze_script_sha256'),
                     'Complete-policy freeze script changed after selection')
        inputs = policy.get('selection_inputs_sha256', {})
        self.require(set(inputs) == {'selected_parameters.json', 'tables/development_baselines.csv'},
                     'Complete-policy selection must fingerprint both development input tables')
        valid_inputs = set(inputs) == {'selected_parameters.json', 'tables/development_baselines.csv'}
        for name, expected in inputs.items():
            source = self.run/name
            valid_inputs &= self.require(source.exists() and fingerprint(source) == expected,
                                         f'Complete-policy selection input changed: {name}')
        self.require(policy.get('selection_period') == '2024 delivery intervals only',
                     'Complete policy was not selected exclusively on the development period')
        entries = pd.DataFrame(policy.get('entries', []))
        required = {'model', 'MAE_MW', 'kind'}
        if not self.require(required.issubset(entries), 'Complete-policy entries lack model/loss/kind'):
            return
        self.require(len(entries) == 27 and entries.model.nunique() == 27,
                     'Complete-policy choice must cover all 23 model variants plus four baselines')
        self.require(np.isfinite(pd.to_numeric(entries.MAE_MW, errors='coerce')).all(),
                     'Complete-policy comparison contains nonfinite development losses')
        if valid_inputs:
            selected = json.loads((self.run/'selected_parameters.json').read_text())
            baseline = pd.read_csv(self.run/'tables/development_baselines.csv')
            self.require(len(selected) == 23 and all(e.get('status') == 'selected' for e in selected.values()),
                         'Complete-policy choice has unfinished model variants')
            independent = pd.DataFrame([{'model': key, 'MAE_MW': entry['MAE_MW'], 'kind': 'independent_model'}
                                        for key, entry in selected.items()])
            baseline = baseline[['model', 'MAE_MW']].assign(kind='predeclared_baseline')
            expected = pd.concat([independent, baseline], ignore_index=True)
            self.require(len(baseline) == 4 and baseline.model.nunique() == 4,
                         'Complete-policy choice needs four distinct baseline scores')
            self.require(set(expected.model) == set(entries.model),
                         'Complete-policy alternatives differ from the frozen development inputs')
            if set(expected.model) == set(entries.model) and not entries.model.duplicated().any():
                actual = entries.set_index('model').reindex(expected.model)
                self.require(np.allclose(actual.MAE_MW.to_numpy(dtype=float), expected.MAE_MW.to_numpy(dtype=float), rtol=1e-12, atol=1e-9),
                             'Complete-policy development losses differ from source inputs')
                self.require((actual.kind.to_numpy() == expected.kind.to_numpy()).all(),
                             'Complete-policy baseline/independent labels changed')
            expected_policy = expected.loc[expected.MAE_MW.idxmin(), 'model']
            expected_independent = independent.loc[independent.MAE_MW.idxmin(), 'model']
            self.require(policy.get('selected_policy') == expected_policy,
                         'Selected complete policy is not the development MAE argmin among all 27 alternatives')
            self.require(policy.get('selected_independent_model') == expected_independent,
                         'Selected independent model is not the development MAE argmin among 23 model variants')
            frozen = self.read_json(self.run/'frozen_selection.json', required=False)
            if frozen is not None:
                self.require(frozen.get('primary_candidate_from_validation') == expected_independent,
                             'Independent core selection differs from the policy-extension model choice')
        frozen_time = pd.to_datetime(policy.get('frozen_utc'), utc=True)
        self.require(pd.notna(frozen_time), 'Complete-policy selection lacks a valid freeze timestamp')
        paths = set((self.run/'forecasts').glob('*__test__*'))
        paths.update((self.run/'ledgers').glob('*__test__*'))
        if (self.run/'forecasts/all_test.csv.gz').exists():
            paths.add(self.run/'forecasts/all_test.csv.gz')
        if pd.notna(frozen_time):
            for artifact in paths:
                stat = artifact.stat()
                observed_time = pd.Timestamp(getattr(stat, 'st_birthtime', stat.st_mtime), unit='s', tz='UTC')
                self.require(frozen_time <= observed_time,
                             f'Complete policy frozen after a test artifact was created: {artifact.name}')
                if artifact.suffix == '.json':
                    metadata = self.read_json(artifact, required=False)
                    for name in ('created_utc', 'completed_utc', 'created_at', 'completed_at'):
                        if metadata and metadata.get(name):
                            moment = pd.to_datetime(metadata[name], utc=True)
                            self.require(pd.notna(moment) and frozen_time <= moment,
                                         f'Complete policy frozen after test metadata timestamp: {artifact.name}/{name}')
        self.report['development_complete_policy'] = {
            'selected_policy': policy.get('selected_policy'),
            'selected_independent_model': policy.get('selected_independent_model'),
            'alternatives': len(entries), 'frozen_utc': policy.get('frozen_utc')}

    def peak_extension(self):
        """Validate the separately frozen, pretest cold-evening extension."""
        path = self.run/'peak_risk_specification.json'
        artifacts = (self.run/'forecasts/peak_uncertainty_test.csv.gz').exists() or any((self.run/'tables').glob('peak_*.csv'))
        if not path.exists():
            if artifacts:
                self.unfinished('Peak-risk artifacts exist without a frozen specification')
            return
        spec = self.read_json(path)
        if spec is None:
            return
        source = self.root/'src/research_peak_risk.py'
        self.require(source.exists() and fingerprint(source) == spec.get('source_sha256'),
                     'Peak extension code changed after its separate freeze')
        if source.exists():
            declarations = ast.parse(source.read_text())
            for node in declarations.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SPEC' for t in node.targets):
                    expected_spec = ast.literal_eval(node.value)
                    self.require({k: v for k, v in spec.items() if k not in {'frozen_utc', 'source_sha256'}} == expected_spec,
                                 'Frozen peak definitions differ from the hashed source specification')
        frozen_time = pd.to_datetime(spec.get('frozen_utc'), utc=True)
        self.require(pd.notna(frozen_time), 'Peak extension lacks a valid freeze timestamp')
        selection = self.read_json(self.run/'peak_interval_selection.json')
        if selection is not None:
            self.require(selection.get('spec_source_sha256') == spec.get('source_sha256'),
                         'Peak interval selection uses another extension source version')
            main_frozen = self.read_json(self.run/'frozen_selection.json', required=False)
            if main_frozen is not None:
                self.require(selection.get('primary') == main_frozen.get('primary_candidate_from_validation'),
                             'Peak interval selection uses another primary candidate')
            selected_time = pd.to_datetime(selection.get('selected_utc'), utc=True)
            self.require(pd.notna(selected_time), 'Peak interval selection has no valid timestamp')
            choices = pd.DataFrame(selection.get('records', []))
            self.require({'model', 'candidate', 'selected'}.issubset(choices), 'Peak interval selection records incomplete')
            if {'model', 'candidate', 'selected'}.issubset(choices):
                for model, rows in choices.groupby('model'):
                    self.require(set(rows.candidate) == {'pooled', 'conditional'}, f'Peak interval alternatives missing: {model}')
                    self.require(rows.selected.eq(True).sum() == 1, f'Exactly one frozen peak policy required: {model}')
            # Wall-clock completion timestamps are supplementary evidence; the
            # source-side first-creation guard prevents selection after replay starts.
            for completion in (self.run/'forecasts').glob('*__test__*.json'):
                completed = pd.Timestamp(completion.stat().st_mtime, unit='s', tz='UTC')
                if pd.notna(frozen_time):
                    self.require(frozen_time <= completed, f'Peak specification frozen after test artifact: {completion.name}')
                if pd.notna(selected_time):
                    self.require(selected_time <= completed, f'Peak intervals selected after test artifact: {completion.name}')
        required = ['peak_risk_scorecard.csv', 'peak_proxy_cost_sensitivity.csv',
                    'peak_calibration_pools.csv', 'peak_interval_selection.csv']
        for name in required:
            if not (self.run/'tables'/name).exists():
                self.unfinished(f'Missing peak-risk table: {name}')
        pools_path = self.run/'tables/peak_calibration_pools.csv'
        if pools_path.exists() and self.contract is not None:
            pools = pd.read_csv(pools_path)
            if 'latest_allowed_calibration_available_at' in pools:
                times = pd.to_datetime(pools.latest_allowed_calibration_available_at, utc=True, errors='coerce')
                cutoff = forecast_origin(self.contract['test']['start'], self.contract)
                self.require(times.dropna().lt(cutoff).all(), 'Peak calibration uses outcomes not released before first test origin')
        cost_path = self.run/'tables/peak_proxy_cost_sensitivity.csv'
        if cost_path.exists():
            cost = pd.read_csv(cost_path)
            self.require('unit' in cost and cost.unit.str.contains('not euros', case=False, na=False).all(),
                         'Peak cost scenarios must be labeled non-euro risk proxies')
            if {'model', 'method', 'mean_proxy_loss', 'RTE_same_policy_proxy_loss'}.issubset(cost):
                reference = cost.loc[cost.model.eq('rte_proxy') & cost.method.eq('fixed_past_out_of_sample_residual_distribution')]
                self.require(np.allclose(reference.mean_proxy_loss, reference.RTE_same_policy_proxy_loss, equal_nan=True),
                             'RTE proxy loss differs from its own same-policy reference')
        self.report['peak_extension_source_checked'] = True

    def check(self):
        self.manifest()
        self.forecasts_and_ledgers()
        self.fit_records()
        self.frozen_selection()
        self.notebooks()
        self.scorecards()
        self.development_policy()
        self.peak_extension()
        self.report.update(errors=self.errors, pending=self.pending, warnings=self.warnings,
                           status='failed' if self.errors else ('incomplete' if self.pending else 'passed'))
        return self.report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='Defaults to .runtime/latest_research_run.json')
    parser.add_argument('--allow-incomplete', action='store_true', help='Allow missing/in-progress artifacts; integrity errors still fail')
    args = parser.parse_args(argv)
    if args.run_dir is None:
        pointer = ROOT/'.runtime/latest_research_run.json'
        if not pointer.exists():
            print(json.dumps({'status': 'incomplete' if args.allow_incomplete else 'failed',
                              'errors': [] if args.allow_incomplete else ['No latest research run pointer'],
                              'pending': ['No latest research run pointer'] if args.allow_incomplete else []}, indent=2))
            return 0 if args.allow_incomplete else 1
        args.run_dir = Path(json.loads(pointer.read_text())['run_dir'])
    run = args.run_dir if args.run_dir.is_absolute() else ROOT/args.run_dir
    report = ArtifactChecker(ROOT, run, args.allow_incomplete).check()
    print(json.dumps(report, indent=2, default=str))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
