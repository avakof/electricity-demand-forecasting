"""Frozen, causal RTE feature/risk experiment. Research only; no deployment."""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .common import fingerprint, forecast_origin, write_json
from .features import prediction_fallback
from .research_data import load_research_contract
from .rte_feature_lab import prepare_feature_lab
from .rte_learned_features import build_learned_feature_bank
from .rte_base_replay import load_causal_base_bank


def read_frame(path):
    f = pd.read_csv(path, low_memory=False)
    for col in ('origin', 'delivery_time', 'fit_origin'):
        if col in f:
            f[col] = pd.to_datetime(f[col], utc=True)
    return f


def cold_mask(frame):
    hour = pd.DatetimeIndex(frame.delivery_time).tz_convert('Europe/Paris').hour
    return np.isin(hour, [18, 19, 20]) & frame.realized_temperature_c.lt(5).to_numpy()


def point_scores(predictions, labels):
    f = predictions.merge(labels[['delivery_time', 'actual_mw', 'realized_temperature_c']],
                          on='delivery_time', validate='many_to_one')
    good = f.dropna(subset=['actual_mw', 'prediction_mw'])
    error = good.actual_mw.to_numpy()-good.prediction_mw.to_numpy()
    cold = cold_mask(good)
    weights = 1+4*cold
    return {'n': len(good), 'missing_target_n': int(f.actual_mw.isna().sum()),
            'MAE_MW': float(np.mean(np.abs(error))), 'RMSE_MW': float(np.sqrt(np.mean(error**2))),
            'bias_prediction_minus_actual_MW': float(-np.mean(error)),
            'weighted_MAE_MW': float(np.sum(weights*np.abs(error))/np.sum(weights)),
            'weighted_loss_sum': float(np.sum(weights*np.abs(error))), 'weight_sum': float(np.sum(weights)),
            'cold_MAE_MW': float(np.mean(np.abs(error[cold]))) if cold.any() else np.nan,
            'cold_n': int(cold.sum()), 'cold_days': int(good.loc[cold, 'local_day'].nunique()),
            'raw_availability': float(f.raw_prediction_mw.notna().mean())}


def candidate_variants(base_names):
    variants = {}
    def add(family, arm, tier='stacked', weight=5, mode='residual', base=None, restricted=False):
        name = f'lab__{family}__{arm}__{tier}__w{weight}__{mode}'
        if base:
            name = f'lab__hybrid__{base}'
        elif restricted:
            name = f'lab__hybrid_control__{arm}'
        variants[name] = dict(model=name, family=family, arm=arm, tier=tier,
                              cold_weight=weight, mode=mode, base=base,
                              restrict_to_base_burnin=bool(base or restricted))
    for arm in ('national', 'regional_only', 'combined'):
        for family in ('hist_gradient_boosting', 'lightgbm', 'lightgbm_quantile', 'gam'):
            add(family, arm)
        for tier in ('basic', 'engineered', 'stacked'):
            for weight in (1, 5):
                add('lightgbm', arm, tier, weight)
        add('lightgbm', arm, mode='direct')
    for base in base_names:
        add('lightgbm', 'combined' if base.endswith('__regional') else 'national', base=base)
    for arm in ('national', 'combined'):
        add('lightgbm', arm, restricted=True)
    return variants


def configurations(family):
    # The two candidates change capacity/regularization; window/retraining policy
    # stays constant to avoid conflating a feature ablation with a different policy.
    if family == 'gam':
        return [dict(n_knots=3, alpha=100.), dict(n_knots=4, alpha=1000.)]
    if family == 'hist_gradient_boosting':
        return [dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=96, l2_regularization=5.),
                dict(max_iter=140, max_leaf_nodes=31, min_samples_leaf=144, l2_regularization=20.)]
    return [dict(n_estimators=120, num_leaves=15, min_child_samples=96, reg_lambda=5.),
            dict(n_estimators=200, num_leaves=31, min_child_samples=144, reg_lambda=20.)]


class WeightedAdapter:
    def __init__(self, family, params, q=None):
        self.family, self.params, self.q = family, dict(params), q

    def fit(self, X, y, weights):
        self.columns = list(X.columns)
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        if self.family == 'gam':
            estimator = Ridge(alpha=self.params['alpha'], solver='lsqr', tol=1e-3, max_iter=200)
            steps = [('imputer', imputer), ('spline', SplineTransformer(n_knots=self.params['n_knots'],
                     degree=3, include_bias=False, extrapolation='linear')), ('scale', StandardScaler()), ('model', estimator)]
        elif self.family == 'hist_gradient_boosting':
            args = dict(self.params, learning_rate=.06, early_stopping=False, random_state=42,
                        loss='absolute_error' if self.q is None else 'quantile')
            if self.q is not None:
                args['quantile'] = self.q
            steps = [('imputer', imputer), ('model', HistGradientBoostingRegressor(**args))]
        else:
            from lightgbm import LGBMRegressor
            q = .5 if self.q is None and self.family == 'lightgbm_quantile' else self.q
            args = dict(self.params, learning_rate=.05, n_jobs=1, verbosity=-1, random_state=42,
                        deterministic=True, force_col_wise=True, objective='regression_l1' if q is None else 'quantile')
            if q is not None:
                args['alpha'] = q
            steps = [('imputer', imputer), ('model', LGBMRegressor(**args))]
        self.estimator = Pipeline(steps)
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.estimator.fit(X.replace([np.inf, -np.inf], np.nan), y, model__sample_weight=weights)
        self.warnings = sorted(set(str(x.message) for x in caught))
        return self

    def predict(self, X):
        with threadpool_limits(limits=1):
            return self.estimator.predict(X[self.columns].replace([np.inf, -np.inf], np.nan))


class RiskLabRun:
    def __init__(self, root, path):
        self.root, self.path = Path(root), Path(path)
        self.spec = json.loads((self.root/'config/rte_risk_experiment.json').read_text())
        self.contract = load_research_contract(root)
        self._data = self._frame = None
        self.path.mkdir(parents=True, exist_ok=True)
        for directory in ('forecasts', 'fits', 'ledgers', 'tables', 'figures'):
            (self.path/directory).mkdir(exist_ok=True)

    @classmethod
    def create(cls, root):
        root = Path(root)
        path = root/'reports/rte_risk'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run = cls(root, path)
        run.write_manifest()
        write_json(root/'.runtime/latest_rte_risk_run.json', {'run_dir': str(path)})
        return run

    @classmethod
    def latest(cls, root):
        return cls(root, json.loads((Path(root)/'.runtime/latest_rte_risk_run.json').read_text())['run_dir'])

    def write_manifest(self):
        paths = [self.root/'config/rte_risk_experiment.json']
        paths += [self.root/'src'/f for f in ('rte_risk_lab.py', 'rte_risk_uncertainty.py', 'rte_feature_lab.py', 'rte_learned_features.py', 'rte_base_replay.py')]
        original = json.loads((self.root/self.spec['reference_run']/'manifest.json').read_text())
        source = dict(original['source_sha256'])
        source.update({str(p.relative_to(self.root)): fingerprint(p) for p in paths})
        for name, sha in source.items():
            assert fingerprint(self.root/name) == sha, name
        write_json(self.path/'manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
                   'source_sha256': source, 'raw_sha256': original['raw_sha256'], 'spec': self.spec,
                   'original_manifest_sha256': fingerprint(self.root/self.spec['reference_run']/'manifest.json')})

    def verify_manifest(self):
        manifest = json.loads((self.path/'manifest.json').read_text())
        assert fingerprint(self.root/self.spec['reference_run']/'manifest.json') == manifest['original_manifest_sha256']
        for name, sha in manifest['source_sha256'].items():
            assert fingerprint(self.root/name) == sha, f'Frozen source changed: {name}'
        for name, sha in manifest['raw_sha256'].items():
            assert fingerprint(self.root/'data'/name) == sha, f'Raw data changed: {name}'
        return True

    @property
    def data(self):
        if self._data is None:
            self._data = prepare_feature_lab(self.root)
        return self._data

    @property
    def frame(self):
        if self._frame is None:
            self.verify_manifest()
            print('Preparing causal features, learned feature bank and archived base forecasts', flush=True)
            learned = build_learned_feature_bank(self.data, cache_dir=self.path/'learned_features')
            write_json(self.path/'learned_feature_fits.json', learned.attrs.get('fit_ledger', []))
            base, ledger = load_causal_base_bank(self.root, self.data.labels)
            write_json(self.path/'base_forecast_provenance.json', ledger)
            f = self.data.features.merge(learned, on=['origin', 'delivery_time'], how='left', validate='one_to_one')
            f = f.merge(base, on=['origin', 'delivery_time'], how='left', validate='one_to_one')
            self._frame = f
            self.data.feature_register().to_csv(self.path/'tables/feature_register.csv', index=False)
            names = [c.removeprefix('base__').removesuffix('__prediction') for c in base if c.endswith('__prediction')]
            write_json(self.path/'variant_register.json', candidate_variants(names))
        return self._frame

    def columns(self, variant):
        arm = variant['arm']
        tier = 'basic' if variant['tier'] == 'basic' else 'engineered'
        columns = self.data.columns(arm, tier=tier)
        if variant['tier'] == 'stacked':
            columns += [f'learned_{name}__{arm}' for name in ('ridge_demand_mw', 'cold_probability', 'underprediction_probability')]
        if variant.get('base'):
            columns += [f"base__{variant['base']}__{suffix}" for suffix in ('prediction', 'missing', 'fallback')]
        assert len(columns) == len(set(columns))
        assert not {'actual_mw', 'realized_temperature_c', 'oracle_temperature_c'}.intersection(columns)
        return columns

    def replay(self, variant, params, start, end, phase, q=None, fixed=False):
        self.verify_manifest()
        if pd.Timestamp(end) > pd.Timestamp(self.spec['test']['start']) and not (self.path/'frozen_selection.json').exists():
            raise AssertionError('Freeze model selection before producing any test-overlapping predictions')
        model = variant['model']
        identity = dict(variant=variant, params=params, start=start, end=end, phase=phase, q=q, fixed=fixed)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:14]
        stem = f'{model}__{phase}__{key}'
        path = self.path/'forecasts'/f'{stem}.csv.gz'
        seal = path.with_suffix('').with_suffix('.json')
        fitpath = self.path/'fits'/f'{stem}.json'
        ledgerpath = self.path/'ledgers'/f'{stem}.jsonl.gz'
        if seal.exists():
            saved = json.loads(seal.read_text())
            for p, field in ((path,'sha256'), (fitpath,'fits_sha256'), (ledgerpath,'ledger_sha256')):
                assert fingerprint(p) == saved[field], f'Altered forecast cache: {p}'
            return read_frame(path)
        print(f'REPLAY {model} {phase} q={q}', flush=True)
        features = self.frame
        sample = features.loc[features.local_day.ge(start) & features.local_day.lt(end)].copy()
        if sample.empty:
            raise ValueError(f'Empty phase {start} {end}')
        columns = self.columns(variant)
        groups = [(0, sample)] if fixed else sample.groupby(pd.PeriodIndex(sample.local_day, freq='Q'), sort=True)
        outputs, fits = [], []
        # An interrupted, unsealed replay is rebuilt, never appended twice.
        if ledgerpath.exists():
            ledgerpath.unlink()
        for _, block in groups:
            origin = block.origin.min()
            train = features.loc[features.origin.lt(origin)].merge(self.data.labels[
                ['delivery_time','actual_mw','label_available_at']], on='delivery_time', validate='many_to_one')
            train = train.loc[train.label_available_at.lt(origin) & train.actual_mw.notna() & train.rte_at_origin_mw.gt(0)]
            if variant.get('restrict_to_base_burnin'):
                train = train.loc[train.local_day.ge('2024-01-01')]
            if variant.get('base'):
                train = train.loc[train[f"base__{variant['base']}__prediction"].notna()]
            assert train.label_available_at.max() < origin
            valid_history = train.max_history_available_at.notna()
            assert (train.loc[valid_history,'max_history_available_at'] < train.loc[valid_history,'origin']).all()
            if len(train) < 24*60:
                raise ValueError('Insufficient released training labels')
            target = train.actual_mw.to_numpy().copy()
            if variant['mode'] == 'residual':
                target -= train.rte_at_origin_mw.to_numpy()
            probability = train[f"learned_cold_probability__{variant['arm']}"].fillna(.0).clip(0,1).to_numpy()
            weights = 1+(variant['cold_weight']-1)*probability*train.hour.isin([18,19,20]).to_numpy()
            started = time.perf_counter()
            estimator = WeightedAdapter(variant['family'], params, q).fit(train[columns], target, weights)
            raw = estimator.predict(block[columns])
            if variant['mode'] == 'residual':
                raw += block.rte_at_origin_mw.to_numpy()
            raw[~np.isfinite(raw) | (raw <= 0) | ~block.rte_at_origin_mw.gt(0).to_numpy()] = np.nan
            if variant.get('base'):
                raw[block[f"base__{variant['base']}__prediction"].isna().to_numpy()] = np.nan
            # Regional-only fallback cannot quietly consume national history.
            if variant['arm'] == 'regional_only':
                prediction = pd.Series(raw, index=block.index).fillna(block.rte_at_origin_mw.where(block.rte_at_origin_mw.gt(0)))
                source = pd.Series(np.where(np.isfinite(raw), 'model', 'rte'), index=block.index)
                source.loc[prediction.isna()] = 'unavailable'
            else:
                fallback_frame = block.copy()
                fallback_frame['rte_at_origin_mw'] = fallback_frame.rte_at_origin_mw.where(fallback_frame.rte_at_origin_mw.gt(0))
                prediction, source = prediction_fallback(fallback_frame, raw, rte_assisted=True)
            out = block[['origin','delivery_time','local_day']].copy()
            out['model'], out['family'], out['arm'] = model, variant['family'], variant['arm']
            out['prediction_mw'], out['raw_prediction_mw'] = prediction.to_numpy(), raw
            out['fallback_used'], out['fallback_source'] = ~np.isfinite(raw), source.to_numpy()
            out['fit_origin'] = origin
            out['quantile_level'] = q
            sample_train = train.tail(min(len(train), 2000))
            fitted = estimator.predict(sample_train[columns])
            train_y = sample_train.actual_mw.to_numpy()-(sample_train.rte_at_origin_mw.to_numpy() if variant['mode']=='residual' else 0)
            record = dict(origin=str(origin), training_rows=len(train), predictor_columns=columns,
                          latest_label_available_at=str(train.label_available_at.max()),
                          training_start=str(train.delivery_time.min()), target_mode=variant['mode'],
                          weight_min=float(weights.min()), weight_max=float(weights.max()),
                          mean_training_weight=float(weights.mean()), q=q, status='fit',
                          sampled_train_MAE_MW=float(np.mean(np.abs(train_y-fitted))),
                          training_diagnostic_rows=len(sample_train), warnings=estimator.warnings,
                          elapsed_seconds=time.perf_counter()-started)
            fits.append(record)
            with gzip.open(ledgerpath, 'at', encoding='utf-8') as stream:
                for day_origin, day in out.groupby('origin', sort=True):
                    stream.write(json.dumps({'origin':str(day_origin), 'saved_at_utc':datetime.now(timezone.utc).isoformat(),
                        'predictions':json.loads(day.to_json(orient='records', date_format='iso'))})+'\n')
            outputs.append(out)
            del estimator
        result = pd.concat(outputs, ignore_index=True)
        assert 'actual_mw' not in result and not result.duplicated(['origin','delivery_time']).any()
        result.to_csv(path, index=False)
        write_json(fitpath, fits)
        write_json(seal, dict(**identity, rows=len(result), predictor_columns=columns,
                             created_utc=datetime.now(timezone.utc).isoformat(), sha256=fingerprint(path),
                             fits_sha256=fingerprint(fitpath), ledger_sha256=fingerprint(ledgerpath)))
        return result

    def tune(self):
        self.verify_manifest()
        if (self.path/'frozen_selection.json').exists():
            return self.freeze()['selections']
        _ = self.frame
        variants = json.loads((self.path/'variant_register.json').read_text())
        selections, trials = {}, []
        for model, variant in variants.items():
            options = []
            for trial, params in enumerate(configurations(variant['family'])):
                predictions = []
                for fold in self.spec['validation_folds']:
                    p = self.replay(variant, params, fold['start'], fold['end'], f"{fold['name']}_trial{trial}")
                    scores = point_scores(p, self.data.labels)
                    trials.append(dict(model=model,trial=trial,fold=fold['name'],params_json=json.dumps(params),**scores))
                    predictions.append(p)
                pooled = point_scores(pd.concat(predictions), self.data.labels)
                options.append(dict(variant=variant, params=params, trial=trial, metrics=pooled))
            eligible = [o for o in options if o['metrics']['raw_availability'] >= .99]
            if not eligible:
                raise ValueError(f'No eligible configuration for {model}')
            selections[model] = min(eligible, key=lambda x:(x['metrics']['weighted_MAE_MW'], x['trial']))
            write_json(self.path/'selected_parameters.json', selections)
            pd.DataFrame(trials).to_csv(self.path/'tables/tuning_trials.csv', index=False)
            print(f"SELECTED {model}: {selections[model]['metrics']['weighted_MAE_MW']:.3f}", flush=True)
        return selections

    def freeze(self):
        frozen_path = self.path/'frozen_selection.json'
        if frozen_path.exists():
            saved = json.loads(frozen_path.read_text())
            for name, sha in saved['selection_inputs_sha256'].items():
                assert fingerprint(self.path/name) == sha, f'Frozen selection inputs changed: {name}'
            return saved
        for path in (self.path/'forecasts').glob('lab__*.json'):
            saved = json.loads(path.read_text())
            if pd.Timestamp(saved['end']) > pd.Timestamp(self.spec['test']['start']):
                raise AssertionError('Cannot first freeze after forecasts overlapping test exist')
        selections = json.loads((self.path/'selected_parameters.json').read_text())
        variants = json.loads((self.path/'variant_register.json').read_text())
        assert set(selections) == set(variants)
        trials = pd.read_csv(self.path/'tables/tuning_trials.csv')
        assert len(trials) == len(variants)*2*len(self.spec['validation_folds'])
        for model, group in trials.groupby('model'):
            assert group.groupby('trial').fold.nunique().eq(len(self.spec['validation_folds'])).all()
        use = np.zeros(len(self.frame),dtype=bool)
        for fold in self.spec['validation_folds']:
            use |= (self.frame.local_day.ge(fold['start']) & self.frame.local_day.lt(fold['end'])).to_numpy()
        dev = self.frame.loc[use]
        baseline = dev[['origin','delivery_time','local_day']].copy()
        baseline['prediction_mw'] = dev.rte_at_origin_mw.to_numpy()+dev.rte_bias_recent_mw.to_numpy()
        baseline['raw_prediction_mw'] = baseline.prediction_mw
        bias = point_scores(baseline, self.data.labels)
        eligible = {k:v for k,v in selections.items() if v['metrics']['MAE_MW'] <= 1.02*bias['MAE_MW']
                    and v['metrics']['cold_MAE_MW'] <= bias['cold_MAE_MW']
                    and v['metrics']['cold_n'] >= 60 and v['metrics']['cold_days'] >= 20}
        diagnostic = min(selections, key=lambda k:selections[k]['metrics']['weighted_MAE_MW'])
        winner = min(eligible, key=lambda k:eligible[k]['metrics']['weighted_MAE_MW']) if eligible else None
        probability = []
        from .rte_risk_uncertainty import assemble_quantiles, quantile_scores
        for model, entry in selections.items():
            v = entry['variant']
            if v['family'] != 'lightgbm_quantile' or v.get('base'):
                continue
            folds = [assemble_quantiles(self, entry, f['start'], f['end'], f"probability_{f['name']}") for f in self.spec['validation_folds']]
            p = pd.concat(folds, ignore_index=True)
            score = quantile_scores(p, self.data.labels)
            probability.append(dict(model=model, **score))
        pd.DataFrame(probability).to_csv(self.path/'tables/probability_development.csv', index=False)
        best_quantile = min(probability, key=lambda p:p['WIS_MW'])['model']
        frozen = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), selections=selections,
                      selected_point=winner, diagnostic_best_point=diagnostic, selected_quantile=best_quantile,
                      development_bias_reference=bias, passing_guardrails=list(eligible),
                      test_previously_explored=True, selection_inputs_sha256={name:fingerprint(self.path/name) for name in (
                          'selected_parameters.json','tables/tuning_trials.csv','tables/probability_development.csv')})
        write_json(frozen_path, frozen)
        return frozen

    def evaluate(self):
        self.verify_manifest()
        frozen = self.freeze()
        results = {}
        for phase in ('calibration','test'):
            parts=[]
            for entry in frozen['selections'].values():
                interval = self.spec[phase]
                parts.append(self.replay(entry['variant'], entry['params'], interval['start'], interval['end'], phase))
            results[phase] = pd.concat(parts, ignore_index=True)
            results[phase].to_csv(self.path/'forecasts'/f'lab_{phase}.csv.gz', index=False)
        return results
