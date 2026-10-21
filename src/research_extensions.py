"""Pre-test development sensitivities and input stress tests; research only."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .common import forecast_origin
from .features import HISTORY, PAST_WEATHER, prediction_fallback
from .metrics import point_metrics
from .research_models import make_tabular


def development_sensitivities(run):
    selected = json.loads((run.path/"selected_parameters.json").read_text())
    tabular = {key: value for key, value in selected.items() if value["status"] == "selected"
               and value["arm"] == "national" and value["family"] in ("lightgbm", "hist_gradient_boosting", "gam")}
    neural = {key: value for key, value in selected.items() if value["status"] == "selected"
              and value["family"] in ("nbeats", "nhits", "tft", "gru", "patchtst", "dlinear")}
    rows = []
    best_key = min(tabular, key=lambda k: tabular[k]["MAE_MW"])
    entry = tabular[best_key]
    specifications = [("matched_expanding", dict(entry["params"]), False),
                      ("matched_365day_window", {**entry["params"], "window_days": 365}, False),
                      ("matched_weekly_refit", {**entry["params"], "refit_days": 7}, False),
                      ("national_full_history", dict(entry["params"]), True)]
    for label, params, full_history in specifications:
        predictions = []
        for fold in run.contract["validation_folds"]:
            predictions.append(run.replay(entry["family"], params, fold["start"], fold["end"],
                                          tag=f"{best_key}__{label}", phase=f"sensitivity_{fold['name']}",
                                          full_history=full_history))
        joined = pd.concat(predictions, ignore_index=True).merge(run.data.labels, on="delivery_time", validate="many_to_one")
        rows.append({"model": best_key, "sensitivity": label, "seed": 42, **point_metrics(joined)})
    if neural:
        best_neural = min(neural, key=lambda k: neural[k]["MAE_MW"])
        entry = neural[best_neural]
        # Never select the best random seed. Report every predetermined seed.
        for seed in run.contract["neural_finalist_seeds"]:
            if seed == 42:
                rows.append({"model": best_neural, "sensitivity": "neural_seed", "seed": seed, "MAE_MW": entry["MAE_MW"]})
                continue
            predictions = []
            for fold in run.contract["validation_folds"]:
                predictions.append(run.replay(entry["family"], entry["params"], fold["start"], fold["end"],
                                              arm=entry["arm"], seed=seed, tag=f"{best_neural}__seed{seed}",
                                              phase=f"sensitivity_{fold['name']}"))
            joined = pd.concat(predictions, ignore_index=True).merge(run.data.labels, on="delivery_time", validate="many_to_one")
            rows.append({"model": best_neural, "sensitivity": "neural_seed", "seed": seed, **point_metrics(joined)})
    result = pd.DataFrame(rows)
    result.to_csv(run.path/"tables/development_sensitivities.csv", index=False)
    return result


def stress_inputs(run):
    """Freeze one development fit, perturb only predictor inputs, count fallback."""
    selected = json.loads((run.path/"selected_parameters.json").read_text())
    entry = selected["lightgbm__regional"]
    if entry["status"] != "selected":
        return pd.DataFrame()
    origin = forecast_origin("2024-12-01", run.contract)
    train = run.data.training_rows(origin)
    columns = run.data.columns("regional")
    model = make_tabular("lightgbm", entry["params"]).fit(train[columns], train.actual_mw)
    evaluation = run.data.phase("2024-12-01", "2025-01-01")
    rows = []
    for scenario in ("clean", "missing_demand_history", "missing_regional_inputs", "stale_past_weather", "rapid_demand_change", "extreme_past_cold", "missing_rte"):
        changed = evaluation.copy()
        if scenario == "missing_demand_history":
            changed[HISTORY] = np.nan
        elif scenario == "missing_regional_inputs":
            changed[run.data.region_columns] = np.nan
        elif scenario == "stale_past_weather":
            changed[PAST_WEATHER] = np.nan
        elif scenario == "rapid_demand_change":
            changed[["last_load", "mean_24h"]] *= 1.2
        elif scenario == "extreme_past_cold":
            changed[PAST_WEATHER] -= 10
        elif scenario == "missing_rte":
            changed[["rte_at_origin_mw", "rte_bias_recent_mw"]] = np.nan
        raw = model.predict(changed[columns])
        # Declare a completeness guard for this stress policy; do not call
        # imputed predictions reliable when all demand history disappeared.
        raw[changed[HISTORY].notna().sum(axis=1).to_numpy() == 0] = np.nan
        prediction, source = prediction_fallback(changed, raw)
        out = changed[["delivery_time"]].copy()
        out["prediction_mw"] = prediction.to_numpy()
        joined = out.merge(run.data.labels, on="delivery_time", validate="many_to_one")
        rows.append({"scenario": scenario, "model": "lightgbm__regional", "fallback_fraction": float(np.isnan(raw).mean()),
                     "policy_availability": float(np.isfinite(prediction).mean()), **point_metrics(joined)})
    result = pd.DataFrame(rows)
    result.to_csv(run.path/"tables/development_stress_tests.csv", index=False)
    return result
