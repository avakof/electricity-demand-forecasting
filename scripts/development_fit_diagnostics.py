"""Reporting-only diagnostics from saved development forecasts and fit records.

No models are fitted and no calibration/test predictions or target values are
read. Raw labels are streamed only through the development interval; the first
later row is checked by timestamp and its demand value is never parsed.

An in-sample/later-period gap is descriptive. Different regimes, sample sizes,
selection and overlapping training histories prevent a formal bias/variance
decomposition or a diagnosis of overfitting from this gap alone.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _labels(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    records = []
    previous = None
    with (root / "data/france_electricity_hourly.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            stamp = datetime.fromisoformat(row["datetime_utc"].replace("+0000", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("Raw demand timestamps must have an explicit UTC offset")
            stamp = stamp.astimezone(timezone.utc)
            if previous is not None and stamp <= previous:
                raise ValueError("Raw demand timestamps must be unique and increasing")
            previous = stamp
            if stamp >= end.to_pydatetime():
                break  # Never parse the demand of a later-period row.
            if stamp < start.to_pydatetime():
                continue
            value = float(row["load_mw"]) if row["load_mw"].strip() else np.nan
            records.append({"delivery_time": stamp, "actual_mw": value})
    result = pd.DataFrame(records, columns=["delivery_time", "actual_mw"])
    result["delivery_time"] = pd.to_datetime(result.delivery_time, utc=True)
    return result


def _functional(metadata: dict, family: str) -> tuple[str, str]:
    objective = metadata.get("objective") or metadata.get("training_objective") or "not recorded"
    quantile = metadata.get("quantile")
    if quantile is not None:
        functional = "conditional median" if float(quantile) == .5 else f"conditional quantile q={quantile}"
    elif "median" in objective.lower():
        functional = "conditional median"
    elif "mean" in objective.lower():
        functional = "conditional mean"
    elif metadata.get("implementation") == "neuralforecast":
        functional = "median-oriented MAE on scaled bridge targets"
    else:
        functional = "not recorded"
    return objective, functional


def _metrics(frame: pd.DataFrame) -> dict:
    actual = pd.to_numeric(frame.actual_mw, errors="coerce")
    eligible = np.isfinite(actual)
    result = {"n_forecast_intervals": len(frame), "n_labels": int(eligible.sum()),
              "n_missing_labels": int((~eligible).sum())}
    for prefix, column in (("raw", "raw_prediction_mw"), ("policy", "prediction_mw")):
        prediction = pd.to_numeric(frame[column], errors="coerce")
        observed = eligible & np.isfinite(prediction)
        result[f"n_{prefix}_scored"] = int(observed.sum())
        result[f"{prefix}_availability"] = float(observed.sum()/eligible.sum()) if eligible.any() else np.nan
        result[f"oos_{prefix}_MAE_MW"] = float((actual[observed]-prediction[observed]).abs().mean())
    return result


def _plot_gaps(gaps: pd.DataFrame, path: Path) -> bool:
    available = gaps.loc[np.isfinite(gaps.train_MAE_MW) & np.isfinite(gaps.oos_raw_MAE_MW)]
    if available.empty:
        return False
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import colormaps
    figure = Figure(figsize=(13, 7))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    colors = colormaps["tab10"]
    for i, (model, group) in enumerate(available.groupby("model", sort=True)):
        sizes = 15 + 65*np.sqrt(group.n_raw_scored / max(group.n_raw_scored.max(), 1))
        axes[0].scatter(group.train_MAE_MW, group.oos_raw_MAE_MW, s=sizes,
                        alpha=.65, color=colors(i % 10), label=model)
    limit = float(max(available.train_MAE_MW.max(), available.oos_raw_MAE_MW.max())*1.05)
    axes[0].plot([0, limit], [0, limit], color=".4", linestyle="--", linewidth=1)
    axes[0].set(xlim=(0, limit), ylim=(0, limit), xlabel="Saved in-sample MAE (MW)",
                ylabel="Following 2024 raw-forecast MAE (MW)", title="Same fitted predictor; each point is one refit")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, fontsize=8, loc="lower center",
                  bbox_to_anchor=(.5, .055), ncol=2, frameon=False)
    names = sorted(available.model.unique())
    boxes = axes[1].boxplot([available.loc[available.model.eq(name), "same_fit_gap_MW"] for name in names],
                           orientation="horizontal", patch_artist=True, tick_labels=names)
    for i, box in enumerate(boxes["boxes"]):
        box.set_facecolor(colors(i % 10)); box.set_alpha(.5)
    axes[1].axvline(0, color=".4", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Later raw MAE − saved training MAE (MW)", title="Descriptive fit-gap distribution")
    for axis in axes:
        axis.grid(axis="x", alpha=.15)
    figure.suptitle("Development diagnostics: training error versus later forecast error", fontsize=13)
    figure.text(.5, .015, "Different periods/regimes and overlapping fits: gaps do not prove overfitting or quantify formal model variance.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .22, 1, .95))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return True


def _plot_quarters(summary: pd.DataFrame, path: Path) -> bool:
    if summary.empty:
        return False
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    frame = summary.sort_values("pooled_policy_MAE_MW").reset_index(drop=True)
    figure = Figure(figsize=(10, max(4, .3*len(frame)+1.5)))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    positions = np.arange(len(frame))
    axis.hlines(positions, frame.quarter_MAE_min_MW, frame.quarter_MAE_max_MW, color="#93a9bb", linewidth=3)
    axis.scatter(frame.pooled_policy_MAE_MW, positions, color="#285473", zorder=3, s=25)
    axis.set(yticks=positions, yticklabels=frame.model, xlabel="2024 development MAE (MW)",
             title=f"Independent model variants ({len(frame)}): pooled 2024 MAE and quarterly range")
    axis.invert_yaxis(); axis.grid(axis="x", alpha=.15)
    figure.text(.5, .015, "Independent variants only; RTE and the preselected bias-correction policy are reported elsewhere.\n"
                "Ranges describe quarterly regime/seasonal dispersion, not confidence intervals or formal variance.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .065, 1, 1)); figure.savefig(path, dpi=150, bbox_inches="tight")
    return True


def build_diagnostics(run_dir: str | Path, root: str | Path) -> dict:
    """Compare saved SAME-FIT tabular train MAE with subsequent raw OOS MAE.

    Selected 2024 configurations only; no training or model selection is done.
    Models lacking an unscaled in-sample MAE receive an explicit unavailable gap.
    """
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    manifest = json.loads((run_dir/"manifest.json").read_text())
    contract = manifest["contract"]
    folds = contract["validation_folds"]
    zone = contract["timezone"]
    start = min(pd.Timestamp(f["start"], tz=zone).tz_convert("UTC") for f in folds)
    end = max(pd.Timestamp(f["end"], tz=zone).tz_convert("UTC") for f in folds)
    if start < pd.Timestamp("2023-12-31 23:00", tz="UTC") or end > pd.Timestamp("2025-01-01", tz="UTC"):
        raise ValueError("This reporting script is restricted to the predeclared 2024 development interval")
    labels = _labels(root, start, end)
    selected = json.loads((run_dir/"selected_parameters.json").read_text())
    gap_rows, quarter_rows, missing = [], [], []
    for model, choice in sorted(selected.items()):
        if choice.get("status") != "selected":
            continue
        trial = int(choice["trial"])
        for fold in folds:
            phase = f"{fold['name']}_trial{trial}"
            files = []
            for candidate in (run_dir/"forecasts").glob(f"{model}__{phase}__*.csv.gz"):
                stem = candidate.name.removesuffix(".csv.gz")
                metadata_path = candidate.with_name(stem+".json")
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    if metadata.get("params") == choice["params"] and metadata.get("phase") == phase:
                        files.append(candidate)
            if not files:
                missing.append({"model": model, "trial": trial, "fold": fold["name"], "reason": "completed selected-configuration cache absent"})
                continue
            if len(files) != 1:
                raise ValueError(f"Ambiguous completed cache for {model} {phase}")
            prediction_path = files[0]
            stem = prediction_path.name.removesuffix(".csv.gz")
            frame = pd.read_csv(prediction_path)
            frame["delivery_time"] = pd.to_datetime(frame.delivery_time, utc=True)
            frame["origin"] = pd.to_datetime(frame.origin, utc=True)
            frame["fit_origin"] = pd.to_datetime(frame.fit_origin, utc=True)
            lower = pd.Timestamp(fold["start"], tz=zone).tz_convert("UTC")
            upper = pd.Timestamp(fold["end"], tz=zone).tz_convert("UTC")
            if not frame.delivery_time.ge(lower).all() or not frame.delivery_time.lt(upper).all():
                raise ValueError(f"Prediction cache contains non-development/fold intervals: {prediction_path}")
            if not frame.model.eq(model).all() or frame.duplicated(["origin", "delivery_time"]).any():
                raise ValueError(f"Unexpected model or duplicate origin/target in {prediction_path}")
            frame = frame.merge(labels, on="delivery_time", how="left", validate="many_to_one")
            quarter_rows.append({"model": model, "family": choice["family"], "arm": choice["arm"],
                                 "trial": trial, "fold": fold["name"], **_metrics(frame)})
            fit_path = run_dir/"fits"/(stem+".json")
            fits = json.loads(fit_path.read_text()) if fit_path.exists() else []
            fit_map = {pd.to_datetime(record["origin"], utc=True): record for record in fits}
            for fit_origin, group in frame.groupby("fit_origin", dropna=False, sort=True):
                if pd.isna(fit_origin) or not group.origin.ge(fit_origin).all():
                    raise ValueError("A development forecast has missing/future fit origin")
                record = fit_map.get(fit_origin, {})
                metadata = record.get("metadata") or {}
                objective, functional = _functional(metadata, choice["family"])
                train = record.get("train_MAE_MW", np.nan)
                train = float(train) if train is not None else np.nan
                metrics = _metrics(group)
                gap = metrics["oos_raw_MAE_MW"]-train
                gap_rows.append({
                    "model": model, "family": choice["family"], "arm": choice["arm"],
                    "trial": trial, "fold": fold["name"], "fit_origin": fit_origin,
                    "fit_status": record.get("status", "fit_record_missing"),
                    "n_train_rows": record.get("training_rows", np.nan),
                    "n_train_observed_labels": metadata.get("observed_training_rows",
                        record.get("training_rows", np.nan)-metadata.get("masked_training_rows", 0)),
                    "objective": objective, "point_functional": functional,
                    "train_MAE_MW": train, **metrics, "same_fit_gap_MW": gap,
                    "same_fit_gap_ratio": metrics["oos_raw_MAE_MW"]/train if np.isfinite(train) and train > 0 else np.nan,
                    "gap_status": "available_descriptive_only" if np.isfinite(train) and np.isfinite(gap)
                                  else "unavailable_no_saved_unscaled_training_MAE_or_raw_predictions",
                })
    gaps = pd.DataFrame(gap_rows)
    quarters = pd.DataFrame(quarter_rows)
    summaries = []
    if not quarters.empty:
        for model, frame in quarters.groupby("model", sort=True):
            total = int(frame.n_policy_scored.sum())
            summaries.append({"model": model, "family": frame.family.iloc[0], "arm": frame.arm.iloc[0],
                              "trial": int(frame.trial.iloc[0]), "quarters_recorded": len(frame),
                              "quarters_expected": len(folds), "n_policy_scored": total,
                              "pooled_policy_MAE_MW": float((frame.oos_policy_MAE_MW*frame.n_policy_scored).sum()/total) if total else np.nan,
                              "quarter_MAE_min_MW": frame.oos_policy_MAE_MW.min(),
                              "quarter_MAE_max_MW": frame.oos_policy_MAE_MW.max(),
                              "quarter_MAE_sd_MW": frame.oos_policy_MAE_MW.std(ddof=1)})
    dispersion = pd.DataFrame(summaries)
    seed_path = run_dir/"tables/development_sensitivities.csv"
    seeds = pd.read_csv(seed_path) if seed_path.exists() else pd.DataFrame()
    seeds = seeds.loc[seeds.sensitivity.eq("neural_seed")].copy() if "sensitivity" in seeds else pd.DataFrame(columns=["model", "seed", "MAE_MW"])
    seed_summary = []
    for model, group in seeds.groupby("model", sort=True):
        if group.seed.duplicated().any():
            raise ValueError(f"Duplicate seed diagnostics for {model}")
        seed_summary.append({"model": model, "seeds_recorded": group.seed.nunique(),
                             "seeds_expected": len(contract.get("neural_finalist_seeds", [])),
                             "seed_MAE_mean_MW": group.MAE_MW.mean(), "seed_MAE_sd_MW": group.MAE_MW.std(ddof=1),
                             "seed_MAE_min_MW": group.MAE_MW.min(), "seed_MAE_max_MW": group.MAE_MW.max()})
    stability = pd.DataFrame(seed_summary)
    (run_dir/"tables").mkdir(exist_ok=True); (run_dir/"figures").mkdir(exist_ok=True)
    for name, frame in (("development_fit_gaps", gaps), ("development_quarter_scores", quarters),
                        ("development_quarter_dispersion", dispersion), ("development_seed_scores", seeds),
                        ("development_seed_stability", stability)):
        frame.to_csv(run_dir/"tables"/(name+".csv"), index=False)
    missing_frame = pd.DataFrame(missing, columns=["model", "trial", "fold", "reason"])
    missing_frame.to_csv(run_dir/"tables/development_diagnostic_exclusions.csv", index=False)
    figures = []
    if not gaps.empty and _plot_gaps(gaps, run_dir/"figures/development_fit_gaps.png"):
        figures.append(run_dir/"figures/development_fit_gaps.png")
    if _plot_quarters(dispersion, run_dir/"figures/development_quarter_dispersion.png"):
        figures.append(run_dir/"figures/development_quarter_dispersion.png")
    notes_path = run_dir/"development_diagnostics.md"
    notes_path.write_text(
        "# Development fit diagnostics\n\n"
        "Saved selected-configuration forecasts from the predeclared 2024 folds only. "
        "No model was fitted and no calibration/test target values or forecast files were read.\n\n"
        "Each fit-gap row compares the saved in-sample MAE of one fitted predictor with its raw "
        "forecasts on subsequent observations until the next fit or fold end. Fallback predictions "
        "are reported separately because they are not outputs of the same fitted predictor. "
        "Training MAE is resubstitution error and the later observations can have different regimes; "
        "this gap does not prove overfitting or provide a formal bias/variance decomposition. "
        "Models without a saved unscaled in-sample MAE have an explicitly unavailable gap; "
        "scaled neural training trajectories must not be substituted.\n\n"
        "Quarter ranges and standard deviations describe observed seasonal/regime dispersion; "
        "quarters share overlapping training histories. They are not independent replicates or "
        "confidence intervals. Seed summaries include every recorded predeclared seed without "
        "selecting a lucky seed.\n\n"
        f"Fit groups: {len(gaps)}; completed model-quarter groups: {len(quarters)}; "
        f"missing completed caches: {len(missing)}; recorded seed rows: {len(seeds)}.\n"
    )
    return {"fit_gaps": gaps, "quarter_scores": quarters, "quarter_dispersion": dispersion,
            "seed_scores": seeds, "seed_stability": stability, "exclusions": missing_frame,
            "figure_paths": figures, "notes_path": notes_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = build_diagnostics(args.run_dir, args.root)
    print(f"Saved {len(result['fit_gaps'])} development fit groups; "
          f"{len(result['quarter_scores'])} selected model-quarter scores; "
          f"{len(result['seed_scores'])} seed rows. {result['notes_path']}")


if __name__ == "__main__":
    main()
