"""Reporting-only plots with roles chosen from pre-test selection records.

Functions receive already computed score tables. They neither read outcomes nor
choose a model/interval method from historical comparison scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EMPIRICAL = "fixed_past_out_of_sample_residual_distribution"


def _name(model: str) -> str:
    names = {"rte_bias_proxy": "RTE + bias correction", "rte_proxy": "RTE",
             "gam__national": "GAM (national)", "gam__regional": "GAM (regional history)"}
    return names.get(model, model.replace("__", " / ").replace("_", " "))


def predeclared_plot_roles(run_dir: str | Path) -> list[dict]:
    """Read selection seals only; never select using a comparison-period score."""
    run_dir = Path(run_dir)
    policy = json.loads((run_dir/"development_policy_selection.json").read_text())
    frozen = json.loads((run_dir/"frozen_selection.json").read_text())
    peak = json.loads((run_dir/"peak_interval_selection.json").read_text())
    primary = frozen["primary_candidate_from_validation"]
    if peak["primary"] != primary:
        raise ValueError("Peak interval seal does not match the independent-model selection")
    roles = [{"role": "complete_policy", "model": policy["selected_policy"],
              "method": EMPIRICAL, "title": f"Complete policy: {_name(policy['selected_policy'])}",
              "method_label": "Fixed past OOS residuals", "color": "#24744e", "marker": "D"}]
    for role, model, color, marker, title in (
        ("independent_model", primary, "#236b8e", "o", f"Independent model: {_name(primary)}"),
        ("rte_reference", "rte_proxy", "#c28635", "s", "RTE reference"),
    ):
        choices = [row for row in peak["records"] if row["model"] == model and row["selected"] is True]
        if len(choices) != 1 or choices[0]["candidate"] not in ("conditional", "pooled"):
            raise ValueError(f"Exactly one predeclared peak interval method is required for {model}")
        conditional = choices[0]["candidate"] == "conditional"
        roles.append({"role": role, "model": model,
                      "method": "peak_conditioned_empirical_residuals" if conditional else "peak_pooled_empirical_residuals",
                      "title": title, "method_label": "Selected conditional residuals" if conditional else "Selected pooled residuals",
                      "color": color, "marker": marker})
    return roles


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _count(value) -> str:
    number = _number(value)
    return str(int(number)) if number is not None else "unavailable"


def _figure(size):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=size)
    FigureCanvasAgg(figure)
    return figure


def _point(frame, spec, coverage_column):
    missing = {**spec, "width_MW": np.nan, coverage_column: np.nan,
               "interval_nobs": np.nan, "ndays": np.nan}
    rows = frame.loc[frame.model.eq(spec["model"]) & frame.method.eq(spec["method"])]
    if len(rows) != 1:
        return None, {**missing, "plot_status": "missing_or_ambiguous_predeclared_point"}
    row = rows.iloc[0]
    x, y = _number(row.get("width_MW")), _number(row.get(coverage_column))
    if x is None or x < 0 or y is None or not 0 <= y <= 1:
        return None, {**missing, "plot_status": "nonfinite_or_invalid_predeclared_point"}
    return row, {**spec, "plot_status": "plotted", "width_MW": x,
                 coverage_column: y, "interval_nobs": row.get("interval_nobs"), "ndays": row.get("ndays")}


def _background(axis, frame, y_column):
    x = pd.to_numeric(frame.width_MW, errors="coerce")
    y = pd.to_numeric(frame[y_column], errors="coerce")
    valid = np.isfinite(x) & x.ge(0) & np.isfinite(y) & y.between(0, 1)
    axis.scatter(x[valid], y[valid], color="#b7bdc4", s=24, alpha=.55, zorder=1,
                 label="Other model/method pairs")
    if valid.any():
        left, right = float(x[valid].min()), float(x[valid].max())
        span = max(right-left, right*.15, 100.)
        axis.set_xlim(max(0., left-.06*span), right+.08*span)
    return y[valid]


def _annotate(axis, row, spec, y_column, position, extra=""):
    label = (f"{spec['title']}\n{spec['method_label']}\n"
             f"n={_count(row.get('interval_nobs'))}; observed days={_count(row.get('ndays'))}")
    if extra:
        label += "\n"+extra
    axis.annotate(label, xy=(float(row.width_MW), float(row[y_column])),
                  xytext=(1.045, position), textcoords="axes fraction",
                  ha="left", va="top", fontsize=9, color=spec["color"],
                  bbox={"boxstyle": "round,pad=.4", "facecolor": "white", "edgecolor": spec["color"], "alpha": .96},
                  arrowprops={"arrowstyle": "-", "color": spec["color"], "linewidth": 1.1},
                  annotation_clip=False, zorder=5)


def plot_cold_evening(scorecard: pd.DataFrame, roles: list[dict], output: str | Path) -> pd.DataFrame:
    """Highlight frozen role/method pairs and available 14-day block intervals."""
    from matplotlib.ticker import PercentFormatter
    critical = scorecard.loc[scorecard.segment.eq("realized_cold_evening")].copy()
    figure = _figure((12, 6.3)); axis = figure.subplots()
    rates = _background(axis, critical, "upper_exceedance_rate")
    records, bounds = [], [.05]
    if len(rates):
        bounds.extend(rates.tolist())
    for i, spec in enumerate(roles):
        row, record = _point(critical, spec, "upper_exceedance_rate")
        record["ci14_status"] = "point_not_plotted"
        if row is None:
            records.append(record); continue
        x, y = float(row.width_MW), float(row.upper_exceedance_rate)
        axis.scatter([x], [y], color=spec["color"], marker=spec["marker"], s=75,
                     edgecolor="white", linewidth=.7, zorder=4)
        low, high = _number(row.get("upper_rate_ci14_low")), _number(row.get("upper_rate_ci14_high"))
        valid_ci = low is not None and high is not None and 0 <= low <= high <= 1
        record["ci14_status"] = "shown" if valid_ci else "unavailable_or_invalid"
        if valid_ci:
            cap = (axis.get_xlim()[1]-axis.get_xlim()[0])*.008
            axis.vlines(x, low, high, color=spec["color"], linewidth=2, alpha=.85, zorder=3)
            axis.hlines([low, high], x-cap, x+cap, color=spec["color"], linewidth=1.5, zorder=3)
            bounds.extend([low, high])
            record.update(upper_rate_ci14_low=low, upper_rate_ci14_high=high)
        _annotate(axis, row, spec, "upper_exceedance_rate", .94-i*.31,
                  "14-day block 95% CI shown" if valid_ci else "14-day CI unavailable")
        records.append(record)
    axis.axhline(.05, color="#72523c", linestyle="--", linewidth=1.2, label="Nominal upper-tail target: 5%")
    axis.set_ylim(-.012, min(1.025, max(.12, max(bounds)+.04)))
    axis.yaxis.set_major_formatter(PercentFormatter(1.))
    axis.set(xlabel="Mean interval width on realized cold evenings (MW)", ylabel="Upper-bound exceedance rate",
             title="Cold-evening upper-tail risk: predeclared policy and interval methods")
    axis.grid(alpha=.15); axis.legend(loc="best", fontsize=8, frameon=False)
    figure.subplots_adjust(left=.09, right=.62, bottom=.19, top=.90)
    figure.text(.09, .035,
                "Realized temperature <5 °C, 18:00–20:00 local: retrospective diagnosis, not an operational alert.\n"
                "Colored methods were selected before scoring; vertical segments show 14-day moving-block 95% intervals.",
                ha="left", fontsize=9)
    figure.savefig(Path(output), dpi=150, bbox_inches="tight")
    return pd.DataFrame(records)


def plot_overall_comparison(point_scores: pd.DataFrame, peak_scorecard: pd.DataFrame,
                            roles: list[dict], output: str | Path) -> pd.DataFrame:
    """Tall point leaderboard and overall coverage using the SAME frozen roles."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter
    ordered = point_scores.sort_values("MAE_MW", ascending=False)
    figure = _figure((17, max(9., .31*len(ordered)+1.3)))
    axes = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1.]})
    model_colors = {spec["model"]: spec["color"] for spec in reversed(roles)}
    colors = [model_colors.get(model, "#b7bdc4") for model in ordered.model]
    axes[0].barh(ordered.model, ordered.MAE_MW, color=colors)
    axes[0].tick_params(axis="y", labelsize=8.5)
    axes[0].set(xlabel="Matched MAE (MW)", title=f"Historical comparison: {len(ordered)} complete forecast policies")
    axes[0].grid(axis="x", alpha=.15)
    maximum = pd.to_numeric(ordered.MAE_MW, errors="coerce").max()
    if np.isfinite(maximum) and maximum > 0:
        axes[0].set_xlim(0, maximum*1.13)
        for position, (_, row) in enumerate(ordered.iterrows()):
            if row.model in model_colors and np.isfinite(row.MAE_MW):
                axes[0].text(row.MAE_MW+maximum*.015, position, f"{row.MAE_MW:,.0f}",
                             ha="left", va="center", fontsize=8, color=model_colors[row.model])
    overall = peak_scorecard.loc[peak_scorecard.segment.eq("all")].copy()
    coverage = _background(axes[1], overall, "coverage90")
    records = []
    for i, spec in enumerate(roles):
        row, record = _point(overall, spec, "coverage90")
        if row is not None:
            axes[1].scatter([float(row.width_MW)], [float(row.coverage90)], color=spec["color"],
                            marker=spec["marker"], s=75, edgecolor="white", linewidth=.7, zorder=4)
            _annotate(axes[1], row, spec, "coverage90", .94-i*.30)
        records.append(record)
    axes[1].axhline(.9, color="#72523c", linestyle="--", linewidth=1.2)
    low = max(-.012, min(float(coverage.min())-.04, .86)) if len(coverage) else .0
    high = min(1.025, max(float(coverage.max()), .9)+.04) if len(coverage) else 1.025
    axes[1].set_ylim(low, high); axes[1].yaxis.set_major_formatter(PercentFormatter(1.))
    axes[1].set(xlabel="Mean interval width (MW)", ylabel="Empirical coverage",
                title="Overall coverage and width: same predeclared methods")
    axes[1].grid(alpha=.15)
    handles = [Patch(facecolor=spec["color"], label=spec["title"]) for spec in roles]
    handles.append(Patch(facecolor="#b7bdc4", label="Other policies / methods"))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .055), ncol=2, fontsize=9, frameon=False)
    figure.subplots_adjust(left=.19, right=.77, bottom=.15, top=.94, wspace=.55)
    figure.text(.19, .018,
                "Identical eligible observations for point errors; highlights and interval methods were fixed before historical scoring.\n"
                "Dashed coverage line: nominal 90%. Marginal hourly coverage does not establish daily-trajectory coverage.",
                ha="left", fontsize=9)
    figure.savefig(Path(output), dpi=150, bbox_inches="tight")
    return pd.DataFrame(records)
