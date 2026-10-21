"""Limitations of the retrospective research evidence, without deployment logic."""
from __future__ import annotations

import pandas as pd


def risk_register() -> pd.DataFrame:
    """Describe evidence limitations and the research needed to resolve them."""
    return pd.DataFrame([
        ("unknown vintages", "revised data can leak into historical inputs", "obtain original publication and issue archives; label current results retrospective"),
        ("unknown hourly aggregation", "target and benchmark may represent different intervals", "verify extraction recipe, aggregation and interval labels"),
        ("previously explored holdout", "selection may be influenced by full-history EDA", "freeze this retrospective comparison; confirm on fresh later data"),
        ("national proxy", "utility customer mix may behave differently", "limit conclusions to national demand; validate a customer portfolio separately"),
        ("weather miss", "realized temperature cannot establish forecast-weather quality", "obtain authentic weather forecast archives; keep oracle sensitivity separate"),
        ("rare extremes", "few cold or hot observations", "report segment counts and trajectory-aware stress results"),
        ("settlement alignment", "hourly research targets do not represent quarter-hour settlement", "state the hourly scope; do not infer settlement performance"),
        ("financial inference", "lower forecast loss need not mean lower costs", "report skill and risk proxies; financial evaluation requires prices, applicable rules and a common decision policy"),
    ], columns=["risk", "consequence", "mitigation"])
