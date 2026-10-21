from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Paris"


def project_root(start=None):
    here = Path(start or Path.cwd()).resolve()
    for path in (here, *here.parents):
        if (path / "config/forecast_contract.json").exists():
            return path
    raise FileNotFoundError("Run from the project or a child directory.")


def load_contract(root=None):
    root = project_root(root)
    return json.loads((root / "config/forecast_contract.json").read_text())


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def prepare_directories(root):
    for name in ("data/raw", "data/processed", "data/origin_snapshots", "reports/audit", "reports/figures", "reports/forecasts", "reports/scorecards", "reports/models", "reports/uncertainty"):
        (Path(root) / name).mkdir(parents=True, exist_ok=True)


def local_day_grid(day):
    day = pd.Timestamp(day)
    if day.tzinfo is not None:
        day = day.tz_convert(TZ).tz_localize(None)
    start = day.normalize().tz_localize(TZ)
    end = (day.normalize() + pd.DateOffset(days=1)).tz_localize(TZ)
    return pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")


def forecast_origin(day, contract):
    day = pd.Timestamp(day)
    if day.tzinfo is not None:
        day = day.tz_convert(TZ).tz_localize(None)
    origin_date = day.normalize() - pd.DateOffset(days=1)
    return (origin_date + pd.Timedelta(hours=contract["origin_hour_local"], minutes=contract["origin_minute_local"])).tz_localize(TZ).tz_convert("UTC")


def holiday_dates(year):
    """Metropolitan national public holidays; excludes regional and school holidays.

    Gregorian Easter: Meeus/Jones/Butcher algorithm. Calendar assumptions are
    documented; holiday status is a national proxy, not every customer's schedule.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, rem = divmod(h + ell - 7 * m + 114, 31)
    easter = pd.Timestamp(year=year, month=month, day=rem + 1)
    fixed = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    return {pd.Timestamp(year=year, month=mth, day=day).date() for mth, day in fixed} | {(easter + pd.Timedelta(days=n)).date() for n in (1, 39, 50)}


def calendar_features(index):
    local = pd.DatetimeIndex(index).tz_convert(TZ)
    holidays = set().union(*(holiday_dates(y) for y in np.unique(local.year)))
    month_to_season = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    f = pd.DataFrame(index=index)
    f["hour"] = local.hour
    f["weekday"] = local.dayofweek
    f["month"] = local.month
    f["year"] = local.year
    f["weekend"] = (local.dayofweek >= 5).astype(int)
    f["holiday"] = [int(day in holidays) for day in local.date]
    f["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    f["annual_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    f["annual_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    f["utc_offset_hours"] = [d.utcoffset().total_seconds() / 3600 for d in local]
    f["local_day"] = local.strftime("%Y-%m-%d")
    f["season"] = [month_to_season[m] for m in local.month]
    return f


def plotting_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (11, 4), "figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18, "axes.axisbelow": True})
