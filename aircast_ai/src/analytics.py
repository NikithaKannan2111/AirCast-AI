"""
analytics.py
------------
Summary statistics, trends, location comparison and PM2.5 alert logic.

ALERT THRESHOLDS (PM2.5, 24-hour average, ug/m3)
These follow the category breakpoints of the US EPA Air Quality Index (2024
revision) and are used here for EDUCATIONAL purposes only. They are not an
official health advisory; always follow your local authority's guidance.

    Good                              0.0 -  9.0
    Moderate                          9.1 - 35.4
    Unhealthy for Sensitive Groups   35.5 - 55.4
    Unhealthy                        55.5 - 125.4
    Very Unhealthy                  125.5 - 225.4
    Hazardous                       225.5 and above
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_processing import POLLUTANT_COLUMNS

# (upper bound inclusive, label, colour, short advice)
ALERT_LEVELS = [
    (9.0, "Good", "#2e9e4f", "Air quality is satisfactory."),
    (35.4, "Moderate", "#e6c200", "Acceptable; unusually sensitive people may be affected."),
    (55.4, "Unhealthy for Sensitive Groups", "#f28c28", "Sensitive groups should reduce prolonged outdoor exertion."),
    (125.4, "Unhealthy", "#d62728", "Everyone may begin to experience health effects."),
    (225.4, "Very Unhealthy", "#8e44ad", "Health alert: increased risk for everyone."),
    (float("inf"), "Hazardous", "#7e0023", "Emergency conditions: avoid outdoor activity."),
]
LEVEL_ORDER = [lvl[1] for lvl in ALERT_LEVELS]
LEVEL_COLORS = {lvl[1]: lvl[2] for lvl in ALERT_LEVELS}
LEVEL_ADVICE = {lvl[1]: lvl[3] for lvl in ALERT_LEVELS}
NO_DATA = "No data"
# Levels from this one upward raise a visible alert in the dashboard.
ALERT_FROM_LEVEL = "Unhealthy for Sensitive Groups"


def classify_pm25(value) -> str:
    """Map a PM2.5 value to its alert category name."""
    if value is None or pd.isna(value):
        return NO_DATA
    for upper, label, _, _ in ALERT_LEVELS:
        if value <= upper:
            return label
    return ALERT_LEVELS[-1][1]


def is_alert_level(level: str) -> bool:
    """True when the category is at or above the alert starting level."""
    return level in LEVEL_ORDER and LEVEL_ORDER.index(level) >= LEVEL_ORDER.index(ALERT_FROM_LEVEL)


def alert_threshold_table() -> pd.DataFrame:
    """Documented thresholds as a table for display."""
    rows, lower = [], 0.0
    for upper, label, _, advice in ALERT_LEVELS:
        rng = f"{lower:.1f} - {upper:.1f}" if np.isfinite(upper) else f"> {lower:.1f}"
        rows.append({"Category": label, "PM2.5 range (ug/m3)": rng, "Meaning": advice})
        lower = upper + 0.1 if np.isfinite(upper) else lower
    return pd.DataFrame(rows)


def available_pollutants(df: pd.DataFrame) -> list[str]:
    """Pollutant columns that exist and contain at least one value."""
    return [c for c in POLLUTANT_COLUMNS if c in df.columns and df[c].notna().any()]


def overview_metrics(df: pd.DataFrame) -> dict:
    """Headline numbers for the overview page."""
    pm10_mean = df["pm10"].mean() if "pm10" in df.columns else np.nan
    return {
        "total_records": int(len(df)),
        "avg_pm25": float(df["pm25"].mean()) if df["pm25"].notna().any() else float("nan"),
        "avg_pm10": float(pm10_mean) if pd.notna(pm10_mean) else float("nan"),
        "n_locations": int(df["location"].nunique()),
        "start": df["timestamp"].min() if len(df) else pd.NaT,
        "end": df["timestamp"].max() if len(df) else pd.NaT,
        "pm25_missing_pct": float(df["pm25"].isna().mean() * 100) if len(df) else float("nan"),
    }


def location_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per location with averages, peak PM2.5 and data completeness."""
    grouped = df.groupby("location")
    summary = pd.DataFrame(
        {
            "readings": grouped.size(),
            "avg_pm25": grouped["pm25"].mean(),
            "max_pm25": grouped["pm25"].max(),
            "pm25_missing_pct": grouped["pm25"].apply(lambda s: s.isna().mean() * 100),
        }
    )
    if "pm10" in df.columns:
        summary["avg_pm10"] = grouped["pm10"].mean()
    if "no2" in df.columns:
        summary["avg_no2"] = grouped["no2"].mean()
    summary = summary.round(2).sort_values("avg_pm25", ascending=False).reset_index()
    return summary


def trend_series(df: pd.DataFrame, pollutant: str = "pm25", freq: str = "D") -> pd.DataFrame:
    """Average of ``pollutant`` per location and time bucket (D, W or MS)."""
    if pollutant not in df.columns:
        raise ValueError(f"Column '{pollutant}' is not in the data.")
    out = (
        df.groupby(["location", pd.Grouper(key="timestamp", freq=freq)])[pollutant]
        .mean()
        .reset_index()
        .dropna(subset=[pollutant])
    )
    return out


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and percentage for every column."""
    report = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        }
    )
    report["missing_pct"] = (report["missing_count"] / max(len(df), 1) * 100).round(2)
    return report


def daily_pm25(df: pd.DataFrame) -> pd.DataFrame:
    """Daily mean PM2.5 per location with an alert category."""
    daily = trend_series(df, "pm25", "D").rename(columns={"timestamp": "date"})
    daily["alert_level"] = daily["pm25"].apply(classify_pm25)
    return daily


def latest_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent daily PM2.5 per location and its alert category."""
    daily = daily_pm25(df).sort_values("date")
    latest = daily.groupby("location").tail(1).reset_index(drop=True)
    return latest[["location", "date", "pm25", "alert_level"]]


def alert_day_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Number of days per location in each alert category (long format)."""
    daily = daily_pm25(df)
    counts = daily.groupby(["location", "alert_level"]).size().reset_index(name="days")
    counts["alert_level"] = pd.Categorical(counts["alert_level"], categories=LEVEL_ORDER, ordered=True)
    return counts.sort_values(["location", "alert_level"]).reset_index(drop=True)


def worst_days(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """The ``n`` highest daily-mean PM2.5 values."""
    daily = daily_pm25(df)
    return daily.sort_values("pm25", ascending=False).head(n).reset_index(drop=True)
