"""
prediction.py
-------------
PM2.5 prediction with scikit-learn.

Task: predict the daily mean PM2.5 of a location for day D using ONLY
information available before day D.

Features (all built from the past, so there is no data leakage):
  * pm25 of the previous 1, 2, 3 and 7 days (lags)
  * mean pm25 of the previous 7 days (the window ends the day before D)
  * previous-day pm10, no2, o3, temperature, humidity, wind_speed (if present)
  * calendar features (day of week, month and day of year as sin/cos)
  * the location (one-hot encoded)

Evaluation: chronological split. The earliest ~80 % of dates are used for
training and the latest ~20 % are held out for testing. The model is compared
with a simple "persistence" baseline that predicts today's PM2.5 = yesterday's.

Preprocessing (median imputation, one-hot encoding) lives inside a scikit-learn
Pipeline, so it is fitted on the training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.analytics import classify_pm25

PM25_LAGS = (1, 2, 3, 7)
LAGGED_SOURCES = ["pm10", "no2", "o3", "temperature", "humidity", "wind_speed"]
NON_FEATURE_COLUMNS = ["date", "location", "pm25"]


class InsufficientDataError(ValueError):
    """Raised when there is not enough data to train or evaluate a model."""


@dataclass
class ModelResult:
    pipeline: Pipeline
    numeric_features: list
    metrics: dict               # {"model": {...}, "baseline": {...}}
    test_predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    n_train: int
    n_test: int


def _daily_by_location(df: pd.DataFrame, add_forecast_day: bool):
    """Yield (location, daily-mean DataFrame with a continuous daily index)."""
    for location, group in df.groupby("location"):
        daily = group.set_index("timestamp").drop(columns="location").resample("D").mean()
        if add_forecast_day:
            daily.loc[daily.index.max() + pd.Timedelta(days=1)] = np.nan
        yield location, daily


def build_feature_frame(df: pd.DataFrame, add_forecast_day: bool = False) -> pd.DataFrame:
    """
    Build the supervised-learning table (one row per location per day).

    Every feature of day D is computed with ``shift`` so that it only uses data
    from days before D. With ``add_forecast_day=True`` one extra, empty day is
    appended per location; its target is unknown (NaN) and it is used to
    produce a genuine forecast.
    """
    sources = [c for c in LAGGED_SOURCES if c in df.columns and df[c].notna().any()]
    frames = []
    for location, daily in _daily_by_location(df, add_forecast_day):
        feat = pd.DataFrame(index=daily.index)
        feat["pm25"] = daily["pm25"]  # target
        for lag in PM25_LAGS:
            feat[f"pm25_lag{lag}"] = daily["pm25"].shift(lag)
        feat["pm25_roll7_mean"] = daily["pm25"].shift(1).rolling(7, min_periods=3).mean()
        for col in sources:
            feat[f"{col}_lag1"] = daily[col].shift(1)
        idx = feat.index
        feat["day_of_week"] = idx.dayofweek
        feat["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        feat["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
        feat["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
        feat["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
        feat.insert(0, "location", location)
        feat.index.name = "date"
        frames.append(feat.reset_index())
    return pd.concat(frames, ignore_index=True)


def _make_pipeline(numeric_features: list, random_state: int) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("loc", OneHotEncoder(handle_unknown="ignore"), ["location"]),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, random_state=random_state, n_jobs=-1
    )
    return Pipeline([("prep", prep), ("model", model)])


def _scores(actual, predicted) -> dict:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def train_and_evaluate(df: pd.DataFrame, test_fraction: float = 0.2, random_state: int = 42,
                       min_rows: int = 60) -> ModelResult:
    """Train on the earliest dates, evaluate on the latest dates."""
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5.")

    frame = build_feature_frame(df)
    frame = frame.dropna(subset=["pm25", "pm25_lag1"])  # need a target and yesterday's value
    if len(frame) < min_rows or frame["date"].nunique() < 20:
        raise InsufficientDataError(
            f"Not enough usable history to train a model ({len(frame)} rows, "
            f"{frame['date'].nunique()} distinct days). At least {min_rows} rows and "
            "20 distinct days with consecutive daily data are needed."
        )

    dates = np.sort(frame["date"].unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * (1 - test_fraction))])
    train = frame[frame["date"] < cutoff]
    test = frame[frame["date"] >= cutoff]
    if len(train) < 30 or len(test) < 5:
        raise InsufficientDataError("The chronological split leaves too few rows for training or testing.")

    numeric_features = [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]
    columns = numeric_features + ["location"]

    pipeline = _make_pipeline(numeric_features, random_state)
    pipeline.fit(train[columns], train["pm25"])
    predicted = np.clip(pipeline.predict(test[columns]), 0, None)

    baseline = test["pm25_lag1"].to_numpy()  # persistence: "same as yesterday"
    metrics = {
        "model": _scores(test["pm25"], predicted),
        "baseline": _scores(test["pm25"], baseline),
    }

    test_predictions = pd.DataFrame(
        {
            "date": test["date"].to_numpy(),
            "location": test["location"].to_numpy(),
            "actual_pm25": test["pm25"].to_numpy(),
            "predicted_pm25": predicted,
            "baseline_pm25": baseline,
        }
    )
    test_predictions["abs_error"] = (test_predictions["actual_pm25"] - test_predictions["predicted_pm25"]).abs()

    names = [n.split("__", 1)[-1] for n in pipeline.named_steps["prep"].get_feature_names_out()]
    importance = (
        pd.DataFrame({"feature": names, "importance": pipeline.named_steps["model"].feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return ModelResult(
        pipeline=pipeline,
        numeric_features=numeric_features,
        metrics=metrics,
        test_predictions=test_predictions,
        feature_importance=importance,
        train_end=pd.Timestamp(train["date"].max()),
        test_start=pd.Timestamp(test["date"].min()),
        n_train=int(len(train)),
        n_test=int(len(test)),
    )


def forecast_next_day(df: pd.DataFrame, result: ModelResult) -> pd.DataFrame:
    """
    Forecast the day after each location's last observation.
    The output is a FORECAST, not an observation.
    """
    frame = build_feature_frame(df, add_forecast_day=True)
    last = frame.sort_values("date").groupby("location").tail(1)
    last = last[last["pm25"].isna() & last["pm25_lag1"].notna()]
    if last.empty:
        return pd.DataFrame(columns=["location", "forecast_date", "predicted_pm25", "alert_level"])
    columns = result.numeric_features + ["location"]
    predicted = np.clip(result.pipeline.predict(last[columns]), 0, None)
    out = pd.DataFrame(
        {
            "location": last["location"].to_numpy(),
            "forecast_date": last["date"].to_numpy(),
            "predicted_pm25": np.round(predicted, 2),
        }
    )
    out["alert_level"] = out["predicted_pm25"].apply(classify_pm25)
    return out.reset_index(drop=True)


def metrics_table(result: ModelResult) -> pd.DataFrame:
    """Model vs baseline metrics as a tidy table."""
    table = pd.DataFrame(result.metrics).T.round(3)
    table.index = ["Random Forest model", "Baseline: same as yesterday"]
    return table


def describe_performance(result: ModelResult) -> str:
    """Plain-language verdict that only claims what the numbers support."""
    m, b = result.metrics["model"], result.metrics["baseline"]
    change = (b["MAE"] - m["MAE"]) / b["MAE"] * 100 if b["MAE"] > 0 else 0.0
    if change > 5:
        verdict = f"The model's MAE is {change:.1f}% lower than the 'same as yesterday' baseline, so it adds some value on the held-out period."
    elif change < -5:
        verdict = f"The model's MAE is {abs(change):.1f}% HIGHER than the 'same as yesterday' baseline, so it does not beat a trivial method on the held-out period."
    else:
        verdict = "The model performs about the same as the 'same as yesterday' baseline (MAE within 5 %), so it adds little over a trivial method."
    return (
        f"{verdict} On the test period the model's typical error is about {m['MAE']:.1f} ug/m3 (MAE) "
        f"and RMSE is {m['RMSE']:.1f} ug/m3; R2 = {m['R2']:.2f}."
    )
