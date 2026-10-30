"""
Basic tests for AirCast AI. Run from the project root:   pytest
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # makes imports work even without pytest.ini
    sys.path.insert(0, str(ROOT))

from src import analytics as an  # noqa: E402
from src import data_processing as dp  # noqa: E402
from src import prediction as pr  # noqa: E402
from src import pyspark_processing as ps  # noqa: E402
from src import sql_runner  # noqa: E402

SAMPLE_CSV = ROOT / "data" / "sample_air_quality.csv"


def make_raw():
    return pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02", "not-a-date", "2024-01-03", "2024-01-04"],
            "Station": ["A", "A", "A", "A", "  ", "B"],
            "PM2.5": [10, 99, "abc", 20, 30, -5],
            "PM10": [20, 40, 30, 50, 60, 70],
            "humidity": [50, 60, 70, 80, 90, 150],
        }
    )


# ----------------------------- data processing -----------------------------
def test_column_aliases_are_normalised():
    df = dp.normalize_columns(make_raw())
    assert {"timestamp", "location", "pm25", "pm10"}.issubset(df.columns)


def test_missing_required_column_raises():
    with pytest.raises(dp.DataValidationError):
        dp.clean_data(pd.DataFrame({"timestamp": ["2024-01-01"], "pm25": [10]}))


def test_clean_data_handles_bad_rows():
    clean, report = dp.clean_data(make_raw())
    assert report["invalid_timestamps_dropped"] == 1      # 'not-a-date'
    assert report["missing_location_dropped"] == 1        # blank station
    assert report["duplicates_dropped"] == 1              # A on 2024-01-01 twice
    assert report["non_numeric_values_set_missing"]["pm25"] == 1   # 'abc'
    assert report["out_of_range_values_set_missing"]["pm25"] == 1  # -5
    assert report["out_of_range_values_set_missing"]["humidity"] == 1  # 150
    assert (clean["pm25"].dropna() >= 0).all()
    assert clean["timestamp"].is_monotonic_increasing or clean.groupby("location")["timestamp"].apply(
        lambda s: s.is_monotonic_increasing).all()


def test_all_invalid_pm25_raises():
    raw = pd.DataFrame({"timestamp": ["2024-01-01"], "location": ["A"], "pm25": ["x"]})
    with pytest.raises(dp.DataValidationError):
        dp.clean_data(raw)


def test_load_csv_errors():
    with pytest.raises(dp.DataValidationError):
        dp.load_csv(io.StringIO(""))
    with pytest.raises(dp.DataValidationError):
        dp.load_csv(ROOT / "does_not_exist.csv")


def test_filter_data_is_inclusive():
    df, _ = dp.clean_data(dp.load_csv(SAMPLE_CSV))
    out = dp.filter_data(df, ["Station A (Urban Core)"], "2023-01-01", "2023-01-31")
    assert set(out["location"]) == {"Station A (Urban Core)"}
    assert out["timestamp"].min() == pd.Timestamp("2023-01-01")
    assert out["timestamp"].max() == pd.Timestamp("2023-01-31")


def test_synthetic_data_is_reproducible_and_has_gaps():
    a = dp.generate_synthetic_data(periods=60, seed=1)
    b = dp.generate_synthetic_data(periods=60, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert a["pm25"].isna().any()
    assert a["location"].nunique() == len(dp.SYNTHETIC_STATIONS)


def test_sample_csv_loads():
    assert SAMPLE_CSV.exists()
    df, report = dp.clean_data(dp.load_csv(SAMPLE_CSV))
    assert report["rows_out"] == len(df) > 1000


# -------------------------------- analytics --------------------------------
def test_classify_pm25_boundaries():
    assert an.classify_pm25(9.0) == "Good"
    assert an.classify_pm25(9.1) == "Moderate"
    assert an.classify_pm25(35.4) == "Moderate"
    assert an.classify_pm25(35.5) == "Unhealthy for Sensitive Groups"
    assert an.classify_pm25(125.4) == "Unhealthy"
    assert an.classify_pm25(300) == "Hazardous"
    assert an.classify_pm25(np.nan) == an.NO_DATA


def test_overview_and_location_summary():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "location": ["A", "A", "B", "B"],
            "pm25": [10.0, 20.0, 40.0, np.nan],
            "pm10": [20.0, 40.0, 80.0, 100.0],
        }
    )
    m = an.overview_metrics(df)
    assert m["total_records"] == 4 and m["n_locations"] == 2
    assert m["avg_pm25"] == pytest.approx(70 / 3)
    assert m["avg_pm10"] == pytest.approx(60.0)
    summary = an.location_summary(df).set_index("location")
    assert summary.loc["A", "avg_pm25"] == 15.0
    assert summary.loc["B", "pm25_missing_pct"] == 50.0
    report = an.missing_value_report(df).set_index("column")
    assert report.loc["pm25", "missing_count"] == 1


def test_alert_helpers():
    assert an.is_alert_level("Unhealthy") and not an.is_alert_level("Moderate")
    assert len(an.alert_threshold_table()) == len(an.ALERT_LEVELS)
    df, _ = dp.clean_data(dp.load_csv(SAMPLE_CSV))
    assert set(an.latest_readings(df)["location"]) == set(df["location"])
    assert not an.worst_days(df, 5).empty


# ------------------------------- prediction --------------------------------
@pytest.fixture(scope="module")
def clean_sample():
    df, _ = dp.clean_data(dp.load_csv(SAMPLE_CSV))
    return df


def test_features_use_only_past_data(clean_sample):
    frame = pr.build_feature_frame(clean_sample)
    loc = "Station A (Urban Core)"
    one = frame[frame["location"] == loc].set_index("date")
    # lag1 of day D must equal the actual pm25 of day D-1.
    d = pd.Timestamp("2023-03-10")
    assert one.loc[d, "pm25_lag1"] == pytest.approx(one.loc[d - pd.Timedelta(days=1), "pm25"], nan_ok=True)
    # The 7-day mean covers days D-7 ... D-1 only (D itself is excluded).
    window = one.loc[d - pd.Timedelta(days=7): d - pd.Timedelta(days=1), "pm25"]
    assert one.loc[d, "pm25_roll7_mean"] == pytest.approx(window.mean())


def test_chronological_split_has_no_overlap(clean_sample):
    result = pr.train_and_evaluate(clean_sample)
    assert result.train_end < result.test_start
    assert result.test_predictions["date"].min() >= result.test_start
    for key in ("MAE", "RMSE", "R2"):
        assert np.isfinite(result.metrics["model"][key])
        assert np.isfinite(result.metrics["baseline"][key])
    assert (result.test_predictions["predicted_pm25"] >= 0).all()
    assert "pm25" not in result.numeric_features  # the target is never a feature
    assert "MAE" in pr.describe_performance(result)


def test_forecast_is_after_last_observation(clean_sample):
    result = pr.train_and_evaluate(clean_sample)
    forecast = pr.forecast_next_day(clean_sample, result)
    assert len(forecast) == clean_sample["location"].nunique()
    assert (pd.to_datetime(forecast["forecast_date"]) > clean_sample["timestamp"].max()).all()


def test_too_little_data_raises():
    tiny = dp.generate_synthetic_data(periods=15, seed=3)
    df, _ = dp.clean_data(tiny)
    with pytest.raises(pr.InsufficientDataError):
        pr.train_and_evaluate(df)


# ----------------------------------- SQL -----------------------------------
def test_sql_queries_run_on_sqlite():
    results = sql_runner.run_queries(SAMPLE_CSV, ROOT / "sql" / "analysis_queries.sql")
    for name in ("average_pollution_by_location", "highest_pm25_readings",
                 "daily_trend_all_locations", "missing_value_analysis"):
        assert name in results and not results[name].empty
    assert len(results["highest_pm25_readings"]) == 10
    assert results["missing_value_analysis"].loc[0, "total_rows"] == 5480


def test_sql_matches_pandas(clean_sample):
    sql_avg = sql_runner.run_queries(SAMPLE_CSV, ROOT / "sql" / "analysis_queries.sql")[
        "average_pollution_by_location"].set_index("location")["avg_pm25"]
    pandas_avg = clean_sample.groupby("location")["pm25"].mean().round(2)
    for loc, value in pandas_avg.items():
        assert sql_avg[loc] == pytest.approx(value, abs=0.011)


# ---------------------------------- PySpark ---------------------------------
def test_pyspark_availability_flag_is_boolean():
    assert isinstance(ps.is_pyspark_available(), bool)


@pytest.mark.skipif(ps.is_pyspark_available(), reason="PySpark is installed")
def test_pyspark_missing_gives_clear_message():
    with pytest.raises(ps.PySparkNotInstalledError, match="pip install pyspark"):
        ps.create_spark_session()
    assert ps.main(["--input", str(SAMPLE_CSV)]) == 1
