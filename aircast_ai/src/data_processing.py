"""
data_processing.py
------------------
Loading, validating, cleaning and filtering air-quality data, plus a generator
for a clearly labelled SYNTHETIC demo dataset.

Expected CSV schema (column names are case-insensitive; a few aliases such as
"PM2.5", "date" or "station" are understood):

    timestamp     (required)  date or date-time of the measurement
    location      (required)  monitoring station / city name
    pm25          (required)  PM2.5 in ug/m3
    pm10, no2, o3 (optional)  other pollutants in ug/m3
    temperature   (optional)  degrees Celsius
    humidity      (optional)  relative humidity, 0-100 %
    wind_speed    (optional)  metres per second
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "location", "pm25"]
POLLUTANT_COLUMNS = ["pm25", "pm10", "no2", "o3"]
WEATHER_COLUMNS = ["temperature", "humidity", "wind_speed"]
NUMERIC_COLUMNS = POLLUTANT_COLUMNS + WEATHER_COLUMNS

# Physically plausible (min, max) ranges. Values outside become missing (NaN).
VALID_RANGES = {
    "pm25": (0, 1000),
    "pm10": (0, 2000),
    "no2": (0, 2000),
    "o3": (0, 1000),
    "temperature": (-60, 60),
    "humidity": (0, 100),
    "wind_speed": (0, 120),
}

# Alternative column names that users commonly have in their files.
COLUMN_ALIASES = {
    "pm2.5": "pm25",
    "pm2_5": "pm25",
    "pm_25": "pm25",
    "pm_2_5": "pm25",
    "pm 2.5": "pm25",
    "pm_10": "pm10",
    "pm 10": "pm10",
    "date": "timestamp",
    "datetime": "timestamp",
    "date_time": "timestamp",
    "time": "timestamp",
    "station": "location",
    "city": "location",
    "site": "location",
    "temp": "temperature",
    "wind": "wind_speed",
    "windspeed": "wind_speed",
}


class DataValidationError(ValueError):
    """Raised when the input data cannot be used by the application."""


def normalize_column_name(name: str) -> str:
    """Lower-case a column name, trim spaces and map known aliases."""
    cleaned = str(name).strip().lower()
    if cleaned in COLUMN_ALIASES:
        return COLUMN_ALIASES[cleaned]
    cleaned = cleaned.replace(" ", "_")
    return COLUMN_ALIASES.get(cleaned, cleaned)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with standardised column names."""
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    return out


def validate_columns(df: pd.DataFrame) -> None:
    """Raise DataValidationError if a required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            "Missing required column(s): "
            + ", ".join(missing)
            + ". Required columns are: "
            + ", ".join(REQUIRED_COLUMNS)
            + f". Columns found: {', '.join(map(str, df.columns))}"
        )


def load_csv(source) -> pd.DataFrame:
    """Read a CSV from a path or file-like object. Raises DataValidationError."""
    try:
        df = pd.read_csv(source)
    except FileNotFoundError as exc:
        raise DataValidationError(f"File not found: {source}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The CSV file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"The file could not be read as a CSV: {exc}") from exc
    if df.empty:
        raise DataValidationError("The CSV file has a header but no data rows.")
    return df


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Validate and clean a raw DataFrame.

    Steps: standardise names -> check required columns -> parse timestamps ->
    drop rows with bad timestamp/location -> convert numbers -> blank out
    impossible values -> drop duplicate (timestamp, location) rows -> sort.

    Returns (clean_dataframe, report_dictionary).
    """
    df = normalize_columns(raw)
    validate_columns(df)

    report: dict = {"rows_in": int(len(df))}
    # If a file has duplicate column names after normalising, keep the first one.
    df = df.loc[:, ~df.columns.duplicated()]
    keep = [c for c in ["timestamp", "location"] + NUMERIC_COLUMNS if c in df.columns]
    df = df[keep].copy()

    # --- timestamp -------------------------------------------------------
    parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = parsed.dt.tz_convert(None)
    bad_ts = df["timestamp"].isna()
    report["invalid_timestamps_dropped"] = int(bad_ts.sum())
    df = df[~bad_ts].copy()

    # --- location --------------------------------------------------------
    loc_text = df["location"].astype("string").str.strip()
    bad_loc = (loc_text.isna() | (loc_text == "") | (loc_text.str.lower() == "nan")).fillna(True)
    bad_loc = bad_loc.to_numpy(dtype=bool)
    report["missing_location_dropped"] = int(bad_loc.sum())
    df = df[~bad_loc].copy()
    df["location"] = loc_text[~bad_loc].astype(str).to_numpy()

    # --- numeric columns -------------------------------------------------
    non_numeric: dict[str, int] = {}
    out_of_range: dict[str, int] = {}
    for col in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        was_missing = df[col].isna()
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        non_numeric[col] = int((df[col].isna() & ~was_missing).sum())
        low, high = VALID_RANGES[col]
        invalid = df[col].notna() & ((df[col] < low) | (df[col] > high))
        out_of_range[col] = int(invalid.sum())
        df.loc[invalid, col] = np.nan
    report["non_numeric_values_set_missing"] = non_numeric
    report["out_of_range_values_set_missing"] = out_of_range

    # --- duplicates, sorting ---------------------------------------------
    dup = df.duplicated(subset=["timestamp", "location"], keep="first")
    report["duplicates_dropped"] = int(dup.sum())
    df = df[~dup]

    df = df.sort_values(["location", "timestamp"]).reset_index(drop=True)
    report["rows_out"] = int(len(df))

    if df.empty:
        raise DataValidationError("No valid rows remain after cleaning the data.")
    if df["pm25"].notna().sum() == 0:
        raise DataValidationError("The 'pm25' column has no valid numeric values.")

    report["pm25_missing_pct"] = round(float(df["pm25"].isna().mean() * 100), 2)
    return df, report


def filter_data(df: pd.DataFrame, locations=None, start=None, end=None) -> pd.DataFrame:
    """Filter by locations and an inclusive start/end date."""
    out = df
    if locations is not None:
        out = out[out["location"].isin(list(locations))]
    if start is not None:
        out = out[out["timestamp"] >= pd.Timestamp(start).normalize()]
    if end is not None:
        out = out[out["timestamp"] < pd.Timestamp(end).normalize() + pd.Timedelta(days=1)]
    return out.copy()


# ---------------------------------------------------------------------------
# Synthetic data generator (clearly NOT real measurements)
# ---------------------------------------------------------------------------
SYNTHETIC_STATIONS = {
    "Station A (Urban Core)": 55.0,
    "Station B (Industrial Zone)": 70.0,
    "Station C (Suburban)": 35.0,
    "Station D (Riverside Park)": 25.0,
    "Station E (Highway Corridor)": 60.0,
}


def generate_synthetic_data(start: str = "2022-01-01", periods: int = 1096, seed: int = 42) -> pd.DataFrame:
    """
    Create a SYNTHETIC daily dataset for five made-up stations.

    The numbers are produced by a simple simulation (yearly seasonality +
    day-to-day persistence + wind effect + noise). They are NOT real
    measurements and do not describe any real place. About 1-3 % of values are
    deliberately blanked to exercise the missing-value handling.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="D")
    n = len(dates)
    doy = dates.dayofyear.to_numpy()

    # Weather shared by the region, with small station-level differences.
    temp_region = 24 + 8 * np.sin(2 * np.pi * (doy - 100) / 365.25) + rng.normal(0, 1.5, n)
    hum_region = 60 + 18 * np.sin(2 * np.pi * (doy - 200) / 365.25) + rng.normal(0, 5, n)
    # Pollution episodes shared by the region (e.g. stagnant air).
    shared = np.zeros(n)
    shared_eps = rng.normal(0, 0.12, n)
    for i in range(1, n):
        shared[i] = 0.8 * shared[i - 1] + shared_eps[i]

    frames = []
    for name, base in SYNTHETIC_STATIONS.items():
        wind = np.clip(rng.gamma(4.0, 2.0, n), 0.5, 40.0)
        local = np.zeros(n)
        local_eps = rng.normal(0, 0.12, n)
        for i in range(1, n):
            local[i] = 0.7 * local[i - 1] + local_eps[i]

        season = 1 + 0.45 * np.cos(2 * np.pi * (doy - 15) / 365.25)  # winter peak
        wind_effect = np.clip(1 - 0.025 * (wind - 8), 0.5, 1.5)
        pm25 = np.clip(base * season * np.exp(shared + local) * wind_effect, 2, None)
        pm10 = pm25 * rng.normal(1.6, 0.12, n) + rng.normal(5, 3, n)
        no2 = np.clip(20 + 0.35 * pm25 + rng.normal(0, 6, n), 1, None)
        temperature = temp_region + rng.normal(0, 0.8, n)
        o3 = np.clip(30 + 1.4 * temperature - 0.2 * no2 + rng.normal(0, 8, n), 1, None)
        humidity = np.clip(hum_region + rng.normal(0, 3, n), 15, 100)

        df = pd.DataFrame(
            {
                "timestamp": dates.strftime("%Y-%m-%d"),
                "location": name,
                "pm25": pm25.round(2),
                "pm10": np.clip(pm10, 3, None).round(2),
                "no2": no2.round(2),
                "o3": o3.round(2),
                "temperature": temperature.round(1),
                "humidity": humidity.round(1),
                "wind_speed": wind.round(1),
            }
        )
        # Deliberately blank some values.
        for col, frac in [("pm25", 0.02), ("pm10", 0.025), ("no2", 0.02), ("o3", 0.02),
                          ("temperature", 0.01), ("humidity", 0.01), ("wind_speed", 0.01)]:
            df.loc[rng.random(n) < frac, col] = np.nan
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def save_synthetic_csv(path) -> Path:
    """Generate the synthetic dataset and write it to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_data().to_csv(path, index=False)
    return path
