"""
pyspark_processing.py
---------------------
A SEPARATE data-processing module written with PySpark (DataFrame API).
It performs: loading -> cleaning -> location-wise statistics -> daily
aggregates -> missing-value counts.

WHAT THIS MODULE IS
  * Spark running in LOCAL mode on one machine (master = "local[*]").
  * It reads a CSV from your normal file system and saves results as CSV files
    through pandas (this avoids the Hadoop "winutils" problem on Windows).

WHAT THIS MODULE IS NOT
  * It does NOT use HDFS, a Hadoop cluster, Hive or any database server.
    Those need extra infrastructure that is not part of this project.

REQUIREMENTS (only for this module, NOT for the dashboard)
  * pip install pyspark
  * Java (JDK 17 recommended) and the JAVA_HOME environment variable set.

USAGE (from the project root)
    python -m src.pyspark_processing --input data/sample_air_quality.csv --output outputs/spark
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

try:
    from src.data_processing import (NUMERIC_COLUMNS, REQUIRED_COLUMNS, VALID_RANGES,
                                     normalize_column_name)
except ModuleNotFoundError:  # allows `python src/pyspark_processing.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data_processing import (NUMERIC_COLUMNS, REQUIRED_COLUMNS, VALID_RANGES,
                                     normalize_column_name)


class PySparkNotInstalledError(RuntimeError):
    """Raised when PySpark is not installed."""


class SparkStartupError(RuntimeError):
    """Raised when a local Spark session cannot be started (usually Java)."""


def is_pyspark_available() -> bool:
    """True if the pyspark package can be imported."""
    return importlib.util.find_spec("pyspark") is not None


def _require_pyspark() -> None:
    if not is_pyspark_available():
        raise PySparkNotInstalledError(
            "PySpark is not installed. Install it with:  pip install pyspark\n"
            "PySpark also needs Java (JDK 17 recommended) and the JAVA_HOME variable. "
            "The Streamlit dashboard works without PySpark."
        )


def create_spark_session(app_name: str = "AirCastAI", master: str = "local[*]"):
    """Create (or reuse) a local Spark session."""
    _require_pyspark()
    from pyspark.sql import SparkSession

    try:
        return (
            SparkSession.builder.appName(app_name)
            .master(master)
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
    except Exception as exc:  # Spark raises several Java-related exception types
        raise SparkStartupError(
            "Could not start a local Spark session. Check that Java (JDK 17) is installed "
            "and JAVA_HOME is set (run `java -version` in a new terminal). "
            f"Original error: {exc}"
        ) from exc


def load_and_clean_spark(spark, csv_path: str):
    """Read the CSV with Spark and apply the same cleaning rules as the Pandas module."""
    from pyspark.sql import functions as F

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    raw = spark.read.option("header", True).csv(str(path))
    for old in raw.columns:
        new = normalize_column_name(old)
        if new != old:
            raw = raw.withColumnRenamed(old, new)

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    df = (
        raw.withColumn("timestamp", F.to_timestamp(F.col("timestamp")))
        .withColumn("location", F.trim(F.col("location")))
        .filter(F.col("timestamp").isNotNull())
        .filter(F.col("location").isNotNull() & (F.col("location") != ""))
    )
    for col in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        low, high = VALID_RANGES[col]
        numeric = F.col(col).cast("double")  # non-numeric text becomes null
        df = df.withColumn(
            col, F.when((numeric < low) | (numeric > high), F.lit(None).cast("double")).otherwise(numeric)
        )
    return df.dropDuplicates(["timestamp", "location"])


def location_statistics(df):
    """Per-location count, mean, min, max, standard deviation and missing PM2.5."""
    from pyspark.sql import functions as F

    aggs = [
        F.count("*").alias("readings"),
        F.round(F.avg("pm25"), 2).alias("avg_pm25"),
        F.round(F.min("pm25"), 2).alias("min_pm25"),
        F.round(F.max("pm25"), 2).alias("max_pm25"),
        F.round(F.stddev("pm25"), 2).alias("std_pm25"),
        F.sum(F.when(F.col("pm25").isNull(), 1).otherwise(0)).alias("missing_pm25"),
    ]
    if "pm10" in df.columns:
        aggs.append(F.round(F.avg("pm10"), 2).alias("avg_pm10"))
    return df.groupBy("location").agg(*aggs).orderBy(F.col("avg_pm25").desc())


def daily_aggregates(df):
    """Daily mean pollutant values per location."""
    from pyspark.sql import functions as F

    cols = [c for c in ["pm25", "pm10", "no2", "o3"] if c in df.columns]
    return (
        df.withColumn("date", F.to_date("timestamp"))
        .groupBy("date", "location")
        .agg(*[F.round(F.avg(c), 2).alias(f"avg_{c}") for c in cols])
        .orderBy("location", "date")
    )


def missing_value_counts(df):
    """Missing-value count per column, returned as a small Spark DataFrame."""
    from pyspark.sql import functions as F

    total = df.count()
    counts = df.agg(*[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]).collect()[0]
    rows = [(c, int(counts[c]), round(100.0 * int(counts[c]) / total, 2) if total else 0.0) for c in df.columns]
    return df.sparkSession.createDataFrame(rows, ["column", "missing_count", "missing_pct"])


def run_pipeline(input_csv: str, output_dir: str = "outputs/spark", master: str = "local[*]") -> dict:
    """Run the whole PySpark job and save three CSV result files."""
    spark = create_spark_session(master=master)
    try:
        spark.sparkContext.setLogLevel("ERROR")
        cleaned = load_and_clean_spark(spark, input_csv).cache()
        results = {
            "location_statistics": location_statistics(cleaned).toPandas(),
            "daily_aggregates": daily_aggregates(cleaned).toPandas(),
            "missing_values": missing_value_counts(cleaned).toPandas(),
        }
        results["rows_after_cleaning"] = cleaned.count()
    finally:
        spark.stop()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("location_statistics", "daily_aggregates", "missing_values"):
        results[name].to_csv(out / f"{name}.csv", index=False)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AirCast AI - local PySpark processing")
    parser.add_argument("--input", default="data/sample_air_quality.csv", help="input CSV path")
    parser.add_argument("--output", default="outputs/spark", help="folder for result CSV files")
    parser.add_argument("--master", default="local[*]", help='Spark master (default "local[*]")')
    args = parser.parse_args(argv)
    try:
        results = run_pipeline(args.input, args.output, args.master)
    except (PySparkNotInstalledError, SparkStartupError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Rows after cleaning: {results['rows_after_cleaning']}")
    print("\nLocation statistics:")
    print(results["location_statistics"].to_string(index=False))
    print(f"\nResults saved in: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
