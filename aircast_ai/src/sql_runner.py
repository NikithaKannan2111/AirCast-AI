"""
sql_runner.py
-------------
Runs the queries in sql/analysis_queries.sql against the cleaned CSV data using
Python's built-in SQLite (an in-memory database, no server needed).

    python -m src.sql_runner
    python -m src.sql_runner --csv my_data.csv --query highest_pm25_readings
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

try:
    from src.data_processing import DataValidationError, clean_data, load_csv
except ModuleNotFoundError:  # allows `python src/sql_runner.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data_processing import DataValidationError, clean_data, load_csv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "sample_air_quality.csv"
DEFAULT_SQL = ROOT / "sql" / "analysis_queries.sql"
TABLE_NAME = "air_quality"


def load_queries(sql_path=DEFAULT_SQL) -> dict:
    """Split the SQL file into {query_name: sql_text} using the '-- name:' markers."""
    text = Path(sql_path).read_text(encoding="utf-8")
    parts = re.split(r"^-- name:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    # parts = [header, name1, body1, name2, body2, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def build_database(df: pd.DataFrame) -> sqlite3.Connection:
    """Create an in-memory SQLite database with the table ``air_quality``."""
    conn = sqlite3.connect(":memory:")
    table = df.copy()
    table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    table.to_sql(TABLE_NAME, conn, index=False, if_exists="replace")
    return conn


def run_queries(csv_path=DEFAULT_CSV, sql_path=DEFAULT_SQL) -> dict:
    """Clean the CSV, load it into SQLite and run every query. Returns {name: DataFrame}."""
    df, _ = clean_data(load_csv(csv_path))
    conn = build_database(df)
    try:
        return {name: pd.read_sql_query(sql, conn) for name, sql in load_queries(sql_path).items()}
    finally:
        conn.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run AirCast AI SQL queries on SQLite")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="input CSV file")
    parser.add_argument("--sql", default=str(DEFAULT_SQL), help="SQL file")
    parser.add_argument("--query", default=None, help="run only the query with this name")
    args = parser.parse_args(argv)
    try:
        results = run_queries(args.csv, args.sql)
    except (DataValidationError, FileNotFoundError, sqlite3.Error, pd.errors.DatabaseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.query:
        if args.query not in results:
            print(f"ERROR: unknown query '{args.query}'. Available: {', '.join(results)}", file=sys.stderr)
            return 1
        results = {args.query: results[args.query]}
    for name, frame in results.items():
        print(f"\n=== {name} ===")
        print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
