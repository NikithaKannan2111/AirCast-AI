# AirCast AI — Big Data Analytics and Air Quality Prediction System

A college mini-project that analyses air-pollution measurements, compares monitoring
stations, raises threshold-based alerts and predicts **PM2.5** with a real scikit-learn model.
Everything runs locally on a Windows laptop with Python — **no API key, database server or Hadoop needed.**

> **Data notice:** the bundled file `data/sample_air_quality.csv` is **synthetic** (produced by
> `data/generate_sample_data.py`, which uses a random simulation). It is **not** real monitoring
> data and does not describe any real city or station. You can load your own real CSV in the dashboard.

---

## 1. Features

| # | Requirement | Where |
|---|-------------|-------|
| 1 | Overview: total records, average PM2.5, average PM10, number of locations | *Overview* tab |
| 2 | Pollution-trend chart (daily / weekly / monthly, any pollutant) | *Overview* tab |
| 3 | Station comparison (bar, box plot, summary table, grouped pollutants) | *Station comparison* tab |
| 4 | Location + date-range filters | Sidebar |
| 5 | Pollutant distribution (histogram + box, optional split by location) | *Distribution* tab |
| 6 | PM2.5 prediction (held-out test + next-day forecast) | *PM2.5 prediction* tab |
| 7 | Alerts with documented thresholds | *Alerts* tab |
| 8 | Data table + CSV download + data-quality report | *Data table* tab |
| 9 | Model performance, methodology and limitations | *Model performance & limits* tab |
| – | Upload your own compatible CSV | Sidebar → "Upload my own CSV" |

## 2. Architecture and project structure

```
aircast_ai/
├── app.py                      # Streamlit dashboard (entry point)
├── requirements.txt            # core dependencies
├── requirements-spark.txt      # OPTIONAL: pyspark only
├── pytest.ini                  # makes imports work from the project root
├── README.md
├── .gitignore
├── data/
│   ├── sample_air_quality.csv  # SYNTHETIC demo data (5 stations x 1096 days)
│   └── generate_sample_data.py # regenerates the CSV
├── src/
│   ├── data_processing.py      # load / validate / clean / filter + synthetic generator
│   ├── analytics.py            # statistics, trends, comparison, alert thresholds
│   ├── prediction.py           # scikit-learn model, chronological evaluation, forecast
│   ├── pyspark_processing.py   # SEPARATE PySpark module (local mode)
│   └── sql_runner.py           # runs the SQL file on in-memory SQLite
├── sql/
│   └── analysis_queries.sql    # analytical SQL queries
└── tests/
    └── test_processing.py      # pytest tests
```

```
CSV ──► data_processing (validate + clean) ──► analytics ─────┐
                         │                                    ├──► app.py (Streamlit + Plotly)
                         ├──► prediction (scikit-learn) ──────┘
                         ├──► sql_runner  (SQLite + sql/analysis_queries.sql)
                         └──► pyspark_processing (optional, local Spark)
```

### What is implemented and what is NOT

| Component | Status |
|-----------|--------|
| Streamlit dashboard, Pandas/NumPy processing, Plotly charts | Implemented |
| scikit-learn Random Forest regression with chronological evaluation | Implemented |
| SQL queries | Implemented, executed on **SQLite** (bundled with Python) |
| PySpark cleaning / aggregation / location statistics | Implemented in **local mode** on one machine; needs `pip install pyspark` + Java |
| HDFS, Hadoop cluster, Hive, Spark cluster, MySQL/PostgreSQL server | **NOT configured.** They require extra infrastructure that is not part of this project. The SQL is written so it can be adapted to MySQL/PostgreSQL (notes in the file). |
| Real-time data / external APIs / authentication / cloud deployment | **Not included** (out of scope) |

## 3. Dataset schema

Column names are case-insensitive; aliases such as `PM2.5`, `date`, `station`, `city` are accepted.

| Column | Required | Description |
|--------|----------|-------------|
| `timestamp` | yes | date or date-time, e.g. `2024-03-15` |
| `location` | yes | station / city name |
| `pm25` | yes | PM2.5 in µg/m³ |
| `pm10`, `no2`, `o3` | no | other pollutants in µg/m³ |
| `temperature` | no | °C |
| `humidity` | no | % (0–100) |
| `wind_speed` | no | m/s |

**Validation and cleaning** (`src/data_processing.py`): required-column check, unparseable timestamps
dropped, blank locations dropped, non-numeric values → missing, impossible values (e.g. negative PM2.5,
humidity > 100) → missing, duplicate `(timestamp, location)` rows dropped. A cleaning report is shown in
the dashboard (*Data table → Data-quality report*). Errors such as an empty file or missing columns
produce a readable message instead of a crash.

The synthetic sample deliberately contains about 1–3 % missing values to exercise this handling.

## 4. Installation on Windows (VS Code)

Prerequisite: **Python 3.10 – 3.12** from python.org (tick *"Add Python to PATH"* during setup).

1. Unzip the project and open the folder in VS Code (*File → Open Folder…*).
2. Open a terminal (*Terminal → New Terminal*, PowerShell) and create a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the script, run once:
   `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or use Command Prompt and `.venv\Scripts\activate.bat`.
3. Install the dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. In VS Code press `Ctrl+Shift+P` → *Python: Select Interpreter* → choose the `.venv` one.

## 5. Running the project

All commands are run **from the project root folder** (the one containing `app.py`).

**Dashboard**
```powershell
streamlit run app.py
```
A browser tab opens at http://localhost:8501. (Press `Ctrl+C` in the terminal to stop it.)

**Tests**
```powershell
pytest
```

**SQL queries** (no database server needed)
```powershell
python -m src.sql_runner
python -m src.sql_runner --query highest_pm25_readings
```
Table schema and instructions are at the top of `sql/analysis_queries.sql`. You can also import the
CSV into "DB Browser for SQLite" and run the queries there.

**Regenerate the synthetic sample data** (optional)
```powershell
python data/generate_sample_data.py
```

**PySpark module** (optional — the dashboard does not need it)
1. Install Java JDK 17 (e.g. Eclipse Temurin) and make sure `java -version` works in a *new* terminal.
2. `pip install -r requirements-spark.txt`
3. Run:
   ```powershell
   python -m src.pyspark_processing --input data/sample_air_quality.csv --output outputs/spark
   ```
   It prints per-location statistics and writes `location_statistics.csv`, `daily_aggregates.csv` and
   `missing_values.csv` to `outputs/spark/`. If PySpark or Java is missing you get a clear error message
   explaining what to install. Results are saved through pandas so the Windows Hadoop *winutils* tool is not needed.

## 6. Machine-learning methodology

* **Task:** predict the **daily mean PM2.5** of a location for day *D*.
* **Model:** `RandomForestRegressor` (200 trees) inside a scikit-learn `Pipeline`.
* **Features** — built only from information available *before* day *D*:
  PM2.5 of the previous 1, 2, 3 and 7 days; mean PM2.5 of days *D-7…D-1*; previous-day PM10, NO₂, O₃,
  temperature, humidity and wind speed (if present); calendar features (day of week, month and day-of-year as sin/cos); location (one-hot).
* **Avoiding leakage:** all lags use `shift`, so day *D*'s own values never enter its features; median
  imputation and one-hot encoding are part of the Pipeline and are **fitted on training rows only**.
* **Split:** *chronological* — the earliest ~80 % of dates train the model, the latest ~20 % are held out.
  No shuffling and no random split.
* **Baseline:** "tomorrow = today" (persistence). A model that cannot beat this is not useful.
* **Forecast:** the dashboard also produces a next-day forecast for each location. It is labelled as a
  *forecast* and is visually separate from observed values.

## 7. Model evaluation

The dashboard reports **MAE**, **RMSE** and **R²** for the model and the baseline on the held-out dates, and
generates a verdict *from the numbers* (it says the model is better, equal or worse than the baseline).

Example from one run on the bundled synthetic data (values can vary slightly with library versions):

| | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | ≈ 7.6 | ≈ 10.8 | ≈ 0.83 |
| Baseline (yesterday) | ≈ 8.3 | ≈ 11.7 | ≈ 0.80 |

Interpretation: the model improves on the baseline by roughly 9 % in MAE — a modest gain, mostly because
yesterday's PM2.5 is already a strong predictor. These numbers describe **simulated** data and must not be
read as real-world accuracy. Use your own data to get meaningful results.

## 8. Alert thresholds

Alerts use the **daily mean PM2.5** and the category breakpoints of the US EPA Air Quality Index (2024 revision),
applied here for educational purposes only:

| Category | PM2.5 (µg/m³) |
|----------|---------------|
| Good | 0.0 – 9.0 |
| Moderate | 9.1 – 35.4 |
| Unhealthy for Sensitive Groups | 35.5 – 55.4 |
| Unhealthy | 55.5 – 125.4 |
| Very Unhealthy | 125.5 – 225.4 |
| Hazardous | ≥ 225.5 |

They are not an official health advisory; follow your local authority's guidance. Edit `ALERT_LEVELS` in
`src/analytics.py` to use another standard (e.g. WHO or your national limits).

## 9. Testing

`pytest` runs `tests/test_processing.py`, which covers: column normalisation and validation errors, cleaning of
bad rows, filtering, reproducibility of the synthetic generator, alert classification boundaries, overview and
location statistics, leakage checks on features, the chronological split (train dates all precede test dates),
forecast dates, the "too little data" error, the SQL file executing on SQLite and matching Pandas, and the
PySpark "not installed" message.

## 10. Limitations

* The sample data is synthetic, so evaluation results are about the simulation, not the real atmosphere.
* Predictions are **one day ahead** and depend strongly on yesterday's PM2.5.
* The model has no information about future weather, traffic, fires, festivals or industrial events.
* One chronological test window can be optimistic or pessimistic; other seasons/years may differ.
* Gaps in a user's data reduce the number of usable training rows (lags need consecutive days). Data finer than
  daily is averaged to daily values.
* Median imputation of missing features is simple and can hide real patterns.
* PySpark runs on one machine here; it shows the programming model, not cluster-scale performance.

## 11. Uploading to GitHub

1. Create an empty repository on github.com (no README/licence, so there are no conflicts).
2. In the VS Code terminal, from the project root:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: AirCast AI air quality analytics project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
3. `.gitignore` already excludes `.venv/`, caches and the `outputs/` folder, so only source code and the small sample CSV are uploaded.
4. If you use real data, document its source and usage terms here before publishing it.

## 12. Troubleshooting

| Problem | Fix |
|---------|-----|
| `streamlit` is not recognised | Activate the virtual environment, then `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root, using `python -m src.…` for modules |
| Prediction tab says not enough data | Your file needs consecutive daily history (at least ~60 usable rows over 20+ days) |
| PySpark error about Java | Install JDK 17, open a new terminal, check `java -version` |
