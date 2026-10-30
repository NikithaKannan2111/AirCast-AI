-- =============================================================================
-- AirCast AI - analytical SQL queries
-- =============================================================================
-- DIALECT: SQLite (ships with Python, no server needed). Notes for MySQL /
--          PostgreSQL are given in comments where the syntax differs.
--
-- EXPECTED TABLE SCHEMA
--   CREATE TABLE air_quality (
--       timestamp   TEXT,   -- 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
--       location    TEXT,   -- monitoring station name
--       pm25        REAL,   -- ug/m3   (may be NULL)
--       pm10        REAL,   -- ug/m3   (may be NULL)
--       no2         REAL,   -- ug/m3   (may be NULL)
--       o3          REAL,   -- ug/m3   (may be NULL)
--       temperature REAL,   -- degrees Celsius
--       humidity    REAL,   -- percent
--       wind_speed  REAL    -- m/s
--   );
--
-- HOW TO RUN (from the project root, no database server required):
--   python -m src.sql_runner
--   This cleans data/sample_air_quality.csv, loads it into an in-memory SQLite
--   table called air_quality and runs every query below.
--   Alternatively open the file in any SQLite client (e.g. "DB Browser for
--   SQLite") after importing the CSV into a table named air_quality.
--
-- Each query starts with a "-- name:" line. The runner uses that name.
-- =============================================================================

-- name: average_pollution_by_location
-- Average pollution levels per monitoring station, highest PM2.5 first.
SELECT location,
       COUNT(*)                 AS readings,
       ROUND(AVG(pm25), 2)      AS avg_pm25,
       ROUND(AVG(pm10), 2)      AS avg_pm10,
       ROUND(AVG(no2), 2)       AS avg_no2,
       ROUND(AVG(o3), 2)        AS avg_o3
FROM air_quality
GROUP BY location
ORDER BY avg_pm25 DESC;

-- name: highest_pm25_readings
-- The ten highest PM2.5 readings in the data set.
SELECT timestamp, location, pm25
FROM air_quality
WHERE pm25 IS NOT NULL
ORDER BY pm25 DESC
LIMIT 10;

-- name: daily_trend_all_locations
-- Network-wide daily average PM2.5 (first 31 days shown; remove LIMIT for all).
-- MySQL/PostgreSQL: DATE(timestamp) also works if the column is a date type.
SELECT DATE(timestamp)          AS day,
       ROUND(AVG(pm25), 2)      AS avg_pm25,
       ROUND(MAX(pm25), 2)      AS max_pm25
FROM air_quality
WHERE pm25 IS NOT NULL
GROUP BY DATE(timestamp)
ORDER BY day
LIMIT 31;

-- name: monthly_trend_by_location
-- Monthly average PM2.5 per location.
-- PostgreSQL: use TO_CHAR(timestamp, 'YYYY-MM');  MySQL: DATE_FORMAT(timestamp, '%Y-%m')
SELECT STRFTIME('%Y-%m', timestamp) AS month,
       location,
       ROUND(AVG(pm25), 2)          AS avg_pm25
FROM air_quality
WHERE pm25 IS NOT NULL
GROUP BY STRFTIME('%Y-%m', timestamp), location
ORDER BY month, location
LIMIT 40;

-- name: days_above_moderate_threshold
-- Number of readings above 35.4 ug/m3 (upper end of the "Moderate" category).
SELECT location,
       SUM(CASE WHEN pm25 > 35.4 THEN 1 ELSE 0 END)                          AS readings_above_35_4,
       COUNT(pm25)                                                           AS readings_with_pm25,
       ROUND(100.0 * SUM(CASE WHEN pm25 > 35.4 THEN 1 ELSE 0 END) / COUNT(pm25), 1) AS percent_above
FROM air_quality
GROUP BY location
ORDER BY percent_above DESC;

-- name: missing_value_analysis
-- Missing values per column for the whole table.
SELECT COUNT(*)                                                    AS total_rows,
       SUM(CASE WHEN pm25 IS NULL THEN 1 ELSE 0 END)               AS missing_pm25,
       SUM(CASE WHEN pm10 IS NULL THEN 1 ELSE 0 END)               AS missing_pm10,
       SUM(CASE WHEN no2 IS NULL THEN 1 ELSE 0 END)                AS missing_no2,
       SUM(CASE WHEN o3 IS NULL THEN 1 ELSE 0 END)                 AS missing_o3,
       SUM(CASE WHEN temperature IS NULL THEN 1 ELSE 0 END)        AS missing_temperature,
       SUM(CASE WHEN humidity IS NULL THEN 1 ELSE 0 END)           AS missing_humidity,
       SUM(CASE WHEN wind_speed IS NULL THEN 1 ELSE 0 END)         AS missing_wind_speed
FROM air_quality;

-- name: missing_pm25_by_location
-- Share of missing PM2.5 values per location.
SELECT location,
       COUNT(*)                                                      AS total_rows,
       SUM(CASE WHEN pm25 IS NULL THEN 1 ELSE 0 END)                 AS missing_pm25,
       ROUND(100.0 * SUM(CASE WHEN pm25 IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS missing_pct
FROM air_quality
GROUP BY location
ORDER BY missing_pct DESC;
