-- create_table.sql
--
-- Target Oracle ADB table for India's daily weather data.
--
-- The CITY column holds an Indian state or union territory name (up to
-- 40 chars, e.g. "Dadra and Nagar Haveli and Daman and Diu"), hence
-- VARCHAR2(64) to leave headroom.
--
-- Populated by ingest_data.py (fetches from the Open-Meteo Archive API and
-- bulk-inserts via cursor.executemany()).
--
-- The table name here (INDIA_WEATHER) must match the TABLE_NAME env var
-- read by ingest_data.py and compare_models.py — if you rename the table,
-- update TABLE_NAME in .env accordingly.
--
-- "DATE" is an Oracle reserved word and must be quoted whenever referenced.

CREATE TABLE INDIA_WEATHER (
  "DATE"            DATE          NOT NULL,
  CITY              VARCHAR2(64)  NOT NULL,
  TEMPERATURE_C     NUMBER(6,2),
  PRECIPITATION_MM  NUMBER(8,2),
  WIND_SPEED_KMH    NUMBER(6,2),
  CONSTRAINT PK_INDIA_WEATHER PRIMARY KEY ("DATE", CITY)
);

CREATE INDEX IX_INDIA_WEATHER_CITY ON INDIA_WEATHER (CITY);
