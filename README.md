# AIDP Weather Forecasting — Oracle AICEC Demo

An end-to-end weather forecasting system for Indian states and union territories, built on Oracle Autonomous Database (ADB) and the AIDP Medallion architecture. The system ingests raw weather data, trains machine learning models to produce 30-day forecasts with anomaly detection, and serves the results through an interactive Next.js dashboard.

---

## Architecture Overview

```
Open-Meteo API → Bronze → Silver → Gold → Delta Sharing → Frontend
```

| Layer | Description |
|---|---|
| **Bronze** | Raw ingestion from Open-Meteo API (historical weather data for 36 Indian states and union territories) |
| **Silver** | Data cleaning — handles missing values, removes duplicates |
| **Gold** | ML forecasting, anomaly detection, and LLM-generated safety recommendations |
| **Delta Sharing** | Gold table published via OCI Delta Sharing (Delta Sharing Protocol) |
| **Frontend** | Next.js dashboard consuming Delta Sharing data |

Medallion pipeline notebooks are located in `medallion/`.

---

## ML Model

The Gold layer uses a **Multivariate Autoregressive LSTM** (`models/ar_lstm_model.py`) trained on the full historical data for each region. It produces 30-day forecasts for temperature, precipitation, and wind speed, with Gaussian noise for uncertainty estimation.

Forecast script: `lstm_ar_forecast.py`

---

## Frontend

Located in `frontend/`. Built with Next.js 16, Tailwind CSS, Recharts, and Leaflet.

**Pages:**
- `/` — Dashboard with per-region weather charts (temperature, precipitation, wind speed), anomaly highlights, and LLM recommendation dialogs
- `/map` — Interactive Leaflet map showing all regions with pulsing anomaly markers (pulse intensity scales with anomaly count)
- `/architecture` — System architecture explanation

**API routes:**
- `/api/weather` — Fetches and caches forecast data from Delta Sharing (1-hour TTL)
- `/api/coords` — Geocodes region names dynamically via Open-Meteo geocoding API (1-hour TTL)

---

## Setup

### Python (ML pipeline)

```bash
cd /raid/aidp-weather-forecast-india
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
npm run build
npm start
```

> Requires Node.js ≥ 20. Set environment variables in `frontend/.env.local` (see Delta Sharing config).

---

## Environment Variables

Create `frontend/.env.local`:

```
DELTA_SHARE_ENDPOINT=
DELTA_SHARE_TOKEN=
DELTA_SHARE_NAME=
DELTA_SCHEMA_NAME=
DELTA_TABLE_NAME=
```

Create `.env` in the project root for the ML pipeline:

```
DB_USER=
DB_PASS=
DB_WALLET_LOCATION=
DB_WALLET_PASSWORD=
CONNECTION_STRING=
TABLE_NAME=INDIA_WEATHER
```

`TABLE_NAME` must match the Oracle table created by `sql/create_table.sql`, which
defines the target `INDIA_WEATHER` table read/written by `ingest_data.py` and
`compare_models.py`.

---

## Regions

The pipeline covers all 28 states and 8 union territories of India (36 regions
total). Each region is represented by its administrative capital, with two
exceptions: Haryana and Punjab share their capital, Chandigarh, with the
Chandigarh union territory itself, so Haryana uses Gurugram and Punjab uses
Ludhiana instead — otherwise those three regions would produce identical
weather series.

---

## Data Source

Weather data sourced from [Open-Meteo](https://open-meteo.com/) (free, no API key required).
