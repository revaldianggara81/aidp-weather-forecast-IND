# PLAN.md

## Project

30-day weather forecasting (temperature, precipitation, wind speed) for all
36 Indian states and union territories, backed by Oracle ADB with a
medallion (bronze/silver/gold) data pipeline and a Next.js dashboard.
Adapted from the Indonesian version at `/raid/aidp`.

## Regions

- 28 states + 8 union territories (36 regions total).
- Each region is represented by the coordinates of its administrative
  capital.
- The Oracle `CITY` column stores the region name (e.g. "Tamil Nadu"), not
  the capital city name.
- Two documented exceptions: Haryana uses Gurugram and Punjab uses Ludhiana
  instead of their shared capital Chandigarh, so that Haryana, Punjab, and
  the union territory of Chandigarh do not produce three identical time
  series.

## Database

Shares an existing Oracle ADB instance with the Indonesian project — no new
instance or wallet was provisioned.

| Setting | Value |
|---|---|
| Wallet | see `DB_WALLET_LOCATION` in `.env` (reused, not copied into this repo) |
| DSN | see `CONNECTION_STRING` in `.env` |
| Schema / user | see `DB_USER` in `.env` |
| Table | `INDIA_WEATHER` |

Concrete connection values live in `.env`, which is gitignored — copy
`.env.example` and fill it in. `INDIA_WEATHER` shares a schema with the
Indonesian project's `INDONESIA_WEATHER` table. `ingest_data.py` issues a
`TRUNCATE` against whatever `TABLE_NAME` resolves to, so
`TABLE_NAME=INDIA_WEATHER` in `.env` is the single control keeping the
Indonesian data safe. Verify it before every ingest run.

Table created with `sql/create_table.sql`: PK `("DATE", CITY)` plus index
`IX_INDIA_WEATHER_CITY`.

## Milestones

- **M1 — Config + ingestion layer for India** — DONE
- **M2 — Frontend adaptation** — DONE
  - `public/india_states.geojson`: geoBoundaries gbOpen IND ADM1, 36 features,
    coordinates rounded to 4dp (4.9 MB -> 1.98 MB). Replaces the unused
    `cities.geojson`.
  - `/api/boundary` rewritten to serve from that local file instead of querying
    Nominatim per region (Nominatim forbids systematic querying; 36 lookups per
    dashboard session would abuse it). Diacritic-insensitive lookup.
  - Map centre `[22, 82.5]` zoom 4; per-region fallback zoom 9 -> 6.
  - Branding + "City" -> "Region" labels; architecture page and README copy.
  - `npm run build` passes, no TypeScript errors. Boundary route tested live:
    all 36 regions resolve, 404 on unknown, 400 on missing param.
- **M3 — Medallion notebook adaptation** (bronze/silver/gold tables + region
  names) — TODO
- **M4 — Data ingestion into Oracle ADB** — DONE
  - `INDIA_WEATHER`: 65,772 rows, 36 regions x 1827 days,
    2021-08-31 -> 2026-08-31. No NULLs, no negative precipitation.
  - Attribution validated against climate: coldest = Ladakh (4.05 C),
    Himachal Pradesh, Jammu & Kashmir (all Himalayan); wettest = Nagaland
    (4063 mm/yr), Sikkim, Arunachal Pradesh (all NE India); driest =
    Ladakh (326 mm/yr), Rajasthan (677 mm/yr).
  - `INDONESIA_WEATHER` unchanged at 10,962 rows.
- **M5 — git init and push** — DONE
  - Repo: https://github.com/revaldianggara81/aidp-weather-forecast-IND
  - 73 files on `main`. Notebook outputs cleared before the first commit so
    the public repo carries no stale Indonesian execution results.
  - `.env`, wallet files, `node_modules/`, `.next/`, `__pycache__/` and
    Claude artefacts are all gitignored and verified absent from the remote.
  - Concrete DSN / wallet path / schema names are kept out of this file;
    they live in `.env` only.

## Carried-over items

Docstrings in `forecast.py`, `prophet_forecast.py`, and `lstm_forecast.py`
still mention Indonesia. They are cosmetic (these scripts read the table
name from `TABLE_NAME` and derive regions from the data) but should be
cleaned up alongside M2/M3.

## Known adaptation risks

Hyperparameters in `configuration.yaml` were tuned on Indonesia's uniformly
tropical climate. India spans desert, Himalayan, and monsoon regimes, so
precipitation accuracy in particular is expected to need retuning after the
first data load.

Open-Meteo's free tier rate-limits per minute and per hour. With 36 regions
the archive fetch must batch multiple coordinates per request; one request
per region trips HTTP 429.
