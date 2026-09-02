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

## Post-M5 work

Deployment and defect fixes after the initial push.

### Docker

Stack namespaced apart from the Indonesian one so both run on the same host:
`aidp-india-weather-forecast` / `aidp-india-weather-nginx` on port **9334**
(Indonesia uses 9333), image `aidp-india-web`. `env_file` is `required: false`
so the stack starts before the Delta Share exists. See README for details.

### Defects found and fixed

1. **NaN crash in gold training.** `df.pivot` on a DATE x CITY grid leaves NaN
   wherever a region is missing a date; `_global_zscore` used `nanmean`/`nanstd`,
   which tolerate NaN but propagate it. Validation loss went NaN, so
   `vl < best_val` was never true, `best_state` stayed `None`, and
   `load_state_dict(None)` raised a TypeError 20 epochs after the real problem.
   Silver now drops dates not present for every region and asserts the grid is
   complete; gold raises early with a message naming the cause.

2. **Global statistics on a climatically diverse country.** Normalisation and
   anomaly thresholds were computed across all 36 regions pooled. Ladakh (4 C
   annual mean against a 12-29 C national range) was forecast 8.5 C too warm,
   and 95th-percentile anomalies concentrated on Lakshadweep (23/30 days) and
   the Andamans (21/30) while 21 regions never flagged anything. Both are now
   per-region. Latent in the Indonesian original, where 6 uniformly tropical
   cities made pooling harmless.

3. **Map markers geocoded to the wrong continent.** `/api/coords` geocoded
   region names with `count=1` and no country filter: Assam landed in
   Mozambique, Goa in Genova, Odisha in Germany, Kerala in Finland, and only
   16 of 36 regions resolved at all. Coordinates now come from
   `frontend/public/india_region_coords.json`, generated from
   `configuration.yaml` -- the same verified source that produced the weather
   data, so markers and forecasts cannot diverge. The route makes no network
   calls.

### Known remaining risk

Fixes 1 and 2 are in the notebooks but the AIDP gold table still holds output
from the pre-fix run. Re-run gold in AIDP to pick them up.
