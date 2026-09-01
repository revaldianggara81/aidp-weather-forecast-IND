"""
ingest_data.py — Fetch 5 years of daily weather data from Open-Meteo and
load it into the Oracle ADB table named by the TABLE_NAME env var
(default INDIA_WEATHER), covering India's 28 states and 8 union territories.

Steps:
  1. Fetch N years of daily data from Open-Meteo Archive API
  2. Ask for confirmation, then TRUNCATE the table
  3. Bulk-insert all rows with cursor.executemany()
  4. Print a verification summary
"""

import argparse
import os
import re
import sys
import time
from datetime import date, datetime, timedelta

import oracledb
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

TABLE = os.getenv("TABLE_NAME", "INDIA_WEATHER")
if not re.match(r"^[A-Za-z][A-Za-z0-9_$#]*$", TABLE):
    raise SystemExit(
        f"Invalid TABLE_NAME env var: {TABLE!r}. Must be a valid Oracle "
        "identifier matching ^[A-Za-z][A-Za-z0-9_$#]*$"
    )

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
]

# Open-Meteo's free tier rate-limits aggressively (HTTP 429) if we issue one
# request per region. Batch multiple regions into a single request via
# comma-separated latitude/longitude instead.
CHUNK_SIZE = 10


# ── Config ────────────────────────────────────────────────────────────────────

def load_config():
    with open("configuration.yaml") as f:
        return yaml.safe_load(f)


# ── Weather fetch ─────────────────────────────────────────────────────────────

def _get_with_retry(url: str, params: dict, timeout: int = 30) -> requests.Response:
    """
    GET with retry on HTTP 429 / 5xx: exponential backoff of 30, 60, 120, 240
    seconds (4 retries). Re-raises if all attempts fail. Other 4xx errors are
    not retried.
    """
    backoffs = [30, 60, 120, 240]
    for i in range(len(backoffs) + 1):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            if i == len(backoffs):
                resp.raise_for_status()
            wait = backoffs[i]
            print(f"  rate limited ({resp.status_code}) — retrying in {wait}s "
                  f"(attempt {i + 1}/{len(backoffs)})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp


def fetch_weather_batch(regions: list[dict], start: date, end: date) -> list[tuple]:
    """
    Fetch daily weather for multiple regions in a single Open-Meteo request
    (comma-separated latitude/longitude). Returns a flat list of
    (date_obj, region, temperature_c, precipitation_mm, wind_speed_kmh) tuples,
    covering all regions, in region order.
    """
    lats = ",".join(str(r["lat"]) for r in regions)
    lons = ",".join(str(r["lon"]) for r in regions)

    resp = _get_with_retry(
        ARCHIVE_URL,
        params={
            "latitude":  lats,
            "longitude": lons,
            "start_date": start.isoformat(),
            "end_date":   end.isoformat(),
            "daily":      ",".join(DAILY_VARS),
            "timezone":   "auto",
        },
    )
    payload = resp.json()
    results = payload if isinstance(payload, list) else [payload]

    if len(results) != len(regions):
        raise RuntimeError(
            f"Open-Meteo returned {len(results)} result(s) for {len(regions)} "
            "requested region(s) — refusing to zip by index, since a mismatch "
            "would misattribute weather data to the wrong region."
        )

    rows = []
    for region, data in zip(regions, results):
        daily  = data["daily"]
        times  = daily["time"]
        temps  = daily["temperature_2m_mean"]
        precip = daily["precipitation_sum"]
        wind   = daily["wind_speed_10m_max"]

        for t, te, pr, wi in zip(times, temps, precip, wind):
            # Replace None (missing values) with 0 for precipitation, skip row if
            # temperature or wind are missing
            if te is None or wi is None:
                continue
            pr = pr if pr is not None else 0.0
            date_obj = datetime.strptime(t, "%Y-%m-%d").date()
            rows.append((date_obj, region["name"], float(te), float(pr), float(wi)))

    return rows


# ── Oracle helpers ─────────────────────────────────────────────────────────────

def connect():
    return oracledb.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        dsn=os.getenv("CONNECTION_STRING"),
        config_dir=os.getenv("DB_WALLET_LOCATION"),
        wallet_location=os.getenv("DB_WALLET_LOCATION"),
        wallet_password=os.getenv("DB_WALLET_PASSWORD"),
    )


def current_table_info(conn) -> dict:
    """Return {city: row_count} for the existing table."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                f'SELECT CITY, COUNT(*), MIN("DATE"), MAX("DATE") '
                f'FROM {TABLE} GROUP BY CITY ORDER BY CITY'
            )
            rows = cur.fetchall()
    except oracledb.DatabaseError as e:
        (error,) = e.args
        if getattr(error, "code", None) == 942:  # ORA-00942: table or view does not exist
            return {}
        raise
    return {r[0]: {"count": r[1], "first": r[2], "last": r[3]} for r in rows}


def truncate_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {TABLE}")
    conn.commit()


def bulk_insert(conn, rows: list[tuple]):
    sql = (
        f'INSERT INTO {TABLE} ("DATE", CITY, TEMPERATURE_C, '
        f'PRECIPITATION_MM, WIND_SPEED_KMH) VALUES (:1, :2, :3, :4, :5)'
    )
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch and load India weather data into Oracle ADB."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch from Open-Meteo and print a summary WITHOUT opening any "
             "database connection.",
    )
    parser.add_argument(
        "--years", type=int, default=5,
        help="Years of history to fetch (default: 5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg     = load_config()
    regions = cfg["regions"]

    end_date   = date.today() - timedelta(days=1)   # yesterday (archive lag)
    start_date = end_date.replace(year=end_date.year - args.years)

    print(f"Target date range : {start_date} → {end_date}  ({args.years} years)")
    print(f"Regions           : {len(regions)}\n")

    # ── Step 1: Print regions and coordinates from config ────────────────────
    print("Regions (from configuration.yaml):")
    for r in regions:
        print(f"  {r['name']:<42}  lat={r['lat']:>9.4f}  lon={r['lon']:>10.4f}")

    # ── Step 2: Fetch weather data ────────────────────────────────────────────
    print(f"\nFetching {args.years}-year daily weather from Open-Meteo Archive "
          f"(batched, {CHUNK_SIZE} regions/request)...")
    all_rows = []
    for i in range(0, len(regions), CHUNK_SIZE):
        chunk = regions[i:i + CHUNK_SIZE]
        chunk_rows = fetch_weather_batch(chunk, start_date, end_date)
        all_rows.extend(chunk_rows)

        for r in chunk:
            region = r["name"]
            region_rows = [row for row in chunk_rows if row[1] == region]
            if region_rows:
                print(f"  {region:<42}  {len(region_rows):>5} rows  "
                      f"({region_rows[0][0]} → {region_rows[-1][0]})")
            else:
                print(f"  {region:<42}  {0:>5} rows  (no data)")

        if i + CHUNK_SIZE < len(regions):
            time.sleep(2)   # be polite to the free API between chunks

    print(f"\n  Total rows fetched: {len(all_rows):,}")

    if args.dry_run:
        print("\nDry run complete. No database connection was opened.")
        sys.exit(0)

    # ── Step 3: Confirm before truncating ─────────────────────────────────────
    conn = connect()
    print("\n── Current table contents ───────────────────────────────────────────")
    info = current_table_info(conn)
    if not info:
        print("  (table is empty or does not exist yet)")
    for region, d in info.items():
        print(f"  {region:<42}  {d['count']:>5} rows  ({str(d['first'])[:10]} → {str(d['last'])[:10]})")

    print(f"\n⚠️  WARNING: TRUNCATE TABLE {TABLE} is irreversible.")
    answer = input("Type 'yes' to confirm truncate and re-insert: ").strip().lower()
    if answer != "yes":
        print("Aborted. No changes made.")
        conn.close()
        sys.exit(0)

    # ── Step 4: Truncate and insert ───────────────────────────────────────────
    print("\nTruncating table...")
    truncate_table(conn)
    print("  Done.")

    print(f"Inserting {len(all_rows):,} rows...")
    bulk_insert(conn, all_rows)
    print("  Done.")

    # ── Step 5: Verify ────────────────────────────────────────────────────────
    print("\n── Verification — new table contents ────────────────────────────────")
    info_new = current_table_info(conn)
    total = 0
    for region, d in info_new.items():
        print(f"  {region:<42}  {d['count']:>5} rows  ({str(d['first'])[:10]} → {str(d['last'])[:10]})")
        total += d["count"]
    print(f"\n  Total rows in table: {total:,}")

    conn.close()
    print(f"\nDone. Table successfully updated with {args.years}-year data.")


if __name__ == "__main__":
    main()
