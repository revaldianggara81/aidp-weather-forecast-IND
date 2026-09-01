"""
explore_data.py — Profile the raw data to inform preprocessing decisions.

Checks:
  - Date range and completeness per city
  - Duplicate rows
  - Outliers (Z-score and IQR)
  - Distribution skewness (especially precipitation)
  - Zero/near-zero precipitation days
"""

import os
import warnings
import numpy as np
import pandas as pd
import oracledb
from dotenv import load_dotenv
from scipy import stats

warnings.filterwarnings("ignore")
load_dotenv()

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]


def load_data():
    conn = oracledb.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        dsn=os.getenv("CONNECTION_STRING"),
        config_dir=os.getenv("DB_WALLET_LOCATION"),
        wallet_location=os.getenv("DB_WALLET_LOCATION"),
        wallet_password=os.getenv("DB_WALLET_PASSWORD"),
    )
    table = os.getenv("TABLE_NAME")
    with conn.cursor() as cur:
        cur.execute(
            f'SELECT "DATE", CITY, TEMPERATURE_C, PRECIPITATION_MM, WIND_SPEED_KMH '
            f'FROM {table} ORDER BY CITY, "DATE"'
        )
        cols = [c[0].upper() for c in cur.description]
        rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=cols)
    df["DATE"] = pd.to_datetime(df["DATE"])
    for t in TARGETS:
        df[t] = pd.to_numeric(df[t])
    return df


def check_completeness(df):
    print("\n── 1. Date Range & Completeness per City ─────────────────────────────")
    full_range = pd.date_range(df["DATE"].min(), df["DATE"].max(), freq="D")
    print(f"  Overall: {df['DATE'].min().date()} → {df['DATE'].max().date()} "
          f"({len(full_range)} expected days)\n")

    print(f"  {'City':<12} {'Rows':>6} {'Missing days':>13} {'Duplicates':>11}")
    print("  " + "─" * 46)
    for city in sorted(df["CITY"].unique()):
        cdf = df[df["CITY"] == city].sort_values("DATE")
        missing = len(full_range) - len(cdf)
        dups = cdf.duplicated(subset=["DATE"]).sum()
        print(f"  {city:<12} {len(cdf):>6} {missing:>13} {dups:>11}")


def check_nulls(df):
    print("\n── 2. Null / NaN Values ───────────────────────────────────────────────")
    null_counts = df[TARGETS].isnull().sum()
    if null_counts.sum() == 0:
        print("  No nulls found.")
    else:
        for col, cnt in null_counts.items():
            if cnt > 0:
                print(f"  {col}: {cnt} nulls")


def check_outliers(df):
    print("\n── 3. Outliers ────────────────────────────────────────────────────────")
    print(f"  {'Target':<22} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} "
          f"{'Z>3':>6} {'IQR×3':>7}")
    print("  " + "─" * 72)

    for target in TARGETS:
        vals = df[target].dropna()
        z_outliers   = (np.abs(stats.zscore(vals)) > 3).sum()
        q1, q3       = vals.quantile(0.25), vals.quantile(0.75)
        iqr          = q3 - q1
        iqr_outliers = ((vals < q1 - 3 * iqr) | (vals > q3 + 3 * iqr)).sum()
        print(f"  {target:<22} {vals.min():>8.2f} {vals.max():>8.2f} "
              f"{vals.mean():>8.2f} {vals.std():>8.2f} "
              f"{z_outliers:>6} {iqr_outliers:>7}")


def check_skewness(df):
    print("\n── 4. Skewness & Distribution ────────────────────────────────────────")
    print(f"  {'Target':<22} {'Skewness':>10} {'Kurtosis':>10}  Note")
    print("  " + "─" * 58)

    for target in TARGETS:
        vals = df[target].dropna()
        skew = vals.skew()
        kurt = vals.kurtosis()
        note = ""
        if abs(skew) > 1:
            note = "HIGH SKEW — consider log transform"
        elif abs(skew) > 0.5:
            note = "moderate skew"
        print(f"  {target:<22} {skew:>10.3f} {kurt:>10.3f}  {note}")


def check_precipitation_zeros(df):
    print("\n── 5. Precipitation Zero-days per City ───────────────────────────────")
    print(f"  {'City':<12} {'Total':>7} {'Zero days':>10} {'Zero %':>8} {'Max':>8} {'p95':>8}")
    print("  " + "─" * 56)
    for city in sorted(df["CITY"].unique()):
        vals = df[df["CITY"] == city]["PRECIPITATION_MM"]
        zero_pct = (vals == 0).mean() * 100
        print(f"  {city:<12} {len(vals):>7} {(vals == 0).sum():>10} "
              f"{zero_pct:>7.1f}% {vals.max():>8.2f} {vals.quantile(0.95):>8.2f}")


def check_cross_correlations(df):
    print("\n── 6. Cross-variable Pearson Correlations (all cities) ───────────────")
    corr = df[TARGETS].corr().round(3)
    print(corr.to_string())


def main():
    print("Loading data from Oracle ADB...")
    df = load_data()
    print(f"  {len(df):,} rows loaded.")

    check_completeness(df)
    check_nulls(df)
    check_outliers(df)
    check_skewness(df)
    check_precipitation_zeros(df)
    check_cross_correlations(df)

    print("\n── Done ───────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
