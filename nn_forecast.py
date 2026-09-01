"""
nn_forecast.py — Neural network weather forecasting with anomaly detection.

Models: LSTM | N-HiTS | TFT
For each model:
  - Trains on all data except the last 30 days (test evaluation)
  - Retrains on full data and forecasts the next 30 days
  - Flags forecast days where the predicted value exceeds the 90th percentile
    of the historical distribution for that city × variable

Plots are generated only for cities that have at least one anomaly in any model.
Each plot shows all three model forecasts together with the threshold line.
"""

import logging
import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import oracledb
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("neuralforecast").setLevel(logging.ERROR)

load_dotenv()

from models import lstm_model, nhits_model, tft_model

# ── Config ────────────────────────────────────────────────────────────────────

with open("configuration.yaml") as f:
    _cfg = yaml.safe_load(f)

TEST_DAYS     = _cfg["forecasting"]["test_days"]
FORECAST_DAYS = _cfg["forecasting"]["forecast_days"]
CONTEXT_DAYS  = _cfg["forecasting"]["context_days"]

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]
TARGET_LABELS = {
    "TEMPERATURE_C":    "Temperature (°C)",
    "PRECIPITATION_MM": "Precipitation (mm)",
    "WIND_SPEED_KMH":   "Wind Speed (km/h)",
}

MODELS = {
    "LSTM":   lstm_model,
    "N-HiTS": nhits_model,
    "TFT":    tft_model,
}
MODEL_COLORS = {
    "LSTM":   "#8E24AA",   # purple
    "N-HiTS": "#1565C0",   # blue
    "TFT":    "#00897B",   # teal
}

ANOMALY_PERCENTILE = 90
THRESHOLD_COLOR    = "#FF6F00"   # orange
ANOMALY_COLOR      = "#FF6F00"

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
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


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(test_preds: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city in sorted(test_preds["CITY"].unique()):
        tp = test_preds[test_preds["CITY"] == city].sort_values("DATE").reset_index(drop=True)
        ar = (
            actuals[(actuals["CITY"] == city) & (actuals["DATE"].isin(tp["DATE"]))]
            .sort_values("DATE")
            .reset_index(drop=True)
        )
        for target in TARGETS:
            mae  = mean_absolute_error(ar[target], tp[target])
            rmse = np.sqrt(mean_squared_error(ar[target], tp[target]))
            rows.append({"City": city, "Target": target, "MAE": mae, "RMSE": rmse})
    return pd.DataFrame(rows)


def print_metrics_table(all_metrics: dict):
    print("\n── Test Set MAE Summary " + "─" * 56)
    print(f"  {'Model':<10} {'City':<12} {'Temp':>8} {'Precip':>10} {'Wind':>8}")
    print("  " + "─" * 52)
    for model_name, metrics in all_metrics.items():
        for city in sorted(metrics["City"].unique()):
            m = metrics[metrics["City"] == city]
            vals = {t: m[m["Target"] == t]["MAE"].values[0] for t in TARGETS}
            print(
                f"  {model_name:<10} {city:<12} "
                f"{vals['TEMPERATURE_C']:>8.2f} "
                f"{vals['PRECIPITATION_MM']:>10.2f} "
                f"{vals['WIND_SPEED_KMH']:>8.2f}"
            )
        # Print per-model average
        avgs = {t: metrics[metrics["Target"] == t]["MAE"].mean() for t in TARGETS}
        print(
            f"  {'':10} {'  AVERAGE':<12} "
            f"{avgs['TEMPERATURE_C']:>8.2f} "
            f"{avgs['PRECIPITATION_MM']:>10.2f} "
            f"{avgs['WIND_SPEED_KMH']:>8.2f}"
        )
        print()


# ── Anomaly Detection ─────────────────────────────────────────────────────────

def compute_thresholds(df: pd.DataFrame) -> dict:
    """90th percentile threshold per city × target from historical data."""
    thresholds = {}
    for city in sorted(df["CITY"].unique()):
        city_df = df[df["CITY"] == city]
        thresholds[city] = {
            t: city_df[t].quantile(ANOMALY_PERCENTILE / 100)
            for t in TARGETS
        }
    return thresholds


def detect_anomalies(future_preds: pd.DataFrame, thresholds: dict,
                     model_name: str) -> pd.DataFrame:
    """Return a DataFrame of forecast days exceeding the 90th percentile threshold."""
    rows = []
    for city in sorted(future_preds["CITY"].unique()):
        fp = future_preds[future_preds["CITY"] == city].sort_values("DATE")
        for target in TARGETS:
            thr = thresholds[city][target]
            for _, day in fp[fp[target] > thr].iterrows():
                rows.append({
                    "Model":     model_name,
                    "City":      city,
                    "Date":      day["DATE"].date(),
                    "Variable":  TARGET_LABELS[target],
                    "Forecast":  round(day[target], 2),
                    "Threshold": round(thr, 2),
                    "Above by":  round(day[target] - thr, 2),
                })
    return pd.DataFrame(rows)


def print_anomaly_report(all_anomalies: dict):
    combined = pd.concat(all_anomalies.values(), ignore_index=True) if all_anomalies else pd.DataFrame()

    if combined.empty:
        print(f"\n  No anomalies detected at {ANOMALY_PERCENTILE}th percentile across any model.")
        return

    print(f"\n── Anomaly Report — forecast values above {ANOMALY_PERCENTILE}th percentile "
          + "─" * 16)

    for model_name, anomalies in all_anomalies.items():
        if anomalies.empty:
            print(f"\n  [{model_name}]  No anomalies detected.")
            continue
        print(f"\n  [{model_name}]  {len(anomalies)} anomaly day(s) across "
              f"{anomalies['City'].nunique()} city(ies)")
        print(f"  {'City':<12} {'Date':<12} {'Variable':<22} "
              f"{'Forecast':>10} {'Threshold':>10} {'Above by':>9}")
        print("  " + "─" * 78)
        for city in sorted(anomalies["City"].unique()):
            for _, row in anomalies[anomalies["City"] == city].iterrows():
                print(f"  {row['City']:<12} {str(row['Date']):<12} "
                      f"{row['Variable']:<22} "
                      f"{row['Forecast']:>10.2f} {row['Threshold']:>10.2f} "
                      f"{row['Above by']:>9.2f}")

    total = sum(len(a) for a in all_anomalies.values())
    cities = set(c for a in all_anomalies.values() for c in a["City"].unique())
    print(f"\n  Total anomalies across all models: {total}  |  "
          f"Cities affected: {', '.join(sorted(cities)) if cities else 'none'}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_city(actuals: pd.DataFrame, results: dict, city: str,
              thresholds: dict, out_dir: str):
    """
    3-panel plot for one city showing all models' forecasts plus threshold lines.
    Historical context + test actuals shown in black.
    Each model's forecast shown in its colour (dashed).
    Threshold shown as orange dotted line.
    Anomaly days marked with orange dots per model.
    """
    real = actuals[actuals["CITY"] == city].sort_values("DATE")

    # Derive date boundaries from any model's output
    first_test   = list(results.values())[0][0]
    first_future = list(results.values())[0][1]
    test_start   = first_test[first_test["CITY"] == city]["DATE"].min()
    test_end     = first_test[first_test["CITY"] == city]["DATE"].max()
    future_start = first_future[first_future["CITY"] == city]["DATE"].min()
    future_end   = first_future[first_future["CITY"] == city]["DATE"].max()

    context   = real[real["DATE"] < test_start].tail(CONTEXT_DAYS)
    test_real = real[(real["DATE"] >= test_start) & (real["DATE"] <= test_end)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle(f"Neural Network Forecast — {city}", fontsize=14,
                 fontweight="bold", y=0.99)

    for ax, target in zip(axes, TARGETS):
        thr = thresholds[city][target]

        # Actual data
        ax.plot(context["DATE"], context[target],
                color="black", linewidth=1.6, label="Actual", zorder=5)
        ax.plot(test_real["DATE"], test_real[target],
                color="black", linewidth=1.6, zorder=5)

        # Each model's forecast
        for model_name, (test_preds, future_preds) in results.items():
            fp = future_preds[future_preds["CITY"] == city].sort_values("DATE")
            tp = test_preds[test_preds["CITY"] == city].sort_values("DATE")
            color = MODEL_COLORS[model_name]

            ax.plot(tp["DATE"], tp[target],
                    color=color, linewidth=1.3, linestyle="--",
                    alpha=0.7, zorder=3)
            ax.plot(fp["DATE"], fp[target],
                    color=color, linewidth=1.8, linestyle="--",
                    label=model_name, alpha=0.9, zorder=4)

            # Anomaly markers
            anomaly_fp = fp[fp[target] > thr]
            if not anomaly_fp.empty:
                ax.scatter(anomaly_fp["DATE"], anomaly_fp[target],
                           color=color, zorder=6, s=50,
                           edgecolors="white", linewidths=0.8, marker="o")

        # Threshold line
        ax.axhline(thr, color=THRESHOLD_COLOR, linewidth=1.3,
                   linestyle=":", label=f"90th pctile ({thr:.1f})", zorder=2)

        # Shaded windows
        ax.axvspan(test_start,   test_end,   alpha=0.05, color="#E53935", zorder=1)
        ax.axvspan(future_start, future_end, alpha=0.05, color="#2E7D32", zorder=1)
        ax.axvline(test_start,   color="grey", linewidth=0.7, linestyle=":")
        ax.axvline(future_start, color="grey", linewidth=0.7, linestyle=":")

        ax.set_ylabel(TARGET_LABELS[target], fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.22, linestyle="--")
        ax.legend(loc="upper left", fontsize=8, ncol=3)

    ax = axes[-1]
    ylim = ax.get_ylim()
    mid_test   = test_start   + (test_end   - test_start)   / 2
    mid_future = future_start + (future_end - future_start) / 2
    ax.text(mid_test,   ylim[0], "← evaluation →",
            ha="center", fontsize=7, color="#E53935", alpha=0.8)
    ax.text(mid_future, ylim[0], "← forecast →",
            ha="center", fontsize=7, color="#2E7D32", alpha=0.8)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{city.lower().replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data from Oracle ADB...")
    df = load_data()
    print(
        f"  {len(df):,} rows | {df['CITY'].nunique()} cities | "
        f"{df['DATE'].min().date()} → {df['DATE'].max().date()}"
    )

    test_cutoff = df["DATE"].max() - pd.Timedelta(days=TEST_DAYS)
    train_df    = df[df["DATE"] <= test_cutoff].copy()
    print(f"  Train: up to {test_cutoff.date()} | Test: last {TEST_DAYS} days\n")

    results     = {}   # model_name -> (test_preds, future_preds)
    all_metrics = {}

    for model_name, module in MODELS.items():
        print(f"── Running {model_name} " + "─" * (52 - len(model_name)))
        try:
            test_preds, future_preds = module.run(df, train_df, _cfg)
            results[model_name]     = (test_preds, future_preds)
            all_metrics[model_name] = compute_metrics(test_preds, df)
            print(f"  {model_name} done.\n")
        except Exception as e:
            print(f"  {model_name} FAILED: {e}\n")

    if not results:
        print("All models failed — nothing to report.")
        return

    # ── Metrics ───────────────────────────────────────────────────────────────
    print_metrics_table(all_metrics)

    # ── Anomaly detection ─────────────────────────────────────────────────────
    print(f"Computing {ANOMALY_PERCENTILE}th percentile thresholds from historical data...")
    thresholds = compute_thresholds(df)

    cities = sorted(df["CITY"].unique())
    print(f"\n── {ANOMALY_PERCENTILE}th Percentile Thresholds per City ─────────────────────────────")
    print(f"  {'City':<12} {'Temp (°C)':>10} {'Precip (mm)':>12} {'Wind (km/h)':>12}")
    print("  " + "─" * 50)
    for city in cities:
        t = thresholds[city]
        print(f"  {city:<12} {t['TEMPERATURE_C']:>10.2f} "
              f"{t['PRECIPITATION_MM']:>12.2f} {t['WIND_SPEED_KMH']:>12.2f}")

    all_anomalies = {}
    for model_name, (_, future_preds) in results.items():
        all_anomalies[model_name] = detect_anomalies(future_preds, thresholds, model_name)

    print_anomaly_report(all_anomalies)

    # ── Plots — only cities with anomalies in any model ───────────────────────
    out_dir = "plots/nn"
    anomaly_cities = sorted(set(
        city
        for anomalies in all_anomalies.values()
        for city in anomalies["City"].unique()
    ))

    if anomaly_cities:
        print(f"\nGenerating plots for {len(anomaly_cities)} city(ies) with anomalies...")
        for city in anomaly_cities:
            plot_city(df, results, city, thresholds, out_dir)
    else:
        print("\nNo anomalies detected — plotting all cities for reference...")
        for city in cities:
            plot_city(df, results, city, thresholds, out_dir)

    print(f"\nDone. Plots saved to ./{out_dir}/")


if __name__ == "__main__":
    main()
