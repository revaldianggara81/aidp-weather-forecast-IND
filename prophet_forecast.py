"""
prophet_forecast.py — Prophet weather forecasting for Indian states and union territories.

For each city × target variable:
  - Trains a Prophet model on all data except the last 30 days
  - Evaluates on the held-out test period (last 30 days)
  - Retrains on full data and forecasts the next 30 days

Anomaly detection:
  - Computes 90th percentile threshold per city × variable from historical data
  - Flags any forecast day where the predicted value exceeds its threshold
  - Prints an anomaly report and plots only cities with at least one anomaly,
    with threshold lines and anomaly markers overlaid on the forecast
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
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

with open("configuration.yaml") as f:
    _cfg = yaml.safe_load(f)

PROPHET_CFG   = _cfg["prophet"]
TEST_DAYS     = _cfg["forecasting"]["test_days"]
FORECAST_DAYS = _cfg["forecasting"]["forecast_days"]
CONTEXT_DAYS  = _cfg["forecasting"]["context_days"]

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]
TARGET_LABELS = {
    "TEMPERATURE_C":    "Temperature (°C)",
    "PRECIPITATION_MM": "Precipitation (mm)",
    "WIND_SPEED_KMH":   "Wind Speed (km/h)",
}

ANOMALY_PERCENTILE = 90   # flag forecast values above this historical percentile

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


# ── Prophet ───────────────────────────────────────────────────────────────────

def fit_prophet(series_df: pd.DataFrame, target: str) -> Prophet:
    """Fit a Prophet model on a single city/target series."""
    df_p = series_df[["DATE", target]].rename(
        columns={"DATE": "ds", target: "y"}
    ).copy()
    df_p["ds"] = pd.to_datetime(df_p["ds"])

    m = Prophet(
        seasonality_mode=PROPHET_CFG["seasonality_mode"],
        yearly_seasonality=PROPHET_CFG["yearly_seasonality"],
        weekly_seasonality=PROPHET_CFG["weekly_seasonality"],
        daily_seasonality=PROPHET_CFG["daily_seasonality"],
        changepoint_prior_scale=float(PROPHET_CFG["changepoint_prior_scale"]),
    )
    m.fit(df_p)
    return m


def predict_prophet(model: Prophet, n_days: int, target: str) -> pd.DataFrame:
    """Forecast n_days beyond the last training date."""
    future = model.make_future_dataframe(periods=n_days, freq="D")
    fc = model.predict(future)
    last_train = model.history["ds"].max()

    result = fc[fc["ds"] > last_train][["ds", "yhat", "yhat_lower", "yhat_upper"]].head(n_days).copy()
    result = result.rename(columns={"ds": "DATE", "yhat": target,
                                    "yhat_lower": f"{target}_lower",
                                    "yhat_upper": f"{target}_upper"})
    result["DATE"] = pd.to_datetime(result["DATE"])

    if target == "PRECIPITATION_MM":
        result[target]              = result[target].clip(lower=0)
        result[f"{target}_lower"]   = result[f"{target}_lower"].clip(lower=0)
        result[f"{target}_upper"]   = result[f"{target}_upper"].clip(lower=0)

    return result.reset_index(drop=True)


# ── Train / Evaluate ──────────────────────────────────────────────────────────

def run_city(city_df: pd.DataFrame, city: str):
    """
    For one city:
      - Train on data up to test_cutoff, predict test period
      - Retrain on full data, predict future 30 days
    Returns (test_preds, future_preds) as DataFrames.
    """
    last_date   = city_df["DATE"].max()
    test_cutoff = last_date - pd.Timedelta(days=TEST_DAYS)

    train_df = city_df[city_df["DATE"] <= test_cutoff].copy()
    full_df  = city_df.copy()

    test_by_target   = {}
    future_by_target = {}

    for target in TARGETS:
        model_eval  = fit_prophet(train_df, target)
        test_fc     = predict_prophet(model_eval, TEST_DAYS, target)
        test_by_target[target] = test_fc

        model_full  = fit_prophet(full_df, target)
        future_fc   = predict_prophet(model_full, FORECAST_DAYS, target)
        future_by_target[target] = future_fc

    def merge_targets(preds_by_target):
        base = preds_by_target[TARGETS[0]][[
            "DATE", TARGETS[0], f"{TARGETS[0]}_lower", f"{TARGETS[0]}_upper"
        ]]
        for t in TARGETS[1:]:
            base = base.merge(
                preds_by_target[t][["DATE", t, f"{t}_lower", f"{t}_upper"]],
                on="DATE"
            )
        base.insert(1, "CITY", city)
        return base

    return merge_targets(test_by_target), merge_targets(future_by_target)


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


def print_metrics(metrics: pd.DataFrame):
    print("\n── Prophet Test Set Metrics (last 30 days) " + "─" * 36)
    print(f"{'City':<12} {'Temp MAE':>9} {'Temp RMSE':>10} {'Precip MAE':>11} "
          f"{'Precip RMSE':>12} {'Wind MAE':>9} {'Wind RMSE':>10}")
    print("─" * 78)
    for city in sorted(metrics["City"].unique()):
        m = metrics[metrics["City"] == city]
        def g(t, col): return m[m["Target"] == t][col].values[0]
        print(
            f"{city:<12} "
            f"{g('TEMPERATURE_C','MAE'):>9.2f} {g('TEMPERATURE_C','RMSE'):>10.2f} "
            f"{g('PRECIPITATION_MM','MAE'):>11.2f} {g('PRECIPITATION_MM','RMSE'):>12.2f} "
            f"{g('WIND_SPEED_KMH','MAE'):>9.2f} {g('WIND_SPEED_KMH','RMSE'):>10.2f}"
        )

    print("\n── Averages across cities " + "─" * 52)
    for target in TARGETS:
        sub = metrics[metrics["Target"] == target]
        print(f"  {TARGET_LABELS[target]:<25}  MAE: {sub['MAE'].mean():.2f}   RMSE: {sub['RMSE'].mean():.2f}")


# ── Anomaly Detection ─────────────────────────────────────────────────────────

def compute_thresholds(df: pd.DataFrame) -> dict:
    """
    Compute the 90th percentile threshold per city × target from historical data.
    Returns {city: {target: threshold_value}}.
    """
    thresholds = {}
    for city in sorted(df["CITY"].unique()):
        city_df = df[df["CITY"] == city]
        thresholds[city] = {
            target: city_df[target].quantile(ANOMALY_PERCENTILE / 100)
            for target in TARGETS
        }
    return thresholds


def detect_anomalies(future_preds: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Flag forecast days where the predicted value exceeds the 90th percentile threshold.
    Returns a DataFrame with one row per anomaly.
    """
    rows = []
    for city in sorted(future_preds["CITY"].unique()):
        fp = future_preds[future_preds["CITY"] == city].sort_values("DATE")
        for target in TARGETS:
            thr = thresholds[city][target]
            anomaly_days = fp[fp[target] > thr]
            for _, day in anomaly_days.iterrows():
                rows.append({
                    "City":      city,
                    "Date":      day["DATE"].date(),
                    "Variable":  TARGET_LABELS[target],
                    "Forecast":  round(day[target], 2),
                    "Threshold": round(thr, 2),
                    "Above by":  round(day[target] - thr, 2),
                })
    return pd.DataFrame(rows)


def print_anomaly_report(anomalies: pd.DataFrame):
    """Print a formatted anomaly table."""
    if anomalies.empty:
        print("\n  No anomalies detected in the 30-day forecast.")
        return

    print(f"\n── Anomaly Report (forecast values above {ANOMALY_PERCENTILE}th percentile) "
          + "─" * 20)
    print(f"  {'City':<12} {'Date':<12} {'Variable':<22} "
          f"{'Forecast':>10} {'Threshold':>10} {'Above by':>9}")
    print("  " + "─" * 80)

    for city in sorted(anomalies["City"].unique()):
        city_rows = anomalies[anomalies["City"] == city]
        for _, row in city_rows.iterrows():
            print(f"  {row['City']:<12} {str(row['Date']):<12} {row['Variable']:<22} "
                  f"{row['Forecast']:>10.2f} {row['Threshold']:>10.2f} {row['Above by']:>9.2f}")
        print()

    print(f"  Total anomalies: {len(anomalies)}  across {anomalies['City'].nunique()} "
          f"{'city' if anomalies['City'].nunique() == 1 else 'cities'}")


# ── Plots ─────────────────────────────────────────────────────────────────────

COLORS = {
    "actual":    "#1565C0",   # dark blue
    "test_pred": "#E53935",   # red
    "forecast":  "#2E7D32",   # dark green
    "band":      "#A5D6A7",   # light green for uncertainty band
    "threshold": "#FF6F00",   # orange for threshold line
    "anomaly":   "#FF6F00",   # orange dots for anomaly forecast days
}


def plot_city(actuals: pd.DataFrame, test_preds: pd.DataFrame,
              future_preds: pd.DataFrame, city: str, out_dir: str,
              thresholds: dict = None):
    """
    3-panel figure for one city.
    If thresholds provided, draws a dashed orange threshold line on each panel
    and marks forecast days that exceed the threshold with orange dots.
    """
    real   = actuals[actuals["CITY"] == city].sort_values("DATE")
    tp     = test_preds[test_preds["CITY"] == city].sort_values("DATE")
    fp     = future_preds[future_preds["CITY"] == city].sort_values("DATE")

    test_start   = tp["DATE"].min()
    test_end     = tp["DATE"].max()
    future_start = fp["DATE"].min()
    future_end   = fp["DATE"].max()

    context   = real[real["DATE"] < test_start].tail(CONTEXT_DAYS)
    test_real = real[(real["DATE"] >= test_start) & (real["DATE"] <= test_end)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle(f"Prophet Forecast — {city}", fontsize=14, fontweight="bold", y=0.99)

    for ax, target in zip(axes, TARGETS):
        lower_col = f"{target}_lower"
        upper_col = f"{target}_upper"

        # Historical + test actual
        ax.plot(context["DATE"], context[target],
                color=COLORS["actual"], linewidth=1.6, label="Actual")
        ax.plot(test_real["DATE"], test_real[target],
                color=COLORS["actual"], linewidth=1.6)

        # Test prediction
        ax.plot(tp["DATE"], tp[target],
                color=COLORS["test_pred"], linewidth=1.6,
                linestyle="--", label="Prophet (test)")

        # Future forecast + uncertainty band
        ax.fill_between(fp["DATE"],
                        fp[lower_col], fp[upper_col],
                        color=COLORS["band"], alpha=0.5, label="80% interval")
        ax.plot(fp["DATE"], fp[target],
                color=COLORS["forecast"], linewidth=2,
                linestyle="--", label="Prophet (forecast)")

        # Threshold line + anomaly markers
        if thresholds and city in thresholds:
            thr = thresholds[city][target]
            ax.axhline(thr, color=COLORS["threshold"], linewidth=1.2,
                       linestyle=":", label=f"90th pctile ({thr:.1f})")
            anomaly_fp = fp[fp[target] > thr]
            if not anomaly_fp.empty:
                ax.scatter(anomaly_fp["DATE"], anomaly_fp[target],
                           color=COLORS["anomaly"], zorder=5,
                           s=40, label="Anomaly", edgecolors="white", linewidths=0.5)

        # Shaded windows
        ax.axvspan(test_start,   test_end,   alpha=0.06, color=COLORS["test_pred"])
        ax.axvspan(future_start, future_end, alpha=0.06, color=COLORS["forecast"])
        ax.axvline(test_start,   color="grey", linewidth=0.7, linestyle=":")
        ax.axvline(future_start, color="grey", linewidth=0.7, linestyle=":")

        ax.set_ylabel(TARGET_LABELS[target], fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(loc="upper left", fontsize=8, ncol=3)

    ax = axes[-1]
    ylim = ax.get_ylim()
    mid_test   = test_start   + (test_end   - test_start)   / 2
    mid_future = future_start + (future_end - future_start) / 2
    ax.text(mid_test,   ylim[0], "← evaluation →",
            ha="center", fontsize=7, color=COLORS["test_pred"], alpha=0.8)
    ax.text(mid_future, ylim[0], "← forecast →",
            ha="center", fontsize=7, color=COLORS["forecast"], alpha=0.8)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{city.lower().replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_metrics_summary(metrics: pd.DataFrame, out_dir: str):
    """Bar chart of MAE per city, one subplot per target."""
    cities = sorted(metrics["City"].unique())
    x      = np.arange(len(cities))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Prophet — Test Set MAE by City", fontsize=13, fontweight="bold")

    for ax, target in zip(axes, TARGETS):
        maes = [
            metrics[(metrics["City"] == c) & (metrics["Target"] == target)]["MAE"].values[0]
            for c in cities
        ]
        bars = ax.bar(x, maes, color=COLORS["test_pred"], alpha=0.8, edgecolor="white")

        for bar, val in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(TARGET_LABELS[target], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(cities, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("MAE")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, "metrics_summary.png")
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

    cities       = sorted(df["CITY"].unique())
    test_parts   = []
    future_parts = []

    for city in cities:
        print(f"Training Prophet — {city}...")
        city_df = df[df["CITY"] == city].sort_values("DATE").reset_index(drop=True)
        test_p, future_p = run_city(city_df, city)
        test_parts.append(test_p)
        future_parts.append(future_p)

    test_preds   = pd.concat(test_parts,   ignore_index=True)
    future_preds = pd.concat(future_parts, ignore_index=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(test_preds, df)
    print_metrics(metrics)

    # ── Forecast summary ──────────────────────────────────────────────────────
    future_start = future_preds["DATE"].min().date()
    future_end   = future_preds["DATE"].max().date()
    print(f"\n── 30-Day Forecast Summary ({future_start} → {future_end}) " + "─" * 20)
    summary = future_preds.groupby("CITY")[TARGETS].mean().round(2)
    summary.columns = ["Avg Temp (°C)", "Avg Precip (mm)", "Avg Wind (km/h)"]
    print(summary.to_string())

    # ── Anomaly Detection ─────────────────────────────────────────────────────
    print(f"\nComputing {ANOMALY_PERCENTILE}th percentile thresholds from historical data...")
    thresholds = compute_thresholds(df)

    print(f"\n── {ANOMALY_PERCENTILE}th Percentile Thresholds per City ──────────────────────────────")
    print(f"  {'City':<12} {'Temp (°C)':>10} {'Precip (mm)':>12} {'Wind (km/h)':>12}")
    print("  " + "─" * 50)
    for city in cities:
        t = thresholds[city]
        print(f"  {city:<12} {t['TEMPERATURE_C']:>10.2f} "
              f"{t['PRECIPITATION_MM']:>12.2f} {t['WIND_SPEED_KMH']:>12.2f}")

    anomalies = detect_anomalies(future_preds, thresholds)
    print_anomaly_report(anomalies)

    # ── Plots — only cities with anomalies ────────────────────────────────────
    out_dir = "plots/prophet"
    anomaly_cities = sorted(anomalies["City"].unique()) if not anomalies.empty else []

    if anomaly_cities:
        print(f"\nGenerating plots for {len(anomaly_cities)} cities with anomalies...")
        for city in anomaly_cities:
            plot_city(df, test_preds, future_preds, city, out_dir, thresholds=thresholds)
    else:
        print("\nNo anomalies detected — plotting all cities for reference...")
        for city in cities:
            plot_city(df, test_preds, future_preds, city, out_dir, thresholds=thresholds)

    print("Generating metrics summary chart...")
    plot_metrics_summary(metrics, out_dir)

    print(f"\nDone. Plots saved to ./{out_dir}/")


if __name__ == "__main__":
    main()
