"""
compare.py — Run all 4 forecasting models and generate comparison plots.

Models: LightGBM | Prophet | N-HiTS | LSTM
Output: plots/comparison/<city>_comparison.png
        plots/comparison/metrics_summary.png
"""

import os
import warnings
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error, mean_squared_error
import oracledb
from dotenv import load_dotenv

from models import prophet_model, lstm_model, tft_model

warnings.filterwarnings("ignore")
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]
TARGET_LABELS = {
    "TEMPERATURE_C":    "Temperature (°C)",
    "PRECIPITATION_MM": "Precipitation (mm)",
    "WIND_SPEED_KMH":   "Wind Speed (km/h)",
}
MODELS = {
    "Prophet": prophet_model,
    "LSTM":    lstm_model,
    "TFT":     tft_model,
}
MODEL_COLORS = {
    "Prophet": "#FB8C00",
    "LSTM":    "#8E24AA",
    "TFT":     "#00897B",
}


# ── Data ──────────────────────────────────────────────────────────────────────

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


def load_cfg():
    with open("configuration.yaml") as f:
        return yaml.safe_load(f)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(test_preds, actuals):
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


def print_metrics_table(all_metrics):
    print("\n── Test Set MAE Summary " + "─" * 56)
    header = f"{'Model':<12} {'City':<12} {'Temp':>8} {'Precip':>10} {'Wind':>8}"
    print(header)
    print("─" * 56)
    for model_name, metrics in all_metrics.items():
        for city in sorted(metrics["City"].unique()):
            m = metrics[metrics["City"] == city]
            vals = {t: m[m["Target"] == t]["MAE"].values[0] for t in TARGETS}
            print(
                f"{model_name:<12} {city:<12} "
                f"{vals['TEMPERATURE_C']:>8.2f} "
                f"{vals['PRECIPITATION_MM']:>10.2f} "
                f"{vals['WIND_SPEED_KMH']:>8.2f}"
            )
        print()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_city_comparison(actuals, results, context_days):
    """
    One figure per city. 3 rows (one per target).
    Solid black = actual. Dashed coloured = test-period predictions.
    Dotted coloured = future forecast.
    Grey shading marks the test and forecast windows.
    """
    os.makedirs("plots/comparison", exist_ok=True)
    cities = sorted(actuals["CITY"].unique())

    # Derive test and future date boundaries from any model's output
    first_test   = list(results.values())[0][0]
    first_future = list(results.values())[0][1]
    test_start   = first_test["DATE"].min()
    test_end     = first_test["DATE"].max()
    future_start = first_future["DATE"].min()
    future_end   = first_future["DATE"].max()

    for city in cities:
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle(f"Model Comparison — {city}", fontsize=14, fontweight="bold", y=0.99)

        real       = actuals[actuals["CITY"] == city].sort_values("DATE")
        context    = real[real["DATE"] < test_start].tail(context_days)
        test_real  = real[(real["DATE"] >= test_start) & (real["DATE"] <= test_end)]

        for ax, target in zip(axes, TARGETS):
            # Actual data (context + test period)
            ax.plot(
                context["DATE"], context[target],
                color="black", linewidth=1.6, label="Actual", zorder=5,
            )
            ax.plot(
                test_real["DATE"], test_real[target],
                color="black", linewidth=1.6, zorder=5,
            )

            # Each model — test predictions (dashed) and future forecast (dotted)
            for model_name, (test_preds, future_preds) in results.items():
                tp = test_preds[test_preds["CITY"] == city].sort_values("DATE")
                fp = future_preds[future_preds["CITY"] == city].sort_values("DATE")
                color = MODEL_COLORS[model_name]

                ax.plot(tp["DATE"], tp[target],
                        color=color, linewidth=1.5, linestyle="--",
                        label=model_name, alpha=0.9, zorder=4)
                ax.plot(fp["DATE"], fp[target],
                        color=color, linewidth=1.5, linestyle=":",
                        alpha=0.9, zorder=4)

            # Shade test and forecast windows
            ax.axvspan(test_start,   test_end,   alpha=0.06, color="#E53935", zorder=1)
            ax.axvspan(future_start, future_end, alpha=0.06, color="#2E7D32", zorder=1)
            ax.axvline(test_start,   color="gray", linewidth=0.7, linestyle=":", zorder=2)
            ax.axvline(future_start, color="gray", linewidth=0.7, linestyle=":", zorder=2)

            ax.set_ylabel(TARGET_LABELS[target], fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
            ax.grid(True, alpha=0.22, linestyle="--")
            ax.legend(loc="upper left", fontsize=8, ncol=3)

        # Annotate shaded regions in bottom axis
        ax = axes[-1]
        ylim = ax.get_ylim()
        mid_test   = test_start   + (test_end   - test_start)   / 2
        mid_future = future_start + (future_end - future_start) / 2
        ax.text(mid_test,   ylim[0], "← test →",     ha="center", fontsize=7, color="#E53935", alpha=0.8)
        ax.text(mid_future, ylim[0], "← forecast →", ha="center", fontsize=7, color="#2E7D32", alpha=0.8)

        # Shared legend for line styles
        legend_extra = [
            mpatches.Patch(color="none", label="-- test pred  ··· future fcst"),
        ]
        axes[0].legend(
            handles=axes[0].get_legend_handles_labels()[0] + legend_extra,
            labels=axes[0].get_legend_handles_labels()[1] + ["-- test  ··· forecast"],
            loc="upper left", fontsize=8, ncol=3,
        )

        plt.tight_layout()
        fname = f"plots/comparison/{city.lower().replace(' ', '_')}_comparison.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


def plot_metrics_summary(all_metrics):
    """Grouped bar chart: MAE per city, one subplot per target, one bar colour per model."""
    os.makedirs("plots/comparison", exist_ok=True)

    model_names = list(all_metrics.keys())
    cities      = sorted(list(all_metrics.values())[0]["City"].unique())
    x     = np.arange(len(cities))
    width = 0.72 / len(model_names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Comparison — Test Set MAE by City", fontsize=13, fontweight="bold")

    for ax, target in zip(axes, TARGETS):
        for i, model_name in enumerate(model_names):
            df   = all_metrics[model_name]
            maes = [
                df[(df["City"] == c) & (df["Target"] == target)]["MAE"].values[0]
                for c in cities
            ]
            offset = (i - len(model_names) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, maes, width,
                label=model_name,
                color=MODEL_COLORS[model_name],
                alpha=0.85, edgecolor="white", linewidth=0.5,
            )
            # Value labels on bars
            for bar, val in zip(bars, maes):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5,
                )

        ax.set_title(TARGET_LABELS[target], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(cities, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("MAE")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fname = "plots/comparison/metrics_summary.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_cfg()

    print("Loading data from Oracle ADB...")
    df = load_data()
    print(
        f"  {len(df):,} rows | {df['CITY'].nunique()} cities | "
        f"{df['DATE'].min().date()} → {df['DATE'].max().date()}"
    )

    test_cutoff = df["DATE"].max() - pd.Timedelta(days=cfg["forecasting"]["test_days"])
    train_df    = df[df["DATE"] <= test_cutoff].copy()
    print(f"  Train: up to {test_cutoff.date()} | Test: last {cfg['forecasting']['test_days']} days\n")

    results     = {}   # model_name -> (test_preds, future_preds)
    all_metrics = {}

    for model_name, module in MODELS.items():
        print(f"── Running {model_name} " + "─" * (50 - len(model_name)))
        try:
            test_preds, future_preds = module.run(df, train_df, cfg)
            results[model_name]     = (test_preds, future_preds)
            all_metrics[model_name] = compute_metrics(test_preds, df)
            print(f"  {model_name} done.\n")
        except Exception as e:
            print(f"  {model_name} FAILED: {e}\n")

    if not results:
        print("All models failed — nothing to plot.")
        return

    print_metrics_table(all_metrics)

    print("\nGenerating per-city comparison plots...")
    plot_city_comparison(df, results, cfg["forecasting"]["context_days"])

    print("Generating metrics summary bar chart...")
    plot_metrics_summary(all_metrics)

    print(f"\nDone. Plots saved to ./plots/comparison/")


if __name__ == "__main__":
    main()
