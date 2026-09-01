"""
India Weather Forecasting — LightGBM
Predicts Temperature (°C), Precipitation (mm), Wind Speed (km/h)
for the next 30 days per city using recursive multi-step forecasting.
"""

import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
import oracledb
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

with open("configuration.yaml") as f:
    _cfg = yaml.safe_load(f)

LAGS = _cfg["features"]["lags"]
WINDOWS = _cfg["features"]["rolling_windows"]
TEST_DAYS = _cfg["forecasting"]["test_days"]
FORECAST_DAYS = _cfg["forecasting"]["forecast_days"]
CONTEXT_DAYS = _cfg["forecasting"]["context_days"]
MODEL_PARAMS = _cfg["lgbm"]

# ── Constants ─────────────────────────────────────────────────────────────────

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]
TARGET_LABELS = {
    "TEMPERATURE_C": "Temperature (°C)",
    "PRECIPITATION_MM": "Precipitation (mm)",
    "WIND_SPEED_KMH": "Wind Speed (km/h)",
}


# ── 1. Load Data ──────────────────────────────────────────────────────────────

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


# ── 2. Feature Engineering ────────────────────────────────────────────────────

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["CITY", "DATE"]).reset_index(drop=True)

    # Calendar
    df["day_of_year"] = df["DATE"].dt.dayofyear
    df["month"] = df["DATE"].dt.month
    df["day_of_week"] = df["DATE"].dt.dayofweek

    # Fourier terms — capture annual seasonality
    doy = df["day_of_year"]
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    df["sin_doy2"] = np.sin(4 * np.pi * doy / 365.25)
    df["cos_doy2"] = np.cos(4 * np.pi * doy / 365.25)

    # City as integer category
    df["city_code"] = df["CITY"].astype("category").cat.codes

    # Per-city lag and rolling features (shift(1) prevents data leakage)
    for target in TARGETS:
        grp = df.groupby("CITY")[target]
        for lag in LAGS:
            df[f"{target}_lag{lag}"] = grp.shift(lag)
        for w in WINDOWS:
            df[f"{target}_rmean{w}"] = grp.transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            df[f"{target}_rstd{w}"] = grp.transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).std()
            )

    return df


def feature_cols() -> list[str]:
    calendar = [
        "day_of_year", "month", "day_of_week",
        "sin_doy", "cos_doy", "sin_doy2", "cos_doy2", "city_code",
    ]
    lags = [f"{t}_lag{l}" for t in TARGETS for l in LAGS]
    rolls = (
        [f"{t}_rmean{w}" for t in TARGETS for w in WINDOWS]
        + [f"{t}_rstd{w}" for t in TARGETS for w in WINDOWS]
    )
    return calendar + lags + rolls


# ── 3. Train ──────────────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame, feats: list[str]) -> dict:
    models = {}
    valid = df[feats].notna().all(axis=1)  # drop rows with incomplete lag history
    X = df.loc[valid, feats]
    for target in TARGETS:
        y = df.loc[valid, target]
        model = lgb.LGBMRegressor(**MODEL_PARAMS, verbose=-1)
        model.fit(X, y)
        models[target] = model
    return models


# ── 4. Recursive Forecast ─────────────────────────────────────────────────────

def recursive_forecast(
    models: dict,
    history: pd.DataFrame,
    feats: list[str],
    city_map: dict,
    n_days: int,
) -> pd.DataFrame:
    """
    Recursively forecast n_days ahead for every city.
    Each predicted day is appended to the history buffer so subsequent
    steps can use it as a lag feature.
    """
    buffer = history.copy()
    last_date = buffer["DATE"].max()
    cities = sorted(buffer["CITY"].unique())
    all_preds = []

    for step in range(1, n_days + 1):
        forecast_date = last_date + pd.Timedelta(days=step)
        new_rows = []

        for city in cities:
            city_vals = buffer[buffer["CITY"] == city].sort_values("DATE")

            doy = forecast_date.dayofyear
            row = {
                "DATE": forecast_date,
                "CITY": city,
                "city_code": city_map[city],
                "day_of_year": doy,
                "month": forecast_date.month,
                "day_of_week": forecast_date.dayofweek,
                "sin_doy": np.sin(2 * np.pi * doy / 365.25),
                "cos_doy": np.cos(2 * np.pi * doy / 365.25),
                "sin_doy2": np.sin(4 * np.pi * doy / 365.25),
                "cos_doy2": np.cos(4 * np.pi * doy / 365.25),
            }

            for target in TARGETS:
                vals = city_vals[target].values
                for lag in LAGS:
                    row[f"{target}_lag{lag}"] = vals[-lag] if len(vals) >= lag else np.nan
                for w in WINDOWS:
                    window_vals = vals[-w:] if len(vals) >= w else vals
                    row[f"{target}_rmean{w}"] = float(np.mean(window_vals))
                    row[f"{target}_rstd{w}"] = float(np.std(window_vals)) if len(window_vals) > 1 else 0.0

            # Predict each target using the row's features
            feat_row = pd.DataFrame([row])[feats]
            for target in TARGETS:
                pred = models[target].predict(feat_row)[0]
                # Precipitation cannot be negative
                if target == "PRECIPITATION_MM":
                    pred = max(0.0, pred)
                row[target] = pred

            new_rows.append(row)

        new_df = pd.DataFrame(new_rows)
        buffer = pd.concat([buffer, new_df[list(buffer.columns)]], ignore_index=True)
        all_preds.append(new_df[["DATE", "CITY"] + TARGETS])

    return pd.concat(all_preds, ignore_index=True)


# ── 5. Metrics ────────────────────────────────────────────────────────────────

def print_metrics(test_preds: pd.DataFrame, actuals: pd.DataFrame) -> None:
    print("\n── Test Set Metrics (last 30 days) " + "─" * 44)
    header = f"{'City':<12}"
    for t in TARGETS:
        short = t.split("_")[0][:5]
        header += f"  {short}_MAE  {short}_RMSE"
    print(header)
    print("─" * 80)

    for city in sorted(test_preds["CITY"].unique()):
        tp = test_preds[test_preds["CITY"] == city].sort_values("DATE").reset_index(drop=True)
        ar = (
            actuals[(actuals["CITY"] == city) & (actuals["DATE"].isin(tp["DATE"]))]
            .sort_values("DATE")
            .reset_index(drop=True)
        )
        line = f"{city:<12}"
        for target in TARGETS:
            mae = mean_absolute_error(ar[target], tp[target])
            rmse = np.sqrt(mean_squared_error(ar[target], tp[target]))
            line += f"  {mae:6.2f}  {rmse:6.2f}"
        print(line)


# ── 6. Plot ───────────────────────────────────────────────────────────────────

def plot_forecasts(
    actuals: pd.DataFrame,
    test_preds: pd.DataFrame,
    future_preds: pd.DataFrame,
) -> None:
    os.makedirs("plots", exist_ok=True)
    cities = sorted(actuals["CITY"].unique())

    colors = {"real": "#1565C0", "test_pred": "#E53935", "future": "#2E7D32"}

    for city in cities:
        fig, axes = plt.subplots(3, 1, figsize=(14, 11))
        fig.suptitle(f"Weather Forecast — {city}", fontsize=14, fontweight="bold", y=0.99)

        real = actuals[actuals["CITY"] == city].sort_values("DATE")
        tp = test_preds[test_preds["CITY"] == city].sort_values("DATE")
        fp = future_preds[future_preds["CITY"] == city].sort_values("DATE")

        test_start = tp["DATE"].min()
        context = real[real["DATE"] < test_start].tail(CONTEXT_DAYS)
        test_real = real[real["DATE"].isin(tp["DATE"])].sort_values("DATE")

        for ax, target in zip(axes, TARGETS):
            # Historical context
            ax.plot(
                context["DATE"], context[target],
                color=colors["real"], linewidth=1.5, label="Historical (actual)",
            )
            # Test period — actual
            ax.plot(
                test_real["DATE"], test_real[target],
                color=colors["real"], linewidth=1.5,
            )
            # Test period — predicted
            ax.plot(
                tp["DATE"], tp[target],
                color=colors["test_pred"], linewidth=1.5,
                linestyle="--", label="Test prediction",
            )
            # Future forecast
            ax.plot(
                fp["DATE"], fp[target],
                color=colors["future"], linewidth=2,
                linestyle="--", label="30-day forecast",
            )

            # Shaded regions
            if len(tp):
                ax.axvspan(tp["DATE"].min(), tp["DATE"].max(), alpha=0.07, color=colors["test_pred"])
                ax.axvline(tp["DATE"].min(), color=colors["test_pred"], linewidth=0.8, linestyle=":")
            if len(fp):
                ax.axvspan(fp["DATE"].min(), fp["DATE"].max(), alpha=0.07, color=colors["future"])
                ax.axvline(fp["DATE"].min(), color=colors["future"], linewidth=0.8, linestyle=":")

            ax.set_ylabel(TARGET_LABELS[target], fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.legend(loc="upper left", fontsize=8)

        # Annotate test / forecast windows in the bottom panel
        for ax in axes:
            if len(tp):
                ax.annotate(
                    "← test →",
                    xy=(tp["DATE"].min() + (tp["DATE"].max() - tp["DATE"].min()) / 2, ax.get_ylim()[0]),
                    fontsize=7, ha="center", color=colors["test_pred"], alpha=0.7,
                )
            if len(fp):
                ax.annotate(
                    "← forecast →",
                    xy=(fp["DATE"].min() + (fp["DATE"].max() - fp["DATE"].min()) / 2, ax.get_ylim()[0]),
                    fontsize=7, ha="center", color=colors["future"], alpha=0.7,
                )

        plt.tight_layout()
        fname = f"plots/{city.lower().replace(' ', '_')}_forecast.png"
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

    print("Engineering features...")
    df_feat = make_features(df)
    feats = feature_cols()

    city_map = {city: i for i, city in enumerate(sorted(df["CITY"].unique()))}
    df_feat["city_code"] = df_feat["CITY"].map(city_map)

    # Train / test split — hold out last 30 days
    last_date = df_feat["DATE"].max()
    test_cutoff = last_date - pd.Timedelta(days=TEST_DAYS)

    train_df = df_feat[df_feat["DATE"] <= test_cutoff]
    print(f"  Training up to {test_cutoff.date()} | Test period: last {TEST_DAYS} days")

    print("Training LightGBM models (Temperature, Precipitation, Wind Speed)...")
    models_eval = train_models(train_df, feats)

    print("Running recursive forecast over test period...")
    test_preds = recursive_forecast(models_eval, train_df, feats, city_map, TEST_DAYS)

    print_metrics(test_preds, df_feat)

    print("\nRetraining on full dataset for future forecast...")
    models_full = train_models(df_feat, feats)

    print(f"Forecasting next {FORECAST_DAYS} days from {last_date.date()}...")
    future_preds = recursive_forecast(models_full, df_feat, feats, city_map, FORECAST_DAYS)

    # ── Forecast Summary ──
    future_start = future_preds["DATE"].min().date()
    future_end = future_preds["DATE"].max().date()
    print(f"\n── 30-Day Forecast Summary ({future_start} → {future_end}) " + "─" * 25)
    summary = future_preds.groupby("CITY")[TARGETS].mean().round(2)
    summary.columns = ["Avg Temp (°C)", "Avg Precip (mm)", "Avg Wind (km/h)"]
    print(summary.to_string())

    print("\nGenerating plots...")
    plot_forecasts(df_feat, test_preds, future_preds)
    print(f"\nDone. Plots saved to ./plots/")


if __name__ == "__main__":
    main()
