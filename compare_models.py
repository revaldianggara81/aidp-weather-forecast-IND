"""
compare_models.py — 4-Model Weather Forecasting Comparison

Fetches 5 years of daily weather data from Open-Meteo for 36 Indian states and
union territories (same as the bronze layer), then trains and evaluates 4 models:

  1. MV AR-LSTM with Gaussian Noise  (current production)
  2. Temporal Fusion Transformer (TFT)
  3. SARIMAX
  4. Prophet

Evaluation: last 30 days held out.  Metrics: MAE, RMSE per city × target.
Plots saved to plots/comparison/.
"""

import os
import random
import time
import warnings
from datetime import date, datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

with open("medallion/configuration.yaml") as f:
    CFG = yaml.safe_load(f)

REGIONS       = CFG["regions"]
CITIES        = [r["name"] for r in REGIONS]
COORDS        = {r["name"]: (r["lat"], r["lon"]) for r in REGIONS}
TEST_DAYS     = CFG["forecasting"]["test_days"]       # 30
FORECAST_DAYS = CFG["forecasting"]["forecast_days"]   # 30
CONTEXT_DAYS  = CFG["forecasting"]["context_days"]    # 60

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]
TARGET_LABELS = {
    "TEMPERATURE_C":    "Temperature (°C)",
    "PRECIPITATION_MM": "Precipitation (mm)",
    "WIND_SPEED_KMH":   "Wind Speed (km/h)",
}

INPUT_DAYS  = 90
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
BATCH_SIZE  = 32
EPOCHS      = 150
LR          = 0.001
ES_PATIENCE = 20

OUT_DIR = "plots/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_COLORS = {
    "AR-LSTM":  "#E53935",   # red
    "TFT":      "#8E24AA",   # purple
    "SARIMAX":  "#FB8C00",   # orange
    "Prophet":  "#43A047",   # green
    "Actual":   "#1565C0",   # blue
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (same as bronze layer — Open-Meteo API)
# ══════════════════════════════════════════════════════════════════════════════

ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS    = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
YEARS         = 5

# Open-Meteo's free tier rate-limits aggressively (HTTP 429) if we issue one
# request per city. Batch multiple cities into a single request via
# comma-separated latitude/longitude instead.
CHUNK_SIZE = 10


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


def fetch_data() -> pd.DataFrame:
    """Fetch 5 years of daily weather from Open-Meteo for all cities (batched)."""
    end_date   = date.today() - timedelta(days=1)
    start_date = end_date.replace(year=end_date.year - YEARS)

    print(f"Fetching data: {start_date} → {end_date}  ({YEARS} years, {len(CITIES)} cities)")

    # Fetch weather, batched CHUNK_SIZE cities per request
    all_rows = []
    for i in range(0, len(REGIONS), CHUNK_SIZE):
        chunk = REGIONS[i:i + CHUNK_SIZE]
        lats = ",".join(str(r["lat"]) for r in chunk)
        lons = ",".join(str(r["lon"]) for r in chunk)

        resp = _get_with_retry(
            ARCHIVE_URL,
            params={
                "latitude": lats, "longitude": lons,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "daily": ",".join(DAILY_VARS),
                "timezone": "auto",
            },
        )
        payload = resp.json()
        results = payload if isinstance(payload, list) else [payload]

        if len(results) != len(chunk):
            raise RuntimeError(
                f"Open-Meteo returned {len(results)} result(s) for {len(chunk)} "
                "requested city/cities — refusing to zip by index, since a "
                "mismatch would misattribute weather data to the wrong city."
            )

        for region, data in zip(chunk, results):
            city = region["name"]
            daily = data["daily"]
            city_rows = []
            for t_str, te, pr, wi in zip(daily["time"], daily["temperature_2m_mean"],
                                         daily["precipitation_sum"], daily["wind_speed_10m_max"]):
                if te is None or wi is None:
                    continue
                pr = pr if pr is not None else 0.0
                city_rows.append((datetime.strptime(t_str, "%Y-%m-%d"), city,
                                  float(te), float(pr), float(wi)))
            all_rows.extend(city_rows)
            print(f"  {city:<42} {len(city_rows):>5} rows")

        if i + CHUNK_SIZE < len(REGIONS):
            time.sleep(2)   # be polite to the free API between chunks

    df = pd.DataFrame(all_rows, columns=[
        "DATE", "CITY", "TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH",
    ])
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["PRECIPITATION_MM"] = df["PRECIPITATION_MM"].clip(lower=0.0)
    df = df.drop_duplicates(subset=["CITY", "DATE"], keep="first")
    df = df.sort_values(["CITY", "DATE"]).reset_index(drop=True)

    print(f"\nTotal: {len(df):,} rows | {df['CITY'].nunique()} cities | "
          f"{df['DATE'].min().date()} → {df['DATE'].max().date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _pivot(df, cities):
    """Return (T, n_cities, 3), sorted city list, DatetimeIndex."""
    frames = [df.pivot(index="DATE", columns="CITY", values=t)[cities].values
              for t in TARGETS]
    data  = np.stack(frames, axis=-1)
    dates = df.pivot(index="DATE", columns="CITY", values=TARGETS[0]).index
    return data, dates


def _global_zscore(data):
    flat = data.reshape(-1, 3)
    mean = np.nanmean(flat, axis=0)
    std  = np.nanstd(flat, axis=0)
    return (data - mean) / (std + 1e-8), mean, std


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: MV AR-LSTM WITH GAUSSIAN NOISE
# ══════════════════════════════════════════════════════════════════════════════

class LSTMAr(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, HIDDEN_SIZE, NUM_LAYERS, batch_first=True,
                            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0)
        self.fc = nn.Linear(HIDDEN_SIZE, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class SampleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def _mv_samples(data_norm, city_onehot):
    T, nc, _ = data_norm.shape
    samples = []
    for ci in range(nc):
        tgt = data_norm[:, ci, :]
        oh  = np.tile(city_onehot[ci], (T, 1))
        feat = np.concatenate([tgt, oh], axis=1)
        for i in range(T - INPUT_DAYS - 1):
            samples.append((feat[i:i+INPUT_DAYS].astype(np.float32),
                            tgt[i+INPUT_DAYS].astype(np.float32)))
    return samples


def _train_lstm(samples, in_dim, out_dim, label=""):
    random.seed(42)
    random.shuffle(samples)
    n_val  = max(1, int(len(samples) * 0.15))
    tr_ld  = DataLoader(SampleDataset(samples[n_val:]),  batch_size=BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(SampleDataset(samples[:n_val]),  batch_size=BATCH_SIZE, shuffle=False)

    model   = LSTMAr(in_dim, out_dim).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=7)
    loss_fn = nn.HuberLoss(delta=1.0)
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x_b, y_b in tr_ld:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x_b), y_b).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vl = sum(loss_fn(model(x.to(DEVICE)), y.to(DEVICE)).item()
                 for x, y in val_ld) / len(val_ld)
        sched.step(vl)

        if epoch % 50 == 0 and label:
            print(f"      [{label}] epoch {epoch}/{EPOCHS}  "
                  f"val={vl:.4f}  lr={opt.param_groups[0]['lr']:.1e}")

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= ES_PATIENCE:
                if label:
                    print(f"      [{label}] early stop at epoch {epoch}  best val={best_val:.4f}")
                break

    model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    return model


def _rollout_mv(model, seed, city_oh, n_steps):
    window = seed.copy()
    preds = []
    for _ in range(n_steps):
        oh_tiled = np.tile(city_oh, (INPUT_DAYS, 1))
        x = np.concatenate([window, oh_tiled], axis=1).astype(np.float32)
        with torch.no_grad():
            p = model(torch.FloatTensor(x).unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()
        preds.append(p)
        window = np.vstack([window[1:], p.reshape(1, 3)])
    return np.array(preds)


def run_ar_lstm(train_df, test_actuals_3d, cities):
    """Train MV AR-LSTM, return test predictions as DataFrame rows."""
    print("\n  Training MV AR-LSTM...")
    n_cities = len(cities)
    city_onehot = np.eye(n_cities)
    in_dim = 3 + n_cities

    train_data, _ = _pivot(train_df, cities)
    train_norm, mean_tr, std_tr = _global_zscore(train_data)

    samples = _mv_samples(train_norm, city_onehot)
    model = _train_lstm(samples, in_dim=in_dim, out_dim=3, label="AR-LSTM")

    # Test rollout (no noise — clean MAE comparison)
    rows = []
    for ci, city in enumerate(cities):
        seed = train_norm[-INPUT_DAYS:, ci, :]
        preds_norm = _rollout_mv(model, seed, city_onehot[ci], TEST_DAYS)
        preds_raw = preds_norm * (std_tr + 1e-8) + mean_tr
        preds_raw[:, 1] = np.maximum(0.0, preds_raw[:, 1])  # clip precip
        for step in range(TEST_DAYS):
            rows.append({"step": step, "CITY": city,
                         "TEMPERATURE_C": preds_raw[step, 0],
                         "PRECIPITATION_MM": preds_raw[step, 1],
                         "WIND_SPEED_KMH": preds_raw[step, 2]})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: TFT (Temporal Fusion Transformer)
# ══════════════════════════════════════════════════════════════════════════════

def run_tft(train_df, test_dates, cities):
    """Train TFT via pytorch_forecasting, return test predictions."""
    # Use lightning.pytorch (not pytorch_lightning) for compatibility with
    # pytorch_forecasting which inherits from lightning.pytorch.LightningModule
    import lightning.pytorch as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import MAE

    print("\n  Training TFT...")

    tft_cfg = CFG.get("tft", {})
    max_encoder_length = tft_cfg.get("input_size", 60)
    max_prediction_length = TEST_DAYS

    # Prepare long-format DataFrame for pytorch_forecasting
    prep = train_df.copy()
    prep = prep.sort_values(["CITY", "DATE"]).reset_index(drop=True)

    # Add time_idx (integer time index per group)
    min_date = prep["DATE"].min()
    prep["time_idx"] = (prep["DATE"] - min_date).dt.days

    # Add calendar features as known reals
    prep["month"]       = prep["DATE"].dt.month.astype(float)
    prep["day_of_year"] = prep["DATE"].dt.dayofyear.astype(float)
    prep["sin_doy"]     = np.sin(2 * np.pi * prep["day_of_year"] / 365.25)
    prep["cos_doy"]     = np.cos(2 * np.pi * prep["day_of_year"] / 365.25)

    all_test_preds = []

    for target in TARGETS:
        print(f"    TFT — {TARGET_LABELS[target]}...")

        training = TimeSeriesDataSet(
            prep,
            time_idx="time_idx",
            target=target,
            group_ids=["CITY"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=["time_idx", "sin_doy", "cos_doy", "month"],
            time_varying_unknown_reals=[target],
            static_categoricals=["CITY"],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        train_dl = training.to_dataloader(train=True, batch_size=tft_cfg.get("batch_size", 32),
                                          num_workers=0)

        # Validation: last max_prediction_length days of training data
        val_data = TimeSeriesDataSet.from_dataset(training, prep, predict=True,
                                                  stop_randomization=True)
        val_dl = val_data.to_dataloader(train=False, batch_size=tft_cfg.get("batch_size", 32),
                                        num_workers=0)

        pl.seed_everything(tft_cfg.get("random_seed", 42))

        tft_model = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=tft_cfg.get("hidden_size", 64),
            attention_head_size=tft_cfg.get("n_head", 4),
            dropout=tft_cfg.get("dropout", 0.1),
            hidden_continuous_size=tft_cfg.get("hidden_size", 64) // 2,
            loss=MAE(),
            learning_rate=tft_cfg.get("learning_rate", 0.001),
            reduce_on_plateau_patience=5,
        )

        trainer = pl.Trainer(
            max_epochs=15,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            enable_progress_bar=True,
            enable_model_summary=False,
            gradient_clip_val=0.1,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            ],
        )

        trainer.fit(tft_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # Predict: extend data with future slots for test period
        test_start = train_df["DATE"].max() + pd.Timedelta(days=1)
        pred_dates = pd.date_range(start=test_start, periods=TEST_DAYS, freq="D")

        pred_rows = []
        for city in cities:
            for d in pred_dates:
                pred_rows.append({
                    "DATE": d, "CITY": city,
                    "TEMPERATURE_C": 0.0, "PRECIPITATION_MM": 0.0, "WIND_SPEED_KMH": 0.0,
                    "time_idx": (d - min_date).days,
                    "month": float(d.month),
                    "day_of_year": float(d.timetuple().tm_yday),
                    "sin_doy": np.sin(2 * np.pi * d.timetuple().tm_yday / 365.25),
                    "cos_doy": np.cos(2 * np.pi * d.timetuple().tm_yday / 365.25),
                })
        pred_ext = pd.concat([prep, pd.DataFrame(pred_rows)], ignore_index=True)
        pred_ext = pred_ext.sort_values(["CITY", "DATE"]).reset_index(drop=True)

        pred_ds = TimeSeriesDataSet.from_dataset(training, pred_ext, predict=True,
                                                 stop_randomization=True)
        pred_dl = pred_ds.to_dataloader(train=False, batch_size=128, num_workers=0)

        raw_preds = tft_model.predict(pred_dl, mode="raw")
        point_preds = raw_preds["prediction"][:, :, 0].cpu().numpy()  # (n_groups, horizon)

        # Map predictions back to city × step
        for ci, city in enumerate(sorted(cities)):
            for step in range(TEST_DAYS):
                val = float(point_preds[ci, step])
                if target == "PRECIPITATION_MM":
                    val = max(0.0, val)
                all_test_preds.append({
                    "step": step, "CITY": city, "target": target, "value": val,
                })

    # Pivot predictions into standard format
    pred_df = pd.DataFrame(all_test_preds)
    result_rows = []
    for city in cities:
        for step in range(TEST_DAYS):
            row = {"step": step, "CITY": city}
            for target in TARGETS:
                mask = (pred_df["CITY"] == city) & (pred_df["step"] == step) & (pred_df["target"] == target)
                row[target] = pred_df.loc[mask, "value"].values[0]
            result_rows.append(row)
    return pd.DataFrame(result_rows)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: SARIMAX
# ══════════════════════════════════════════════════════════════════════════════

def run_sarimax(train_df, cities):
    """
    Fit SARIMAX per city × target.
    Order: (2,1,1) with seasonal (1,0,1,7) for weekly seasonality.
    Annual seasonality via Fourier exogenous regressors.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    print("\n  Training SARIMAX...")
    order = (2, 1, 1)
    seasonal_order = (1, 0, 1, 7)

    n_total = len(cities) * len(TARGETS)
    done = 0
    rows = []

    for city in cities:
        city_df = train_df[train_df["CITY"] == city].sort_values("DATE").reset_index(drop=True)
        doy = city_df["DATE"].dt.dayofyear.values.astype(float)

        # Fourier terms for annual seasonality (2 harmonics)
        exog = np.column_stack([
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            np.sin(4 * np.pi * doy / 365.25),
            np.cos(4 * np.pi * doy / 365.25),
        ])

        # Future exog for TEST_DAYS
        last_date = city_df["DATE"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=TEST_DAYS, freq="D")
        future_doy = future_dates.dayofyear.values.astype(float)
        exog_future = np.column_stack([
            np.sin(2 * np.pi * future_doy / 365.25),
            np.cos(2 * np.pi * future_doy / 365.25),
            np.sin(4 * np.pi * future_doy / 365.25),
            np.cos(4 * np.pi * future_doy / 365.25),
        ])

        for target in TARGETS:
            done += 1
            print(f"    SARIMAX {done}/{n_total}: {city}/{target[:8]}...", end=" ", flush=True)

            y = city_df[target].values.astype(float)

            try:
                model = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=200)
                preds = result.forecast(steps=TEST_DAYS, exog=exog_future)
                if target == "PRECIPITATION_MM":
                    preds = np.maximum(0.0, preds)
                print(f"done")
            except Exception as e:
                print(f"FAILED ({e}), using naive forecast")
                # Fallback: repeat last 30 days
                preds = y[-TEST_DAYS:]

            for step in range(TEST_DAYS):
                rows.append({"step": step, "CITY": city,
                             target: float(preds[step])})

    # Merge target columns
    result = pd.DataFrame(rows)
    merged_rows = []
    for city in cities:
        for step in range(TEST_DAYS):
            row = {"step": step, "CITY": city}
            for target in TARGETS:
                mask = (result["CITY"] == city) & (result["step"] == step) & result[target].notna()
                row[target] = result.loc[mask, target].values[0]
            merged_rows.append(row)
    return pd.DataFrame(merged_rows)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4: PROPHET
# ══════════════════════════════════════════════════════════════════════════════

def run_prophet(train_df, cities):
    """Fit Prophet per city × target."""
    import logging
    from prophet import Prophet

    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    print("\n  Training Prophet...")
    prophet_cfg = CFG.get("prophet", {})
    n_total = len(cities) * len(TARGETS)
    done = 0
    rows = []

    for city in cities:
        city_df = train_df[train_df["CITY"] == city].sort_values("DATE").reset_index(drop=True)

        for target in TARGETS:
            done += 1
            print(f"    Prophet {done}/{n_total}: {city}/{target[:8]}...", end=" ", flush=True)

            df_p = city_df[["DATE", target]].rename(columns={"DATE": "ds", target: "y"})
            df_p["ds"] = pd.to_datetime(df_p["ds"])

            m = Prophet(
                seasonality_mode=prophet_cfg.get("seasonality_mode", "additive"),
                yearly_seasonality=prophet_cfg.get("yearly_seasonality", True),
                weekly_seasonality=prophet_cfg.get("weekly_seasonality", True),
                daily_seasonality=prophet_cfg.get("daily_seasonality", False),
                changepoint_prior_scale=float(prophet_cfg.get("changepoint_prior_scale", 0.05)),
            )
            m.fit(df_p)
            future = m.make_future_dataframe(periods=TEST_DAYS, freq="D")
            fc = m.predict(future)
            last_train = df_p["ds"].max()
            preds = fc[fc["ds"] > last_train]["yhat"].head(TEST_DAYS).values

            if target == "PRECIPITATION_MM":
                preds = np.maximum(0.0, preds)

            print(f"done")
            for step in range(TEST_DAYS):
                rows.append({"step": step, "CITY": city, target: float(preds[step])})

    # Merge target columns
    result = pd.DataFrame(rows)
    merged_rows = []
    for city in cities:
        for step in range(TEST_DAYS):
            row = {"step": step, "CITY": city}
            for target in TARGETS:
                mask = (result["CITY"] == city) & (result["step"] == step) & result[target].notna()
                row[target] = result.loc[mask, target].values[0]
            merged_rows.append(row)
    return pd.DataFrame(merged_rows)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(preds_dict, actuals_df, test_dates, cities):
    """
    preds_dict: {"model_name": DataFrame with step, CITY, targets}
    Returns DataFrame: City, Target, model_name_MAE, model_name_RMSE ...
    """
    rows = []
    for city in cities:
        actual = actuals_df[(actuals_df["CITY"] == city) &
                            (actuals_df["DATE"].isin(test_dates))].sort_values("DATE")
        for target in TARGETS:
            row = {"City": city, "Target": target}
            for model_name, pred_df in preds_dict.items():
                p = pred_df[pred_df["CITY"] == city].sort_values("step")
                n = min(len(actual), len(p))
                a_vals = actual[target].values[:n]
                p_vals = p[target].values[:n]
                row[f"{model_name}_MAE"]  = mean_absolute_error(a_vals, p_vals)
                row[f"{model_name}_RMSE"] = np.sqrt(mean_squared_error(a_vals, p_vals))
            rows.append(row)
    return pd.DataFrame(rows)


def print_metrics(metrics_df, model_names):
    """Print formatted comparison table."""
    print(f"\n{'═' * 90}")
    print("TEST SET METRICS — MAE (avg across all cities)")
    print(f"{'═' * 90}")

    print(f"\n  {'Target':<22}", end="")
    for m in model_names:
        print(f" {m:>12}", end="")
    print(f" {'Winner':>10}")
    print("  " + "─" * (22 + 12 * len(model_names) + 12))

    for target in TARGETS:
        sub = metrics_df[metrics_df["Target"] == target]
        print(f"  {TARGET_LABELS[target]:<22}", end="")
        maes = {}
        for m in model_names:
            avg_mae = sub[f"{m}_MAE"].mean()
            maes[m] = avg_mae
            print(f" {avg_mae:>12.3f}", end="")
        winner = min(maes, key=maes.get)
        print(f" {winner:>10}")

    # Per-city detail
    print(f"\n{'─' * 90}")
    print("PER-CITY BREAKDOWN")
    print(f"{'─' * 90}")

    for target in TARGETS:
        print(f"\n  {TARGET_LABELS[target]}:")
        sub = metrics_df[metrics_df["Target"] == target]
        print(f"  {'City':<14}", end="")
        for m in model_names:
            print(f" {m:>12}", end="")
        print(f" {'Winner':>10}")
        print("  " + "─" * (14 + 12 * len(model_names) + 12))

        for _, row in sub.iterrows():
            print(f"  {row['City']:<14}", end="")
            maes = {}
            for m in model_names:
                v = row[f"{m}_MAE"]
                maes[m] = v
                print(f" {v:>12.3f}", end="")
            winner = min(maes, key=maes.get)
            print(f" {winner:>10}")

        # Average row
        print(f"  {'Average':<14}", end="")
        for m in model_names:
            print(f" {sub[f'{m}_MAE'].mean():>12.3f}", end="")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_city_comparison(actuals_df, preds_dict, city, test_dates, out_dir):
    """3-panel figure: one subplot per target, all models overlaid."""
    real = actuals_df[actuals_df["CITY"] == city].sort_values("DATE")
    test_start = test_dates[0]
    test_end   = test_dates[-1]

    context = real[real["DATE"] < test_start].tail(CONTEXT_DAYS)
    test_real = real[(real["DATE"] >= test_start) & (real["DATE"] <= test_end)]

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f"4-Model Comparison — {city}", fontsize=14, fontweight="bold", y=0.99)

    for ax, target in zip(axes, TARGETS):
        # Actual (context + test)
        ax.plot(context["DATE"], context[target],
                color=MODEL_COLORS["Actual"], linewidth=1.5, label="Actual")
        ax.plot(test_real["DATE"], test_real[target],
                color=MODEL_COLORS["Actual"], linewidth=1.5)

        # Each model's test predictions
        for model_name, pred_df in preds_dict.items():
            p = pred_df[pred_df["CITY"] == city].sort_values("step")
            p_dates = test_dates[:len(p)]
            ax.plot(p_dates, p[target].values[:len(p_dates)],
                    color=MODEL_COLORS.get(model_name, "#999"),
                    linewidth=1.5, linestyle="--", label=model_name)

        ax.axvspan(test_start, test_end, alpha=0.06, color="#E53935")
        ax.axvline(test_start, color="grey", linewidth=0.7, linestyle=":")

        ax.set_ylabel(TARGET_LABELS[target], fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(loc="upper left", fontsize=8, ncol=5)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"{city.lower().replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_mae_bars(metrics_df, model_names, out_dir):
    """Grouped bar chart: MAE per city, one subplot per target."""
    cities = sorted(metrics_df["City"].unique())
    x = np.arange(len(cities))
    n_models = len(model_names)
    w = 0.8 / n_models

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Test MAE by City — 4-Model Comparison",
                 fontsize=14, fontweight="bold")

    for ax, target in zip(axes, TARGETS):
        sub = metrics_df[metrics_df["Target"] == target].set_index("City").loc[cities]
        for i, m in enumerate(model_names):
            vals = sub[f"{m}_MAE"].values
            bars = ax.bar(x + i * w - (n_models - 1) * w / 2, vals, w,
                          label=m, color=MODEL_COLORS.get(m, "#999"),
                          alpha=0.85, edgecolor="white")
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{bar.get_height():.2f}",
                        ha="center", va="bottom", fontsize=6)

        ax.set_title(TARGET_LABELS[target], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(cities, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("MAE")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, "mae_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_avg_mae_summary(metrics_df, model_names, out_dir):
    """Simple horizontal bar chart showing average MAE per model per target."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Average MAE Across All Cities — 4-Model Comparison",
                 fontsize=14, fontweight="bold")

    for ax, target in zip(axes, TARGETS):
        sub = metrics_df[metrics_df["Target"] == target]
        avg_maes = [sub[f"{m}_MAE"].mean() for m in model_names]
        colors = [MODEL_COLORS.get(m, "#999") for m in model_names]

        bars = ax.barh(model_names, avg_maes, color=colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, avg_maes):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", ha="left", va="center", fontsize=10, fontweight="bold")

        ax.set_title(TARGET_LABELS[target], fontsize=11)
        ax.set_xlabel("MAE")
        ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, "avg_mae_summary.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("4-MODEL WEATHER FORECASTING COMPARISON")
    print("=" * 70)

    df = fetch_data()
    cities = sorted(df["CITY"].unique())

    # ── Train/test split ──────────────────────────────────────────────────────
    last_date   = df["DATE"].max()
    test_cutoff = last_date - pd.Timedelta(days=TEST_DAYS)
    train_df    = df[df["DATE"] <= test_cutoff].copy()
    test_df     = df[df["DATE"] > test_cutoff].copy()
    test_dates  = sorted(test_df["DATE"].unique())

    print(f"\nTrain: {len(train_df):,} rows  |  "
          f"Test: {len(test_df):,} rows ({test_dates[0].date()} → {test_dates[-1].date()})")
    print(f"Device: {DEVICE}")

    # ── Get 3D test actuals ───────────────────────────────────────────────────
    test_3d, _ = _pivot(test_df, cities)

    # ── Run all 4 models ──────────────────────────────────────────────────────
    preds = {}
    model_names = ["AR-LSTM", "TFT", "SARIMAX", "Prophet"]

    # Model 1: AR-LSTM
    print(f"\n{'═' * 70}")
    print("MODEL 1: MV AR-LSTM WITH GAUSSIAN NOISE")
    print(f"{'═' * 70}")
    preds["AR-LSTM"] = run_ar_lstm(train_df, test_3d, cities)

    # Model 2: TFT
    print(f"\n{'═' * 70}")
    print("MODEL 2: TEMPORAL FUSION TRANSFORMER (TFT)")
    print(f"{'═' * 70}")
    try:
        preds["TFT"] = run_tft(train_df, test_dates, cities)
    except Exception as e:
        print(f"  TFT FAILED: {e}")
        print("  Skipping TFT...")
        model_names.remove("TFT")

    # Model 3: SARIMAX
    print(f"\n{'═' * 70}")
    print("MODEL 3: SARIMAX")
    print(f"{'═' * 70}")
    preds["SARIMAX"] = run_sarimax(train_df, cities)

    # Model 4: Prophet
    print(f"\n{'═' * 70}")
    print("MODEL 4: PROPHET")
    print(f"{'═' * 70}")
    preds["Prophet"] = run_prophet(train_df, cities)

    # ── Metrics ───────────────────────────────────────────────────────────────
    active_models = [m for m in model_names if m in preds]
    metrics_df = compute_metrics(preds, df, test_dates, cities)
    print_metrics(metrics_df, active_models)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\nGenerating plots to {OUT_DIR}/...")

    # Per-city time series plots
    for city in cities:
        plot_city_comparison(df, preds, city, test_dates, OUT_DIR)

    # Bar charts
    plot_mae_bars(metrics_df, active_models, OUT_DIR)
    plot_avg_mae_summary(metrics_df, active_models, OUT_DIR)

    print(f"\nDone. All plots saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
