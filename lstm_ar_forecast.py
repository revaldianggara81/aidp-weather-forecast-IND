"""
lstm_ar_forecast.py — Autoregressive LSTM: multi-variate vs uni-variate comparison.

Two approaches, same architecture (2-layer LSTM, hidden=128), same lookback (90d):

  A) Multi-variate global  (1 model)
     Input per step : 3 targets + 6 city one-hot  (9 features)
     Output         : next day's 3 targets
     AR rollout     : all 3 targets predicted together at each step

  B) Uni-variate per city × target  (18 models = 6 cities × 3 targets)
     Input per step : 1 target's own history  (1 feature)
     Output         : next day's value of that target
     AR rollout     : each target forecasted independently

Both run autoregressively for 30 steps with no ground-truth leakage.
Plots saved to plots/lstm_ar/ — both approaches on the same axes.
"""

import os
import random
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import oracledb
import pandas as pd
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset, random_split

warnings.filterwarnings("ignore")
load_dotenv()

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

INPUT_DAYS  = 90
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
BATCH_SIZE  = 32
EPOCHS      = 150
LR          = 0.001
ES_PATIENCE = 20

ANOMALY_PERCENTILE = 95

COLORS = {
    "actual":      "#1565C0",   # dark blue
    "mv_test":     "#E53935",   # red
    "mv_forecast": "#2E7D32",   # dark green
    "uv_test":     "#6A1B9A",   # purple
    "uv_forecast": "#E65100",   # deep orange
    "threshold":   "#78909C",   # grey
}


# ── Data loading ──────────────────────────────────────────────────────────────

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


# ── Array helpers ─────────────────────────────────────────────────────────────

def _pivot(df):
    """Return (T, n_cities, 3), sorted city list, DatetimeIndex."""
    cities = sorted(df["CITY"].unique())
    frames = [df.pivot(index="DATE", columns="CITY", values=t)[cities].values
              for t in TARGETS]
    data  = np.stack(frames, axis=-1)           # (T, n_cities, 3)
    dates = df.pivot(index="DATE", columns="CITY", values=TARGETS[0]).index
    return data, cities, dates


def _global_zscore(data):
    """Per-target Z-score across all cities × time.  data: (T, n_cities, 3)."""
    flat = data.reshape(-1, 3)
    mean = np.nanmean(flat, axis=0)   # (3,)
    std  = np.nanstd( flat, axis=0)   # (3,)
    return (data - mean) / (std + 1e-8), mean, std


def _city_zscore_1d(series):
    """Z-score a single 1-D city/target series.  Returns (normed, mean, std)."""
    m = series.mean()
    s = series.std() + 1e-8
    return (series - m) / s, m, s


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMAr(nn.Module):
    """1-step-ahead LSTM.  Input (batch, seq, in_dim) → output (batch, out_dim)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, HIDDEN_SIZE, NUM_LAYERS, batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.fc = nn.Linear(HIDDEN_SIZE, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Dataset ───────────────────────────────────────────────────────────────────

class SampleDataset(Dataset):
    """Wraps a pre-built list of (x_array, y_array) pairs."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ── Sample builders ───────────────────────────────────────────────────────────

def _mv_samples(data_norm, city_onehot):
    """
    Build multi-variate training samples from all cities.
    x : (INPUT_DAYS, 9)  — 3 normalised targets + 6 city one-hot
    y : (3,)             — next day's 3 targets ONLY  (no one-hot in output)
    """
    T, n_cities, _ = data_norm.shape
    samples = []
    for city_idx in range(n_cities):
        targets  = data_norm[:, city_idx, :]                          # (T, 3)
        oh_col   = np.tile(city_onehot[city_idx], (T, 1))            # (T, 6)
        features = np.concatenate([targets, oh_col], axis=1)         # (T, 9)
        for i in range(T - INPUT_DAYS - 1):
            x = features[i : i + INPUT_DAYS].astype(np.float32)      # (90, 9)
            y = targets[i + INPUT_DAYS].astype(np.float32)           # (3,)
            samples.append((x, y))
    return samples


def _uv_samples(series_norm):
    """
    Build uni-variate training samples from a 1-D normalised series.
    x : (INPUT_DAYS, 1)
    y : (1,)
    """
    T = len(series_norm)
    samples = []
    for i in range(T - INPUT_DAYS - 1):
        x = series_norm[i : i + INPUT_DAYS].reshape(-1, 1).astype(np.float32)
        y = series_norm[i + INPUT_DAYS : i + INPUT_DAYS + 1].astype(np.float32)
        samples.append((x, y))
    return samples


# ── Generic training loop ─────────────────────────────────────────────────────

def _train(samples, in_dim, out_dim, device, label=""):
    """Train one LSTMAr model on a list of (x, y) samples."""
    random.seed(42)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * 0.15))
    tr_ds  = SampleDataset(samples[n_val:])
    val_ds = SampleDataset(samples[:n_val])

    tr_ld  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model   = LSTMAr(in_dim, out_dim).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=7)
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x_b, y_b in tr_ld:
            x_b, y_b = x_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss_fn(model(x_b), y_b).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vl = (sum(loss_fn(model(x.to(device)), y.to(device)).item()
                  for x, y in val_ld) / len(val_ld))
        sched.step(vl)

        if epoch % 50 == 0 and label:
            print(f"      [{label}] epoch {epoch}/{EPOCHS}  "
                  f"val={vl:.4f}  lr={opt.param_groups[0]['lr']:.1e}")

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= ES_PATIENCE:
                if label:
                    print(f"      [{label}] early stop at epoch {epoch}  "
                          f"best val={best_val:.4f}")
                break

    model.load_state_dict(best_state)
    model.to(device).eval()
    return model


# ── Noise estimation from training residuals ──────────────────────────────────

def _noise_std_mv(model, data_norm, city_onehot, device, max_samples=2000):
    """
    Estimate per-target residual std in normalised space from 1-step-ahead
    predictions on the training data.  Returns array (3,).
    """
    samples = _mv_samples(data_norm, city_onehot)
    if len(samples) > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in idx]

    xs = np.stack([s[0] for s in samples])   # (N, INPUT_DAYS, 9)
    ys = np.stack([s[1] for s in samples])   # (N, 3)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(xs), 256):
            preds.append(
                model(torch.FloatTensor(xs[i:i+256]).to(device)).cpu().numpy()
            )
    preds = np.concatenate(preds)            # (N, 3)
    return (ys - preds).std(axis=0)          # (3,)


def _noise_std_uv(model, series_norm, device, max_samples=2000):
    """
    Estimate residual std in normalised space for a single 1-D series.
    Returns scalar float.
    """
    samples = _uv_samples(series_norm)
    if len(samples) > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in idx]

    xs = np.stack([s[0] for s in samples])   # (N, INPUT_DAYS, 1)
    ys = np.stack([s[1] for s in samples])   # (N, 1)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(xs), 256):
            preds.append(
                model(torch.FloatTensor(xs[i:i+256]).to(device)).cpu().numpy()
            )
    preds = np.concatenate(preds)
    return float((ys - preds)[:, 0].std())


# ── Autoregressive rollout ────────────────────────────────────────────────────

def _rollout_mv(model, seed_targets_norm, city_oh, n_steps, device,
                noise_std=None, rng_seed=0):
    """
    Multi-variate AR rollout.
    seed_targets_norm : (INPUT_DAYS, 3) normalised
    city_oh           : (6,) city one-hot
    noise_std         : (3,) per-target std in normalised space, or None
    Returns           : (n_steps, 3) normalised predictions
    """
    window = seed_targets_norm.copy()    # (90, 3)
    preds  = []
    rng    = np.random.default_rng(rng_seed)
    for _ in range(n_steps):
        oh_tiled = np.tile(city_oh, (INPUT_DAYS, 1))                  # (90, 6)
        x = np.concatenate([window, oh_tiled], axis=1).astype(np.float32)
        with torch.no_grad():
            p = model(torch.FloatTensor(x).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        if noise_std is not None:
            p = p + rng.normal(0, noise_std).astype(np.float32)
        preds.append(p)                                               # (3,)
        window = np.vstack([window[1:], p.reshape(1, 3)])
    return np.array(preds)               # (n_steps, 3)


def _rollout_uv(model, seed_norm, n_steps, device,
                noise_std=None, rng_seed=0):
    """
    Uni-variate AR rollout.
    seed_norm : (INPUT_DAYS,) normalised
    noise_std : scalar std in normalised space, or None
    Returns   : (n_steps,) normalised predictions
    """
    window = seed_norm.reshape(-1, 1).copy()   # (90, 1)
    preds  = []
    rng    = np.random.default_rng(rng_seed)
    for _ in range(n_steps):
        x = torch.FloatTensor(window).unsqueeze(0).to(device)
        with torch.no_grad():
            p = model(x).squeeze(0).cpu().numpy()   # (1,)
        if noise_std is not None:
            p = p + rng.normal(0, noise_std, size=p.shape).astype(np.float32)
        preds.append(p[0])
        window = np.vstack([window[1:], p.reshape(1, 1)])
    return np.array(preds)                          # (n_steps,)


# ══════════════════════════════════════════════════════════════════════════════
# Approach A — Multi-variate global model
# ══════════════════════════════════════════════════════════════════════════════

def run_multivariate(train_df, full_df, cities, city_onehot, device):
    n_cities = len(cities)

    # ── Train/test split ──────────────────────────────────────────────────────
    train_data, _, _ = _pivot(train_df)
    train_norm, mean_tr, std_tr = _global_zscore(train_data)

    print(f"  Building dataset and training (eval split)…")
    model_eval = _train(_mv_samples(train_norm, city_onehot),
                        in_dim=9, out_dim=3, device=device, label="MV eval")

    # Test rollout — no noise (clean MAE comparison against actuals)
    test_rows = []
    for city_idx, city in enumerate(cities):
        seed = train_norm[-INPUT_DAYS:, city_idx, :]                  # (90, 3)
        preds_norm = _rollout_mv(model_eval, seed, city_onehot[city_idx],
                                 TEST_DAYS, device)                   # (30, 3)
        for step, p in enumerate(preds_norm):
            raw = p * (std_tr + 1e-8) + mean_tr
            raw[TARGETS.index("PRECIPITATION_MM")] = max(0.0, raw[TARGETS.index("PRECIPITATION_MM")])
            test_rows.append({"step": step, "CITY": city,
                               "TEMPERATURE_C": raw[0],
                               "PRECIPITATION_MM": raw[1],
                               "WIND_SPEED_KMH": raw[2]})
    test_preds = pd.DataFrame(test_rows)

    # ── Full retrain ──────────────────────────────────────────────────────────
    full_data, _, _ = _pivot(full_df)
    full_norm, mean_full, std_full = _global_zscore(full_data)

    print(f"  Retraining on full data…")
    model_full = _train(_mv_samples(full_norm, city_onehot),
                        in_dim=9, out_dim=3, device=device, label="MV full")

    # Estimate residual noise from full training data
    ns = _noise_std_mv(model_full, full_norm, city_onehot, device)
    print(f"  MV noise std (normalised): temp={ns[0]:.3f}  "
          f"precip={ns[1]:.3f}  wind={ns[2]:.3f}")

    # Future rollout — with noise to restore realistic dynamics
    future_rows = []
    for city_idx, city in enumerate(cities):
        seed = full_norm[-INPUT_DAYS:, city_idx, :]
        preds_norm = _rollout_mv(model_full, seed, city_onehot[city_idx],
                                 FORECAST_DAYS, device,
                                 noise_std=ns, rng_seed=city_idx)
        for step, p in enumerate(preds_norm):
            raw = p * (std_full + 1e-8) + mean_full
            raw[TARGETS.index("PRECIPITATION_MM")] = max(0.0, raw[TARGETS.index("PRECIPITATION_MM")])
            future_rows.append({"step": step, "CITY": city,
                                 "TEMPERATURE_C": raw[0],
                                 "PRECIPITATION_MM": raw[1],
                                 "WIND_SPEED_KMH": raw[2]})
    future_preds = pd.DataFrame(future_rows)
    return test_preds, future_preds


# ══════════════════════════════════════════════════════════════════════════════
# Approach B — Uni-variate per city × target  (18 models)
# ══════════════════════════════════════════════════════════════════════════════

def run_univariate(train_df, full_df, cities, device):
    test_store   = {c: {} for c in cities}
    future_store = {c: {} for c in cities}

    n_total = len(cities) * len(TARGETS)
    done = 0

    for city in cities:
        tr_city = train_df[train_df["CITY"] == city].sort_values("DATE")
        fu_city = full_df[full_df["CITY"]  == city].sort_values("DATE")

        for target in TARGETS:
            done += 1
            label = f"{city}/{target[:4]}  ({done}/{n_total})"

            # ── Eval model ────────────────────────────────────────────────────
            tr_series = tr_city[target].values.astype(np.float64)
            tr_norm, m_tr, s_tr = _city_zscore_1d(tr_series)

            model_eval = _train(_uv_samples(tr_norm), in_dim=1, out_dim=1,
                                device=device, label=label)

            # Test rollout — no noise
            preds_norm = _rollout_uv(model_eval, tr_norm[-INPUT_DAYS:],
                                     TEST_DAYS, device)
            preds = preds_norm * s_tr + m_tr
            if target == "PRECIPITATION_MM":
                preds = np.maximum(0.0, preds)
            test_store[city][target] = preds.tolist()

            # ── Full model ────────────────────────────────────────────────────
            fu_series = fu_city[target].values.astype(np.float64)
            fu_norm, m_fu, s_fu = _city_zscore_1d(fu_series)

            model_full = _train(_uv_samples(fu_norm), in_dim=1, out_dim=1,
                                device=device, label=label + " full")

            # Estimate noise from full-data residuals, apply to future rollout
            ns = _noise_std_uv(model_full, fu_norm, device)
            rng_seed = done  # unique per (city, target) pair

            preds_norm_f = _rollout_uv(model_full, fu_norm[-INPUT_DAYS:],
                                       FORECAST_DAYS, device,
                                       noise_std=ns, rng_seed=rng_seed)
            preds_f = preds_norm_f * s_fu + m_fu
            if target == "PRECIPITATION_MM":
                preds_f = np.maximum(0.0, preds_f)
            future_store[city][target] = preds_f.tolist()

    def _assemble(store, n_days):
        recs = []
        for city in cities:
            for step in range(n_days):
                recs.append({"step": step, "CITY": city,
                             "TEMPERATURE_C":    store[city]["TEMPERATURE_C"][step],
                             "PRECIPITATION_MM": store[city]["PRECIPITATION_MM"][step],
                             "WIND_SPEED_KMH":   store[city]["WIND_SPEED_KMH"][step]})
        return pd.DataFrame(recs)

    return _assemble(test_store, TEST_DAYS), _assemble(future_store, FORECAST_DAYS)


# ── Attach real forecast dates to step-indexed DataFrames ─────────────────────

def _attach_dates(df, anchor_date, n_days):
    dates = [anchor_date + pd.Timedelta(days=s + 1) for s in range(n_days)]
    out = df.copy()
    out["DATE"] = out["step"].map(lambda s: dates[s])
    return out.drop(columns="step")


# ── Metrics ───────────────────────────────────────────────────────────────────

def build_mae_table(mv_test, uv_test, actuals, cities):
    rows = []
    for city in cities:
        actual = (actuals[(actuals["CITY"] == city) & actuals["DATE"].isin(mv_test["DATE"])]
                  .sort_values("DATE"))
        mv = mv_test[mv_test["CITY"] == city].sort_values("DATE")
        uv = uv_test[uv_test["CITY"] == city].sort_values("DATE")
        for t in TARGETS:
            n = min(len(actual), len(mv), len(uv))
            rows.append({
                "City": city, "Target": t,
                "MAE_MV":  mean_absolute_error(actual[t].values[:n], mv[t].values[:n]),
                "MAE_UV":  mean_absolute_error(actual[t].values[:n], uv[t].values[:n]),
                "RMSE_MV": np.sqrt(mean_squared_error(actual[t].values[:n], mv[t].values[:n])),
                "RMSE_UV": np.sqrt(mean_squared_error(actual[t].values[:n], uv[t].values[:n])),
            })
    return pd.DataFrame(rows)


def print_anomaly_report(future_preds, thresholds, label):
    rows = []
    for city in sorted(future_preds["CITY"].unique()):
        fp = future_preds[future_preds["CITY"] == city].sort_values("DATE")
        for target in TARGETS:
            thr = thresholds[city][target]
            for _, day in fp[fp[target] > thr].iterrows():
                rows.append({
                    "City":      city,
                    "Date":      day["DATE"].date(),
                    "Variable":  TARGET_LABELS[target],
                    "Forecast":  round(day[target], 2),
                    "Threshold": round(thr, 2),
                    "Above by":  round(day[target] - thr, 2),
                })
    anomalies = pd.DataFrame(rows)

    print(f"\n── Anomaly Report — {label} (>{ANOMALY_PERCENTILE}th percentile) " + "─" * 18)
    if anomalies.empty:
        print("  No anomalies detected.")
        return anomalies

    print(f"  {'City':<12} {'Date':<12} {'Variable':<22} "
          f"{'Forecast':>10} {'Threshold':>10} {'Above by':>9}")
    print("  " + "─" * 80)
    for city in sorted(anomalies["City"].unique()):
        for _, r in anomalies[anomalies["City"] == city].iterrows():
            print(f"  {r['City']:<12} {str(r['Date']):<12} {r['Variable']:<22} "
                  f"{r['Forecast']:>10.2f} {r['Threshold']:>10.2f} {r['Above by']:>9.2f}")
        print()
    print(f"  Total: {len(anomalies)} anomalies across "
          f"{anomalies['City'].nunique()} cities")
    return anomalies


def print_comparison(mae_table):
    prophet = {"TEMPERATURE_C": 0.69, "PRECIPITATION_MM": 7.65, "WIND_SPEED_KMH": 2.82}
    print("\n── Test MAE: Multi-variate vs Uni-variate " + "─" * 36)
    for target in TARGETS:
        sub = mae_table[mae_table["Target"] == target]
        print(f"\n  {TARGET_LABELS[target]}")
        print(f"  {'City':<12} {'Multi-var (MV)':>14} {'Uni-var (UV)':>13}  Winner")
        print("  " + "─" * 48)
        for _, row in sub.iterrows():
            winner = "MV" if row["MAE_MV"] <= row["MAE_UV"] else "UV"
            marker = " ←" if winner == "UV" else ""
            print(f"  {row['City']:<12} {row['MAE_MV']:>14.3f} {row['MAE_UV']:>13.3f}  "
                  f"{winner}{marker}")
        avg_mv = sub["MAE_MV"].mean()
        avg_uv = sub["MAE_UV"].mean()
        winner = "MV" if avg_mv <= avg_uv else "UV"
        print(f"  {'Average':<12} {avg_mv:>14.3f} {avg_uv:>13.3f}  {winner}  "
              f"(Prophet baseline: {prophet[target]:.3f})")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_city(actuals, mv_test, mv_future, uv_test, uv_future,
              city, out_dir, thresholds):
    real = actuals[actuals["CITY"] == city].sort_values("DATE")
    mt   = mv_test[mv_test["CITY"] == city].sort_values("DATE")
    mf   = mv_future[mv_future["CITY"] == city].sort_values("DATE")
    ut   = uv_test[uv_test["CITY"] == city].sort_values("DATE")
    uf   = uv_future[uv_future["CITY"] == city].sort_values("DATE")

    test_start   = mt["DATE"].min()
    test_end     = mt["DATE"].max()
    future_start = mf["DATE"].min()
    future_end   = mf["DATE"].max()

    context   = real[real["DATE"] < test_start].tail(CONTEXT_DAYS)
    test_real = real[(real["DATE"] >= test_start) & (real["DATE"] <= test_end)]

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f"AR-LSTM: Multi-variate vs Uni-variate — {city}",
                 fontsize=14, fontweight="bold", y=0.99)

    for ax, target in zip(axes, TARGETS):
        # Actual
        ax.plot(context["DATE"], context[target],
                color=COLORS["actual"], linewidth=1.6, label="Actual")
        ax.plot(test_real["DATE"], test_real[target],
                color=COLORS["actual"], linewidth=1.6)

        # Multi-variate
        ax.plot(mt["DATE"], mt[target], color=COLORS["mv_test"],
                linewidth=1.5, linestyle="--", label="MV test")
        ax.plot(mf["DATE"], mf[target], color=COLORS["mv_forecast"],
                linewidth=2.0, linestyle="--", label="MV forecast")

        # Uni-variate
        ax.plot(ut["DATE"], ut[target], color=COLORS["uv_test"],
                linewidth=1.5, linestyle="-.", label="UV test")
        ax.plot(uf["DATE"], uf[target], color=COLORS["uv_forecast"],
                linewidth=2.0, linestyle="-.", label="UV forecast")

        # Threshold + anomaly dots
        thr = thresholds[city][target]
        ax.axhline(thr, color=COLORS["threshold"], linewidth=1.0,
                   linestyle=":", label=f"90th pctile ({thr:.1f})")
        for fp_df, color in [(mf, COLORS["mv_forecast"]), (uf, COLORS["uv_forecast"])]:
            anom = fp_df[fp_df[target] > thr]
            if not anom.empty:
                ax.scatter(anom["DATE"], anom[target], color=color, zorder=5,
                           s=50, edgecolors="white", linewidths=0.7)

        # Shaded windows
        ax.axvspan(test_start,   test_end,   alpha=0.05, color=COLORS["mv_test"])
        ax.axvspan(future_start, future_end, alpha=0.05, color=COLORS["mv_forecast"])
        ax.axvline(test_start,   color="grey", linewidth=0.7, linestyle=":")
        ax.axvline(future_start, color="grey", linewidth=0.7, linestyle=":")

        ax.set_ylabel(TARGET_LABELS[target], fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(loc="upper left", fontsize=8, ncol=4)

    ax = axes[-1]
    ylim = ax.get_ylim()
    ax.text(test_start + (test_end - test_start) / 2, ylim[0],
            "← evaluation →", ha="center", fontsize=7,
            color=COLORS["mv_test"], alpha=0.8)
    ax.text(future_start + (future_end - future_start) / 2, ylim[0],
            "← forecast →", ha="center", fontsize=7,
            color=COLORS["mv_forecast"], alpha=0.8)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{city.lower().replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_mae_comparison(mae_table, out_dir):
    cities = sorted(mae_table["City"].unique())
    x      = np.arange(len(cities))
    w      = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("AR-LSTM Test MAE: Multi-variate vs Uni-variate",
                 fontsize=13, fontweight="bold")

    for ax, target in zip(axes, TARGETS):
        sub    = mae_table[mae_table["Target"] == target].set_index("City").loc[cities]
        mv_mae = sub["MAE_MV"].values
        uv_mae = sub["MAE_UV"].values

        b_mv = ax.bar(x - w/2, mv_mae, w, label="Multi-variate",
                      color=COLORS["mv_forecast"], alpha=0.85, edgecolor="white")
        b_uv = ax.bar(x + w/2, uv_mae, w, label="Uni-variate",
                      color=COLORS["uv_forecast"], alpha=0.85, edgecolor="white")

        for bars in (b_mv, b_uv):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{bar.get_height():.2f}",
                        ha="center", va="bottom", fontsize=7.5)

        ax.set_title(TARGET_LABELS[target], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(cities, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("MAE")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, "mae_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data from Oracle ADB...")
    df = load_data()
    print(f"  {len(df):,} rows | {df['CITY'].nunique()} cities | "
          f"{df['DATE'].min().date()} → {df['DATE'].max().date()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    cities      = sorted(df["CITY"].unique())
    city_onehot = np.eye(len(cities))
    last_date   = df["DATE"].max()
    test_cutoff = last_date - pd.Timedelta(days=TEST_DAYS)
    train_df    = df[df["DATE"] <= test_cutoff].copy()
    full_df     = df.copy()

    # ── Approach A ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("Approach A — Multi-variate global  (1 model)")
    print(f"  Input: 3 targets + {len(cities)} city one-hot → predicts 3 targets")
    print(f"{'═'*60}")
    mv_test_raw, mv_future_raw = run_multivariate(
        train_df, full_df, cities, city_onehot, device
    )
    mv_test   = _attach_dates(mv_test_raw,   train_df["DATE"].max(), TEST_DAYS)
    mv_future = _attach_dates(mv_future_raw, full_df["DATE"].max(),  FORECAST_DAYS)

    # ── Approach B ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("Approach B — Uni-variate per city × target  (18 models)")
    print("  Input: 1 target's own history → predicts that same target")
    print(f"{'═'*60}")
    uv_test_raw, uv_future_raw = run_univariate(
        train_df, full_df, cities, device
    )
    uv_test   = _attach_dates(uv_test_raw,   train_df["DATE"].max(), TEST_DAYS)
    uv_future = _attach_dates(uv_future_raw, full_df["DATE"].max(),  FORECAST_DAYS)

    # ── Metrics ───────────────────────────────────────────────────────────────
    mae_table = build_mae_table(mv_test, uv_test, df, cities)
    print_comparison(mae_table)

    # ── Forecast summary ──────────────────────────────────────────────────────
    fs, fe = mv_future["DATE"].min().date(), mv_future["DATE"].max().date()
    print(f"\n── 30-Day Forecast Summary ({fs} → {fe}) " + "─" * 20)
    for label, fp in [("Multi-variate", mv_future), ("Uni-variate", uv_future)]:
        print(f"\n  {label}:")
        s = fp.groupby("CITY")[TARGETS].mean().round(2)
        s.columns = ["Avg Temp (°C)", "Avg Precip (mm)", "Avg Wind (km/h)"]
        for line in s.to_string().split("\n"):
            print(f"    {line}")

    # ── Thresholds ────────────────────────────────────────────────────────────
    # Combined threshold across ALL cities per target variable
    combined_thr = {t: df[t].quantile(ANOMALY_PERCENTILE / 100) for t in TARGETS}
    thresholds   = {city: combined_thr for city in cities}

    print(f"\n── {ANOMALY_PERCENTILE}th Percentile Thresholds (combined across all cities) " + "─" * 14)
    print(f"  {'Variable':<25} {'Threshold':>10}")
    print("  " + "─" * 37)
    for t in TARGETS:
        print(f"  {TARGET_LABELS[t]:<25} {combined_thr[t]:>10.2f}")

    mv_anomalies = print_anomaly_report(mv_future, thresholds, "Multi-variate")
    uv_anomalies = print_anomaly_report(uv_future, thresholds, "Uni-variate")

    # ── Plots ─────────────────────────────────────────────────────────────────
    out_dir = "plots/lstm_ar"
    print(f"\nGenerating plots for all {len(cities)} cities...")
    for city in cities:
        plot_city(df, mv_test, mv_future, uv_test, uv_future,
                  city, out_dir, thresholds)

    print("Generating MAE comparison chart...")
    plot_mae_comparison(mae_table, out_dir)
    print(f"\nDone. Plots saved to ./{out_dir}/")


if __name__ == "__main__":
    main()
