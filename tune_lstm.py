"""
tune_lstm.py — Bayesian hyperparameter optimisation for the LSTM-Attn forecaster.

Data split (no leakage):
  Train : all data except the last 60 days  → used to fit the model
  Val   : days −60 to −30                   → Optuna objective (MAE)
  Test  : last 30 days                      → held out, never seen during tuning

Outputs:
  - Prints trial progress and best hyperparameters
  - Writes best_lstm_config.yaml
"""

import logging
import os
import warnings

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import oracledb
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml
from dotenv import load_dotenv

from models.lstm_attn_model import (
    WeatherLSTMAttn,
    SlidingWindowDataset,
    _pivot,
    _calendar_features,
    _normalize_per_city,
    _denormalize_per_city,
    TARGETS,
    N_TARGETS,
    N_CALENDAR,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

N_TRIALS     = 50
N_VAL        = 30    # validation period length (days −60 to −30)
N_TEST       = 30    # test period length       (held out)
MAX_EPOCHS   = 100   # per-trial epoch cap; early stopping typically cuts this short
ES_PATIENCE  = 10    # early-stop patience inside each trial


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


# ── Per-trial training (with Optuna pruning support) ─────────────────────────

def _train_trial(trial, train_norm, city_onehot, train_cal, input_len,
                 hidden_size, num_layers, dropout, lr, batch_size,
                 n_heads, attention_dropout, device):
    """Train LSTM-Attn for one Optuna trial; prunes early if val loss is poor."""
    n_cities  = len(city_onehot)
    feat_size = N_TARGETS + n_cities + N_CALENDAR

    full_ds = SlidingWindowDataset(
        train_norm, city_onehot, train_cal, input_len, N_VAL,
    )
    if len(full_ds) < 10:
        raise optuna.exceptions.TrialPruned()

    n_total    = len(full_ds)
    n_val_sp   = max(1, int(n_total * 0.15))
    n_train_sp = n_total - n_val_sp
    train_ds, val_ds = random_split(
        full_ds, [n_train_sp, n_val_sp],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = WeatherLSTMAttn(
        feat_size=feat_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        forecast_horizon=N_VAL,
        n_heads=n_heads,
        attention_dropout=attention_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val  = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                val_loss += loss_fn(model(x_b), y_b).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Report to Optuna for median pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val:
            best_val  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= ES_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model


# ── Val-period prediction ──────────────────────────────────────────────────────

def _predict_val(model, train_norm, city_onehot, train_cal, input_len,
                 mean_tr, std_tr, cities, device):
    """Predict N_VAL days from the end of train_norm for all cities."""
    rows = []
    for city_idx, city in enumerate(cities):
        window   = train_norm[-input_len:, city_idx, :]       # (input_len, 3)
        oh_tiled = np.tile(city_onehot[city_idx], (input_len, 1))
        cal      = train_cal[-input_len:]                     # (input_len, 3)
        x = np.concatenate([window, oh_tiled, cal], axis=1).astype(np.float32)
        x_t = torch.FloatTensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = model(x_t).squeeze(0).cpu().numpy()   # (N_VAL, 3)

        pred = _denormalize_per_city(pred_norm, mean_tr, std_tr, city_idx)
        pred[:, TARGETS.index("PRECIPITATION_MM")] = np.maximum(
            0.0, pred[:, TARGETS.index("PRECIPITATION_MM")]
        )
        for step in range(N_VAL):
            rows.append({
                "step":             step,
                "CITY":             city,
                "TEMPERATURE_C":    pred[step, 0],
                "PRECIPITATION_MM": pred[step, 1],
                "WIND_SPEED_KMH":   pred[step, 2],
            })
    return pd.DataFrame(rows)


# ── Objective factory ─────────────────────────────────────────────────────────

def make_objective(train_norm, train_cal, val_df, city_onehot,
                   mean_tr, std_tr, cities, device):

    def objective(trial):
        # ── Sample hyperparameters ─────────────────────────────────────────
        hidden_size       = trial.suggest_categorical("hidden_size", [64, 128, 256])
        num_layers        = trial.suggest_int("num_layers", 1, 3)
        dropout           = trial.suggest_float("dropout", 0.05, 0.4)
        lr                = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size        = trial.suggest_categorical("batch_size", [16, 32, 64])
        input_size        = trial.suggest_categorical("input_size", [30, 60, 90, 120])
        n_heads           = trial.suggest_categorical("n_heads", [2, 4])
        attention_dropout = trial.suggest_float("attention_dropout", 0.0, 0.3)

        # ── Train ──────────────────────────────────────────────────────────
        try:
            model = _train_trial(
                trial, train_norm, city_onehot, train_cal, input_size,
                hidden_size, num_layers, dropout, lr, batch_size,
                n_heads, attention_dropout, device,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            raise

        # ── Predict val period ─────────────────────────────────────────────
        try:
            preds_df = _predict_val(
                model, train_norm, city_onehot, train_cal, input_size,
                mean_tr, std_tr, cities, device,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            raise

        # ── Compute avg MAE across all cities × targets ────────────────────
        maes = []
        for city in cities:
            actual = (
                val_df[val_df["CITY"] == city]
                .sort_values("DATE")
                .reset_index(drop=True)
            )
            pred_c = (
                preds_df[preds_df["CITY"] == city]
                .sort_values("step")
                .reset_index(drop=True)
            )
            n = min(len(actual), len(pred_c), N_VAL)
            for t in TARGETS:
                mae = np.mean(np.abs(actual[t].values[:n] - pred_c[t].values[:n]))
                maes.append(mae)

        return float(np.mean(maes))

    return objective


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data from Oracle ADB...")
    df = load_data()
    print(
        f"  {len(df):,} rows | {df['CITY'].nunique()} cities | "
        f"{df['DATE'].min().date()} → {df['DATE'].max().date()}"
    )

    # ── Data splits ────────────────────────────────────────────────────────────
    max_date     = df["DATE"].max()
    val_end      = max_date - pd.Timedelta(days=N_TEST)      # last day of val
    train_cutoff = max_date - pd.Timedelta(days=N_TEST + N_VAL)  # last train day

    train_df = df[df["DATE"] <= train_cutoff].copy()
    val_df   = df[(df["DATE"] > train_cutoff) & (df["DATE"] <= val_end)].copy()

    print(f"\n  Train : {train_df['DATE'].min().date()} → {train_df['DATE'].max().date()} "
          f"({len(train_df)} rows)")
    print(f"  Val   : {val_df['DATE'].min().date()} → {val_df['DATE'].max().date()} "
          f"({len(val_df)} rows)")
    print(f"  Test  : last {N_TEST} days — held out, never used")

    # ── Prepare arrays ─────────────────────────────────────────────────────────
    train_data, cities, train_dates = _pivot(train_df)
    city_onehot = np.eye(len(cities))
    train_cal   = _calendar_features(train_dates)
    train_norm, mean_tr, std_tr = _normalize_per_city(train_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print(f"\nRunning {N_TRIALS} Optuna trials …\n")

    # ── Study ──────────────────────────────────────────────────────────────────
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )

    obj_fn = make_objective(
        train_norm, train_cal, val_df, city_onehot,
        mean_tr, std_tr, cities, device,
    )
    study.optimize(obj_fn, n_trials=N_TRIALS, show_progress_bar=True)

    # ── Results ────────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'─' * 60}")
    print(f"Best trial : #{best.number}  |  Val avg MAE = {best.value:.4f}")
    print(f"{'─' * 60}")
    for k, v in best.params.items():
        print(f"  {k:<24} = {v}")

    # ── Write best_lstm_config.yaml ────────────────────────────────────────────
    best_params = best.params
    config = {
        "lstm_attn": {
            "input_size":          int(best_params["input_size"]),
            "hidden_size":         int(best_params["hidden_size"]),
            "num_layers":          int(best_params["num_layers"]),
            "dropout":             float(round(best_params["dropout"], 4)),
            "learning_rate":       float(best_params["learning_rate"]),
            "epochs":              200,
            "batch_size":          int(best_params["batch_size"]),
            "n_heads":             int(best_params["n_heads"]),
            "attention_dropout":   float(round(best_params["attention_dropout"], 4)),
            "early_stop_patience": 15,
            "gradient_clip":       1.0,
        }
    }
    out_path = "best_lstm_config.yaml"
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest config written to: {out_path}")

    # ── Top-5 summary ──────────────────────────────────────────────────────────
    trials_df = (
        study.trials_dataframe()
        .dropna(subset=["value"])
        .sort_values("value")
        .reset_index(drop=True)
    )
    param_cols = [c for c in trials_df.columns if c.startswith("params_")]
    display_cols = ["number", "value"] + param_cols
    print(f"\n── Top 5 trials " + "─" * 44)
    print(trials_df[display_cols].head(5).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
