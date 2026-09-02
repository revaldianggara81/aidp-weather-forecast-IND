"""Autoregressive LSTM forecaster — predicts one step at a time.

Unlike the direct LSTM (which predicts all 30 days in one shot), this model:
  1. Trains to predict only the NEXT single timestep
  2. At inference, feeds each prediction back as input for the following step
     (rolling window update), repeating for the full forecast horizon

This makes the forecast truly autoregressive: each day's prediction
conditions on all previously predicted days. The trade-off vs. direct LSTM is
that errors can compound across the 30-step rollout.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]
N_TARGETS = len(TARGETS)


# ── Model ─────────────────────────────────────────────────────────────────────

class ARWeatherLSTM(nn.Module):
    """Single-step LSTM: predicts the next timestep only."""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, N_TARGETS)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # (batch, n_targets)


# ── Dataset ───────────────────────────────────────────────────────────────────

class OneStepDataset(Dataset):
    """Each sample: (input_window, next_single_step) — one-step-ahead target."""

    def __init__(self, data_norm, city_onehot, input_len):
        T, n_cities, _ = data_norm.shape
        self.samples = []

        for city_idx in range(n_cities):
            city_data = data_norm[:, city_idx, :]       # (T, 3)
            oh        = city_onehot[city_idx]           # (n_cities,)
            oh_tiled  = np.tile(oh, (input_len, 1))     # (input_len, n_cities)

            for i in range(T - input_len):
                x = city_data[i : i + input_len]        # (input_len, 3)
                y = city_data[i + input_len]            # (3,) — next step only
                x_full = np.concatenate([x, oh_tiled], axis=1).astype(np.float32)
                self.samples.append((x_full, y.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ── Helpers (shared with lstm_model) ─────────────────────────────────────────

def _pivot(df):
    cities = sorted(df["CITY"].unique())
    frames = []
    for t in TARGETS:
        pivot = df.pivot(index="DATE", columns="CITY", values=t)[cities]
        frames.append(pivot.values)
    data  = np.stack(frames, axis=-1)       # (T, n_cities, 3)
    dates = pivot.index
    return data, cities, dates


def _normalize(data, mean=None, std=None):
    if mean is None:
        mean = np.nanmean(data, axis=0)
        std  = np.nanstd(data, axis=0)
    normed = (data - mean[None, :, :]) / (std[None, :, :] + 1e-8)
    return normed, mean, std


def _denormalize(normed, mean, std):
    return normed * (std[None, :] + 1e-8) + mean[None, :]


# ── Train ─────────────────────────────────────────────────────────────────────

def _train(data_norm, city_onehot, input_len, lcfg, device):
    feat_size = N_TARGETS + len(city_onehot)
    dataset   = OneStepDataset(data_norm, city_onehot, input_len)
    loader    = DataLoader(dataset, batch_size=lcfg["batch_size"], shuffle=True)

    model = ARWeatherLSTM(
        input_size  = feat_size,
        hidden_size = lcfg["hidden_size"],
        num_layers  = lcfg["num_layers"],
        dropout     = lcfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lcfg["learning_rate"])
    loss_fn   = nn.MSELoss()
    epochs    = lcfg["epochs"]

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"    epoch {epoch}/{epochs}  loss={total_loss/len(loader):.4f}")

    model.eval()
    return model


# ── Autoregressive rollout ────────────────────────────────────────────────────

def _ar_forecast(model, seed_norm, city_onehot_vec, input_len, n_days, device):
    """
    Roll one step at a time.
    seed_norm: (T, 3) normalised history for one city — we take the last input_len rows.
    Returns: (n_days, 3) normalised predictions.
    """
    buffer = seed_norm[-input_len:].copy()   # (input_len, 3)
    preds  = []

    for _ in range(n_days):
        oh_tiled = np.tile(city_onehot_vec, (input_len, 1))    # (input_len, n_cities)
        x_full   = np.concatenate([buffer, oh_tiled], axis=1).astype(np.float32)
        x_t      = torch.FloatTensor(x_full).unsqueeze(0).to(device)

        with torch.no_grad():
            step_pred = model(x_t).squeeze(0).cpu().numpy()    # (3,)

        preds.append(step_pred)
        # Slide the window: drop oldest row, append new prediction
        buffer = np.vstack([buffer[1:], step_pred[None, :]])

    return np.array(preds)   # (n_days, 3)


def _build_predictions(model, data_norm, city_onehot, input_len, n_days,
                        mean, std, cities, last_date, device):
    rows = []
    for city_idx, city in enumerate(cities):
        pred_norm = _ar_forecast(
            model, data_norm[:, city_idx, :],
            city_onehot[city_idx], input_len, n_days, device,
        )
        pred = _denormalize(pred_norm, mean[city_idx], std[city_idx])
        pred[:, TARGETS.index("PRECIPITATION_MM")] = np.maximum(
            0.0, pred[:, TARGETS.index("PRECIPITATION_MM")]
        )
        for step in range(n_days):
            rows.append({
                "DATE": last_date + pd.Timedelta(days=step + 1),
                "CITY": city,
                "TEMPERATURE_C":    pred[step, 0],
                "PRECIPITATION_MM": pred[step, 1],
                "WIND_SPEED_KMH":   pred[step, 2],
            })
    return pd.DataFrame(rows)


# ── Public interface ──────────────────────────────────────────────────────────

def run(full_df, train_df, cfg):
    lcfg    = cfg["ar_lstm"]
    n_test  = cfg["forecasting"]["test_days"]
    n_fcst  = cfg["forecasting"]["forecast_days"]
    inp_len = lcfg["input_size"]

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    AR-LSTM using device: {device}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    train_data, cities, _ = _pivot(train_df)
    n_cities    = len(cities)
    city_onehot = np.eye(n_cities)

    train_norm, mean_tr, std_tr = _normalize(train_data)
    print(f"    Training AR-LSTM for test evaluation ({len(train_df)} rows)…")
    model_eval = _train(train_norm, city_onehot, inp_len, lcfg, device)

    test_preds = _build_predictions(
        model_eval, train_norm, city_onehot, inp_len,
        n_test, mean_tr, std_tr, cities, train_df["DATE"].max(), device,
    )

    # ── Future forecast ───────────────────────────────────────────────────────
    full_data, _, _ = _pivot(full_df)
    full_norm, mean_full, std_full = _normalize(full_data)
    print(f"    Retraining AR-LSTM on full data ({len(full_df)} rows)…")
    model_full = _train(full_norm, city_onehot, inp_len, lcfg, device)

    future_preds = _build_predictions(
        model_full, full_norm, city_onehot, inp_len,
        n_fcst, mean_full, std_full, cities, full_df["DATE"].max(), device,
    )

    return test_preds, future_preds
