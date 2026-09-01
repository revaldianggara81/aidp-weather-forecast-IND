"""Seq-to-seq LSTM forecaster — PyTorch global model with city one-hot encoding.

Architecture:
  - Input:  (batch, input_size, 3 targets + n_cities one-hot)
  - Encoder: stacked LSTM
  - Head:   Linear(hidden → forecast_horizon × 3 targets)
  - Output: (batch, forecast_horizon, 3)

The model predicts all 30 days and all 3 targets in a single forward pass
(no recursive step), which avoids error accumulation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]
N_TARGETS = len(TARGETS)


# ── Model ─────────────────────────────────────────────────────────────────────

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, forecast_horizon):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, forecast_horizon * N_TARGETS)
        self.forecast_horizon = forecast_horizon

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]                          # (batch, hidden)
        pred = self.fc(last)                          # (batch, horizon * 3)
        return pred.view(-1, self.forecast_horizon, N_TARGETS)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    def __init__(self, data_norm, city_onehot, input_len, output_len):
        """
        data_norm:   (T, n_cities, 3) normalised array
        city_onehot: (n_cities, n_cities) identity matrix
        """
        T, n_cities, _ = data_norm.shape
        self.samples = []

        for city_idx in range(n_cities):
            city_data = data_norm[:, city_idx, :]          # (T, 3)
            oh = city_onehot[city_idx]                     # (n_cities,)
            oh_tiled = np.tile(oh, (input_len, 1))         # (input_len, n_cities)

            for i in range(T - input_len - output_len + 1):
                x = city_data[i : i + input_len]           # (input_len, 3)
                y = city_data[i + input_len : i + input_len + output_len]
                x_full = np.concatenate([x, oh_tiled], axis=1).astype(np.float32)
                self.samples.append((x_full, y.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pivot(df):
    """Return (T, n_cities, 3) array and aligned city list + date index."""
    cities = sorted(df["CITY"].unique())
    frames = []
    for t in TARGETS:
        pivot = df.pivot(index="DATE", columns="CITY", values=t)[cities]
        frames.append(pivot.values)                         # (T, n_cities)
    data = np.stack(frames, axis=-1)                        # (T, n_cities, 3)
    dates = pivot.index
    return data, cities, dates


def _normalize(data, mean=None, std=None):
    """Normalise per target (over time × cities). Returns normalised array + stats."""
    flat = data.reshape(-1, N_TARGETS)
    if mean is None:
        mean = np.nanmean(flat, axis=0)   # (3,)
        std  = np.nanstd(flat, axis=0)    # (3,)
    normed = (data - mean[None, None, :]) / (std[None, None, :] + 1e-8)
    return normed, mean, std


def _denormalize(normed, mean, std):
    return normed * (std[None, :] + 1e-8) + mean[None, :]   # (T, 3)


# ── Train / Predict ───────────────────────────────────────────────────────────

def _train(data_norm, city_onehot, input_len, output_len, lcfg, device):
    feat_size = N_TARGETS + len(city_onehot)
    dataset = SlidingWindowDataset(data_norm, city_onehot, input_len, output_len)
    loader  = DataLoader(dataset, batch_size=lcfg["batch_size"], shuffle=True)

    model = WeatherLSTM(
        input_size=feat_size,
        hidden_size=lcfg["hidden_size"],
        num_layers=lcfg["num_layers"],
        dropout=lcfg["dropout"],
        forecast_horizon=output_len,
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


def _predict_all_cities(model, data_norm, city_onehot, input_len, n_days, mean, std, cities, device):
    """One forward pass per city → returns DataFrame [DATE placeholder, CITY, targets]."""
    rows = []
    n_cities = len(cities)
    for city_idx, city in enumerate(cities):
        city_window = data_norm[-input_len:, city_idx, :]   # (input_len, 3)
        oh = city_onehot[city_idx]                          # (n_cities,)
        oh_tiled = np.tile(oh, (input_len, 1))              # (input_len, n_cities)
        x = np.concatenate([city_window, oh_tiled], axis=1).astype(np.float32)
        x_t = torch.FloatTensor(x).unsqueeze(0).to(device) # (1, input_len, feat_size)

        with torch.no_grad():
            pred_norm = model(x_t).squeeze(0).cpu().numpy() # (n_days, 3)

        pred = _denormalize(pred_norm, mean, std)           # (n_days, 3)
        pred[:, TARGETS.index("PRECIPITATION_MM")] = np.maximum(
            0.0, pred[:, TARGETS.index("PRECIPITATION_MM")]
        )
        for step in range(n_days):
            rows.append({
                "step": step,
                "CITY": city,
                "TEMPERATURE_C":    pred[step, 0],
                "PRECIPITATION_MM": pred[step, 1],
                "WIND_SPEED_KMH":   pred[step, 2],
            })
    return pd.DataFrame(rows)


# ── Public interface ──────────────────────────────────────────────────────────

def run(full_df, train_df, cfg):
    lcfg     = cfg["lstm"]
    n_test   = cfg["forecasting"]["test_days"]
    n_fcst   = cfg["forecasting"]["forecast_days"]
    inp_len  = lcfg["input_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    LSTM using device: {device}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    train_data, cities, train_dates = _pivot(train_df)
    n_cities   = len(cities)
    city_onehot = np.eye(n_cities)

    train_norm, mean_tr, std_tr = _normalize(train_data)
    print(f"    Training LSTM for test evaluation ({len(train_df)} rows)…")
    model_eval = _train(train_norm, city_onehot, inp_len, n_test, lcfg, device)

    raw_test = _predict_all_cities(
        model_eval, train_norm, city_onehot, inp_len, n_test, mean_tr, std_tr, cities, device
    )
    # Assign actual forecast dates
    last_train_date = train_df["DATE"].max()
    test_dates = [last_train_date + pd.Timedelta(days=s + 1) for s in range(n_test)]
    test_preds = raw_test.copy()
    test_preds["DATE"] = test_preds["step"].map(lambda s: test_dates[s])
    test_preds = test_preds.drop(columns="step")

    # ── Future forecast ───────────────────────────────────────────────────────
    full_data, cities_full, _ = _pivot(full_df)
    full_norm, mean_full, std_full = _normalize(full_data)
    print(f"    Retraining LSTM on full data ({len(full_df)} rows)…")
    model_full = _train(full_norm, city_onehot, inp_len, n_fcst, lcfg, device)

    raw_future = _predict_all_cities(
        model_full, full_norm, city_onehot, inp_len, n_fcst, mean_full, std_full, cities, device
    )
    last_full_date = full_df["DATE"].max()
    future_dates = [last_full_date + pd.Timedelta(days=s + 1) for s in range(n_fcst)]
    future_preds = raw_future.copy()
    future_preds["DATE"] = future_preds["step"].map(lambda s: future_dates[s])
    future_preds = future_preds.drop(columns="step")

    return test_preds, future_preds
