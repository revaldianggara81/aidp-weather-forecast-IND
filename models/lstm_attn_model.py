"""Bidirectional LSTM with multi-head attention — improved weather forecaster.

Architecture:
  Input (batch, lookback, features)
    → Bidirectional LSTM encoder (batch, lookback, 2×hidden)
    → Multi-head self-attention (batch, lookback, 2×hidden)  [+ residual + LayerNorm]
    → Mean pooling over time (batch, 2×hidden)
    → 3 separate MLP heads → per-target output (batch, horizon)

Improvements over base LSTM (models/lstm_model.py):
  - Bidirectional encoder
  - Multi-head self-attention with residual connection & layer norm
  - Per-target MLP heads (instead of single linear)
  - Calendar features: sin/cos(day_of_year), normalised month
  - Per-city Z-score normalisation
  - Huber loss, ReduceLROnPlateau scheduler, early stopping, gradient clipping
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]
N_TARGETS = len(TARGETS)
N_CALENDAR = 3  # sin(doy), cos(doy), month/12


# ── Model ─────────────────────────────────────────────────────────────────────

class WeatherLSTMAttn(nn.Module):
    def __init__(self, feat_size, hidden_size, num_layers, dropout,
                 forecast_horizon, n_heads, attention_dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_size, hidden_size, num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        enc_dim = hidden_size * 2  # bidirectional doubles the output

        self.attention = nn.MultiheadAttention(
            embed_dim=enc_dim,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(enc_dim)

        # Per-target MLP heads
        self.head_temp   = self._make_head(enc_dim, dropout, forecast_horizon)
        self.head_precip = self._make_head(enc_dim, dropout, forecast_horizon)
        self.head_wind   = self._make_head(enc_dim, dropout, forecast_horizon)
        self.forecast_horizon = forecast_horizon

    @staticmethod
    def _make_head(in_dim, dropout, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, out_dim),
        )

    def forward(self, x):
        # x: (batch, lookback, features)
        lstm_out, _ = self.lstm(x)                        # (batch, lookback, 2*hidden)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)   # residual + norm
        pooled = attn_out.mean(dim=1)                     # (batch, 2*hidden)

        temp   = self.head_temp(pooled)                   # (batch, horizon)
        precip = self.head_precip(pooled)
        wind   = self.head_wind(pooled)
        return torch.stack([temp, precip, wind], dim=-1)  # (batch, horizon, 3)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    def __init__(self, data_norm, city_onehot, calendar_feats, input_len, output_len):
        """
        data_norm:      (T, n_cities, 3) per-city normalised targets
        city_onehot:    (n_cities, n_cities) identity matrix
        calendar_feats: (T, 3) sin_doy, cos_doy, month_norm
        """
        T, n_cities, _ = data_norm.shape
        self.samples = []

        for city_idx in range(n_cities):
            city_data = data_norm[:, city_idx, :]           # (T, 3)
            oh = city_onehot[city_idx]                      # (n_cities,)

            for i in range(T - input_len - output_len + 1):
                x_targets = city_data[i : i + input_len]    # (input_len, 3)
                y = city_data[i + input_len : i + input_len + output_len]

                oh_tiled = np.tile(oh, (input_len, 1))      # (input_len, n_cities)
                cal = calendar_feats[i : i + input_len]     # (input_len, 3)

                x_full = np.concatenate(
                    [x_targets, oh_tiled, cal], axis=1
                ).astype(np.float32)
                self.samples.append((x_full, y.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pivot(df):
    """Return (T, n_cities, 3) array, city list, and DatetimeIndex."""
    cities = sorted(df["CITY"].unique())
    frames = []
    for t in TARGETS:
        pivot = df.pivot(index="DATE", columns="CITY", values=t)[cities]
        frames.append(pivot.values)
    data = np.stack(frames, axis=-1)   # (T, n_cities, 3)
    dates = pivot.index
    return data, cities, dates


def _calendar_features(dates):
    """Compute calendar features from DatetimeIndex → (T, 3)."""
    doy = dates.dayofyear.values.astype(np.float32)
    month = dates.month.values.astype(np.float32)
    sin_doy = np.sin(2 * np.pi * doy / 365.25)
    cos_doy = np.cos(2 * np.pi * doy / 365.25)
    month_norm = month / 12.0
    return np.stack([sin_doy, cos_doy, month_norm], axis=-1)  # (T, 3)


def _normalize_per_city(data, mean=None, std=None):
    """Per-city Z-score normalisation.

    data: (T, n_cities, 3)
    Returns normalised array and per-city stats (n_cities, 3).
    """
    if mean is None:
        mean = np.nanmean(data, axis=0)  # (n_cities, 3)
        std  = np.nanstd(data, axis=0)   # (n_cities, 3)
    normed = (data - mean[None, :, :]) / (std[None, :, :] + 1e-8)
    return normed, mean, std


def _denormalize_per_city(normed, mean, std, city_idx):
    """Denormalize predictions for a single city.

    normed: (T, 3)  |  mean/std: (n_cities, 3)
    """
    return normed * (std[city_idx, :] + 1e-8) + mean[city_idx, :]


# ── Train / Predict ───────────────────────────────────────────────────────────

def _train(data_norm, city_onehot, calendar_feats, input_len, output_len,
           lcfg, device):
    n_cities = len(city_onehot)
    feat_size = N_TARGETS + n_cities + N_CALENDAR

    full_dataset = SlidingWindowDataset(
        data_norm, city_onehot, calendar_feats, input_len, output_len
    )

    # 85/15 train/val split for early stopping
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * 0.15))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=lcfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=lcfg["batch_size"], shuffle=False)

    model = WeatherLSTMAttn(
        feat_size=feat_size,
        hidden_size=lcfg["hidden_size"],
        num_layers=lcfg["num_layers"],
        dropout=lcfg["dropout"],
        forecast_horizon=output_len,
        n_heads=lcfg["n_heads"],
        attention_dropout=lcfg["attention_dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lcfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    loss_fn   = nn.HuberLoss(delta=1.0)
    epochs    = lcfg["epochs"]
    patience  = lcfg.get("early_stop_patience", 15)
    max_norm  = lcfg.get("gradient_clip", 1.0)

    best_val_loss = float("inf")
    best_state    = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                val_loss += loss_fn(model(x_batch), y_batch).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if epoch % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"    epoch {epoch}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.1e}")

        # ── Early stopping ─────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stopping at epoch {epoch} "
                      f"(best val={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model


def _predict_all_cities(model, data_norm, city_onehot, calendar_feats,
                        input_len, n_days, mean, std, cities, device):
    """One forward pass per city → returns DataFrame."""
    rows = []
    for city_idx, city in enumerate(cities):
        city_window = data_norm[-input_len:, city_idx, :]    # (input_len, 3)
        oh = city_onehot[city_idx]
        oh_tiled = np.tile(oh, (input_len, 1))               # (input_len, n_cities)
        cal = calendar_feats[-input_len:]                     # (input_len, 3)

        x = np.concatenate([city_window, oh_tiled, cal], axis=1).astype(np.float32)
        x_t = torch.FloatTensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = model(x_t).squeeze(0).cpu().numpy()  # (n_days, 3)

        pred = _denormalize_per_city(pred_norm, mean, std, city_idx)
        pred[:, TARGETS.index("PRECIPITATION_MM")] = np.maximum(
            0.0, pred[:, TARGETS.index("PRECIPITATION_MM")]
        )
        for step in range(n_days):
            rows.append({
                "step":             step,
                "CITY":             city,
                "TEMPERATURE_C":    pred[step, 0],
                "PRECIPITATION_MM": pred[step, 1],
                "WIND_SPEED_KMH":   pred[step, 2],
            })
    return pd.DataFrame(rows)


# ── Public interface ──────────────────────────────────────────────────────────

def run(full_df, train_df, cfg):
    """Same interface as lstm_model.run() — returns (test_preds, future_preds)."""
    lcfg    = cfg["lstm_attn"]
    n_test  = cfg["forecasting"]["test_days"]
    n_fcst  = cfg["forecasting"]["forecast_days"]
    inp_len = lcfg["input_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    LSTM-Attn using device: {device}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    train_data, cities, train_dates = _pivot(train_df)
    n_cities    = len(cities)
    city_onehot = np.eye(n_cities)

    train_cal  = _calendar_features(train_dates)
    train_norm, mean_tr, std_tr = _normalize_per_city(train_data)

    print(f"    Training LSTM-Attn for test evaluation ({len(train_df)} rows)…")
    model_eval = _train(train_norm, city_onehot, train_cal,
                        inp_len, n_test, lcfg, device)

    raw_test = _predict_all_cities(
        model_eval, train_norm, city_onehot, train_cal,
        inp_len, n_test, mean_tr, std_tr, cities, device,
    )
    last_train_date = train_df["DATE"].max()
    test_dates = [last_train_date + pd.Timedelta(days=s + 1)
                  for s in range(n_test)]
    test_preds = raw_test.copy()
    test_preds["DATE"] = test_preds["step"].map(lambda s: test_dates[s])
    test_preds = test_preds.drop(columns="step")

    # ── Future forecast ───────────────────────────────────────────────────────
    full_data, _, full_dates = _pivot(full_df)
    full_cal  = _calendar_features(full_dates)
    full_norm, mean_full, std_full = _normalize_per_city(full_data)

    print(f"    Retraining LSTM-Attn on full data ({len(full_df)} rows)…")
    model_full = _train(full_norm, city_onehot, full_cal,
                        inp_len, n_fcst, lcfg, device)

    raw_future = _predict_all_cities(
        model_full, full_norm, city_onehot, full_cal,
        inp_len, n_fcst, mean_full, std_full, cities, device,
    )
    last_full_date = full_df["DATE"].max()
    future_dates = [last_full_date + pd.Timedelta(days=s + 1)
                    for s in range(n_fcst)]
    future_preds = raw_future.copy()
    future_preds["DATE"] = future_preds["step"].map(lambda s: future_dates[s])
    future_preds = future_preds.drop(columns="step")

    return test_preds, future_preds
