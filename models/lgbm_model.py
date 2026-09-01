"""LightGBM forecaster — Direct Multi-Step (DMS) forecasting.

Instead of recursively predicting one step at a time (which causes lag/rolling
features to fill with model predictions, converging to a flat mean), we train
one separate LGBMRegressor per forecast horizon h=1..30.

At training time for horizon h:
  - Lag/rolling features are computed from actual historical values at origin t
  - Calendar/Fourier features are computed for the FORECAST date t+h
    (so the model knows what season it is predicting for)
  - Target is the actual value at t+h

At inference time:
  - Lag/rolling features are computed once from the last real observation
  - Each horizon model receives its own forecast-date calendar features
  - All 30 predictions are made independently — no feedback loop
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]


# ── Feature Engineering ───────────────────────────────────────────────────────

def _make_base_features(df, lags, windows):
    """Build lag, rolling, and city features — calendar features added per horizon."""
    df = df.sort_values(["CITY", "DATE"]).reset_index(drop=True)
    df["city_code"] = df["CITY"].astype("category").cat.codes

    for target in TARGETS:
        grp = df.groupby("CITY")[target]
        for lag in lags:
            df[f"{target}_lag{lag}"] = grp.shift(lag)
        for w in windows:
            df[f"{target}_rmean{w}"] = grp.transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            df[f"{target}_rstd{w}"] = grp.transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).std()
            )
    return df


def _add_calendar(df, date_series):
    """
    Add calendar + Fourier features to df, computed from date_series.
    date_series must be a pandas Series aligned to df's index.
    """
    doy = date_series.dt.dayofyear
    df  = df.copy()
    df["day_of_year"] = doy.values
    df["month"]       = date_series.dt.month.values
    df["day_of_week"] = date_series.dt.dayofweek.values
    df["sin_doy"]     = np.sin(2 * np.pi * doy.values / 365.25)
    df["cos_doy"]     = np.cos(2 * np.pi * doy.values / 365.25)
    df["sin_doy2"]    = np.sin(4 * np.pi * doy.values / 365.25)
    df["cos_doy2"]    = np.cos(4 * np.pi * doy.values / 365.25)
    return df


def _feature_cols(lags, windows):
    calendar  = [
        "day_of_year", "month", "day_of_week",
        "sin_doy", "cos_doy", "sin_doy2", "cos_doy2", "city_code",
    ]
    lag_feats  = [f"{t}_lag{l}"   for t in TARGETS for l in lags]
    roll_feats = (
        [f"{t}_rmean{w}" for t in TARGETS for w in windows]
        + [f"{t}_rstd{w}"  for t in TARGETS for w in windows]
    )
    return calendar + lag_feats + roll_feats


# ── Training ──────────────────────────────────────────────────────────────────

def _train_direct(df, base_feat, feats, model_cfg, n_horizons):
    """
    Train one LGBMRegressor per (target, horizon h).

    For horizon h:
      X = base lag/rolling features + calendar features for date t+h
      y = actual value at t+h  (shift target backward by h)
    """
    # Ensure aligned integer index between df and base_feat
    df        = df.reset_index(drop=True)
    base_feat = base_feat.reset_index(drop=True)

    models = {}
    print(
        f"    Training {n_horizons} horizons × {len(TARGETS)} targets "
        f"= {n_horizons * len(TARGETS)} models…"
    )

    for h in range(1, n_horizons + 1):
        # Calendar features for the forecast date t+h
        forecast_dates = (df["DATE"] + pd.Timedelta(days=h)).reset_index(drop=True)
        df_h = _add_calendar(base_feat, forecast_dates)

        valid_feats = df_h[feats].notna().all(axis=1)

        for target in TARGETS:
            y_h   = df.groupby("CITY")[target].shift(-h).reset_index(drop=True)
            valid = valid_feats & y_h.notna()

            m = lgb.LGBMRegressor(**model_cfg, verbose=-1)
            m.fit(df_h.loc[valid, feats], y_h.loc[valid])
            models[(target, h)] = m

    return models


# ── Inference ─────────────────────────────────────────────────────────────────

def _direct_forecast(models, df, base_feat, feats, city_map, n_days):
    """
    Predict n_days ahead for every city.

    For each city, the lag/rolling features are taken from the last real row.
    Calendar features are recomputed per horizon to reflect the forecast date.
    No predictions are ever fed back as inputs.
    """
    last_date = df["DATE"].max()
    cities    = sorted(df["CITY"].unique())
    all_preds = []

    for city in cities:
        city_base = base_feat[base_feat["CITY"] == city].sort_values("DATE")
        last_row  = city_base.iloc[[-1]].copy()   # last real feature row for this city

        for h in range(1, n_days + 1):
            forecast_date = last_date + pd.Timedelta(days=h)
            doy = forecast_date.dayofyear

            # Override calendar features for this forecast date
            row = last_row.copy()
            row["day_of_year"] = doy
            row["month"]       = forecast_date.month
            row["day_of_week"] = forecast_date.dayofweek
            row["sin_doy"]     = np.sin(2 * np.pi * doy / 365.25)
            row["cos_doy"]     = np.cos(2 * np.pi * doy / 365.25)
            row["sin_doy2"]    = np.sin(4 * np.pi * doy / 365.25)
            row["cos_doy2"]    = np.cos(4 * np.pi * doy / 365.25)

            feat_row = row[feats]
            pred     = {"DATE": forecast_date, "CITY": city}

            for target in TARGETS:
                p = models[(target, h)].predict(feat_row)[0]
                if target == "PRECIPITATION_MM":
                    p = max(0.0, p)
                pred[target] = p

            all_preds.append(pred)

    return pd.DataFrame(all_preds)


# ── Convenience function (used by ensemble) ───────────────────────────────────

def fit_predict(df, n_days, lags, windows, model_cfg, city_map):
    """Train DMS models on df and return an n_days forecast DataFrame."""
    feats     = _feature_cols(lags, windows)
    base_feat = _make_base_features(df.copy(), lags, windows)
    base_feat["city_code"] = base_feat["CITY"].map(city_map)
    models    = _train_direct(df, base_feat, feats, model_cfg, n_days)
    return _direct_forecast(models, df, base_feat, feats, city_map, n_days)


# ── Public interface ──────────────────────────────────────────────────────────

def run(full_df, train_df, cfg):
    lags       = cfg["features"]["lags"]
    windows    = cfg["features"]["rolling_windows"]
    model_cfg  = cfg["lgbm"]
    n_test     = cfg["forecasting"]["test_days"]
    n_forecast = cfg["forecasting"]["forecast_days"]

    city_map = {c: i for i, c in enumerate(sorted(full_df["CITY"].unique()))}
    feats    = _feature_cols(lags, windows)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("  Training on train_df…")
    base_feat_train = _make_base_features(train_df.copy(), lags, windows)
    base_feat_train["city_code"] = base_feat_train["CITY"].map(city_map)
    models_eval = _train_direct(train_df, base_feat_train, feats, model_cfg, n_test)
    test_preds  = _direct_forecast(
        models_eval, train_df, base_feat_train, feats, city_map, n_test
    )

    # ── Future forecast — retrain on full data ────────────────────────────────
    print("  Retraining on full_df…")
    base_feat_full = _make_base_features(full_df.copy(), lags, windows)
    base_feat_full["city_code"] = base_feat_full["CITY"].map(city_map)
    models_full  = _train_direct(full_df, base_feat_full, feats, model_cfg, n_forecast)
    future_preds = _direct_forecast(
        models_full, full_df, base_feat_full, feats, city_map, n_forecast
    )

    return test_preds, future_preds
