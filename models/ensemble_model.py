"""Stacking ensemble — LightGBM + Prophet blended by a Ridge meta-model.

Training procedure (avoids data leakage):
  1. Split train_df into base_train (all but last 30 days) and meta_val (last 30 days).
  2. Train both base models on base_train.
  3. Predict meta_val with each base model → out-of-fold (OOF) predictions.
  4. Fit Ridge regression on (lgbm_oof, prophet_oof) → actual_meta_val.
     One Ridge model per target variable.
  5. Retrain base models on full train_df, predict test period.
  6. Blend test predictions through the fitted Ridge → ensemble test preds.
  7. Repeat step 5-6 on full_df for the future forecast.

The Ridge coefficients show how much weight each model receives per target.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from models.lgbm_model import fit_predict as lgbm_fit_predict
from models.prophet_model import _predict_series as prophet_predict

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prophet_predict_all(df, n_days, pcfg):
    """Run Prophet for every city × target and return a unified DataFrame."""
    cities = sorted(df["CITY"].unique())
    parts  = []
    for city in cities:
        city_df  = df[df["CITY"] == city].sort_values("DATE").reset_index(drop=True)
        by_target = {}
        for target in TARGETS:
            preds = prophet_predict(city_df[["DATE", target]], n_days, pcfg)
            by_target[target] = preds[target].values
            dates = preds["DATE"].values
        parts.append(pd.DataFrame({"DATE": dates, "CITY": city, **by_target}))
    result = pd.concat(parts, ignore_index=True)
    result["PRECIPITATION_MM"] = result["PRECIPITATION_MM"].clip(lower=0)
    return result


def _lgbm_predict_all(df, n_days, lags, windows, model_cfg, city_map):
    """Train DMS LightGBM on df and forecast n_days ahead."""
    return lgbm_fit_predict(df, n_days, lags, windows, model_cfg, city_map)


def _blend(lgbm_preds, prophet_preds, ridge_models, cities):
    """Apply fitted Ridge models to blend two sets of predictions."""
    parts = []
    for city in cities:
        lc = lgbm_preds[lgbm_preds["CITY"] == city].sort_values("DATE").reset_index(drop=True)
        pc = prophet_preds[prophet_preds["CITY"] == city].sort_values("DATE").reset_index(drop=True)
        row = {"DATE": lc["DATE"].values, "CITY": city}
        for target in TARGETS:
            X = np.column_stack([lc[target].values, pc[target].values])
            row[target] = ridge_models[target].predict(X)
        parts.append(pd.DataFrame(row))
    return pd.concat(parts, ignore_index=True)


# ── Public interface ──────────────────────────────────────────────────────────

def run(full_df, train_df, cfg):
    lags       = cfg["features"]["lags"]
    windows    = cfg["features"]["rolling_windows"]
    model_cfg  = cfg["lgbm"]
    pcfg       = cfg["prophet"]
    alpha      = cfg["ensemble"]["ridge_alpha"]
    test_days  = cfg["forecasting"]["test_days"]
    n_forecast = cfg["forecasting"]["forecast_days"]

    cities   = sorted(full_df["CITY"].unique())
    city_map = {c: i for i, c in enumerate(cities)}

    # ── Step 1: Split train_df → base_train + meta_val ───────────────────────
    last_train  = train_df["DATE"].max()
    meta_split  = last_train - pd.Timedelta(days=test_days - 1)
    base_train  = train_df[train_df["DATE"] < meta_split].copy()
    meta_val    = train_df[train_df["DATE"] >= meta_split].copy()
    n_meta      = meta_val["DATE"].nunique()

    print(f"  Meta split: base_train up to {base_train['DATE'].max().date()}, "
          f"meta_val = {meta_val['DATE'].min().date()} → {meta_val['DATE'].max().date()} "
          f"({n_meta} days)")

    # ── Step 2: OOF predictions on meta_val ──────────────────────────────────
    print("  Getting OOF predictions (LightGBM + Prophet) on meta_val…")
    lgbm_oof   = _lgbm_predict_all(base_train, n_meta, lags, windows, model_cfg, city_map)
    prophet_oof = _prophet_predict_all(base_train, n_meta, pcfg)

    # ── Step 3: Train Ridge meta-model ───────────────────────────────────────
    print("  Training Ridge meta-model…")
    ridge_models = {}
    print(f"  {'Target':<20} {'LGBM coeff':>12} {'Prophet coeff':>14} {'Intercept':>10}")
    print("  " + "─" * 60)

    for target in TARGETS:
        X_oof, y_oof = [], []
        for city in cities:
            lc  = lgbm_oof[lgbm_oof["CITY"] == city].sort_values("DATE").reset_index(drop=True)
            pc  = prophet_oof[prophet_oof["CITY"] == city].sort_values("DATE").reset_index(drop=True)
            ac  = meta_val[meta_val["CITY"] == city].sort_values("DATE").reset_index(drop=True)
            # Align on common dates
            common = lc["DATE"].values
            ac_aligned = ac[ac["DATE"].isin(common)].sort_values("DATE").reset_index(drop=True)
            pc_aligned = pc[pc["DATE"].isin(common)].sort_values("DATE").reset_index(drop=True)
            X_oof.append(np.column_stack([lc[target].values, pc_aligned[target].values]))
            y_oof.append(ac_aligned[target].values)

        X_oof = np.vstack(X_oof)
        y_oof = np.concatenate(y_oof)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_oof, y_oof)
        ridge_models[target] = ridge
        print(f"  {target:<20} {ridge.coef_[0]:>12.4f} {ridge.coef_[1]:>14.4f} {ridge.intercept_:>10.4f}")

    # ── Step 4: Test-period predictions (base models on full train_df) ────────
    print("\n  Getting test-period predictions (base models retrained on full train_df)…")
    lgbm_test   = _lgbm_predict_all(train_df, test_days, lags, windows, model_cfg, city_map)
    prophet_test = _prophet_predict_all(train_df, test_days, pcfg)
    test_preds  = _blend(lgbm_test, prophet_test, ridge_models, cities)

    # ── Step 5: Future forecast (base models on full_df) ─────────────────────
    print("  Getting future predictions (base models retrained on full_df)…")
    lgbm_future   = _lgbm_predict_all(full_df, n_forecast, lags, windows, model_cfg, city_map)
    prophet_future = _prophet_predict_all(full_df, n_forecast, pcfg)
    future_preds  = _blend(lgbm_future, prophet_future, ridge_models, cities)

    return test_preds, future_preds
