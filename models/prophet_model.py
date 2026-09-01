"""Prophet forecaster — one model per city per target variable."""

import logging
import pandas as pd

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]


def _predict_series(series_df, n_days, pcfg):
    """Fit Prophet on one city/target series and return the next n_days forecast."""
    from prophet import Prophet

    target_col = [c for c in series_df.columns if c != "DATE"][0]
    df_p = series_df.rename(columns={"DATE": "ds", target_col: "y"}).copy()
    df_p["ds"] = pd.to_datetime(df_p["ds"])

    m = Prophet(
        seasonality_mode=pcfg["seasonality_mode"],
        yearly_seasonality=pcfg["yearly_seasonality"],
        weekly_seasonality=pcfg["weekly_seasonality"],
        daily_seasonality=pcfg["daily_seasonality"],
        changepoint_prior_scale=float(pcfg["changepoint_prior_scale"]),
    )
    m.fit(df_p)

    future = m.make_future_dataframe(periods=n_days, freq="D")
    fc = m.predict(future)
    last_train = df_p["ds"].max()
    result = (
        fc[fc["ds"] > last_train][["ds", "yhat"]]
        .head(n_days)
        .rename(columns={"ds": "DATE", "yhat": target_col})
        .reset_index(drop=True)
    )
    result["DATE"] = pd.to_datetime(result["DATE"])
    return result


def run(full_df, train_df, cfg):
    pcfg = cfg["prophet"]
    n_test = cfg["forecasting"]["test_days"]
    n_forecast = cfg["forecasting"]["forecast_days"]
    cities = sorted(full_df["CITY"].unique())

    test_parts, future_parts = [], []

    for city in cities:
        city_train = train_df[train_df["CITY"] == city].sort_values("DATE").reset_index(drop=True)
        city_full = full_df[full_df["CITY"] == city].sort_values("DATE").reset_index(drop=True)

        test_by_target, future_by_target = {}, {}
        test_dates, future_dates = None, None

        for target in TARGETS:
            t_preds = _predict_series(city_train[["DATE", target]], n_test, pcfg)
            f_preds = _predict_series(city_full[["DATE", target]], n_forecast, pcfg)
            test_by_target[target] = t_preds[target].values
            future_by_target[target] = f_preds[target].values
            test_dates = t_preds["DATE"].values
            future_dates = f_preds["DATE"].values

        test_parts.append(pd.DataFrame({"DATE": test_dates, "CITY": city, **test_by_target}))
        future_parts.append(pd.DataFrame({"DATE": future_dates, "CITY": city, **future_by_target}))

    test_preds = pd.concat(test_parts, ignore_index=True)
    future_preds = pd.concat(future_parts, ignore_index=True)

    test_preds["PRECIPITATION_MM"] = test_preds["PRECIPITATION_MM"].clip(lower=0)
    future_preds["PRECIPITATION_MM"] = future_preds["PRECIPITATION_MM"].clip(lower=0)

    return test_preds, future_preds
