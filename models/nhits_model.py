"""N-HiTS forecaster — via Nixtla's neuralforecast library.

One NeuralForecast model is trained per target variable, with all cities
treated as separate series (unique_id). This lets the model learn shared
seasonal patterns across cities while keeping predictions city-specific.
"""

import logging
import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("neuralforecast").setLevel(logging.ERROR)

TARGETS = ["TEMPERATURE_C", "PRECIPITATION_MM", "WIND_SPEED_KMH"]


def _to_nf_format(df, target):
    """Convert wide DataFrame to neuralforecast long format."""
    nf = df[["DATE", "CITY", target]].rename(
        columns={"DATE": "ds", "CITY": "unique_id", target: "y"}
    ).copy()
    nf["ds"] = pd.to_datetime(nf["ds"])
    nf = nf.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return nf


def _merge_targets(preds_by_target):
    df = preds_by_target[TARGETS[0]]
    for t in TARGETS[1:]:
        df = df.merge(preds_by_target[t], on=["DATE", "CITY"])
    return df


def run(full_df, train_df, cfg):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    ncfg = cfg["nhits"]
    n_test = cfg["forecasting"]["test_days"]
    n_forecast = cfg["forecasting"]["forecast_days"]

    test_by_target, future_by_target = {}, {}

    for target in TARGETS:
        train_nf = _to_nf_format(train_df, target)
        full_nf = _to_nf_format(full_df, target)

        def make_model(h):
            return NHITS(
                h=h,
                input_size=ncfg["input_size"],
                max_steps=ncfg["max_steps"],
                learning_rate=ncfg["learning_rate"],
                batch_size=ncfg["batch_size"],
                random_seed=ncfg["random_seed"],
            )

        # Test evaluation: train on train_df, forecast n_test days
        nf_eval = NeuralForecast(models=[make_model(n_test)], freq="D")
        nf_eval.fit(train_nf, val_size=0)
        fc_test = nf_eval.predict().reset_index(drop=True)

        # Future forecast: retrain on full_df, forecast n_forecast days
        nf_full = NeuralForecast(models=[make_model(n_forecast)], freq="D")
        nf_full.fit(full_nf, val_size=0)
        fc_future = nf_full.predict().reset_index(drop=True)

        # neuralforecast names the prediction column after the model class
        nhits_col = [c for c in fc_test.columns if c not in ("unique_id", "ds")][0]

        fc_test = fc_test.rename(
            columns={"unique_id": "CITY", "ds": "DATE", nhits_col: target}
        )
        fc_future = fc_future.rename(
            columns={"unique_id": "CITY", "ds": "DATE", nhits_col: target}
        )

        test_by_target[target] = fc_test[["DATE", "CITY", target]]
        future_by_target[target] = fc_future[["DATE", "CITY", target]]

    test_preds = _merge_targets(test_by_target)
    future_preds = _merge_targets(future_by_target)

    test_preds["DATE"] = pd.to_datetime(test_preds["DATE"])
    future_preds["DATE"] = pd.to_datetime(future_preds["DATE"])
    test_preds["PRECIPITATION_MM"] = test_preds["PRECIPITATION_MM"].clip(lower=0)
    future_preds["PRECIPITATION_MM"] = future_preds["PRECIPITATION_MM"].clip(lower=0)

    return test_preds, future_preds
