

### RNN encoder / decoder model (final vintage)

# Only calculate for the final usable vintage
# TODO: Nowcast and 1,2,3 period ahead forecast + performance metric
# TODO: Check long vs wide format -- does this make a difference?

### Package imports ###

from ray import tune
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoRNN

### Ignore warnings ###

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

### Data preprocessing ###


def load_data(file_path):
    df = (pd.read_csv(file_path)
          .rename(columns={"year_quarter": "ds", "GDPC1": "y"})
          .assign(unique_id=np.ones(len(pd.read_csv(file_path))),
                  ds=lambda df: pd.to_datetime(df['ds'])))
    columns_order = ["unique_id", "ds", "y"] + \
        [col for col in df.columns if col not in ["unique_id", "ds", "y"]]
    return df[columns_order]


def separate_covariates(df, point_in_time):
    covariates = df.drop(columns=["unique_id", "ds", "y"])

    if not point_in_time:
        return df[covariates.columns], df[[]]

    mask = covariates.apply(
        lambda col: col.loc[col.index >= point_in_time - 1].isnull().any())

    past_covariates = df[mask.index[mask]]
    future_covariates = df[mask.index[~mask]]

    return past_covariates, future_covariates


def impute_missing_values_interpolate(data, method='linear'):
    imputed_data = data.copy()
    imputed_data.fillna(method='bfill', inplace=True)
    return imputed_data.interpolate(method=method)


### Different vintages ###


vintage_files = [
    f'../../data/FRED/blocked/vintage_{year}_{month:02d}.csv'
    for year in range(2018, 2024)
    for month in range(1, 13)
    if not (
        (year == 2018 and month < 5) or
        (year == 2023 and month > 2)
    )
]


### Forecast across last usable vintage ###


def forecast_vintage(vintage_file, horizon=2):
    results = {}

    df = load_data(vintage_file)

    target_df = df[["unique_id", "ds", "y"]]

    point_in_time = df.index[-2] # explain later

    past_covariates, future_covariates = separate_covariates(
        df, point_in_time)

    df_pc = impute_missing_values_interpolate(past_covariates)
    df_fc = impute_missing_values_interpolate(future_covariates)

    pcc_list = past_covariates.columns.tolist()
    fcc_list = future_covariates.columns.tolist()

    df = (target_df
          .merge(df_fc, left_index=True, right_index=True)
          .merge(df_pc, left_index=True, right_index=True)
          .iloc[:-1])

    futr_df = (target_df
               .merge(df_fc, left_index=True, right_index=True)
               .drop(columns="y")
               .iloc[-1:])

    config = {
        "input_size": tune.choice([-1]),
        "hist_exog_list": tune.choice([pcc_list]),
        "futr_exog_list": tune.choice([fcc_list]),
        "max_steps": tune.choice([500]),
        "scaler_type": tune.choice(["robust"])
    }

    # Some other parts of configuration to consider
    # 

    model = AutoRNN(h=horizon,
                    config=config, num_samples=1, verbose=False)

    nf = NeuralForecast(models=[model], freq='Q')
    nf.fit(df=df, val_size=24)

    Y_hat_df = nf.predict(futr_df=futr_df)
    Y_hat_df = Y_hat_df.reset_index()
    Y_hat_df['ds'] = Y_hat_df['ds'] + pd.Timedelta(days = 1)

    return Y_hat_df

vintage_file = "../../data/FRED/blocked/vintage_2020_05.csv"

forecast_vintage(vintage_file)

