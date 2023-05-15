

### Package imports ###

import logging
import os
import numpy as np
import pandas as pd
import warnings

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
from ray import tune
from statsmodels.tools.sm_exceptions import ValueWarning

### Ignore warnings ###

warnings.simplefilter("ignore", ValueWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

    point_in_time = point_in_time[0]

    mask = covariates.apply(
        lambda col: col.loc[col.index >= point_in_time - 1].isnull().any())

    past_covariates = df[mask.index[mask]]
    future_covariates = df[mask.index[~mask]]

    return past_covariates, future_covariates


def impute_missing_values_interpolate(data, method='linear'):
    imputed_data = data.copy()
    imputed_data.fillna(method='bfill', inplace=True)
    return imputed_data.interpolate(method=method)


def process_vintage_file(file_path):
    df = load_data(file_path)
    target_df = df[["unique_id", "ds", "y"]]
    point_in_time = list(df[df['y'].isnull()].index)
    past_covariates, future_covariates = separate_covariates(df, point_in_time)

    df_pc = impute_missing_values_interpolate(past_covariates)
    df_fc = impute_missing_values_interpolate(future_covariates)

    pcc_list = past_covariates.columns.tolist()
    fcc_list = future_covariates.columns.tolist()

    df = (target_df
          .merge(df_fc, left_index=True, right_index=True)
          .merge(df_pc, left_index=True, right_index=True)
          .iloc[:-1])

    futr_df = (target_df
               .merge(future_covariates, left_index=True, right_index=True)
               .drop(columns="y")
               .iloc[-1:])

    return df, futr_df, pcc_list, fcc_list


#### NHITS model tuning ####

df, futr_df, pcc_list, fcc_list = process_vintage_file(
    "../../../data/FRED/blocked/vintage_2023_02.csv")

horizon = 20

nhits_config = {
    "hist_exog_list": tune.choice([pcc_list]),
    "futr_exog_list": tune.choice([fcc_list]),
    "learning_rate": tune.choice([1e-3]),
    "max_steps": tune.choice([10]),
    "input_size": tune.choice([8 * horizon]),
    "batch_size": tune.choice([7]),
    "windows_batch_size": tune.choice([256]),
    "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),
    "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]),
    "activation": tune.choice(['ReLU']),
    "n_blocks":  tune.choice([[1, 1, 1]]),
    "mlp_units":  tune.choice([[[512, 512], [512, 512], [512, 512]]]),
    "interpolation_mode": tune.choice(['linear']),
    "val_check_steps": tune.choice([100]),
    "random_seed": tune.randint(1, 10),
}


def create_neural_forecast_model(horizon, nhits_config):
    model = AutoNHITS(h=horizon, config=nhits_config, num_samples=5)
    return NeuralForecast(models=[model], freq='Q')


nf = create_neural_forecast_model(horizon, nhits_config=nhits_config)

nf.fit(df=df)

# Y_hat_df = nf.predict(futr_df=futr_df)
# forecast_value = Y_hat_df.iloc[0, 1]
# results = forecast_value
