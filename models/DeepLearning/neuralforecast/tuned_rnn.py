

### Package imports ###

from statsmodels.tools.sm_exceptions import ValueWarning
from ray import tune
import logging
import os
import numpy as np
import pandas as pd
import warnings

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoRNN
from neuralforecast.tsdataset import TimeSeriesDataset
# os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

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


#### RNN model tuning ####

df, futr_df, pcc_list, fcc_list = process_vintage_file(
    "../../../data/FRED/blocked/vintage_2023_02.csv")

horizon = 150

df_train = df[df.ds < '2020-01-01']
df_test = df[df.ds > '2020-01-01']

dataset, *_ = TimeSeriesDataset.from_df(df_train)

config = {
    "hist_exog_list": tune.choice([pcc_list]),
    "futr_exog_list": tune.choice([fcc_list]),
    "learning_rate": tune.choice([1e-3]),
    "max_steps": tune.choice([10]),
    "input_size": tune.choice([8 * horizon]),
    "encoder_hidden_size": tune.choice([256]),
    "val_check_steps": tune.choice([1]),
    "random_seed": tune.randint(1, 10),
}



model = AutoRNN(h=horizon, num_samples=1, cpus=1)

model.fit(dataset=dataset)

### RNN model forecast ###
