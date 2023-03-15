
# Try and model using NeuralForecast package. NHITS model with covariates.

import logging
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from neuralforecast.models import NHITS
from neuralforecast import NeuralForecast

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={"year_quarter": "ds", "GDPC1": "y"})
    df["unique_id"] = np.ones(len(df))
    df["ds"] = pd.to_datetime(df["ds"])
    df = df[["unique_id", "ds", "y"] +
            [col for col in df.columns if col not in ["unique_id", "ds", "y"]]]
    return df


def impute_missing_values_mean(data):
    imputed_data = data.copy()
    imputed_data.apply(lambda x: x.fillna(x.mean()), axis=0)
    return imputed_data


def impute_missing_values_median(data):
    imputed_data = data.copy()
    imputed_data.apply(lambda x: x.fillna(x.median()), axis=0)
    return imputed_data


def impute_missing_values_rolling_mean(data, window_size):
    imputed_data = data.copy()
    imputed_data.fillna(data.rolling(window=window_size,
                        min_periods=1, center=True).mean(), inplace=True)
    return imputed_data


def impute_missing_values_rolling_median(data, window_size):
    imputed_data = data.copy()
    imputed_data.fillna(data.rolling(window=window_size,
                        min_periods=1, center=True).median(), inplace=True)
    return imputed_data


def impute_missing_values_bfill_ffill(data):
    """
    Impute missing values in multiple time series using backfill and forward fill.

    Args:
        data (pd.DataFrame): The time series data with missing values to impute. Each column represents a time series.

    Returns:
        pd.DataFrame: The imputed time series data.
    """
    imputed_data = data.copy()
    imputed_data.fillna(method='bfill', inplace=True)
    imputed_data.fillna(method='ffill', inplace=True)
    return imputed_data


def impute_missing_values_interpolate(data, method='linear'):
    """
    Impute missing values in multiple time series using interpolation.

    Args:
        data (pd.DataFrame): The time series data with missing values to impute. Each column represents a time series.
        method (str): The interpolation method to use. Default is 'linear'.

    Returns:
        pd.DataFrame: The imputed time series data.
    """
    imputed_data = data.copy()
    imputed_data.interpolate(method=method, inplace=True)
    return imputed_data


def impute_missing_values_ar_multiseries(data, lags=1):
    """
    Impute missing values in multiple time series using an autoregressive model.

    Args:
        data (pd.DataFrame): The time series data with missing values to impute. Each column represents a time series.
        lags (int): The number of lags to use in the autoregressive model.

    Returns:
        pd.DataFrame: The imputed time series data.
    """
    imputed_data = data.copy()

    for col in data.columns:
        while imputed_data[col].isnull().any():
            not_null_indices = imputed_data[col].notnull()
            train = imputed_data.loc[not_null_indices, col]
            null_indices = imputed_data[col].isnull()
            test_indices = imputed_data.loc[null_indices, col].index

            model = AutoReg(train, lags=lags)
            result = model.fit()

            for index in test_indices:
                if index - lags < 0:
                    available_data = imputed_data.loc[:index - 1, col].values
                else:
                    available_data = imputed_data.loc[index -
                                                      lags:index - 1, col].values
                if np.isnan(available_data).any():
                    continue
                forecast = result.predict(start=len(train), end=len(train))
                imputed_data.loc[index, col] = forecast.iloc[0]

    return imputed_data


def create_neural_forecast_model(horizon, pcc_list, fcc_list):
    model = NHITS(h=horizon,
                  input_size=50 * horizon,
                  hist_exog_list=pcc_list,
                  futr_exog_list=fcc_list,
                  scaler_type='robust',
                  max_steps=100)
    return NeuralForecast(models=[model], freq='Q')


# df = load_data('../data/FRED/blocked/vintage_2019_01.csv')

# target_df = df[["unique_id", "ds", "y"]]
# covariates = df.drop(columns=["unique_id", "ds", "y"])

# missing_cols = covariates.columns[covariates.isnull().any()]
# past_covariates = covariates.filter(missing_cols)
# future_covariates = covariates.drop(columns=missing_cols)

# pcc_list = past_covariates.columns.tolist()
# fcc_list = future_covariates.columns.tolist()

# df = pd.merge(target_df, future_covariates, left_index=True, right_index=True)

# df_pc = impute_missing_values_ar_multiseries(past_covariates, lags=1)
# df = pd.merge(df, df_pc, left_index=True, right_index=True)
# df = df.iloc[:-1]

# horizon = 5
# nf = create_neural_forecast_model(horizon, pcc_list, fcc_list)
# nf.fit(df=df)

# futr_df = pd.merge(target_df, future_covariates,
#                    left_index=True, right_index=True)
# futr_df = futr_df.drop(columns="y").iloc[-1:]

# Y_hat_df = nf.predict(futr_df=futr_df)

# Y_hat_df.iloc[0, 1]

# Y_hat_df


def forecast_vintages(vintage_files, horizon=1):
    results = {}

    for file_name in vintage_files:
        df = load_data(file_name)

        target_df = df[["unique_id", "ds", "y"]]
        covariates = df.drop(columns=["unique_id", "ds", "y"])

        missing_cols = covariates.columns[covariates.isnull().any()]
        past_covariates = covariates.filter(missing_cols)
        future_covariates = covariates.drop(columns=missing_cols)

        pcc_list = past_covariates.columns.tolist()
        fcc_list = future_covariates.columns.tolist()

        df = pd.merge(target_df, future_covariates,
                      left_index=True, right_index=True)

        df_pc = impute_missing_values_ar_multiseries(past_covariates, lags=1)
        df = pd.merge(df, df_pc, left_index=True, right_index=True)
        df = df.iloc[:-1]

        nf = create_neural_forecast_model(horizon, pcc_list, fcc_list)
        nf.fit(df=df)

        futr_df = pd.merge(target_df, future_covariates,
                           left_index=True, right_index=True)
        futr_df = futr_df.drop(columns="y").iloc[-1:]

        Y_hat_df = nf.predict(futr_df=futr_df)

        forecast_value = Y_hat_df.iloc[0, 1]

        results[file_name] = forecast_value

    return results


vintage_files = [
    '../data/FRED/blocked/vintage_2019_01.csv',
    '../data/FRED/blocked/vintage_2019_02.csv',
    '../data/FRED/blocked/vintage_2019_03.csv',
    '../data/FRED/blocked/vintage_2019_04.csv'
]

forecast_results = forecast_vintages(vintage_files)
for file_name, result in forecast_results.items():
    # Extract year and month from the file path
    year, month = os.path.splitext(os.path.basename(file_name))[
        0].split("_")[1:3]

    print(f"Results for {year}-{month}:")
    print(result)


def get_files_in_directory(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]


data_directory = '../data/FRED/blocked/'
vintage_files = get_files_in_directory(data_directory)
forecast_results = forecast_vintages(vintage_files)

for file_name, result in forecast_results.items():
    # Extract year and month from the file path
    year, month = os.path.splitext(os.path.basename(file_name))[
        0].split("_")[1:3]

    print(f"Results for {year}-{month}:")
    print(result)
