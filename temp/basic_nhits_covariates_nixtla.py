
# Try and model using NeuralForecast package. NHITS model with covariates.

import logging
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from neuralforecast.models import NHITS
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS
from ray import tune

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

# Suppress ValueWarning from Statsmodels
warnings.simplefilter("ignore", ValueWarning)

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


def has_missing_values_beyond_point(column, df, point_in_time):
    return df.loc[df.index > point_in_time, column].isnull().any()


def separate_covariates(df, point_in_time):
    covariates = df.drop(columns=["unique_id", "ds", "y"])

    past_covariates = []
    future_covariates = []

    for column in covariates.columns:
        if has_missing_values_beyond_point(column, covariates, point_in_time):
            past_covariates.append(column)
        else:
            future_covariates.append(column)

    past_covariates_df = df[past_covariates]
    future_covariates_df = df[future_covariates]

    return past_covariates_df, future_covariates_df


def impute_missing_values_ar_multiseries(data, lags=1):
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


def create_neural_forecast_model(horizon, nhits_config):
    model = AutoNHITS(h=horizon, config=nhits_config, num_samples=5)

    # model_1 = NHITS(h=horizon,
    #                 input_size=70 * horizon,
    #                 hist_exog_list=pcc_list,
    #                 futr_exog_list=fcc_list,
    #                 scaler_type='robust',
    #                 max_steps=20)
    return NeuralForecast(models=[model], freq='Q')


def forecast_vintages(vintage_files, horizon=1):
    results = {}

    for file_name in vintage_files:
        df = load_data(file_name)

        target_df = df[["unique_id", "ds", "y"]]

        point_in_time = df.index[-2]

        past_covariates, future_covariates = separate_covariates(
            df, point_in_time)

        pcc_list = past_covariates.columns.tolist()
        fcc_list = future_covariates.columns.tolist()

        df_fc = impute_missing_values_ar_multiseries(future_covariates, lags=1)
        df_pc = impute_missing_values_ar_multiseries(past_covariates, lags=1)

        df = pd.merge(target_df, df_fc,
                      left_index=True, right_index=True)
        df = pd.merge(df, df_pc, left_index=True, right_index=True)

        if pd.isna(df.loc[df.index[-1], 'y']):
            # Remove the last row
            df = df.iloc[:-1]

        nhits_config = {
            "hist_exog_list": tune.choice([pcc_list]),
            "futr_exog_list": tune.choice([fcc_list]),
            "learning_rate": tune.choice([1e-3]),
            "max_steps": tune.choice([1000]),
            "input_size": tune.choice([5 * horizon]),
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

        nf = create_neural_forecast_model(horizon, nhits_config=nhits_config)
        nf.fit(df=df)

        futr_df = pd.merge(target_df, future_covariates,
                           left_index=True, right_index=True)
        futr_df = futr_df.drop(columns="y").iloc[-1:]

        Y_hat_df = nf.predict(futr_df=futr_df)

        forecast_value = Y_hat_df.iloc[0, 1]

        results[file_name] = forecast_value

        print(nf.models[0].results.get_best_result().config)

    return results

# Run the code over selected vintage files.


vintage_files = [
    '../data/FRED/blocked/vintage_2019_01.csv'
    # '../data/FRED/blocked/vintage_2019_02.csv',
    # '../data/FRED/blocked/vintage_2019_03.csv',
    # '../data/FRED/blocked/vintage_2019_04.csv'
]

forecast_results = forecast_vintages(vintage_files)
for file_name, result in forecast_results.items():
    # Extract year and month from the file path
    year, month = os.path.splitext(os.path.basename(file_name))[
        0].split("_")[1:3]

    print(f"Results for {year}-{month}:")
    print(result)


# def get_files_in_directory(directory):
#     return [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]


# data_directory = '../data/FRED/blocked/'
# vintage_files = get_files_in_directory(data_directory)
# forecast_results = forecast_vintages(vintage_files)

# for file_name, result in forecast_results.items():
#     # Extract year and month from the file path
#     year, month = os.path.splitext(os.path.basename(file_name))[
#         0].split("_")[1:3]

#     print(f"Results for {year}-{month}:")
#     print(result)

# numerical_values = list(forecast_results.values())


# For hyperparameter tuning we get the results from nf.models[0].results.get_best_result().config
