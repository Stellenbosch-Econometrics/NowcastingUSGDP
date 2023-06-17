

### Package imports ###

from ray import tune
import time
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoMLP, AutoNBEATS, AutoNBEATSx, AutoNHITS

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
    # df['ds'] = df['ds'] - pd.Timedelta(days=1)
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

def forecast_vintage(vintage_file, horizon=1):
    results = {}

    df = load_data(vintage_file)

    target_df = df[["unique_id", "ds", "y"]]

    point_in_time = df.index[-2]  # explain later

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

    mlp_config = {
        "input_size": tune.choice([1, 2, 4, 4*2, 4*3]),
        "hidden_size": tune.choice([256, 512, 1024]),
        "num_layers": tune.randint(2, 6),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "random_seed": tune.randint(1, 20),
        "hist_exog_list": tune.choice([pcc_list]),
        "futr_exog_list": tune.choice([fcc_list]),
        "max_steps": tune.choice([1000]),
        "scaler_type": tune.choice(["robust"]),
    }

    nbeats_config = {
        "input_size": tune.choice([1, 2, 4, 4*2, 4*3]), # think about this tuning choice
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice(["robust"]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        # "dropout_prob_theta": tune.choice([0.5]),
        # "early_stop_patience_steps": tune.choice([-1, 2, 5]),
        #"hist_exog_list": tune.choice([pcc_list]),
        #"futr_exog_list": tune.choice([fcc_list]),
        "max_steps": tune.choice([1000]),
        "random_seed": tune.randint(1, 20)
    }

    nbeatsx_config = {
        "input_size": tune.choice([1, 2, 4, 4*2, 4*3]), # think about this tuning choice
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice(["robust"]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "dropout_prob_theta": tune.choice([0.5]),
        # "early_stop_patience_steps": tune.choice([-1, 2, 5]),
        "hist_exog_list": tune.choice([pcc_list]),
        "futr_exog_list": tune.choice([fcc_list]),
        "max_steps": tune.choice([1000]),
        "random_seed": tune.randint(1, 20)
    }

    nhits_config = {
        # "n_pool_kernel_size": tune.choice(
        #     [[2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]
        # ),
        # "n_freq_downsample": tune.choice(
        #     [
        #         [168, 24, 1],
        #         [24, 12, 1],
        #         [180, 60, 1],
        #         [60, 8, 1],
        #         [40, 20, 1],
        #         [1, 1, 1],
        #     ]
        # ),
        "input_size": tune.choice([1, 2, 4, 4*2, 4*3]), # think about this tuning choice
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice(["robust"]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "dropout_prob_theta": tune.choice([0.5]),
        #"early_stop_patience_steps": tune.choice([-1, 2, 5]),
        "hist_exog_list": tune.choice([pcc_list]),
        "futr_exog_list": tune.choice([fcc_list]),
        "max_steps": tune.choice([1000]),
        "random_seed": tune.randint(lower=1, upper=20),
    }

    # Define models and their configurations
    models = {
        #"AutoMLP": {"config": mlp_config},
        "AutoNBEATS": {"config": nbeats_config},
        #"AutoNBEATSx": {"config": nbeatsx_config},
        #"AutoNHITS": {"config": nhits_config},
    }

    # Initialize and fit all models
    model_instances = []

    for model_name, kwargs in models.items():
        print(f"Running model: {model_name}")
        model_class = globals()[model_name]
        instance = model_class(h=horizon, num_samples=1,
                               verbose=False, **kwargs)
        model_instances.append(instance)

    n_time = len(df.ds.unique())
    val_size = int(.2 * n_time)

    nf = NeuralForecast(models=model_instances, freq='Q')
    nf.fit(df=df, val_size=val_size)

    Y_hat_df = nf.predict(futr_df=futr_df)
    Y_hat_df = Y_hat_df.reset_index()
    forecast_value = Y_hat_df.iloc[:, 1].values.tolist()
    Y_hat_df['ds'] = Y_hat_df['ds'] + pd.Timedelta(days = 1)

    results[vintage_file] = forecast_value

    return Y_hat_df, results


comparison = pd.DataFrame()
results = {}

vintage_files = [
    f'data/FRED/blocked/vintage_{year}_{month:02d}.csv'
    for year in range(2018, 2024)
    for month in range(1, 13)
    if not (
        (year == 2018 and month < 5) or
        (year == 2023 and month > 5)
    )
]

total_vintages = len(vintage_files)

start_time_whole = time.time()

def write_to_csv(df, block_number):
    df.to_csv(f'mlp_results_{block_number}.csv', index=False)

block_size = 1
for i in range(0, len(vintage_files), block_size):
    block = vintage_files[i:i+block_size]
    for j, vintage_file in enumerate(block):
        print(
            f"Processing {vintage_file} ({j+1}/{block_size}) in block {i//block_size + 1}")
        vintage_comparison, vintage_results = forecast_vintage(vintage_file)

        vintage_file_name = os.path.basename(vintage_file)
        vintage_file_name = os.path.splitext(vintage_file_name)[0]
        vintage_comparison = vintage_comparison.assign(
            vintage_file=vintage_file_name)

        comparison = pd.concat(
            [comparison, vintage_comparison], ignore_index=True)

        results.update(vintage_results)

    write_to_csv(comparison, i//block_size + 1)

end_time_whole = time.time()

time_diff = end_time_whole - start_time_whole
hours, remainder = divmod(time_diff, 3600)
minutes, seconds = divmod(remainder, 60)

print(
    f"Time taken to run the code: {int(hours)} hour(s), {int(minutes)} minute(s), and {seconds:.2f} seconds")

