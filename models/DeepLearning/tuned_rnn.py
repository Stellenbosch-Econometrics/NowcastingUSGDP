
# This code runs the full RNN model across all vintages. The model is tuned for each vintage.


### Package imports ###

# from statsmodels.tools.sm_exceptions import ValueWarning
from ray import tune
import logging
import os
import numpy as np
import pandas as pd
# import warnings
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoRNN
# from neuralforecast.losses.pytorch import MAE
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

### Ignore warnings ###

# warnings.simplefilter("ignore", ValueWarning)
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

    # point_in_time = point_in_time[0]

    mask = covariates.apply(
        lambda col: col.loc[col.index >= point_in_time - 1].isnull().any())

    past_covariates = df[mask.index[mask]]
    future_covariates = df[mask.index[~mask]]

    return past_covariates, future_covariates


def impute_missing_values_interpolate(data, method='linear'):
    imputed_data = data.copy()
    imputed_data.fillna(method='bfill', inplace=True)
    return imputed_data.interpolate(method=method)


### Forecast across vintages ###

def forecast_vintages(vintage_files, horizon=1):
    results = {}

    for file_path in vintage_files:

        df = load_data(file_path)
        df['ds'] = df['ds'] - pd.Timedelta(days=1)
        target_df = df[["unique_id", "ds", "y"]]

        point_in_time = df.index[-2]
        # point_in_time = list(df[df['y'].isnull()].index)

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
                   .merge(future_covariates, left_index=True, right_index=True)
                   .drop(columns="y")
                   .iloc[-1:])

        # default_config = {
        #     "input_size_multiplier": [-1, 4, 16, 64],
        #     "h": None,
        #     "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
        #     "encoder_n_layers": tune.randint(1, 4),
        #     "context_size": tune.choice([5, 10, 50]),
        #     "decoder_hidden_size": tune.choice([64, 128, 256, 512]),
        #     "learning_rate": tune.loguniform(1e-4, 1e-1),
        #     "max_steps": tune.choice([500, 1000]),
        #     "batch_size": tune.choice([16, 32]),
        #     "loss": None,
        #     "random_seed": tune.randint(1, 20),
        # }

        config = {
            "hist_exog_list": tune.choice([pcc_list]),
            "futr_exog_list": tune.choice([fcc_list]),
            # "learning_rate": tune.choice([1e-3]),
            "max_steps": tune.choice([500]),
            # "input_size": tune.choice([150]),
            # "encoder_hidden_size": tune.choice([256]),
            # "val_check_steps": tune.choice([1]),
            # "random_seed": tune.randint(1, 10),
            "scaler_type": tune.choice(["robust"])
        }

        model = AutoRNN(h=horizon,
                        config=config, num_samples=20)

        nf = NeuralForecast(models=[model], freq='Q')
        nf.fit(df=df)

        Y_hat_df = nf.predict(futr_df=futr_df)

        forecast_value = Y_hat_df.iloc[:, 1].values.tolist()

        results[file_path] = forecast_value

    return results

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

### Capture all the results and print ###

forecast_results = forecast_vintages(vintage_files[0]) # Just the first vintage
# forecast_results = forecast_vintages(vintage_files) # All vintages

# Step 1: Create an empty DataFrame with desired column names
columns = [
    f"{os.path.splitext(os.path.basename(file_name))[0].split('_')[1]}-{os.path.splitext(os.path.basename(file_name))[0].split('_')[2]}" for file_name in forecast_results.keys()]
df_results = pd.DataFrame(columns=columns)

# Step 2: Fill the DataFrame with the corresponding values
for file_name, result in forecast_results.items():
    year_month = f"{os.path.splitext(os.path.basename(file_name))[0].split('_')[1]}-{os.path.splitext(os.path.basename(file_name))[0].split('_')[2]}"
    df_results[year_month] = result

# print(df_results)


results = pd.DataFrame(df_results)
results.to_csv('../DeepLearning/results/rnn_results.csv', index=True)


