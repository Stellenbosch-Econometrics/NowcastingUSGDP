

from ray import tune
import logging
import os
import numpy as np
import pandas as pd
# import warnings
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoRNN
from neuralforecast.losses.numpy import mse, mae

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


vintage_files = [
    f'../../data/FRED/blocked/vintage_{year}_{month:02d}.csv'
    for year in range(2018, 2024)
    for month in range(1, 13)
    if not (
        (year == 2018 and month < 5) or
        (year == 2023 and month > 2)
    )
]

### Cross-validation ###

file_path = vintage_files[-1]

df = load_data(file_path)
df['ds'] = df['ds'] - pd.Timedelta(days=1)
target_df = df[["unique_id", "ds", "y"]]

point_in_time = df.index[-2]

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

model = AutoRNN(h=1, config=config, num_samples=5)

nf = NeuralForecast(models=[model], freq='Q')
# nf.fit(df=df)

fcst_df = nf.cross_validation(df=df, n_windows=150, step_size=1)

fcst_df = fcst_df.iloc[:, :5]
print(fcst_df)


plt.figure(figsize=(10, 6))
plt.plot(fcst_df['ds'], fcst_df['y'], label='y',
         marker='o', linestyle='-', markersize=2)
plt.plot(fcst_df['ds'], fcst_df['AutoRNN'],
         label='AutoRNN', marker='o', linestyle='--', markersize=2)

plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Actual vs. RNN estimate')
plt.legend()

plt.show()

### Model performance ###


def evaluate(df):
    eval_ = {}
    models = df.loc[:, ~df.columns.str.contains(
        'unique_id|y|ds|cutoff')].columns
    for model in models:
        eval_[model] = {}
        for metric in [mse, mae]:
            eval_[model][metric.__name__] = metric(
                df['y'].values, df[model].values)
    eval_df = pd.DataFrame(eval_).rename_axis('metric')
    return eval_df


fcst_df.groupby('ds').apply(lambda df: evaluate(df))

# This gives me the individual performance metrics for each time period. 

# Calculate MAE and MSE for each date
eval_df = fcst_df.groupby('ds').apply(lambda df: evaluate(df))

# Extracting data for the plot
dates = eval_df.index.get_level_values('ds').unique()
mae_values = eval_df.loc[pd.IndexSlice[:, 'mae'], :].values.ravel()
mse_values = eval_df.loc[pd.IndexSlice[:, 'mse'], :].values.ravel()

# Plotting MAE over time
plt.figure(figsize=(10, 6))
plt.plot(dates, mae_values, label='MAE', marker='o', linestyle='-', markersize=2)

plt.xlabel('Date')
plt.ylabel('MAE')
plt.title('Mean Absolute Error over time')
plt.legend()

plt.show()

# Plotting MSE over time
plt.figure(figsize=(10, 6))
plt.plot(dates, mse_values, label='MSE', marker='o', linestyle='-', markersize=2)

plt.xlabel('Date')
plt.ylabel('MSE')
plt.title('Mean Squared Error over time')
plt.legend()

plt.show()



# TODO: Get the average performance metrics
