
# Only calculate for the final usable vintage
# TODO: Nowcast and 1,2,3 period ahead forecast + performance metric

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
    df['ds'] = df['ds'] - pd.Timedelta(days=1)
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

vintage_of_interest = vintage_files[-12]
latest_vintage = vintage_files[-1]

### Forecast across last usable vintage ###


def forecast_vintage(vintage_file, horizon=4):
    results = {}

    df = load_data(vintage_file)

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

    futr_df = (target_df
               .merge(future_covariates, left_index=True, right_index=True)
               .drop(columns="y")
               .iloc[-1:])

    config = {
        "hist_exog_list": tune.choice([pcc_list]),
        "futr_exog_list": tune.choice([fcc_list]),
        "max_steps": tune.choice([500]),
        "scaler_type": tune.choice(["robust"])
    }

    model = AutoRNN(h=horizon,
                    config=config, num_samples=5)

    nf = NeuralForecast(models=[model], freq='Q')
    nf.fit(df=df)

    Y_hat_df = nf.predict(futr_df=futr_df)

    forecast_value = Y_hat_df.iloc[:, 1].values.tolist()

    results[vintage_file] = forecast_value

    return results


# Generate forecasts for the vintage_of_interest
vintage_of_interest_forecast = forecast_vintage(vintage_of_interest)

vintage_of_interest_df = load_data(vintage_of_interest)

# Load latest_vintage data
latest_vintage_df = load_data(latest_vintage)

# Extract the true y values from the latest_vintage_df
true_y_values = latest_vintage_df.loc[latest_vintage_df.index.isin(
    latest_vintage_df.index[-4:]), 'y'].tolist()

# Extract the date column (ds) from the latest_vintage_df
date_column = latest_vintage_df.loc[latest_vintage_df.index.isin(
    latest_vintage_df.index[-4:]), 'ds'].tolist()

# Create a DataFrame with the date column, true y values, and forecasted values
comparison_df = pd.DataFrame({
    'ds': date_column,
    'true_y': true_y_values,
    'forecasted_y': vintage_of_interest_forecast[vintage_of_interest]
})

# Shift the forecasted_y column back by one time period
comparison_df['forecasted_y_shifted'] = comparison_df['forecasted_y'].shift(-1)

# Drop the original forecasted_y column
comparison_df.drop(columns='forecasted_y', inplace=True)

print(comparison_df)


## simple plot

# Extracting data for the plot
dates = comparison_df['ds']
y_true = comparison_df['true_y']
y_forecasted = comparison_df['forecasted_y_shifted']

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(dates, y_true, label='True y', marker='o', linestyle='-', markersize=2)
plt.plot(dates, y_forecasted, label='Forecasted y', marker='o', linestyle='--', markersize=2)

plt.xlabel('Date')
plt.ylabel('Values')
plt.title('True y vs. Forecasted y')
plt.legend()

plt.show()
