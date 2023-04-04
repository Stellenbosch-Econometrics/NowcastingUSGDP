
# %%
# Import the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import NBEATSModel
from darts import TimeSeries
from darts.metrics import rmse
from darts.dataprocessing.transformers import Scaler
from statsmodels.tsa.ar_model import AutoReg
# from darts.utils import train_test_split
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("darts.models.forecasting.pl_forecasting_module").setLevel(
    logging.CRITICAL)
logging.getLogger("darts.models.forecasting.nbeats").setLevel(logging.CRITICAL)
logging.getLogger("darts.models.forecasting.torch_forecasting_model").setLevel(
    logging.CRITICAL)
logging.getLogger("darts.models.forecasting.torch_forecasting_model").setLevel(
    logging.CRITICAL)


# %%

def load_data(file_path):
    df = (pd.read_csv(file_path)
          .rename(columns={"year_quarter": "date", "GDPC1": "y"})
          .assign(date=lambda x: pd.to_datetime(x["date"])))
    return df


def separate_covariates(df, point_in_time):
    covariates = df.drop(columns=["y"])

    if not point_in_time:
        return df[covariates.columns], df[[]]

    point_in_time = point_in_time[0]

    mask = covariates.apply(
        lambda col: col.loc[col.index >= point_in_time - 1].isnull().any())

    past_covariates = df[mask.index[mask]]
    future_covariates = df[mask.index[~mask]]

    return past_covariates, future_covariates


def impute_missing_values_interpolate(data, method='linear'):
    return (data.select_dtypes(include='number').interpolate(method=method)
            .pipe(lambda x: pd.concat([data['date'], x], axis=1) if 'date' in data.columns else x))


def process_vintage_file(file_path):
    df = load_data(file_path)
    target_df = df[["date", "y"]].iloc[:-1]
    point_in_time = list(df[df['y'].isnull()].index)
    past_covariates, future_covariates = separate_covariates(df, point_in_time)

    past_df = impute_missing_values_interpolate(past_covariates).iloc[:-1]
    futr_df = impute_missing_values_interpolate(future_covariates).iloc[-1:]

    return target_df, past_df, futr_df

# %%


vintage_files = [
    '../data/FRED/blocked/vintage_2019_08.csv',
    '../data/FRED/blocked/vintage_2019_09.csv',
    '../data/FRED/blocked/vintage_2019_10.csv',
    '../data/FRED/blocked/vintage_2019_11.csv',
    '../data/FRED/blocked/vintage_2019_12.csv'
]

target_df, past_df, futr_df = process_vintage_file(vintage_files[1])

# %%

gdp_data = target_df
gdp_data["date"] = pd.to_datetime(gdp_data["date"])
gdp_data.set_index("date", inplace=True)
ts_gdp = TimeSeries.from_dataframe(gdp_data)


# %%
# Make sure the timestamp column has a datetime64 data type
past_df["date"] = pd.to_datetime(past_df["date"])

# Set the timestamp column as the index
past_df.set_index("date", inplace=True)

# Convert the DataFrame to a TimeSeries object
ts_past_covariates = TimeSeries.from_dataframe(past_df)

# %%
# Make sure the timestamp column has a datetime64 data type
futr_df["date"] = pd.to_datetime(futr_df["date"])

# Set the timestamp column as the index
futr_df.set_index("date", inplace=True)

# Convert the DataFrame to a TimeSeries object
ts_future_covariates = TimeSeries.from_dataframe(futr_df)


# %%

# We can also stack the past covariates, I think this is what Darts requires.
# column_time_series = [TimeSeries.from_dataframe(
#     df_past_covariates[[column]]) for column in df_past_covariates.columns]

# stacked_past_covariates = column_time_series[0]
# for ts in column_time_series[1:]:
#     stacked_past_covariates = TimeSeries.stack(stacked_past_covariates, ts)

# %%
# Scale the variables
scaler_gdp, scaler_pc, scaler_fc = Scaler(), Scaler(), Scaler()
gdp_scaled = scaler_gdp.fit_transform(ts_gdp)
pc_scaled = scaler_pc.fit_transform(ts_past_covariates)
fc_scaled = scaler_fc.fit_transform(ts_future_covariates)


# %%
train_gdp, val_gdp = gdp_scaled[:-12], gdp_scaled[-12:]

# %%
train_past_covariates, val_past_covariates = pc_scaled[:-
                                                       12], pc_scaled[-12:]

# %%
train_future_covariates, val_future_covariates = fc_scaled[:-
                                                           12], fc_scaled[-12:]

# %%
model_pastcov = NBEATSModel(
    input_chunk_length=50,
    output_chunk_length=10,
    n_epochs=10,
    # random_state=0,
)

# %%
model_pastcov.fit(
    series=gdp_scaled,
    past_covariates=pc_scaled,
    # future_covariates=train_future_covariates,
    verbose=True,
)

# %%
pred_cov = model_pastcov.predict(
    n=10, series=gdp_scaled, past_covariates=pc_scaled)

# %%
gdp_scaled.plot(label="actual")
pred_cov.plot(label="forecast")
plt.legend()
# %%
