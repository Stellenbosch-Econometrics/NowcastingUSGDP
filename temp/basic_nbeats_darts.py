
# %%
# Import the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import NBEATSModel
from darts import TimeSeries
from darts.metrics import rmse
from darts.dataprocessing.transformers import Scaler
# from darts.utils import train_test_split
import warnings
import logging


from darts.models import (
    BlockRNNModel,
)


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
df = pd.read_csv('../data/FRED/blocked/vintage_2019_01.csv')
df = df.rename(columns={"year_quarter": "date", "GDPC1": "y"})
df["date"] = pd.to_datetime(df["date"])

gdp_data = df[["date", "y"]]
df = df.loc[:, df.columns != 'y']
# %%

time_column = 'date'
specific_point_in_time = df.iloc[-2][time_column]


def has_missing_values_beyond_point(column, df, point_in_time):
    return df.loc[df[time_column] > point_in_time, column].isnull().any()


past_covariates = []
future_covariates = []

for column in df.columns:
    if column != time_column:
        if has_missing_values_beyond_point(column, df, specific_point_in_time):
            past_covariates.append(column)
        else:
            future_covariates.append(column)

df_past_covariates = df[[time_column] + past_covariates]
df_future_covariates = df[[time_column] + future_covariates]

# %%
gdp_data = gdp_data.iloc[:-1]
df_past_covariates = df_past_covariates.iloc[:-1]

# %%
gdp_data["date"] = pd.to_datetime(gdp_data["date"])
gdp_data.set_index("date", inplace=True)
ts_gdp = TimeSeries.from_dataframe(gdp_data)

# %%
# Make sure the timestamp column has a datetime64 data type
df_past_covariates["date"] = pd.to_datetime(df_past_covariates["date"])

# Set the timestamp column as the index
df_past_covariates.set_index("date", inplace=True)

# Convert the DataFrame to a TimeSeries object
ts_past_covariates = TimeSeries.from_dataframe(df_past_covariates)

# %%

# We can also stack the past covariates, I think this is what Darts requires.
column_time_series = [TimeSeries.from_dataframe(
    df_past_covariates[[column]]) for column in df_past_covariates.columns]

stacked_past_covariates = column_time_series[0]
for ts in column_time_series[1:]:
    stacked_past_covariates = TimeSeries.stack(stacked_past_covariates, ts)

# %%
train_gdp, val_gdp = ts_gdp[:-36], ts_gdp[-36:]

# %%
train_past_covariates, val_past_covariates = ts_past_covariates[:-
                                                                24], ts_past_covariates[-24:]

# %%
model_pastcov = BlockRNNModel(
    model="LSTM",
    input_chunk_length=24,
    output_chunk_length=12,
    n_epochs=100,
    # random_state=0,
)

# %%
model_pastcov.fit(
    series=train_gdp,
    past_covariates=train_past_covariates,
    verbose=False,
)

# %%
# # Calculate the number of time steps required for the past_covariates
# required_covariate_steps = train_gdp.width + model_pastcov.input_chunk_length - 1

# # Slice the ts_past_covariates to have the necessary time steps for the corresponding series
# train_past_covariates_required = ts_past_covariates[:-36].last_n_points(required_covariate_steps)
# val_past_covariates_required = ts_past_covariates[-36:].last_n_points(required_covariate_steps)

# # Combine the train and validation past_covariates
# complete_past_covariates = TimeSeries.from_series(train_past_covariates_required.pd_series().append(val_past_covariates_required.pd_series()))

# # Train the model
# model_pastcov.fit(series=train_gdp, past_covariates=train_past_covariates_required, verbose=False)

# # Predict the next 10 time steps
# pred_cov = model_pastcov.predict(n=10, series=train_gdp, past_covariates=complete_past_covariates)

# %%
pred_cov = model_pastcov.predict(
    n=10, series=train_gdp, past_covariates=val_past_covariates)

# %%
# ts_gdp.plot(label="actual")
pred_cov.plot(label="forecast")
plt.legend()
# %%
