
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


df_past_covariates = impute_missing_values_ar_multiseries(df_past_covariates)


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
# Make sure the timestamp column has a datetime64 data type
df_future_covariates["date"] = pd.to_datetime(df_future_covariates["date"])

# Set the timestamp column as the index
df_future_covariates.set_index("date", inplace=True)

# Convert the DataFrame to a TimeSeries object
ts_future_covariates = TimeSeries.from_dataframe(df_future_covariates)


# %%

# We can also stack the past covariates, I think this is what Darts requires.
column_time_series = [TimeSeries.from_dataframe(
    df_past_covariates[[column]]) for column in df_past_covariates.columns]

stacked_past_covariates = column_time_series[0]
for ts in column_time_series[1:]:
    stacked_past_covariates = TimeSeries.stack(stacked_past_covariates, ts)

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
    n_epochs=100,
    # random_state=0,
)

# %%
model_pastcov.fit(
    series=train_gdp,
    past_covariates=train_past_covariates,
    verbose=True,
)

# %%
pred_cov = model_pastcov.predict(
    n=10, series=train_gdp, past_covariates=train_past_covariates)

# %%
gdp_scaled.plot(label="actual")
pred_cov.plot(label="forecast")
plt.legend()
# %%
