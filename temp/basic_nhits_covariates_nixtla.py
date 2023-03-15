
# Try and model using NeuralForecast package. NHITS model with covariates.

# MWE for demonstration purposes
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


def impute_missing_values(df_pc):
    for col in df_pc.columns:
        train = df_pc[df_pc[col].notnull()]
        test = df_pc[df_pc[col].isnull()]

        model = AutoReg(train[col], lags=2)
        result = model.fit()

        forecast = result.predict(
            start=len(train), end=len(train) + len(test) - 1)
        df_pc.loc[test.index, col] = forecast
    df_pc.fillna(method='bfill', inplace=True)
    return df_pc


def create_neural_forecast_model(horizon, pcc_list, fcc_list):
    model = NHITS(h=horizon,
                  input_size=10 * horizon,
                  hist_exog_list=pcc_list,
                  futr_exog_list=fcc_list,
                  scaler_type='robust',
                  max_steps=100)
    return NeuralForecast(models=[model], freq='Q')


df = load_data('../data/FRED/blocked/vintage_2019_01.csv')

target_df = df[["unique_id", "ds", "y"]]
covariates = df.drop(columns=["unique_id", "ds", "y"])

missing_cols = covariates.columns[covariates.isnull().any()]
past_covariates = covariates.filter(missing_cols)
future_covariates = covariates.drop(columns=missing_cols)

pcc_list = past_covariates.columns.tolist()
fcc_list = future_covariates.columns.tolist()

df = pd.merge(target_df, future_covariates, left_index=True, right_index=True)

df_pc = impute_missing_values(past_covariates)
df = pd.merge(df, df_pc, left_index=True, right_index=True)
df = df.iloc[:-1]

horizon = 5
nf = create_neural_forecast_model(horizon, pcc_list, fcc_list)
nf.fit(df=df)

futr_df = pd.merge(target_df, future_covariates,
                   left_index=True, right_index=True)
futr_df = futr_df.drop(columns="y").iloc[-1:]

Y_hat_df = nf.predict(futr_df=futr_df)
Y_hat_df
