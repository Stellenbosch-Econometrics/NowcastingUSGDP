
# Try and model using NeuralForecast package. NHITS model with covariates.

# MWE for demonstration purposes

# import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# import data
df = pd.read_csv('../data/FRED/blocked/vintage_2019_01.csv')
# df2 = pd.read_csv('../data/FRED/blocked/vintage_2019_02.csv')
# df3 = pd.read_csv('../data/FRED/blocked/vintage_2019_03.csv')
# df4 = pd.read_csv('../data/FRED/blocked/vintage_2019_04.csv')

# rename columns
df = df.rename(columns={"year_quarter": "ds", "GDPC1": "y"})


def nixtla_struct(df):
    """Shape into the required structure for `neuralforecast` model

    Args:
        df (pd.DataFrame): Pandas DataFrame with columns 'ds' and 'y'
    """
    df["unique_id"] = np.ones(len(df))
    df["ds"] = pd.to_datetime(df["ds"])
    df.insert(0, "unique_id", df.pop("unique_id"))
    df.insert(1, "ds", df.pop("ds"))
    df.insert(2, "y", df.pop("y"))


nixtla_struct(df)

# Three problems to solve:
# 1. Need to specify which are the past and future covariates. However, this will change with each of the vintages used.
# 2. Will need to generate values for the missing values that are one period before the missing value for the target variable.
# 3. Need to repeat the nowcasting process for each of the vintages (some iteration over the different files in the directory).

target_cols = ["unique_id", "ds", "y"]

target_df = df[target_cols]
covariates = df.drop(columns=target_cols)

missing_cols = covariates.columns[covariates.isnull().any()]
past_covariates = covariates.filter(missing_cols)
future_covariates = covariates.drop(columns=missing_cols)

# column names of df to list
pcc_list = past_covariates.columns.tolist()
fcc_list = future_covariates.columns.tolist()

df = pd.merge(target_df, future_covariates, left_index=True, right_index=True)

# past covariates are the pain in the ass!
past_covariates = past_covariates[:-1]
past_covariates = past_covariates.iloc[:, :1]

df = pd.merge(df, past_covariates, left_index=True, right_index=True)

# remove the last row of df
df = df.iloc[:-1]

horizon = 2  # day-ahead daily forecast
model = NHITS(h=horizon,
              input_size=10*horizon,
              hist_exog_list=['RPI_m3'],
              futr_exog_list=fcc_list,
              scaler_type='robust',
              max_steps=10  # epochs
              )
nf = NeuralForecast(models=[model], freq='Q')
nf.fit(df=df)

futr_df = pd.merge(target_df, future_covariates,
                   left_index=True, right_index=True)
futr_df = futr_df.drop(columns="y")
futr_df = futr_df.iloc[-1:]

Y_hat_df = nf.predict(futr_df=futr_df)

Y_hat_df
