
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

# Main problems are
# 1. Imputation of missing values
# 2. Need to repeat the nowcasting process for each of the vintages (some iteration over the different files in the directory).

# target vs covariates
target_cols = ["unique_id", "ds", "y"]
target_df = df[target_cols]
covariates = df.drop(columns=target_cols)

# past vs future covariates
missing_cols = covariates.columns[covariates.isnull().any()]
past_covariates = covariates.filter(missing_cols)
future_covariates = covariates.drop(columns=missing_cols)

pcc_list = past_covariates.columns.tolist()
fcc_list = future_covariates.columns.tolist()

# future covariates at same date as target
df = pd.merge(target_df, future_covariates, left_index=True, right_index=True)

# impute / forecast missing values -- this is the simplest approach.
past_covariates.fillna(method='ffill', inplace=True)
past_covariates.fillna(method='bfill', inplace=True)

df = pd.merge(df, past_covariates, left_index=True, right_index=True)
df = df.iloc[:-1]

horizon = 2
model = NHITS(h=horizon,
              input_size=10*horizon,
              hist_exog_list=pcc_list,
              futr_exog_list=fcc_list,
              scaler_type='robust',
              max_steps=100  # epochs
              )
nf = NeuralForecast(models=[model], freq='Q')
nf.fit(df=df)

futr_df = pd.merge(target_df, future_covariates,
                   left_index=True, right_index=True)
futr_df = futr_df.drop(columns="y")
futr_df = futr_df.iloc[-1:]

Y_hat_df = nf.predict(futr_df=futr_df)

Y_hat_df
