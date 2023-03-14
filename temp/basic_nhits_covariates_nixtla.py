
# Try and model using NeuralForecast package. NHITS model with covariates.

# MWE for demonstration purposes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

df = pd.read_csv('../data/FRED/blocked/vintage_2019_01.csv')
# df2 = pd.read_csv('../data/FRED/blocked/vintage_2019_02.csv')
# df3 = pd.read_csv('../data/FRED/blocked/vintage_2019_03.csv')
# df4 = pd.read_csv('../data/FRED/blocked/vintage_2019_04.csv')

df = df.rename(columns={"year_quarter": "ds", "GDPC1": "y"})
df["unique_id"] = np.ones(len(df))
df["ds"] = pd.to_datetime(df["ds"])
df.insert(0, "unique_id", df.pop("unique_id"))
df.insert(1, "ds", df.pop("ds"))
df.insert(2, "y", df.pop("y"))

# drop the last row of df
# df.drop(df.tail(1).index, inplace=True)

# drop the series with missing values
# df.dropna(axis=1, how='any', inplace=True)

# keep the first 5 columns (as an exercise)
# df = df.iloc[:, :10]
# df.tail()

# extract column names in a list
column_names = df.columns.tolist()

horizon = 2  # day-ahead daily forecast
model = NHITS(h=horizon,
              input_size=10*horizon,
              hist_exog_list=['RPI_m1', 'RPI_m2'],
              scaler_type='robust'
              )
nf = NeuralForecast(models=[model], freq='Q')
nf.fit(df=df)

Y_hat_df = nf.predict()

Y_hat_df["NHITS"]
