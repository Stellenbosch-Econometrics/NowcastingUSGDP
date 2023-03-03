
## %%

# Try and model using NeuralForecast package. NHITS model with past covariates. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


df = pd.read_csv('../data/FRED/blocked/vintage_2019_01.csv')

# drop the last row of df
df.drop(df.tail(1).index, inplace=True)

# drop the series with missing values
df.dropna(axis=1, how='any', inplace=True)

df = df.rename(columns={"year_quarter": "ds", "GDPC1": "y"})
# insert `unique_id` column to column of ones
df["unique_id"] = np.ones(len(df))
# set `ds` to datetime format
df["ds"] = pd.to_datetime(df["ds"])
# move unique_id, ds, y to first three columns
df.insert(0, "unique_id", df.pop("unique_id"))
df.insert(1, "ds", df.pop("ds"))
df.insert(2, "y", df.pop("y"))

# keep the first 5 columns (as an exercise)
df = df.iloc[:, :5]
df.tail()

# plt.figure(figsize=(15,5))
# plt.plot(df[df['unique_id']==1.0]['ds'], df[df['unique_id']==1.0]['y'])
# plt.xlabel('Date')
# plt.ylabel('GDP growth rate')
# plt.grid()

horizon = 2 # day-ahead daily forecast
model = NHITS(h = 2,
              input_size = 20,
              hist_exog_list = ['RPI_m1', 'RPI_m2'],
              scaler_type = 'robust'
            )
nf = NeuralForecast(models=[model], freq='Q')
nf.fit(df=df)

Y_hat_df = nf.predict()

Y_hat_df = Y_hat_df.reset_index()
Y_hat_df.head()