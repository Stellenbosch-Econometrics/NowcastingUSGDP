
# %%

## Going to use the NeuralForecast package from Nixtla. Focus on the NBEATS model no covariates. Univariate modelling. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NHITS, RNN

df = pd.read_csv('../data/FRED/blocked/vintage_2019_01.csv')
gdp_data = df[["year_quarter", "GDPC1"]]
gdp_data = gdp_data.rename(columns={"year_quarter": "ds", "GDPC1": "y"})
# insert `unique_id` column to column of ones
gdp_data["unique_id"] = np.ones(len(gdp_data))
# move unique_id to first column
gdp_data = gdp_data[["unique_id", "ds", "y"]]
# set `ds` to datetime format
gdp_data["ds"] = pd.to_datetime(gdp_data["ds"])
# remove all missing values (rows)
Y_df = gdp_data.dropna()
Y_df.tail()

# This should be the preliminary steps required. The last value is missing, might want

# %%
horizon = 6

models = [NHITS(h=horizon,                   # Forecast horizon
                input_size=35 * horizon,     # Length of input sequence
                max_steps=80,                # Number of steps to train
                #n_freq_downsample=[2, 1, 1] # Downsampling factors for each stack output
                ) 
          ]

fcst = NeuralForecast(models=models, freq='Q')

fcst.fit(df=Y_df)

# %%
Y_hat_df = fcst.predict()

# %%
Y_hat_df = Y_hat_df.reset_index()
Y_hat_df.head()


# %%
fig, ax = plt.subplots(1, 1, figsize = (20, 7))
plot_df = pd.concat([Y_df, Y_hat_df]).set_index('ds') # Concatenate the train and forecast dataframes
plot_df[['y', 'NHITS']].plot(ax=ax, linewidth=2)

ax.set_title('GDP forecast', fontsize=22)
ax.set_ylabel('GDP growth rate', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
# %%
