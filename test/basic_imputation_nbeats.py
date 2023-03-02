
# %%
## Import the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import NBEATSModel
from darts import TimeSeries
from darts.metrics import rmse

df = pd.read_csv('../data/FRED/blocked/vintage_2019_01.csv')

gdp_data = df[["year_quarter", "GDPC1"]]
covariate_data = df.loc[:, df.columns != 'GDPC1']

# Last observation carried forward -- simplest method to use
covariate_data.fillna(method='ffill', inplace=True)

# Drop columns of df with missing values
covariate_data.dropna(axis=1, how='any', inplace=True) # Drops about 221 columns -- check this

# %% 
## Place into a TimeSeries object

gdp_series = TimeSeries.from_dataframe(gdp_data, time_col='year_quarter')
covariate_series = TimeSeries.from_dataframe(covariate_data, time_col="year_quarter")

# %%
gdp_train, gdp_val = gdp_series.split_before(0.8) 
cov_train, cov_val = covariate_series.split_before(0.8)

model_nbeats = NBEATSModel(
    input_chunk_length=80, output_chunk_length=1, n_epochs=100, random_state=42
)

model_nbeats.fit(gdp_train, 
                 past_covariates=cov_train,
                 verbose=True)
# %%
pred_beats = model_nbeats.predict(series = gdp_train,
                                  past_covariates=cov_train, 
                                  n = 1)

# %%
#gdp_series[:-20].plot()
#pred_beats.plot()

# %%
pred_series = model_nbeats.historical_forecasts(
    series = gdp_train,
    past_covariates = cov_train,
    # start=pd.Timestamp("1998-06-01"),
    forecast_horizon=1,
    retrain=False, # Should only be set to false in cases where training is time consuming. 
    verbose=False,
)
# %%
def display_forecast(pred_series, ts_transformed, forecast_type, start_date=None):
    plt.figure(figsize=(10, 6))
    if start_date:
        ts_transformed = ts_transformed.drop_before(start_date)
    ts_transformed.univariate_component(0).plot(label="actual")
    pred_series.plot(label=("historic " + forecast_type + " forecasts"))
    plt.title(
        "RMSE: {}".format(rmse(ts_transformed.univariate_component(0), pred_series))
    )
    plt.legend()

display_forecast(pred_series, gdp_series[:-13], "1 quarter")
# %%
