import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.tools import diff


def impute_missing_values_mean(data):
    imputed_data = data.copy()
    imputed_data.apply(lambda x: x.fillna(x.mean()), axis=0)
    return imputed_data


def impute_missing_values_median(data):
    imputed_data = data.copy()
    imputed_data.apply(lambda x: x.fillna(x.median()), axis=0)
    return imputed_data


def impute_missing_values_rolling_mean(data, window_size):
    imputed_data = data.copy()
    imputed_data.fillna(data.rolling(window=window_size,
                        min_periods=1, center=True).mean(), inplace=True)
    return imputed_data


def impute_missing_values_rolling_median(data, window_size):
    imputed_data = data.copy()
    imputed_data.fillna(data.rolling(window=window_size,
                        min_periods=1, center=True).median(), inplace=True)
    return imputed_data


def impute_missing_values_bfill_ffill(data):
    imputed_data = data.copy()
    imputed_data.fillna(method='bfill', inplace=True)
    imputed_data.fillna(method='ffill', inplace=True)
    return imputed_data


def impute_missing_values_interpolate(data, method='linear'):
    imputed_data = data.copy()
    imputed_data.fillna(method='bfill', inplace=True)
    return imputed_data.interpolate(method=method)
