
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning)


def load_data(file_path):
    return (pd.read_csv(file_path)
            .rename(columns={"year_quarter": "ds", "GDPC1": "y"})
            .assign(unique_id=np.ones(len(pd.read_csv(file_path))),
                    ds=lambda df: pd.to_datetime(df['ds']))
            .reorder_columns(["unique_id", "ds", "y"]))


def separate_covariates(df, point_in_time):
    covariates = df.drop(columns=["unique_id", "ds", "y"])

    if not point_in_time:
        return df[covariates.columns], df[[]]

    point_in_time = point_in_time[0]

    mask = covariates.apply(lambda col: col.loc[col.index >= point_in_time - 1].isnull().any())

    past_covariates = df[mask.index[mask]]
    future_covariates = df[mask.index[~mask]]

    return past_covariates, future_covariates


def impute_missing_values_interpolate(data, method='linear'):
    return data.interpolate(method=method).bfill()

def impute_missing_values_ar_multiseries(data, lags=1):
    def impute_col(col, lags):
        if col.isnull().any():
            not_null_indices = col.notnull()
            train = col.loc[not_null_indices]
            null_indices = col.isnull()

            model = AutoReg(train, lags=lags)
            result = model.fit()

            for index, value in col.loc[null_indices].iteritems():
                if index - lags < 0:
                    available_data = col.loc[:index - 1].values
                else:
                    available_data = col.loc[index - lags:index - 1].values
                if np.isnan(available_data).any():
                    continue
                forecast = result.predict(start=len(train), end=len(train))
                col.loc[index] = forecast.iloc[0]
        return col

    return data.apply(lambda col: impute_col(col, lags))


def process_vintage_file(file_path):
    df = load_data(file_path)
    target_df = df[["unique_id", "ds", "y"]]
    point_in_time = list(df[df['y'].isnull()].index)
    past_covariates, future_covariates = separate_covariates(df, point_in_time)

    df_pc = impute_missing_values_interpolate(past_covariates)
    df_fc = impute_missing_values_interpolate(future_covariates)

    df = (target_df
          .merge(df_fc, left_index=True, right_index=True)
          .merge(df_pc, left_index=True, right_index=True)
          .dropna(subset=['y']))

    futr_df = (target_df
               .merge(future_covariates, left_index=True, right_index=True)
               .drop(columns="y")
               .iloc[-1:])

    return df, futr_df



vintage_files = [
    '../data/FRED/blocked/vintage_2019_08.csv',
    '../data/FRED/blocked/vintage_2019_09.csv',
    '../data/FRED/blocked/vintage_2019_10.csv'
]

df, futr_df = process_vintage_file(vintage_files[2])

missing_values_count = df.isnull().sum().sum()
print(f"There are {missing_values_count} missing values in the DataFrame.")

missing_columns = df.columns[df.isnull().any()]
print("Columns with missing values:", missing_columns)

# There are still columns with missing values at the last value. Need to figure this out.
