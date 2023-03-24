
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning)


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={"year_quarter": "ds", "GDPC1": "y"})
    df["unique_id"] = np.ones(len(df))
    df["ds"] = pd.to_datetime(df["ds"])
    df = df[["unique_id", "ds", "y"] +
            [col for col in df.columns if col not in ["unique_id", "ds", "y"]]]
    return df


def separate_covariates(df, point_in_time):
    covariates = df.drop(columns=["unique_id", "ds", "y"])

    if not point_in_time:
        past_covariates_df = df[covariates.columns]
        future_covariates_df = df[[]]
    else:
        point_in_time = point_in_time[0]
        past_covariates = [
            col for col in covariates.columns if df.loc[df.index >= point_in_time, col].isnull().any()]
        future_covariates = [
            col for col in covariates.columns if col not in past_covariates]

        past_covariates_df = df[past_covariates]
        future_covariates_df = df[future_covariates]

    return past_covariates_df, future_covariates_df


def impute_missing_values_interpolate(data, method='linear'):
    imputed_data = data.copy()
    imputed_data.interpolate(method=method, inplace=True)
    imputed_data.bfill(inplace=True)
    return imputed_data


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


def process_vintage_file(file_path):
    df = load_data(file_path)
    target_df = df[["unique_id", "ds", "y"]]
    point_in_time = list(df[df['y'].isnull()].index)
    past_covariates, future_covariates = separate_covariates(df, point_in_time)

    df_pc = impute_missing_values_interpolate(past_covariates)
    df_fc = impute_missing_values_interpolate(future_covariates)

    df = pd.merge(target_df, df_fc, left_index=True, right_index=True)
    df = pd.merge(df, df_pc, left_index=True, right_index=True)

    if pd.isna(df.loc[df.index[-1], 'y']):
        # Remove the last row
        df = df.iloc[:-1]

    futr_df = pd.merge(target_df, future_covariates,
                       left_index=True, right_index=True)
    futr_df = futr_df.drop(columns="y").iloc[-1:]

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
