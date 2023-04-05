

### Package imports ###

from statsmodels.tools.sm_exceptions import ValueWarning
from ray import tune
import pickle
import logging
import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoRNN
#from neuralforecast.tsdataset import TimeSeriesDataset
#from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.losses.pytorch import MAE
# os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

### Ignore warnings ###

warnings.simplefilter("ignore", ValueWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

### Data preprocessing ###


def load_data(file_path):
    df = (pd.read_csv(file_path)
          .rename(columns={"year_quarter": "ds", "GDPC1": "y"})
          .assign(unique_id=np.ones(len(pd.read_csv(file_path))),
                  ds=lambda df: pd.to_datetime(df['ds'])))
    columns_order = ["unique_id", "ds", "y"] + \
        [col for col in df.columns if col not in ["unique_id", "ds", "y"]]
    return df[columns_order]


def separate_covariates(df, point_in_time):
    covariates = df.drop(columns=["unique_id", "ds", "y"])

    if not point_in_time:
        return df[covariates.columns], df[[]]

    point_in_time = point_in_time[0]

    mask = covariates.apply(
        lambda col: col.loc[col.index >= point_in_time - 1].isnull().any())

    past_covariates = df[mask.index[mask]]
    future_covariates = df[mask.index[~mask]]

    return past_covariates, future_covariates


def impute_missing_values_interpolate(data, method='linear'):
    imputed_data = data.copy()
    imputed_data.fillna(method='bfill', inplace=True)
    return imputed_data.interpolate(method=method)


def process_vintage_file(file_path):
    df = load_data(file_path)
    target_df = df[["unique_id", "ds", "y"]]
    point_in_time = list(df[df['y'].isnull()].index)
    past_covariates, future_covariates = separate_covariates(df, point_in_time)

    df_pc = impute_missing_values_interpolate(past_covariates)
    df_fc = impute_missing_values_interpolate(future_covariates)

    pcc_list = past_covariates.columns.tolist()
    fcc_list = future_covariates.columns.tolist()

    df = (target_df
          .merge(df_fc, left_index=True, right_index=True)
          .merge(df_pc, left_index=True, right_index=True)
          .iloc[:-1])

    futr_df = (target_df
               .merge(future_covariates, left_index=True, right_index=True)
               .drop(columns="y")
               .iloc[-1:])

    return df, futr_df, pcc_list, fcc_list


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


#### RNN model tuning ####

df, futr_df, pcc_list, fcc_list = process_vintage_file(
    "../../../data/FRED/blocked/vintage_2018_05.csv")

save_object(df, 'pickle_files/rnn_df.pickle')
save_object(futr_df, 'pickle_files/rnn_futr_df.pickle')
save_object(pcc_list, 'pickle_files/rnn_pcc_list.pickle')
save_object(fcc_list, 'pickle_files/rnn_fcc_list.pickle')

# forecast horizon
horizon = 4

# configuration possibilities
config = {
    "hist_exog_list": tune.choice([pcc_list]),
    "futr_exog_list": tune.choice([fcc_list]),
    "learning_rate": tune.choice([1e-3]),
    "max_steps": tune.choice([500]),
    "input_size": tune.choice([100]),
    "encoder_hidden_size": tune.choice([256]),
    "val_check_steps": tune.choice([1]),
    "random_seed": tune.randint(1, 10),
}


# general rule is to set num_samples > 20

#model = AutoRNN(h=horizon)
model = AutoRNN(h=horizon, loss=MAE(), config=config, num_samples=30)
nf = NeuralForecast(models=[model], freq='Q')
nf.fit(df=df)


# Storage of the best hyperparameter values
best_config = nf.models[0].results.get_best_result().config

save_object(best_config, 'pickle_files/rnn_best_config.pickle')


# save the specfic model
nf.save(path='/saved_models/rnn/',
        model_index=None,
        overwrite=True,
        save_dataset=True)
