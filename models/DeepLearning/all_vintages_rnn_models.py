

from ray import tune
import time
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.losses.pytorch import MAE
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoRNN, AutoLSTM, AutoGRU, AutoTCN, AutoDilatedRNN

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

def load_data(file_path):
    df = (pd.read_csv(file_path)
          .rename(columns={"year_quarter": "ds", "GDPC1": "y"})
          .assign(unique_id=np.ones(len(pd.read_csv(file_path))),
                  ds=lambda df: pd.to_datetime(df['ds'])))
    columns_order = ["unique_id", "ds", "y"] + \
        [col for col in df.columns if col not in ["unique_id", "ds", "y"]]
    df['ds'] = df['ds'] - pd.Timedelta(days=1)
    return df[columns_order]


def separate_covariates(df, point_in_time):
    covariates = df.drop(columns=["unique_id", "ds", "y"])

    if not point_in_time:
        return df[covariates.columns], df[[]]

    mask = covariates.apply(
        lambda col: col.loc[col.index >= point_in_time - 1].isnull().any())

    past_covariates = df[mask.index[mask]]
    future_covariates = df[mask.index[~mask]]

    return past_covariates, future_covariates


def impute_missing_values_interpolate(data, method='linear'):
    imputed_data = data.copy()
    imputed_data.fillna(method='bfill', inplace=True)
    return imputed_data.interpolate(method=method)


def forecast_vintage(vintage_file, horizon=4):
    results = {}

    df = load_data(vintage_file)

    target_df = df[["unique_id", "ds", "y"]]

    point_in_time = df.index[-2]

    past_covariates, future_covariates = separate_covariates(
        df, point_in_time)

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


    # Checked
    rnn_config = {
        "input_size": tune.choice([4, 4*2, 4* 3, 4*5]), # general rule of thumb -- input size = horizon * 5 -- however, the default for RNN is to use all input history
        "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
        "encoder_n_layers": tune.randint(1, 4), # Normally choice between 1, 2 and 3 is good. Avoid risk of overfitting. 
        "encoder_dropout": tune.choice([0.1, 0.3, 0.5]),
        "context_size": tune.choice([5, 10, 50]),
        "decoder_hidden_size": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32]),
        "hist_exog_list": tune.choice([pcc_list]),
        "futr_exog_list": tune.choice([fcc_list]),
        "max_steps": tune.choice([500, 1000]), # 500 seems to be a good default
        "scaler_type": tune.choice(["robust"]), # this should be robust because of the exogenous variables being included. 
        "random_seed": tune.randint(1, 20), 
    }

    # Checked
    lstm_config = {
        "input_size": tune.choice([4, 4*2, 4* 3, 4*5]), # general rule of thumb -- input size = horizon * 5 -- however, the default for RNN is to use all input history
        "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
        "encoder_n_layers": tune.randint(1, 4), # Normally choice between 1, 2 and 3 is good. Avoid risk of overfitting. 
        "encoder_dropout": tune.choice([0.1, 0.3, 0.5]),
        "context_size": tune.choice([5, 10, 50]),
        "decoder_hidden_size": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32]),
        "hist_exog_list": tune.choice([pcc_list]),
        "futr_exog_list": tune.choice([fcc_list]),
        "max_steps": tune.choice([500, 1000), # 500 seems to be a good default
        "scaler_type": tune.choice(["robust"]), # this should be robust because of the exogenous variables being included. 
        "random_seed": tune.randint(1, 20), 
    }

    # Checked
    gru_config = {
        "input_size": tune.choice([4, 4*2, 4* 3, 4*5]), # general rule of thumb -- input size = horizon * 5 -- however, the default for RNN is to use all input history
        "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
        "encoder_n_layers": tune.randint(1, 4), # Normally choice between 1, 2 and 3 is good. Avoid risk of overfitting. 
        "encoder_dropout": tune.choice([0.1, 0.3, 0.5]),
        "context_size": tune.choice([5, 10, 50]),
        "decoder_hidden_size": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32]),
        "hist_exog_list": tune.choice([pcc_list]),
        "futr_exog_list": tune.choice([fcc_list]),
        "max_steps": tune.choice([500, 1000]), # 500 seems to be a good default
        "scaler_type": tune.choice(["robust"]), # this should be robust because of the exogenous variables being included. 
        "random_seed": tune.randint(1, 20), 
    }
    # Checked   
    tcn_config = {
        "input_size": tune.choice([4, 4*2, 4* 3, 4*5]), # general rule of thumb -- input size = horizon * 5 -- however, the default for RNN is to use all input history
        "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
        "context_size": tune.choice([5, 10, 50]),
        "decoder_hidden_size": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32]),
        "hist_exog_list": tune.choice([pcc_list]),
        "futr_exog_list": tune.choice([fcc_list]),
        "max_steps": tune.choice([500, 1000]), # 500 seems to be a good default
        "scaler_type": tune.choice(["robust"]), # this should be robust because of the exogenous variables being included. 
        "random_seed": tune.randint(1, 20)
    }

    # Checked
    dilated_rnn_config = {
        "input_size": tune.choice([4, 4*2, 4* 3, 4*5]), # general rule of thumb -- input size = horizon * 5 -- however, the default for RNN is to use all input history
        "cell_type": tune.choice(["LSTM", "GRU"]),
        "encoder_hidden_size": tune.choice([50, 100, 200, 300]),
        "dilations": tune.choice([[[1, 2], [4, 8]], [[1, 2, 4, 8]]]),
        "context_size": tune.choice([5, 10, 50]),
        "decoder_hidden_size": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32]),
        "hist_exog_list": tune.choice([pcc_list]),
        "futr_exog_list": tune.choice([fcc_list]),
        "max_steps": tune.choice([500, 1000]), # 500 seems to be a good default
        "scaler_type": tune.choice(["robust"]), # this should be robust because of the exogenous variables being included. 
        "random_seed": tune.randint(1, 20)
    }
    

    models = {  
    "AutoRNN": {"config": rnn_config},
    "AutoLSTM": {"config": lstm_config},
    "AutoGRU": {"config": gru_config},
    "AutoTCN": {"config": tcn_config},
    "AutoDilatedRNN": {"config": dilated_rnn_config}
    }

    model_instances = []

    for model_name, kwargs in models.items():
        print(f"Running model: {model_name}")
        model_class = globals()[model_name]
        # instance = model_class(h=horizon, num_samples=1, search_alg=HyperOptSearch(), loss=MAE(), **kwargs) 
        instance = model_class(h=horizon, num_samples=20, loss=MAE(), **kwargs) 
        model_instances.append(instance)

    nf = NeuralForecast(models=model_instances, freq='Q')
    nf.fit(df=df)

    Y_hat_df = nf.predict(futr_df=futr_df)
    forecast_value = Y_hat_df.iloc[:, 1].values.tolist()

    results[vintage_file] = forecast_value

    Y_hat_df = Y_hat_df.reset_index()

    return Y_hat_df, results

comparison = pd.DataFrame()
results = {}

vintage_files = [
    f'../../data/FRED/blocked/vintage_{year}_{month:02d}.csv'
    for year in range(2018, 2024)
    for month in range(1, 13)
    if not (
        (year == 2018 and month < 5) or
        (year == 2023 and month > 1)
    )
]

total_vintages = len(vintage_files)

start_time_whole = time.time()

for i, vintage_file in enumerate(vintage_files):
    print(f"Processing {vintage_file} ({i+1}/{total_vintages})")
    vintage_comparison, vintage_results = forecast_vintage(vintage_file)

    vintage_file_name = os.path.basename(vintage_file)  
    vintage_file_name = os.path.splitext(vintage_file_name)[0] 
    vintage_comparison = vintage_comparison.assign(vintage_file = vintage_file_name)
    
    comparison = pd.concat([comparison, vintage_comparison], ignore_index=True)
    
    results.update(vintage_results)


end_time_whole = time.time()

time_diff = end_time_whole - start_time_whole
hours, remainder = divmod(time_diff, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Time taken to run the code: {int(hours)} hour(s), {int(minutes)} minute(s), and {seconds:.2f} seconds")

# comparison.to_csv('../DeepLearning/results/all_rnn_models_single_vintage.csv', index=True)
# comparison.to_csv('../DeepLearning/results/all_rnn_models_all_vintages.csv', index=True)
# comparison.to_csv('results/all_rnn_models_all_vintages.csv', index=True)